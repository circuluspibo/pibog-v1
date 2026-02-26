import pyaudio
import numpy as np
import time
import os
import datetime
import asyncio
import threading
from openwakeword.model import Model
from openvino.runtime import Core
from scipy.io.wavfile import write as write_wav
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests # <--- requests 라이브러리 추가
import uvicorn
from queue import Queue

processed_queue = Queue(maxsize=5)

# --- 1. 설정 및 전역 변수 ---
# OpenWakeWord 설정
WAKEWORD_MODEL_NAME = "./models/alexa_v0.1.xml"
WAKEWORD_THRESHOLD = 0.4

# ACLNet (Acoustic Event Classification) 설정
ACLNET_MODEL_XML = "./models/aclnet.xml"
ACLNET_CLASSES_TXT = "./models/aclnet_53cl.txt"
DEVICE = "NPU" # NPU 사용 대신 범용적인 CPU 사용
INPUT_SAMPLE_RATE = 16000

# VAD 및 녹음 설정
VAD_ACTIVATION_THRESHOLD = 500  # VAD: 소리 감지 임계값 (pyaudio int16 기준. 약 500~1000)
SILENCE_DURATION = 2            # 무음 지속 시간 (초)
MAX_RECORDING_DURATION = 15     # 최대 녹음 시간 (초)
RECORDING_OUTPUT_DIR = "recordings_wakeword"

# STT 서버 설정 (사용자 요청 curl 명령어 기반)
STT_API_URL = 'http://127.0.0.1:59532/v1/stt'
STT_LANG = 'ko'
STT_IS_PLAY = 0

# PyAudio 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 16000                     # 1280 samples = 0.08 seconds @ 16kHz

# 상태 관리 변수
is_recording = False
recorded_frames = []
last_sound_time = time.time()
recording_start_time = time.time()
model_input_length = None       # ACLNet 모델의 입력 길이에 따라 동적으로 설정

# 디렉토리 생성
os.makedirs(RECORDING_OUTPUT_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,  # 쿠키나 자격 증명 허용
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # 허용할 HTTP 메소드
    allow_headers=["*"],  # 모든 헤더 허용
)

def queue_to_list(queue: Queue):
    return [queue.get() for _ in range(queue.qsize())]

@app.post("/rec/start")
async def manual_record_start():
    global is_recording
    
    if is_recording:
        return JSONResponse({
            "result": False,
            "message": "Already recording"
        })

    set_recording_active()

    return {
        "result": True,
        "message": "Recording started manually"
    }


@app.post("/rec/stop")
async def manual_record_stop():
    global is_recording
    
    if not is_recording:
        return JSONResponse({
            "result": False,
            "message": "Not currently recording"
        })

    stop_recording_and_save()

    return {
        "result": True,
        "message": "Recording stopped and saved"
    }

@app.get("/sound")
async def sound():
    return { "result" : True, "data" : queue_to_list(processed_queue) } 

@app.get("/listen")
async def listen():
    """마이크 입력을 시작하는 엔드포인트, 별도의 스레드에서 run_mic_loop() 실행"""
    
    def start_mic_loop():
        run_mic_loop()

    mic_thread = threading.Thread(target=start_mic_loop)
    mic_thread.daemon = True  # 프로그램 종료 시 스레드도 종료되도록 설정
    mic_thread.start()

    return { "result": True, "message": "Listening started in background" }
# --- 2. 모델 로드 ---

# ACLNet (OpenVINO) 로드
try:
    ie = Core()
    aclnet_compiled_model = ie.compile_model(model=ie.read_model(ACLNET_MODEL_XML), device_name=DEVICE)
    aclnet_output_layer = aclnet_compiled_model.output(0)
    # OpenVINO 모델의 입력 길이를 PyAudio CHUNK로 설정하거나, CHUNK가 모델 입력 길이와 같도록 설정
    model_input_length = aclnet_compiled_model.input(0).shape[-1]
    
    with open(ACLNET_CLASSES_TXT, 'r') as f:
        ACLNET_CLASSES = [line.strip() for line in f.readlines()]
    
    if model_input_length != CHUNK:
          print(f"[WARN] ACLNet model input length ({model_input_length}) differs from PyAudio CHUNK ({CHUNK}). Performance may be affected.")

except Exception as e:
    print(f"[ERROR] Failed to load ACLNet model: {e}")
    aclnet_compiled_model = None

# OpenWakeWord 로드
try:
    owwModel = Model(wakeword_models=[WAKEWORD_MODEL_NAME], inference_framework="openvino")
except Exception as e:
    print(f"[ERROR] Failed to load OpenWakeWord model: {e}")
    owwModel = None
    
# --- 3. 유틸리티 함수 ---

def preprocess_audio_aclnet(audio_chunk: np.ndarray):
    """ACLNet 모델 입력 형식에 맞게 전처리합니다."""
    # ACLNet은 (1, 1, 1, samples) 형태의 float32를 요구
    model_input = audio_chunk.astype(np.float32) / 32768.0 # Int16을 정규화
    model_input = model_input.reshape(1, 1, 1, -1)
    return model_input

def process_stt(filepath: str):
    """
    STT 서버에 WAV 파일을 전송하고,
    인식된 text가 있으면 RAG txt2chat API까지 자동 호출합니다.
    """
    print(f"[STT] 🚀 Sending {os.path.basename(filepath)} to STT server...")

    params = {
        'lang': STT_LANG,
        'isPlay': STT_IS_PLAY
    }

    try:
        with open(filepath, 'rb') as f:
            files = {
                'file': (os.path.basename(filepath), f, 'audio/wav')
            }

            response = requests.post(
                STT_API_URL,
                params=params,
                files=files,
                timeout=30
            )

        # ---------------- STT 응답 처리 ----------------
        if response.status_code == 200:
            print("[STT] ✅ STT API Success!")

            try:
                result_json = response.json()

                if 'text' in result_json and result_json['text'].strip():
                    recognized_text = result_json['text'].strip()

                    print("==================================================")
                    print(f"🗣️ STT Result: {recognized_text}")
                    print("==================================================")

                    # ---------------- RAG 호출 ----------------
                    call_rag_api(recognized_text)

                else:
                    print(f"[STT] ⚠️ No 'text' field found in response: {result_json}")

            except requests.exceptions.JSONDecodeError:
                print(f"[STT] ⚠️ Failed to decode JSON. Raw: {response.text}")

        else:
            print(f"[STT] ❌ STT API Failed. Status: {response.status_code}, Response: {response.text}")

    except requests.exceptions.ConnectionError as e:
        print(f"[STT] ❌ Connection Error: {e}")

    except requests.exceptions.Timeout:
        print("[STT] ❌ Request Timeout")

    except Exception as e:
        print(f"[STT] ❌ Unexpected Error: {e}")
def call_rag_api(prompt_text: str):
    """
    STT 결과를 RAG txt2chat API로 전달
    """
    print(f"[RAG] 🤖 Sending prompt to RAG: {prompt_text}")

    rag_url = "http://127.0.0.1:59532/v1/rag/txt2chat"

    params = {
        "prompt": prompt_text,
        "lang": "ko",
        "isPlay": 1
    }

    try:
        rag_response = requests.get(rag_url, params=params, timeout=60)

        if rag_response.status_code == 200:
            print("[RAG] ✅ RAG API Success")

            try:
                rag_json = rag_response.json()
                print(f"[RAG] 💬 Response: {rag_json}")
            except:
                print(f"[RAG] Raw Response: {rag_response.text}")

        else:
            print(f"[RAG] ❌ Failed. Status: {rag_response.status_code}")
            print(rag_response.text)

    except Exception as e:
        print(f"[RAG] ❌ Error calling RAG API: {e}")

def set_recording_active():
    """녹음 상태를 활성화하고 시간 변수를 초기화합니다."""
    global is_recording, recorded_frames, recording_start_time, last_sound_time
    is_recording = True
    recorded_frames = []
    recording_start_time = time.time()
    last_sound_time = time.time()
    print("\n[REC] ▶️ Recording STARTED by Wakeword or Manual Trigger.")

def stop_recording_and_save():
    """녹음을 중단하고 수집된 오디오를 WAV 파일로 저장한 후, STT를 실행합니다.""" # <--- 기능 추가 명시
    global is_recording, recorded_frames
    
    if not is_recording:
        return

    is_recording = False
    
    if not recorded_frames:
        print("[REC] 📝 No audio was captured. Stopping.")
        return

    # 프레임 연결
    recording_data = np.concatenate(recorded_frames, axis=0)
    
    # WAV 파일로 저장
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"wakeword_recording_{timestamp}.wav"
    filepath = os.path.join(RECORDING_OUTPUT_DIR, filename)
    write_wav(filepath, RATE, recording_data)

    print(f"[REC] 🛑 Recording STOPPED & SAVED to {filepath}")
    
    recorded_frames = []
    
    # --- STT 처리 추가 ---
    # 녹음된 파일을 STT API로 전송
    process_stt(filepath) # <--- STT 함수 호출

# --- 4. 메인 루프 ---
# (run_mic_loop 함수는 변경 없음)

def run_mic_loop():
    """마이크 입력 스트림을 읽고 VAD/Wakeword 감지 및 녹음을 처리합니다."""
    global is_recording, recorded_frames, last_sound_time, recording_start_time
    
    audio = pyaudio.PyAudio()
    mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print(f"Listening for Wakeword: '{WAKEWORD_MODEL_NAME.replace('_v0.1.xml', '').upper()}' (Threshold: {WAKEWORD_THRESHOLD})")

    try:
        while True:
            # 1. 오디오 데이터 읽기
            audio_chunk_int16 = np.frombuffer(mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            current_time = time.time()
            
            # 2. VAD: 소리 감지
            volume = np.max(np.abs(audio_chunk_int16))
            is_active = volume > VAD_ACTIVATION_THRESHOLD

            # 3. Wakeword 및 AEC 추론 (VAD 기반)
            if is_active:
                
                # a) OpenWakeWord 추론
                if owwModel:
                    prediction = owwModel.predict(audio_chunk_int16)


                    for mdl in owwModel.prediction_buffer.keys():
                        # Add scores in formatted table
                        scores = list(owwModel.prediction_buffer[mdl])
                        curr_score = format(scores[-1], '.20f').replace("-", "")

                        if scores[-1] > 0.5: 
                            if not is_recording:
                                print(f"[WKW] 🔔 Wakeword Detected! (Score: {curr_score}). Starting Recording.")
                                set_recording_active()  
                
                # b) ACLNet (AEC) 추론
                # 녹음 중이 아닐 때만 AEC 수행
                if not is_recording and aclnet_compiled_model:
                    processed_input = preprocess_audio_aclnet(audio_chunk_int16)
                    results = aclnet_compiled_model([processed_input])[aclnet_output_layer]
                    class_probabilities = results.flatten()
                    
                    max_prob = np.max(class_probabilities)
                    # AEC 추론 결과를 출력만 하고, 녹음을 시작하지는 않음 (요구사항에 따라 Wakeword만 녹음 시작 트리거로 유지)
                    if max_prob > 0.5:
                        class_idx = np.argmax(class_probabilities)
                        detected_class = ACLNET_CLASSES[class_idx]
                        print(f"[AEC] 🔥 Detected: {detected_class} (Prob: {max_prob:.2f})")


                        if processed_queue.full():
                            processed_queue.get()  # 가장 오래된 프레임 제거   

                        processed_queue.put([detected_class,round(float(max_prob),2)]) 
            
            # 4. 녹음 상태 처리
            if is_recording:
                
                # 프레임 저장
                recorded_frames.append(audio_chunk_int16)

                # 시간 초과 확인
                if current_time - recording_start_time > MAX_RECORDING_DURATION:
                    print(f"[REC] 🛑 Max duration ({MAX_RECORDING_DURATION}s) reached.")
                    stop_recording_and_save()
                    continue
                    
                # VAD 활성 상태에 따라 마지막 소리 시간 업데이트
                if is_active:
                    last_sound_time = current_time
                
                # 무음(비활성) 지속 시간 확인
                elif current_time - last_sound_time > SILENCE_DURATION:
                    print(f"[REC] 🛑 Silence detected for {SILENCE_DURATION} seconds.")
                    stop_recording_and_save()
                    continue

            # (wakeword 감지 후 녹음 시작 시, is_active가 true일 때만 last_sound_time이 업데이트되어,
            # 무음(silence)으로 인한 녹음 종료 처리는 `is_active`가 false일 때만 작동합니다.)

    except IOError as e:
        # 마이크 오버플로우 등의 PyAudio 오류 처리
        if e.errno == pyaudio.paInputOverflowed:
             print("[WARN] PyAudio input overflowed. Discarding chunk.")
             # continue를 통해 루프를 건너뛰고 오버플로우되지 않은 다음 청크를 읽습니다.
        else:
             print(f"[ERROR] PyAudio error: {e}")
    except KeyboardInterrupt:
        print("\n[INFO] Program interrupted by user.")
    except Exception as e:
        print(f"[FATAL ERROR] An unexpected error occurred: {e}")
    finally:
        # 스트림 및 오디오 종료
        if mic_stream.is_active():
            mic_stream.stop_stream()
        mic_stream.close()
        audio.terminate()
        
        if is_recording:
            print("[REC] ⚠️ Force stopping and saving ongoing recording.")
            stop_recording_and_save()


# --- 5. 프로그램 실행 ---

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
    subprocess.Popen(["play", 'intel_inside.mp3']) # async
