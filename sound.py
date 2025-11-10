import asyncio
import sounddevice as sd
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from openvino import Core
from scipy.io.wavfile import write as write_wav
import os
import datetime
import time

# OpenVINO 모델 파일 경로 설정
ACLNET_MODEL_XML = "./models/aclnet.xml"
ACLNET_CLASSES_TXT = "./models/aclnet_53cl.txt"
DEVICE = "NPU"
app = FastAPI()

# 감지된 소리 이벤트의 클래스 리스트 로드
try:
    with open(ACLNET_CLASSES_TXT, 'r') as f:
        ACLNET_CLASSES = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    raise FileNotFoundError(f"{ACLNET_CLASSES_TXT} 파일이 존재하지 않습니다. Open Model Zoo에서 다운로드하세요.")

# OpenVINO 모델 초기화
ov = Core()
print(f"Using device: {DEVICE}")

try:
    aclnet_compiled_model = ov.compile_model(model=ov.read_model(ACLNET_MODEL_XML), device_name=DEVICE)
except Exception as e:
    print(f"Failed to compile model for NPU: {e}. Falling back to CPU.")
    aclnet_compiled_model = ov.compile_model(model=ov.read_model(ACLNET_MODEL_XML), device_name="CPU")

aclnet_input_layer = aclnet_compiled_model.input(0)
aclnet_output_layer = aclnet_compiled_model.output(0)
INPUT_SAMPLE_RATE = 16000
input_shape = aclnet_input_layer.shape
INPUT_LENGTH_SAMPLES = input_shape[-1]

def preprocess_audio(audio_chunk: np.ndarray):
    model_input = audio_chunk.reshape(1, 1, 1, -1).astype(np.float32)
    return model_input

# ===== 단일 스트림 및 상태 관리 변수 =====
is_recording = False
recorded_frames = []
stream_task = None
RECORDING_OUTPUT_DIR = "recordings"
os.makedirs(RECORDING_OUTPUT_DIR, exist_ok=True)

# ===== 자동 종료를 위한 변수 =====
SILENCE_THRESHOLD = 0.1  # 무음을 판단하는 임계값. 이 값보다 작으면 무음으로 간주.
SILENCE_DURATION = 3      # 무음 지속 시간 (초)
MAX_RECORDING_DURATION = 15 # 최대 녹음 시간 (초)

last_sound_time = time.time()
recording_start_time = time.time()

# ===== 녹음 중단 및 저장 함수 =====
async def stop_recording_and_save():
    global is_recording, recorded_frames
    
    if not recorded_frames:
        is_recording = False
        return JSONResponse(content={"message": "Recording stopped, but no audio was captured."}, status_code=200)

    is_recording = False
    
    recording_data = np.concatenate(recorded_frames, axis=0)
    max_abs_val = np.max(np.abs(recording_data))
    
    normalized_data = recording_data / max_abs_val if max_abs_val > 0 else recording_data
    recording_data_int16 = (normalized_data * np.iinfo(np.int16).max).astype(np.int16)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.wav"
    filepath = os.path.join(RECORDING_OUTPUT_DIR, filename)

    await asyncio.to_thread(write_wav, filepath, INPUT_SAMPLE_RATE, recording_data_int16)
    print(f"Recording saved to {filepath}")
    
    recorded_frames = [] # 초기화
    return JSONResponse(content={"message": f"Recording stopped and saved to {filename}."}, status_code=200)

# ===== 단일 마이크 스트림을 처리하는 비동기 함수 =====
async def single_stream_processor(websocket):
    global is_recording, recorded_frames, last_sound_time, recording_start_time
    
    try:
        with sd.InputStream(samplerate=INPUT_SAMPLE_RATE, channels=1, blocksize=INPUT_LENGTH_SAMPLES) as stream:
            while True:
                # 단일 스트림에서 오디오 데이터 읽기
                audio_chunk, _ = stream.read(stream.blocksize)
                audio_chunk = audio_chunk.flatten()

                # --- 녹음 중인 경우 ---
                if is_recording:
                    current_time = time.time()
                    
                    # 15초 시간 초과 확인
                    if current_time - recording_start_time > MAX_RECORDING_DURATION:
                        await websocket.send_text("Recording stopped: Maximum duration reached.")
                        await stop_recording_and_save()
                        continue
                        
                    # 무음 감지 확인
                    volume = np.max(np.abs(audio_chunk))
                    if volume > SILENCE_THRESHOLD:
                        last_sound_time = current_time
                    elif current_time - last_sound_time > SILENCE_DURATION:
                        await websocket.send_text("Recording stopped: Silence detected for 3 seconds.")
                        await stop_recording_and_save()
                        continue

                    recorded_frames.append(audio_chunk)

                # --- 추론 로직 ---
                processed_input = preprocess_audio(audio_chunk)
                results = aclnet_compiled_model([processed_input])[aclnet_output_layer]
                class_probabilities = results
                
                # WebSocket으로 감지 결과 전송
                if np.max(class_probabilities) > 0.5:
                    class_idx = np.argmax(class_probabilities)
                    detected_class = ACLNET_CLASSES[class_idx]
                    await websocket.send_text(f"Detected: {detected_class}")
                #else:
                #    await websocket.send_text("No event detected")
                
                await asyncio.sleep(0.01)

    except Exception as e:
        print(f"Error in single stream processor: {e}")
    finally:
        print("Single stream processor stopped.")

# ===== 녹음 시작/종료 엔드포인트 수정 =====
@app.get("/listen")
async def toggle_recording():
    global is_recording, recorded_frames, recording_start_time, last_sound_time
    
    if not is_recording:
        is_recording = True
        recorded_frames = [] # 새로운 녹음 시작
        recording_start_time = time.time()
        last_sound_time = time.time()
        return JSONResponse(content={"message": "Recording started. Will stop automatically after 15s or 3s of silence."}, status_code=200)
    else:
        return await stop_recording_and_save()

# ===== 기존 WebSocket 코드 (단일 스트림으로 교체) =====
@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    global stream_task
    await websocket.accept()
    
    if stream_task is None or stream_task.done():
        print("Starting single stream processor.")
        stream_task = asyncio.create_task(single_stream_processor(websocket))
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred in websocket: {e}")
    finally:
        pass


@app.get("/")
async def index():
    """
    웹소켓을 위한 HTML 페이지를 제공합니다.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ACLNet-INT8 소리 이벤트 감지 (NPU)</title>
        <script>
            var ws = new WebSocket("ws://localhost:8001/ws/audio");
            ws.onmessage = function(event) {
                var messageElement = document.createElement("p");
                messageElement.textContent = event.data;
                document.body.appendChild(messageElement);
            };
        </script>
    </head>
    <body>
        <h1>실시간 소리 이벤트 감지 (ACLNet-INT8 on NPU)</h1>
        <h2>녹음 제어</h2>
        <p>GET 요청을 보내 녹음을 시작/중지할 수 있습니다. (예: 브라우저에서 <a href="/listen">/listen</a> 방문)</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8001)