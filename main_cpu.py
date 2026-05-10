from fastapi.middleware.cors import CORSMiddleware
from serverinfo import si
from fastapi.responses import FileResponse, StreamingResponse
from huggingface_hub import snapshot_download, hf_hub_download
import time as t
import numpy as np
import utils
from scipy.io.wavfile import write
from text import text_to_sequence
from pydub import AudioSegment
from serverinfo import si
import logging
from openvino import Core
from fastapi.staticfiles import StaticFiles
from queue import Queue
import openvino as ov
from playsound import playsound
from fastapi.middleware.cors import CORSMiddleware
import hashlib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import requests
import time
import csv
import os
from datetime import datetime
from monitor import CPUPowerMonitor
import subprocess
from slide import create_slide_router
import threading
import queue
# 기존 임포트 항목들 (numpy, time, scipy.io.wavfile 등)

app = FastAPI()


def play_sound(file_path):
    subprocess.run(["play", file_path])


def set_volume(val):
    subprocess.run(["wpctl", "set-volume","58",f"{val}"])


def getHash(text):
  hash_func = hashlib.new('md5')
  hash_func.update(text.encode('utf-8'))
  return hash_func.hexdigest()

# Enable logging for debugging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

_IP = "127.0.0.1" #si.getIP()
_PORT = int(open("port.txt", 'r').read())

app = FastAPI()

#pw = CPUPowerMonitor(interval=1.0)
#pw.start()


app.mount("/web", StaticFiles(directory="web"), name="web")
app.mount("/webfonts", StaticFiles(directory="webfonts"), name="webfonts")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,  # 쿠키나 자격 증명 허용
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # 허용할 HTTP 메소드
    allow_headers=["*"],  # 모든 헤더 허용
)


_VOL = 0.5

core = Core()
path_tts = snapshot_download(repo_id="rippertnt/on-vits2-multi-tts-v1", allow_patterns="*ov*")
pipe_tts = core.compile_model(core.read_model(model=f"{path_tts}/all_base_ov.xml"), device_name="CPU", config={"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS" : 4 })
conf_tts = utils.get_hparams_from_file(hf_hub_download(repo_id="rippertnt/on-vits2-multi-tts-v1", filename="all_base.json"))

app.include_router(create_slide_router(pipe_tts, conf_tts))

@app.get("/")
def main():
  return { "result" : True, "data" : "AI-CPU-V2", "ip" : _IP, "port" : _PORT }      

@app.get("/heartbeat")
async def heartbeat():
  global state
  print(state)
  return { "result" : True, "data" : state }        

@app.get("/volume")
async def volume(vol):
  global _VOL
  _VOL = vol
  print(_VOL)
  set_volume(_VOL)
  return { "result" : True, "data" : _VOL }        

@app.get("/monitor")
def monitor():
  return si.getAll()

import time as t
import csv
import os
from datetime import datetime
from scipy.io.wavfile import write

@app.get("/v1/tts", response_class=FileResponse, summary="입력한 문장으로 부터 음성을 생성합니다.")
def tts1(text="", voice=31, lang='ko', static=0, isPlay=0):
    start = t.time()
    print(text, static)
    filename = getHash(text)

    # phoneme 변환
    phoneme_ids = text_to_sequence(text, [f'canvers_{lang}_cleaners'])
    text_arr = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)

    inputs = {
        "input": text_arr,
        "input_lengths": np.array([text_arr.shape[1]], dtype=np.int64),
        "scales": np.array([0.667, 1.0, 0.8], dtype=np.float16),
        "sid": np.array([int(voice)], dtype=np.int64) if voice is not None else None
    }

    # --------------------------
    # 🔥 TTS 추론 시간 측정
    # --------------------------
    start_time = t.time()
    result = pipe_tts(inputs)
    inference_time = t.time() - start_time
    print(f"Inference time: {inference_time:.4f} seconds")

    audio = list(result.values())[0].squeeze((0, 1))

    # --------------------------
    # 🔥 오디오 길이 → RTF 계산
    # --------------------------
    sampling_rate = conf_tts.data.sampling_rate
    audio_duration = len(audio) / sampling_rate
    rtf = audio_duration / inference_time

    print(f"Audio duration: {audio_duration:.4f} sec | RTF: {rtf:.4f}")
    print(f"Total time: {t.time() - start:.4f}")

    # --------------------------
    # 🔥 CSV 로그 저장
    # --------------------------
    log_file = "CPU_log.csv"
    new_file = not os.path.exists(log_file)

    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([
                "timestamp", "text_length","inference_time", "audio_duration", "rtf", "Watt"
            ])
        writer.writerow([
            datetime.now().isoformat(),
            len(text),
            round(inference_time, 6),
            round(audio_duration, 6),
            round(rtf, 6),
            #pw.get_power()
        ])

    # --------------------------
    # 🔥 WAV 파일 저장
    # --------------------------
    if int(static) > 0:
        write(data=audio, rate=sampling_rate, filename="output/human.wav")
        return "output/human.wav"

    if int(isPlay) > 0:
        play_sound(f"output/{filename}.wav")

    write(data=audio, rate=sampling_rate, filename=f"output/{filename}.wav")
    return f"output/{filename}.wav"
    
# 1. 재생할 파일 경로를 담을 큐 생성
playback_queue = queue.Queue()

# 2. 음성 재생을 담당할 워커 함수
def audio_player_worker():
    while True:
        # 큐에서 파일 경로를 하나씩 꺼냄 (없으면 대기)
        file_path = playback_queue.get()
        if file_path is None: # 종료 시그널
            break
        
        try:
            # subprocess.run은 실행이 완료될 때까지 블로킹됩니다.
            # 따라서 자연스럽게 다음 파일은 이전 파일 재생이 끝난 뒤 실행됩니다.
            subprocess.run(["play", file_path])
        except Exception as e:
            print(f"Playback error: {e}")
        finally:
            # 작업 완료 알림
            playback_queue.task_done()

# 3. 워커 스레드 시작 (데몬 스레드로 설정하여 프로그램 종료 시 자동 종료)
worker_thread = threading.Thread(target=audio_player_worker, daemon=True)
worker_thread.start()

@app.get("/v2/tts", response_class=FileResponse, summary="음성 생성 후 로봇에서 재생")
def tts2(text = "", voice=6, lang='ko', static=0):
    start = t.time()
    filename = getHash(text)
    file_full_path = f"output/{filename}.wav"

    # --- TTS 변환 로직 (기존과 동일) ---
    phoneme_ids = text_to_sequence(text, [f'canvers_{lang}_cleaners'])
    text_input = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)

    inputs = {
        "input": text_input,
        "input_lengths":  np.array([text_input.shape[1]], dtype=np.int64),
        "scales": np.array([0.667, 1.0, 0.8], dtype=np.float16),
        "sid" : np.array([int(voice)], dtype=np.int64) if voice is not None else None
    }

    result = pipe_tts(inputs)
    audio = list(result.values())[0].squeeze((0, 1))  
    
    # 파일 저장
    write(data=audio, rate=conf_tts.data.sampling_rate, filename=file_full_path)
    # -------------------------------

    # 4. 재생 큐에 파일 경로 추가
    # API 응답은 즉시 반환되지만, 재생은 워커 스레드에 의해 순차적으로 진행됩니다.
    playback_queue.put(file_full_path)
    
    print(f"Total time: {t.time() - start:.4f}s - Queued for playback: {filename}")
    
    return file_full_path

print("Loading Complete","CPU")
