from fastapi.middleware.cors import CORSMiddleware
from serverinfo import si
from fastapi.responses import FileResponse, StreamingResponse
from huggingface_hub import snapshot_download, hf_hub_download
import time as t
import numpy as np
import utils
from scipy.io.wavfile import write
import soundfile as sf
from text import text_to_sequence
from serverinfo import si
import logging
from openvino import Core
from fastapi.staticfiles import StaticFiles
from queue import Queue
import openvino as ov
from playsound import playsound
from fastapi.middleware.cors import CORSMiddleware
import hashlib
from fastapi import FastAPI
import csv
import os
from datetime import datetime
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



app.mount("/web", StaticFiles(directory="web"), name="web")
app.mount("/webfonts", StaticFiles(directory="webfonts"), name="webfonts")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,  # 쿠키나 자격 증명 허용
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # 허용할 HTTP 메소드
    allow_headers=["*"],  # 모든 헤더 허용
)


core = Core()
path_tts = snapshot_download(repo_id="rippertnt/on-vits2-multi-tts-v1", allow_patterns="*ov*")
pipe_tts = core.compile_model(core.read_model(model=f"{path_tts}/ko_base_ov.xml"), device_name="CPU", config={"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS" : 4 })
conf_tts = utils.get_hparams_from_file(hf_hub_download(repo_id="rippertnt/on-vits2-multi-tts-v1", filename="ko_base.json"))


@app.get("/")
def main():
  return { "result" : True, "data" : "AI-CPU-V2", "ip" : _IP, "port" : _PORT }      


@app.get("/heartbeat")
async def heartbeat():
  global state
  print(state)
  return { "result" : True, "data" : state }        

@app.get("/monitor")
def monitor():
  return si.getAll()

import time as t
import csv
import os
from datetime import datetime
from scipy.io.wavfile import write

@app.get("/v1/tts", response_class=FileResponse, summary="입력한 문장으로 부터 음성을 생성합니다.")
def tts(text="", voice=31, lang='ko', static=0, isPlay=0):
    start = t.time()
    print(text, static)
    filename = f"{getHash(text)}_{lang}_{voice}"

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


    sf.write(f"output/{filename}.mp3", audio, sampling_rate, format='mp3')
    #write(data=audio, rate=sampling_rate, filename=f"output/{filename}.wav")
    return f"output/{filename}.mp3"
    

print("Loading Complete","CPU")
