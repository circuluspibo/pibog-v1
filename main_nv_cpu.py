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
from nv_monitor import CPUPowerMonitor

import onnxruntime as ort


model = {}
conf = {}
sess_options = ort.SessionOptions()

def loadModel(repo, model_name,isCuda=True):
    print(model_name,isCuda)
    conf[model_name] = utils.get_hparams_from_file(hf_hub_download(repo_id=repo, filename=f'{model_name}.json'))

    if isCuda:
        provider = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})] #'TensorrtExecutionProvider' // , {"cudnn_conv_algo_search": "DEFAULT"}
        model[model_name] = ort.InferenceSession(hf_hub_download(repo_id=repo, filename=f'{model_name}_f16.onnx'), sess_options=sess_options, providers=provider) #f16
    else:
        provider = ['CPUExecutionProvider']
        model[model_name] = ort.InferenceSession(hf_hub_download(repo_id=repo, filename=f'{model_name}.onnx'), sess_options=sess_options, providers=provider) #f16
    #provider = "OpenVINOExecutionProvider" # gpu 는 f16이 되나 cpu는 f32

    
loadModel('rippertnt/on-vits2-multi-tts-v1','ko_base',False)
loadModel('rippertnt/on-vits2-multi-tts-v1','en_base',False)


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

pw = CPUPowerMonitor(interval=1.0)
pw.start()

app.mount("/web", StaticFiles(directory="web"), name="web")
app.mount("/webfonts", StaticFiles(directory="webfonts"), name="webfonts")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,  # 쿠키나 자격 증명 허용
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # 허용할 HTTP 메소드
    allow_headers=["*"],  # 모든 헤더 허용
)


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
    filename = getHash(text)

    # phoneme 변환
    phoneme_ids = text_to_sequence(text, [f'canvers_{lang}_cleaners'])
    text_arr = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)

    inputs = {
        "input": text_arr,
        "input_lengths": np.array([text_arr.shape[1]], dtype=np.int64),
        "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32), # cpu
        "sid": np.array([int(voice)], dtype=np.int64) if voice is not None else None
    }

    start = time.time()
    print('Starting...')
     



    # --------------------------
    # 🔥 TTS 추론 시간 측정
    # --------------------------
    start_time = t.time()
    audio = model[f"{lang}_base"].run(
        None, inputs
    )[0].squeeze((0, 1))

    inference_time = t.time() - start_time
    print(f"Inference time: {inference_time:.4f} seconds")

    # --------------------------
    # 🔥 오디오 길이 → RTF 계산
    # --------------------------
    sampling_rate = conf[f"{lang}_base"].data.sampling_rate
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
            pw.get_power()
        ])

    # --------------------------
    # 🔥 WAV 파일 저장
    # --------------------------
    if int(static) > 0:
        write(data=audio, rate=sampling_rate, filename="output/human.wav")
        return "output/human.wav"

    if int(isPlay) > 0:
        playsound(f"output/{filename}.wav")

    write(data=audio, rate=sampling_rate, filename=f"output/{filename}.wav")
    return f"output/{filename}.wav"
    
    # 31 korean
@app.get("/v2/tts", response_class=FileResponse, summary="음성 생성 후 로봇에서 재생")
def tts(text = "", voice=6, lang='ko', static=0, isPlay=0):
    #org_text = parse.quote(text, safe='', encoding="cp949")
    start = t.time()
    print(text, static)
    filename = getHash(text)

    #phoneme_ids = text_to_sequence(text, conf_tts.data.text_cleaners)
    phoneme_ids = text_to_sequence(text, [f'canvers_{lang}_cleaners'])
    text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)

    inputs = {
        "input": text,
        "input_lengths":  np.array([text.shape[1]], dtype=np.int64),
        "scales": np.array([0.667, 1.0, 0.8], dtype=np.float16),
        "sid" : np.array([int(voice)], dtype=np.int64) if voice is not None else None
    }

    start_time = t.time()
    audio = model[lang].run(
        None, inputs
    )[0].squeeze((0, 1))

    inference_time = t.time() - start_time
    print(f"Inference time: {inference_time:.4f} seconds")

    # --------------------------
    # 🔥 오디오 길이 → RTF 계산
    # --------------------------
    sampling_rate = conf[lang].data.sampling_rate
    audio_duration = len(audio) / sampling_rate

    print(t.time() - start)
    write(data=audio, rate=conf[lang].data.sampling_rate, filename=f"output/{filename}.wav")

    return f"output/{filename}.wav"


print("Loading Complete","CPU")
