from fastapi.middleware.cors import CORSMiddleware
from serverinfo import si
import librosa
from fastapi import FastAPI, File, UploadFile
from transformers import AutoTokenizer
from fastapi.responses import FileResponse, StreamingResponse
import langid
import random
import ctranslate2
from PIL import Image
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download, hf_hub_download
import time as t
import collections
from transformers import AutoTokenizer
from pydantic import BaseModel, Field
import numpy as np
import openvino_genai as ov_genai
import onnxruntime as rt
import utils
import commons
from scipy.io.wavfile import write
from text import text_to_sequence
import torch
import json
from pydub import AudioSegment
from serverinfo import si
#import onnxruntime_genai as og
#from llama_cpp import Llama
import asyncio
from go2_webrtc_driver.webrtc_audiohub import WebRTCAudioHub
import logging
import asyncio
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, VUI_COLOR, SPORT_CMD
from aiortc import MediaStreamTrack
from requests import get
import time
import cv2
from openvino import Core
from fastapi.staticfiles import StaticFiles
from queue import Queue
from ultralytics import YOLO, FastSAM
import openvino as ov
from playsound import playsound
from mandro import HadnControler
import threading
from threading import Event, Thread
from transformers import AutoTokenizer
from pydantic import BaseModel, Field
from iterator import IterableStreamer
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
#optimum-cli export openvino --weight-format int4 --task text-generation-with-past --model growdle/HyperCLOVAX-SEED-Text-Instruct-1.5B ./CLOVAX-1.5B-ov-int4
#kakaocorp/kanana-1.5-2.1b-instruct-2505
#https://github.com/Unitree-Go2-Robot/go2_robot

hL = None
hR = None
core = ov.Core()

det_ov_model = core.read_model('yolo12m_int8_openvino_model/yolo12m.xml')
det_model = YOLO('yolo12m_int8_openvino_model', task='detect')
sam_model = FastSAM("./FastSAM-s_int8_openvino_model")  # or FastSAM-x.pt

det_ov_model.reshape({0: [1, 3, 384, 640]})
compiled_model = core.compile_model(det_ov_model, 'NPU')

if det_model.predictor is None:
    custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}  # method defaults
    args = {**det_model.overrides, **custom}
    det_model.predictor = det_model._smart_load("predictor")(overrides=args, _callbacks=det_model.callbacks)
    det_model.predictor.setup_model(model=det_model.model)

det_model.predictor.model.ov_compiled_model = compiled_model

# Enable logging for debugging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

_IP = "127.0.0.1" #si.getIP()
_PORT = int(open("port.txt", 'r').read())

conn = None
audio_hub = None
track = None
lastColor = 'cyan'
state = { "charge" : 0, "temp" : 0, "voltage" : 0, "cnt_live" : 0, "cnt_object" : 0 }

G1_ARM = {
  "clamp": 17, 
  "highFive": 18, 
  "shakeHands_1": 27,
  "makeHeartBothHands": 20, 
  "makeHeartSingleHands": 21,
  "blowKiss": 12, 
  "hug": 19,
  "hightWave": 26, 
  "lowWave" : 25,
  "ultramanRay" : 24, 
  "bothHandsUp" : 15,
  "singleHandsUp" : 23,
  "Refuse" : 22, 
  "Release Arm" : 99,
}

G1_STATE = {
  "ZeroTorque" : 0,
  "Damp" : 1,
  "Preparation": 4,
  "Seating": 3,       
  "Walk_G1": 500,
  "Walk2_G1" : 501,
  "Run_G1" : 801,
  "Squat_G1" : 706,  
  "SquatUp_G1" : 706,
  "LieUp_G1" : 702,
}

G1_BALANCE = {
  "Stand_G1" : 0,
  "Step_G1" : 1 
}

frame_queue = Queue(maxsize=5)
processed_frame_queue = Queue(maxsize=5)

cnt_live = 0
cnt_object = 0
lastTime = 0

app = FastAPI()

app.mount("/web", StaticFiles(directory="web"), name="web")
app.mount("/webfonts", StaticFiles(directory="webfonts"), name="webfonts")

class Param (BaseModel):
  text : str
  hash : str = Field(default='')
  voice : str = Field(default='main') 
  lang : str = Field(default='ko')
  type : str = Field(default='mp3')
  pitch : str = Field(default='medium')
  rate : str = Field(default='medium')
  volume : str = Field(default='medium')


#path_model_tts = hf_hub_download(repo_id="rippertnt/on-vits2-multi-tts-v1", filename="ko_base_f16.onnx")
#path_conf_tts = hf_hub_download(repo_id="rippertnt/on-vits2-multi-tts-v1", filename="ko_base.json")
#path_stt = snapshot_download(repo_id="circulus/whisper-large-v3-turbo-ov-int4")
#path_text = snapshot_download(repo_id="circulus/gemma-3-1b-it-gguf")

_SYSTEM = "당신은 서큘러스에서 만든 파이봇 이라고 하는 강아지 로봇 인공지능 입니다. 젊은 톤의 대화체로 입력된 언어로 사람 같이 짧게 응답하세요."

class Chat(BaseModel):
  prompt : str = ''
  lang : str = 'auto'
  type : str =  "당신은 서큘러스에서 만든 파이봇 이라고 하는 강아지 로봇 인공지능 입니다. 젊은 톤의 대화체로 입력된 언어로 사람 같이 짧게 응답하세요." #" "당신은 데이비드라고 하는 10살 남자아이 성향의 유쾌하고 즐거운 인공지능입니다. 이모티콘도 잘 활용해서 젊은 말투로 대답하세요."
  rag :  str = ''  
  temp : float = 0.5
  top_p : float = 0.92
  top_k : int = 50
  max : int = 256 #16384

# gemma-3-1b-it-Q4_K_M.gguf
#model_txt = Llama("../models/txt/hyperclovax-seed-text-instruct-1.5b-q4_k_m.gguf", n_threads=4, verbose=False) #from_pretrained

token_txt = AutoTokenizer.from_pretrained("../models/CLOVAX-1.5B-ov-int4")
pipe_txt = ov_genai.LLMPipeline("../models/CLOVAX-1.5B-ov-int4", device="GPU")

pipe_stt = ov_genai.WhisperPipeline('../models/stt',device="GPU")

#pipe_tts = rt.InferenceSession('../models/tts/ko_base_f16.onnx', sess_options=rt.SessionOptions(), providers=["OpenVINOExecutionProvider"], provider_options=[{"device_type" : "CPU" }]) #, "precision" : "FP16"
#conf_tts = utils.get_hparams_from_file('../models/tts/ko_base.json')

core = Core()
config = {"PERFORMANCE_HINT": "LATENCY"}
path_tts = snapshot_download(repo_id="rippertnt/on-vits2-multi-tts-v1", allow_patterns="*ov*")
pipe_tts = core.compile_model(core.read_model(model=f"{path_tts}/all_base_ov.xml"), device_name="CPU", config=config)
conf_tts = utils.get_hparams_from_file(hf_hub_download(repo_id="rippertnt/on-vits2-multi-tts-v1", filename="all_base.json"))

def processing_thread():
    global cnt_live, cnt_object, lastTime, state
    processing_times = collections.deque()

    while True:
        if not frame_queue.empty():
            image = np.array(frame_queue.get())
            cnt_live = 0
            cnt_object = 0

            start_time = time.time()
            results = det_model(image, verbose=False)[0]
            result = sam_model(image, verbose=False, device="intel:npu", retina_masks=True, imgsz=640, conf=0.6, iou=0.9)[0]
            stop_time = time.time()

            # 결과 처리 (Bounding box, mask 등)
            names = det_model.names
            output = image.copy()

            highlight_classes = ['person', 'dog', 'cat', 'horse', 'cow', 'sheep', 'bird', 'elephant', 'bear', 'zebra', 'giraffe','teddy bear']
            for box in results.boxes:
                cls_id = int(box.cls.item())
                cls_name = names[cls_id]
                conf = box.conf.item()
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if cls_name in highlight_classes:
                    rgb_color = (0, 0, 255)
                    cnt_live += 1
                    lastTime = time.time()
                else:
                    rgb_color = (255, 255, 0)
                    cnt_object += 1

                cv2.rectangle(output, (x1, y1), (x2, y2), rgb_color, 2)
                label = f'{cls_name} {conf:.2f}'
                cv2.putText(output, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb_color, 2)

            state["cnt_live"] = cnt_live
            state["cnt_object"] = cnt_object

            masks = result.masks.data.cpu().numpy()
            for mask in masks:
                mask = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(output, contours, -1, (0, 255, 0), 3)

            processing_times.append(stop_time - start_time)
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = output.shape[:2]
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(
                output,
                f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                (20, 40),
                cv2.FONT_HERSHEY_COMPLEX,
                f_width / 1000,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

            if processed_frame_queue.full():
              processed_frame_queue.get()  # 가장 오래된 프레임 제거   

            processed_frame_queue.put(output)
        else:
            time.sleep(0.01)


# for genai
async def process_stream(streamer, isStream=True, isPlay=0):
  cnt = 0
  sentence = ""
  print("streaming start...")
  for new_token in streamer:
    if "assistant" in new_token:
      cnt += 1
      if cnt == 1:
          continue  # Skip the first one (don't yield)
      elif cnt == 2:
          break  # Stop stream (don't yield)
        
    if isStream:
      #print(new_token)
      yield new_token
      #await asyncio.sleep(0) 
    elif "." in new_token or "\n" in new_token:
      sentence = sentence + new_token
      if len(sentence) > 3:      
        sentence = sentence.strip()
        print(sentence)
        if int(isPlay) > 0:
          file = tts(sentence,31,"ko",0,0)
        else:
          await speech(sentence, 'Content', 0, 'ko')

        yield sentence
        await asyncio.sleep(0.01) # thread 처리
        sentence = ""
    else:
      sentence = sentence + new_token
  if len(sentence) > 3:
    yield sentence

# llama cpp verion
async def generate_text_stream(chat : Chat, isStream=True, isPlay=0):
    
    if chat.rag is not None and len(chat.rag) > 10: 
      chat.type=  f"{chat.type} 그리고, 다음 내용을 참고하여 대답을 하되 잘 모르는 내용이면 모른다고 솔직하게 대답하세요.\n<|context|>\n{chat.rag}"    

    prompt = token_txt.apply_chat_template([
      {"role": "system", "content": chat.type},
      {"role": "user", "content": chat.prompt}
    ], tokenize=False,add_generation_prompt=True)
	
    response = model_txt.create_completion(prompt, max_tokens=chat.max, temperature=chat.temp, top_k=chat.top_k,top_p=chat.top_p,repeat_penalty=1.1, stream=True)
    sentence = ""
    for chunk in response:
        if "choices" in chunk and chunk["choices"]:
            new_token =  chunk["choices"][0]["text"]
            if isStream:
              #print(new_token)
              yield new_token
              #await asyncio.sleep(0) 
            elif "." in new_token or "\n" in new_token:
              sentence = sentence + new_token
              if len(sentence) > 3:
                
                sentence = sentence.strip()
                print(sentence)
                if int(isPlay) > 0:
                  file = tts(sentence)
                  print('playing', file)
                  playsound(file)
                else:
                  await speech(sentence, 'Content', 0, 'ko')

                yield sentence
                #await asyncio.sleep(0) 
                sentence = ""
            else:
              sentence = sentence + new_token
    if len(sentence) > 3:
      yield sentence
      #await asyncio.sleep(0.01)  # 비동기 처리를 위한 작은 딜레이

origins = [
    "http://canvers.net",
    "https://canvers.net",   
    "http://www.canvers.net",
    "https://www.canvers.net",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],#origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def main():
  return { "result" : True, "data" : "AI-CPU-V2", "ip" : _IP, "port" : _PORT }      

# Async function to receive video frames and put them in the queue
async def recv_camera_stream(track: MediaStreamTrack):
    while True:
        frame = await track.recv()
        # Convert the frame to a NumPy array
        img = frame.to_ndarray(format="bgr24")

        if frame_queue.full():
            frame_queue.get()  # 가장 오래된 프레임 제거        

        frame_queue.put(img)

@app.get("/connect")
async def connect():
  global conn
  global audio_hub
  conn =  Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP) #Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.0.101")
  await conn.connect()
  print(1)
  audio_hub = WebRTCAudioHub(conn, logger)
  await audio_hub.set_play_mode('no_cycle')
  print(2)
  """
  await conn.datachannel.pub_sub.publish_request_new(
    RTC_TOPIC["MOTION_SWITCHER"], 
    {
        "api_id": 1002,
        "parameter": {"name": "normal"}
    }
  )
  """
  conn.video.switchVideoChannel(True)
  conn.video.add_track_callback(recv_camera_stream)
  
  # image processer start
  threading.Thread(target=processing_thread, daemon=True).start()

  def lowstate_callback(message):
    #print(message)
    msg = message['data']      
    state["charge"] = msg['bms_state']['soc']
    state["temp"] = msg['temperature_ntc1']
    state["voltage"] = msg['power_v']

  conn.datachannel.pub_sub.subscribe(RTC_TOPIC['LOW_STATE'], lowstate_callback)

  return { "result" : True, "data" : True }        

@app.get("/connect2")
async def connect2():
  global conn
  #global audio_hub
  conn =  Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP) #Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.0.101")
  await conn.connect()
  print(1)
  #audio_hub = WebRTCAudioHub(conn, logger)
  #await audio_hub.set_play_mode('no_cycle')
  #conn.video.switchVideoChannel(True)
  #conn.video.add_track_callback(recv_camera_stream)
  #print(3)
  def lowstate_callback(message):
    #print(message)
    msg = message['data']      
    state["charge"] = msg['bms_state']['soc']
    state["temp"] = msg['temperature_ntc1']
    state["voltage"] = msg['power_v']

  conn.datachannel.pub_sub.subscribe(RTC_TOPIC['LOW_STATE'], lowstate_callback)

  return { "result" : True, "data" : True }        

@app.get("/prepare")
async def prepare():

  global hL
  global hR

  try:
      hL = HadnControler('/dev/ttyACM0') # L 컨트롤러 L동글 부터 연결
      hR = HadnControler('/dev/ttyACM1') # R 컨트롤러
      print("컨트롤러 초기화 성공")
  except Exception as e:
      print(f"컨트롤러 초기화 실패: {e}")
      exit()

@app.get("/hand")
async def hand(cmd : str):

  global hL
  global hR

  print(cmd)

  if cmd == 'release':
    thread_L = threading.Thread(target=hL.send_release, args=(None,))
    thread_R = threading.Thread(target=hR.send_release, args=(None,))     
  else:
    thread_L = threading.Thread(target=hL.send_motion, args=(cmd,))
    thread_R = threading.Thread(target=hR.send_motion, args=(cmd,))

  thread_L.start()
  thread_R.start()

@app.get("/heartbeat")
async def heartbeat():
  global state
  print(state)
  return { "result" : True, "data" : state }        

@app.get("/video_feed")
async def video_feed():
  def generate():
    while True:
        if not processed_frame_queue.empty():
            output = processed_frame_queue.get()
            _, img_bytes = cv2.imencode('.jpg', output)
            frame = img_bytes.tobytes()

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            time.sleep(0.01)

  return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

lastCmd = {} 

@app.get("/sport")
async def sport(cmd : str, x=0.0, y=0.0, z=0.0, data=None):
  global conn
  print(cmd, f'x:{x}, y:{y}, z:{z}')
  if conn is None:
    print('Disconnected', cmd)
  elif cmd == 'Move':
    out = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"], {
          "api_id": SPORT_CMD[cmd],
          "parameter": {"x": float(x), "y": float(y), "z": float(z)}
        }
    )
  elif data != None: # SPORT_CMD[cmd] != None and
    out = await conn.datachannel.pub_sub.publish_request_new(
      RTC_TOPIC["SPORT_MOD"], {
          "api_id": SPORT_CMD[cmd],
          "parameter": { "data" : float(data) } # if possible
      }
    )
  else:
    if lastCmd.get(cmd, False):
      lastCmd[cmd] = True
      out = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"], {
            "api_id": SPORT_CMD[cmd],
            "parameter": { "data" : True } # if possible
        }
      )       
    else:
      lastCmd[cmd] = False
      out = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"], {
            "api_id": SPORT_CMD[cmd],
            "parameter": { "data" : False } # if possible
        }
      )


  print("response", out)
            
  return { "result" : True, "data" : out }      

def toFloat(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return value

@app.get("/manual")
async def manual(cmd : str, data : str):
  out = await conn.datachannel.pub_sub.publish_request_new(
    RTC_TOPIC["SPORT_MOD"], {
        "api_id": int(cmd),
        "parameter": { "data" : toFloat(data) } # if possible
    }
  )

  print("response", out)
            
  return { "result" : True, "data" : out }      

@app.get("/arm")
async def arm(cmd = "clamp"):
  global conn
  global G1_ARM

  await conn.datachannel.pub_sub.publish_request_new(
    "rt/api/arm/request", {
        "api_id": 7106,
        "parameter" : { "data" : G1_ARM[cmd] }
    }
  )

  return { "result" : True, "data" : True }      

@app.get("/walkG1")
async def walkG1(lx = 0, ly = 0, rx = 0, ry = 0):
  print("walking",f"L : {lx} {ly} | R : {rx} {ry}")
  global conn

  conn.datachannel.pub_sub.publish_without_callback(
     "rt/wirelesscontroller", {
        "lx": float(lx), "ly": float(ly), "rx": float(rx), "ry": float(ry) 
     }
  )
  """
  await conn.datachannel.pub_sub.publish("rt/wirelesscontroller", { 
     "lx": int(lx), "ly": int(ly), "rx": int(rx), "ry": int(ry) 
  })
  """
  
  return { "result" : True, "data" : True }     

@app.get("/stateG1")
async def stateG1(cmd="Walk_G1"):
  global conn
  global G1_STATE

  await conn.datachannel.pub_sub.publish_request_new(
    "rt/api/sport/request", {
        "api_id": 7101,
        "parameter" : { "data" : G1_STATE[cmd] }
    }
  )

  return { "result" : True, "data" : True }      

@app.get("/balanceG1")
async def balanceG1(cmd="Stand_G1"):
  global conn
  global G1_BALANCE

  await conn.datachannel.pub_sub.publish_request_new(
    "rt/api/sport/request", {
        "api_id": 7102,
        "parameter" : { "data" : G1_BALANCE[cmd] }
    }
  )

  return { "result" : True, "data" : True }      


@app.get("/speech")
async def speech(text : str, motion = None, voice=31, lang='ko'):
  print('speech', text)
  global audio_hub
  filename = getHash(text)
  if audio_hub is not None:
    response = await audio_hub.get_audio_list()
    if response and isinstance(response, dict):
        data_str = response.get('data', {}).get('data', '{}')
        audio_list = json.loads(data_str).get('audio_list', [])
        
        #filename = os.path.splitext(audio_file_path)[0]
        existing_audio = next((audio for audio in audio_list if audio['CUSTOM_NAME'] == filename), None)
        if existing_audio:
            print(f"Audio file {filename} already exists, skipping upload")
            uuid = existing_audio['UNIQUE_ID']
        else:
            print(f"Audio file {filename} not found, proceeding with upload")
            audio_file_path = tts(text = text, voice = voice, lang=lang)
            logger.info(f"Using audio file: {audio_file_path}")
            response = await audio_hub.upload_audio_file(audio_file_path)
            uuid = None
            print(response)
            response = await audio_hub.get_audio_list()
            if response and isinstance(response, dict):
                data_str = response.get('data', {}).get('data', '{}')
                audio_list = json.loads(data_str).get('audio_list', [])
            existing_audio = next((audio for audio in audio_list if audio['CUSTOM_NAME'] == filename), None)
            uuid = existing_audio['UNIQUE_ID']
    print(f"Starting audio playback of file: {uuid}")

    """
    if motion is not None:
      conn.datachannel.pub_sub.publish_without_callback(
        RTC_TOPIC["SPORT_MOD"], {
            "api_id": SPORT_CMD[motion]
        }
      )
    """      
    await audio_hub.play_by_uuid(uuid)
      
  return { "result" : True, "data" : True }  


@app.get("/color")
async def color(value = 'purple', warn = 0):
  global conn
  global lastColor 

  if lastColor == value:
    return

  print(warn)
  if int(warn) > 0:
    await speech("저한테 접근하면 위험하니, 조심해 주세요.", 'Content', 0,'ko')

  if conn is None:
    print('brightness', value)
  else:  
    lastColor = value
    await conn.datachannel.pub_sub.publish_request_new(
      RTC_TOPIC["VUI"], 
      {
        "api_id": 1007,
        "parameter": 
        {
            "color": value,
            #"time": 5,
            #"flash_cycle": 1000  # Flash every second
        }
      }
    )

  return { "result" : True, "data" : True }  

@app.get("/brightness")
async def brightness(value = 10):
  global conn

  if conn is None:
    print('brightness', value)
  else:
    await conn.datachannel.pub_sub.publish_request_new(
      RTC_TOPIC["VUI"], 
      {
          "api_id": 1005,
          "parameter": {"brightness": int(value)}
      }
    )

  return { "result" : True, "data" : True } 

@app.get("/mode")
async def mode(value = 'normal'):
  global conn
  if conn is None:
    print('mode', value)
  else:  
    conn.datachannel.pub_sub.publish_without_callback(
      RTC_TOPIC["MOTION_SWITCHER"], 
      {
          "api_id": 1002,
          "parameter": {"name": value}
      }
    )

  return { "result" : True, "data" : True }  

@app.get("/volume")
async def volume(value = 10):
  global conn
  if conn is None:
    print('volume', value)
  else:
    conn.datachannel.pub_sub.publish_without_callback(
      RTC_TOPIC["VUI"], 
      {
          "api_id": 1003,
          "parameter": {"volume": int(value)}
      }
    )

  return { "result" : True, "data" : True }  


@app.get("/monitor")
def monitor():
  return si.getAll()

@app.get("/v1/txt2chat", summary="문장 기반의 chatgpt 스타일 구현")
def txt2chat(prompt : str , isPlay = 0): # gen or med
  streamer = IterableStreamer(pipe_txt.get_tokenizer())

  messages = [
    {"role": "system", "content": _SYSTEM},
    {"role": "user", "content": prompt}
  ] 
  prompt = token_txt.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
  )

  print(prompt)

  generate_kwargs = dict(
      inputs = prompt,
      max_new_tokens= 256,
      temperature= 0.5,
      #do_sample=True,
      repetition_penalty=1.1,
      top_k=50,
      top_p=0.9,
      streamer=streamer, # !do_sample || top_k > 0
  )

  t1 = Thread(target=pipe_txt.generate, kwargs=generate_kwargs)
  t1.start()

  out = process_stream(streamer, False, isPlay)
  return StreamingResponse(out, media_type='text/event-stream')

"""
@app.post("/v1/txt2chat", summary="문장 기반의 chatgpt 스타일 구현 / batch ")
def txt2chat(chat : Chat, isPlay = 0): # gen or med
  print(chat)
  return StreamingResponse(generate_text_stream(chat, False, isPlay), media_type="text/plain")

@app.post("/v2/txt2chat", summary="문장 기반의 chatgpt 스타일 구현 / stream")
def txt2chat2(chat : Chat, isPlay = 0): # gen or med
  print(chat)
  return StreamingResponse(generate_text_stream(chat, True, isPlay), media_type="text/plain")
"""

@app.post("/v1/stt", summary="음성을 인식합니다.")
def stt(file : UploadFile = File(...), lang="ko", isPlay=0):
  start = t.time()
  location = f"uploads/{file.filename}"

  with open(location,"wb+") as file_object:
    file_object.write(file.file.read())
  
  raw_speech, samplerate = librosa.load(location, sr=16000)
  print('length',librosa.get_duration(y=raw_speech, sr=samplerate))
  raw =  raw_speech.tolist()

  out = pipe_stt.generate(
    raw,
    max_new_tokens=100,
    # 'task' and 'language' parameters are supported for multilingual models only
    language=f"<|{lang}|>",
    task="transcribe",
    #return_timestamps=True
    #streamer=streamer,
  )

  print(t.time()-start)


  chat = Chat()
  chat.prompt = str(out)

  return txt2chat(chat, isPlay)

  #return { "result" : True, "data" : str(out) }

import hashlib

def getHash(text):
  hash_func = hashlib.new('md5')
  hash_func.update(text.encode('utf-8'))
  return hash_func.hexdigest()

"""
    start = t.time()
    print(text, static)

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
    result = pipe_tts(inputs)
    print(f"Inference time: {t.time() - start_time:.4f} seconds")

    audio = list(result.values())[0].squeeze((0, 1))  

    if int(static) > 0:
      write(data=audio, rate=conf_tts.data.sampling_rate, filename=f"human.wav")
      return f"human.wav"
    else:
      write(data=audio, rate=conf_tts.data.sampling_rate, filename=f"output/{str(start)}.wav")
      return f"output/{str(start)}.wav"
"""

@app.get("/v1/tts", response_class=FileResponse, summary="입력한 문장으로 부터 음성을 생성합니다.")
def tts(text = "", voice=31, lang='ko', static=0, isPlay=0):
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
    result = pipe_tts(inputs)
    print(f"Inference time: {t.time() - start_time:.4f} seconds")

    audio = list(result.values())[0].squeeze((0, 1))  

    print(t.time() - start)
    
    if int(static) > 0:
      write(data=audio, rate=conf_tts.data.sampling_rate, filename="output/human.wav")
      return "output/human.wav"
    else:
      write(data=audio, rate=conf_tts.data.sampling_rate, filename=f"output/{filename}.wav")
      audio = AudioSegment.from_wav(f"output/{filename}.wav")
      # Set specific audio parameters for compatibility
      audio = audio.set_frame_rate(22050)  # Standard sample rate
      audio = audio.set_sample_width(2)
      audio = audio.set_channels(1)
      #wav_file_path = audiofile_path.replace('.mp3', '.wav')
      audio.export(f"output/{filename}.wav", format='wav', codec="pcm_s16le" )#parameters=["-ar", "44100"])
      if int(isPlay) > 0 :
        playsound(f"output/{filename}.wav")

      return f"output/{filename}.wav"

print("Loading Complete!")
