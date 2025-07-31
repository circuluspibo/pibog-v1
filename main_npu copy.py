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
import utils
import commons
from scipy.io.wavfile import write
from text import text_to_sequence
import torch
import json
from pydub import AudioSegment
from serverinfo import si
#import onnxruntime as rt
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
from fastapi.middleware.cors import CORSMiddleware
import hashlib
import aiohttp
import asyncio
import requests
#optimum-cli export openvino --weight-format int4 --task text-generation-with-past --model growdle/HyperCLOVAX-SEED-Text-Instruct-1.5B ./CLOVAX-1.5B-ov-int4
#kakaocorp/kanana-1.5-2.1b-instruct-2505
#https://github.com/Unitree-Go2-Robot/go2_robot


def getHash(text):
  hash_func = hashlib.new('md5')
  hash_func.update(text.encode('utf-8'))
  return hash_func.hexdigest()

hL = None
hR = None
core = ov.Core()

det_ov_model = core.read_model('yolo12m_int8_openvino_model/yolo12m.xml')
det_model = YOLO('yolo12m_int8_openvino_model', task='detect')
sam_model = FastSAM("./FastSAM-s_int8_openvino_model")  # or FastSAM-x.pt

det_ov_model.reshape({0: [1, 3, 384, 640]})
#det_ov_model.reshape({0: [1, 3, 480, 640]})

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,  # 쿠키나 자격 증명 허용
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # 허용할 HTTP 메소드
    allow_headers=["*"],  # 모든 헤더 허용
)


core = Core()
config = {"PERFORMANCE_HINT": "LATENCY"}
path_tts = snapshot_download(repo_id="rippertnt/on-vits2-multi-tts-v1", allow_patterns="*ov*")
pipe_tts = core.compile_model(core.read_model(model=f"{path_tts}/all_base_ov.xml"), device_name="CPU", config=config)
conf_tts = utils.get_hparams_from_file(hf_hub_download(repo_id="rippertnt/on-vits2-multi-tts-v1", filename="all_base.json"))

# 큐들 - 파이프라인 단계별로 분리
ai_input_queue = Queue(maxsize=5)      # NPU로 보낼 원본 이미지
ai_result_queue = Queue(maxsize=5)     # NPU 결과를 CPU로 전달
final_output_queue = Queue(maxsize=5)  # 최종 처리된 프레임

def npu_processing_thread():
    """NPU에서 AI 모델 추론만 담당"""
    while True:
        if not ai_input_queue.empty():
            try:
                frame_data = ai_input_queue.get(timeout=0.1)
                image = frame_data['image']
                timestamp = frame_data['timestamp']
                
                # NPU에서 AI 모델 실행
                start_time = time.time()
                det_results = det_model(image, verbose=False)[0]
                sam_results = sam_model(image, verbose=False, device="intel:npu", retina_masks=True, imgsz=640, conf=0.6, iou=0.9)[0]
                inference_time = time.time() - start_time
                
                # 결과를 CPU 처리 스레드로 전달
                result_data = {
                    'image': image,
                    'det_results': det_results,
                    'sam_results': sam_results,
                    'inference_time': inference_time,
                    'timestamp': timestamp
                }
                
                if ai_result_queue.full():
                    ai_result_queue.get()  # 오래된 결과 제거
                ai_result_queue.put(result_data)
                
            except ai_result_queue.Empty:
                continue
        else:
            time.sleep(0.001)

def cpu_processing_thread():
    """CPU에서 후처리 및 시각화 담당"""
    global cnt_live, cnt_object, lastTime, state
    processing_times = collections.deque(maxlen=200)
    
    while True:
        if not ai_result_queue.empty():
            try:
                result_data = ai_result_queue.get(timeout=0.1)
                
                image = result_data['image']
                det_results = result_data['det_results']
                sam_results = result_data['sam_results']
                inference_time = result_data['inference_time']
                
                cnt_live = 0
                cnt_object = 0
                
                # CPU에서 결과 처리 (렌더링, 카운팅 등)
                start_post_time = time.time()
                output = process_detection_results(image, det_results)
                output = process_segmentation_results(output, sam_results)
                post_processing_time = time.time() - start_post_time
                
                # 상태 업데이트
                state["cnt_live"] = cnt_live
                state["cnt_object"] = cnt_object
                
                # 성능 정보 추가
                total_time = inference_time + post_processing_time
                processing_times.append(total_time)
                
                output = add_performance_info(output, processing_times)
                
                # 최종 출력 큐에 추가
                if final_output_queue.full():
                    final_output_queue.get()
                final_output_queue.put(output)
                
            except final_output_queue.Empty:
                continue
        else:
            time.sleep(0.001)

def process_detection_results(image, results):
    """Detection 결과 처리 (CPU 작업)"""
    global cnt_live, cnt_object, lastTime
    
    names = det_model.names
    output = image.copy()
    
    highlight_classes = ['person', 'dog', 'cat', 'horse', 'cow', 'sheep', 'bird', 'elephant', 'bear', 'zebra', 'giraffe', 'teddy bear']
    
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
    
    return output

def process_segmentation_results(output, sam_results):
    """Segmentation 결과 처리 (CPU 작업)"""
    if sam_results.masks is None:
        return output
        
    masks = sam_results.masks.data.cpu().numpy()
    
    for mask in masks:
        colored_mask = (mask * 255).astype(np.uint8)
        
        # 컬러 마스크 생성
        color_layer = np.zeros_like(output, dtype=np.uint8)
        color_layer[:, :] = (0, 255, 0)
        
        # 마스크 적용
        mask_3ch = cv2.merge([colored_mask] * 3)
        masked_color = cv2.bitwise_and(color_layer, mask_3ch)
        
        # 오버레이에 컬러 마스크 반영
        output = np.where(mask_3ch > 0, cv2.addWeighted(output, 1 - 0.3, masked_color, 0.3, 0), output)
    
    return output

def add_performance_info(output, processing_times):
    """성능 정보 추가 (CPU 작업)"""
    _, f_width = output.shape[:2]
    
    if processing_times:
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
    
    return output

def input_processing_thread():
    """메인 스레드 - 입력 프레임을 NPU 큐로 전달"""
    while True:
        if not frame_queue.empty():
            image = np.array(frame_queue.get())
            
            frame_data = {'image': image,'timestamp': time.time()}
            
            # NPU 큐가 가득 차면 오래된 프레임 제거
            if ai_input_queue.full():
                ai_input_queue.get()
            ai_input_queue.put(frame_data)
        else:
            time.sleep(0.001)

def processing_thread():
    """기존 인터페이스 호환성을 위한 메인 함수 - 파이프라인 스레드들 시작"""
    npu_thread = threading.Thread(target=npu_processing_thread, daemon=True)
    cpu_thread = threading.Thread(target=cpu_processing_thread, daemon=True)
    input_thread = threading.Thread(target=input_processing_thread, daemon=True)
    
    npu_thread.start()
    cpu_thread.start() 
    input_thread.start()
    
    # 메인 스레드에서 최종 결과를 processed_frame_queue에 전달
    while True:
        if not final_output_queue.empty():
            output = final_output_queue.get()
            
            if processed_frame_queue.full():
                processed_frame_queue.get()  # 가장 오래된 프레임 제거   
            processed_frame_queue.put(output)
        else:
            time.sleep(0.001)


@app.get("/")
def main():
  return { "result" : True, "data" : "AI-CPU-V2", "ip" : _IP, "port" : _PORT }      


def fetch_frames():
    print("streaming start......")
    with requests.get("http://127.0.0.1:59521/video_feed", stream=True) as response:
        if response.status_code == 200:
            # 서버 A에서 오는 스트리밍을 하나씩 받아 큐에 넣음
            frame_buffer = b''  # 비디오 프레임을 이어서 받기 위한 버퍼
            for chunk in response.iter_content(chunk_size=1024):
                frame_buffer += chunk
                # JPEG 데이터가 하나의 프레임을 완성한 경우
                if b'\xff\xd9' in frame_buffer:  # JPEG의 끝 마커
                    try:
                        # 받은 데이터를 디코딩하여 이미지로 변환
                        image = cv2.imdecode(np.frombuffer(frame_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if image is not None:
                            # 큐에 디코딩된 이미지를 넣음

                            if frame_queue.full():
                              frame_queue.get()  # 가장 오래된 프레임 제거  
                            # object detection 이 가능하도록 사이즈 변환
                            frame_queue.put(image)
                    except Exception as e:
                        print(f"Error decoding image: {e}")
                    frame_buffer = b''  # 다음 프레임을 받기 위해 버퍼 초기화
# 비디오 프레임을 가져오는 스레드를 시작
thread = Thread(target=fetch_frames, daemon=True)
thread.start()


# Async function to receive video frames and put them in the queue
async def recv_camera_stream(track: MediaStreamTrack):
    while True:
        frame = await track.recv()
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

@app.get("/connect3")
async def connect3():
    print("connecting 3....")
    url = "http://127.0.0.1:59521/video_feed"
    #await recv_mjpeg_stream(url)


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
  out = 0
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



import httpx  # httpx를 사용하여 비동기 HTTP 요청을 처리합니다.

# 원본 비디오 스트림 URL
SOURCE_VIDEO_URL = "http://127.0.0.1:59521/video_feed"

async def proxy_video_stream():
    """원본 서버에서 비디오 스트림을 받아서 다시 스트리밍"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            async with client.stream("GET", SOURCE_VIDEO_URL) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    yield chunk
        except httpx.RequestError as e:
            print(f"비디오 스트림 연결 오류: {e}")
            # 연결 오류 시 빈 프레임이나 오류 메시지를 반환할 수 있습니다
            yield b""
        except Exception as e:
            print(f"예상치 못한 오류: {e}")
            yield b""

@app.get("/video_feed2")
async def video_feed():
    """비디오 스트림을 프록시하여 제공"""
    return StreamingResponse(
        proxy_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
# 서버 B에서 비디오 스트리밍 시작

is_collecting = False
collection_task = None

def parse_mjpeg_boundary(buffer):
    """boundary=frame 형식의 MJPEG 스트림 파싱"""
    frames = []
    remaining_buffer = buffer
    
    # boundary 패턴들
    boundary_patterns = [
        b'--frame\r\n',
        b'--frame\n', 
        b'\r\n--frame\r\n',
        b'\n--frame\n'
    ]
    
    while True:
        # boundary 찾기
        boundary_pos = -1
        next_boundary_pos = -1
        
        for pattern in boundary_patterns:
            pos = remaining_buffer.find(pattern)
            if pos != -1:
                boundary_pos = pos
                # 다음 boundary 찾기
                next_pos = remaining_buffer.find(pattern, pos + len(pattern))
                if next_pos != -1:
                    next_boundary_pos = next_pos
                    break
        
        if boundary_pos == -1 or next_boundary_pos == -1:
            break
            
        # 현재 프레임 데이터 추출
        frame_data = remaining_buffer[boundary_pos:next_boundary_pos]
        
        # Content-Type과 Content-Length 헤더 건너뛰기
        header_end = frame_data.find(b'\r\n\r\n')
        if header_end == -1:
            header_end = frame_data.find(b'\n\n')
        
        if header_end != -1:
            jpeg_data = frame_data[header_end + 4:]  # 헤더 이후 데이터
            
            # JPEG 시작 마커 확인
            jpeg_start = jpeg_data.find(b'\xff\xd8')
            if jpeg_start != -1:
                jpeg_data = jpeg_data[jpeg_start:]
                
                # JPEG 끝 마커 찾기
                jpeg_end = jpeg_data.find(b'\xff\xd9')
                if jpeg_end != -1:
                    complete_jpeg = jpeg_data[:jpeg_end + 2]
                    
                    try:
                        # OpenCV로 디코딩
                        nparr = np.frombuffer(complete_jpeg, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            # BGR을 RGB로 변환
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame)
                            #print(f"✓ 프레임 추출 성공: {frame.shape}")
                        else:
                            print("프레임 디코딩 실패")
                            
                    except Exception as e:
                        print(f"프레임 디코딩 오류: {e}")
        
        # 처리된 부분 제거
        remaining_buffer = remaining_buffer[next_boundary_pos:]
    
    return frames, remaining_buffer

async def collect_frames():
    """원본 서버에서 프레임을 수집하여 큐에 저장"""
    global is_collecting
    
    buffer = b""
    frame_count = 0
    chunk_count = 0
    
    try:
        print("비디오 스트림 연결 시작...")
        
        # 더 긴 타임아웃 설정
        timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
        
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            print(f"연결 시도: {SOURCE_VIDEO_URL}")
            
            async with client.stream("GET", SOURCE_VIDEO_URL) as response:
                print(f"응답 상태: {response.status_code}")
                print(f"응답 헤더: {dict(response.headers)}")
                
                if response.status_code != 200:
                    print(f"HTTP 오류: {response.status_code}")
                    return
                
                response.raise_for_status()
                
                print("스트림 읽기 시작...")
                
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    if not is_collecting:
                        print("수집 중지 요청")
                        break
                    
                    chunk_count += 1
                    buffer += chunk
                    
                    # 5초마다 상태 출력
                    if chunk_count % 100 == 0:
                        #print(f"청크 {chunk_count}개 수신, 버퍼 크기: {len(buffer)} bytes")
                        
                        # 버퍼에서 boundary 패턴 확인 (디버깅용)
                        if b'--frame' in buffer:
                            boundary_count = buffer.count(b'--frame')
                            #print(f"발견된 boundary 개수: {boundary_count}")
                        
                        # 버퍼의 처음 200바이트 출력 (디버깅용)
                        if len(buffer) > 200:
                            sample = buffer[:200]
                            #print(f"버퍼 샘플: {sample[:100]}")
                            #if b'Content-Type' in sample:
                            #    print("Content-Type 헤더 발견")
                    
                    # 버퍼가 너무 커지지 않도록 제한
                    if len(buffer) > 2 * 1024 * 1024:  # 2MB 제한
                        #print("버퍼 크기 제한, 일부 제거")
                        # 버퍼의 앞쪽 절반 제거
                        buffer = buffer[len(buffer)//2:]
                    
                    # 최소 버퍼 크기에 도달했을 때만 파싱 시도
                    if len(buffer) > 1000:  # 최소 1KB
                        try:
                            frames, buffer = parse_mjpeg_boundary(buffer)
                            
                            for frame in frames:
                                frame_count += 1
                                
                                # 큐가 꽉 찬 경우 오래된 프레임 제거
                                while frame_queue.full():
                                    try:
                                        dropped = frame_queue.get_nowait()
                                        #print("오래된 프레임 드랍")
                                    except frame_queue.Empty:
                                        break
                                
                                try:
                                    frame_queue.put_nowait(cv2.resize(frame, (640, 384)) )
                                    #print(f"✓ 프레임 #{frame_count} 큐에 추가 (크기: {frame_queue.qsize()}/{frame_queue.maxsize})")
                                except frame_queue.Full:
                                    print("큐 풀 - 프레임 스킵")
                        
                        except Exception as e:
                            print(f"프레임 파싱 오류: {e}")
                            # 파싱 오류시 버퍼 일부 제거
                            if len(buffer) > 5000:
                                buffer = buffer[1000:]
                    
                    # CPU 사용률 조절
                    if chunk_count % 50 == 0:
                        await asyncio.sleep(0.01)
                        
                print("스트림 종료")
                    
    except httpx.TimeoutException as e:
        print(f"타임아웃 오류: {e}")
    except httpx.ConnectError as e:
        print(f"연결 오류: {e}")
    except httpx.RequestError as e:
        print(f"요청 오류: {e}")
    except Exception as e:
        print(f"예상치 못한 오류: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        is_collecting = False
        print(f"프레임 수집 종료. 총 {frame_count}개 프레임 처리됨")

@app.get("/start_collection")
async def start_frame_collection():
    """프레임 수집 시작"""
    global is_collecting, collection_task
    
    if is_collecting:
        return {"message": "이미 프레임 수집이 진행 중입니다"}
    
    # 기존 큐 비우기
    cleared = 0
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
            cleared += 1
        except frame_queue.Empty:
            break
    
    print(f"큐 초기화: {cleared}개 프레임 제거")
    
    is_collecting = True
    task1 = asyncio.create_task(collect_frames())
    threading.Thread(target=processing_thread, daemon=True).start()
    
    return {"message": "프레임 수집을 시작했습니다"}



print("Loading Complete","NPU")
