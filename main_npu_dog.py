from fastapi.middleware.cors import CORSMiddleware
from serverinfo import si
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from huggingface_hub import snapshot_download, hf_hub_download
import time as t
import collections
from pydantic import BaseModel, Field
import numpy as np
import utils
from playsound import playsound
from scipy.io.wavfile import write
from text import text_to_sequence
import json
from pydub import AudioSegment
from serverinfo import si
from go2_webrtc_driver.webrtc_audiohub import WebRTCAudioHub
import logging
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
#from playsound import playsound
from mandro import HadnControler
import threading
from fastapi.middleware.cors import CORSMiddleware
import hashlib
import asyncio
import requests

def getHash(text):
  hash_func = hashlib.new('md5')
  hash_func.update(text.encode('utf-8'))
  return hash_func.hexdigest()

hL = None
hR = None

ov = Core()

FACE_DETECTION_MODEL_XML = "./models/face-detection-retail-0005/FP16-INT8/face-detection-retail-0005.xml"
AGE_GENDER_MODEL_XML = "./models/age-gender-recognition-retail-0013/FP16-INT8/age-gender-recognition-retail-0013.xml"
EMOTION_MODEL_XML = "./models/emotions-recognition-retail-0003/FP16-INT8/emotions-recognition-retail-0003.xml"

DEVICE = "NPU"
# 생명체로 간주할 클래스명 (클래스 이름은 모델에 따라 다를 수 있음!)
LIVING_CLASSES = {'person', 'cat', 'dog', 'bird', 'teddy bear', 'cow', 'sheep', 'horse'}
EMOTIONS = ['neutral', 'happy', 'sad', 'surprise', 'anger']

# 얼굴 탐지 모델
face_det_model = ov.read_model(model=FACE_DETECTION_MODEL_XML)
face_det_compiled_model = ov.compile_model(model=face_det_model, device_name=DEVICE)
face_det_input_layer = face_det_compiled_model.input(0)
face_det_output_layer = face_det_compiled_model.output(0)
face_det_height, face_det_width = list(face_det_input_layer.shape)[2:]

# 나이/성별 모델
age_gender_model = ov.read_model(model=AGE_GENDER_MODEL_XML)
age_gender_compiled_model = ov.compile_model(model=age_gender_model, device_name=DEVICE)
age_gender_input_layer = age_gender_compiled_model.input(0)
age_output_layer = age_gender_compiled_model.output("age_conv3")
gender_output_layer = age_gender_compiled_model.output("prob")
age_gender_height, age_gender_width = list(age_gender_input_layer.shape)[2:]

# 감정 모델
emotion_model = ov.read_model(model=EMOTION_MODEL_XML)
emotion_compiled_model = ov.compile_model(model=emotion_model, device_name=DEVICE)
emotion_input_layer = emotion_compiled_model.input(0)
emotion_output_layer = emotion_compiled_model.output(0)
emotion_height, emotion_width = list(emotion_input_layer.shape)[2:]

det_model = YOLO('./models/yolo11s-seg_int8_openvino_model')
det_model("capture.jpg", device='intel:cpu', imgsz=640) # error 방지용
class_names = det_model.names

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
  "Release_Arm" : 99,
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
cnt_image = 0

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

def visualize_face(frame,face_det_results):
    global state
    h, w, _ = frame.shape

    state["human"]["gender"] = ""
    state["human"]["age"] = ""
    state["human"]["emotion"] = ""   

    for detection in face_det_results[0][0]:  # OpenVINO 출력 형식에 맞게 인덱싱
        confidence = detection[2]  # 신뢰도
        if confidence > 0.5:  # 신뢰도 임계값
            xmin = int(detection[3] * w)
            ymin = int(detection[4] * h)
            xmax = int(detection[5] * w)
            ymax = int(detection[6] * h)
            
            # 얼굴 이미지 자르기 (유효한 범위 내에서)
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            face_img = frame[ymin:ymax, xmin:xmax]
            
            if face_img.size > 0:
                # 나이/성별 모델 추론
                resized_age_gender = cv2.resize(face_img, (age_gender_width, age_gender_height))
                # 입력 형태: [1, 3, H, W]
                ag_input_tensor = np.expand_dims(resized_age_gender.transpose((2, 0, 1)), 0)
                ag_results = age_gender_compiled_model(ag_input_tensor)
                
                # 결과 파싱
                age_pred = int(ag_results[age_output_layer].reshape(1)[0] * 100) # OpenVINO age-gender 모델은 나이값을 100으로 나눈 값으로 출력.
                gender_prob = ag_results[gender_output_layer].reshape(-1) # [female_prob, male_prob] 형태로 변환
                
                # OpenVINO documentation: prob output across 2 type classes [0 - female, 1 - male].
                gender_idx = np.argmax(gender_prob)
                gender = "W" if gender_idx == 0 else "M"
                
                # 감정 모델 추론
                resized_emotion = cv2.resize(face_img, (emotion_width, emotion_height))
                emotion_input_tensor = np.expand_dims(resized_emotion.transpose((2, 0, 1)), 0)
                emotion_results = emotion_compiled_model(emotion_input_tensor)
                
                # 결과 파싱
                emotion_prob = emotion_results[emotion_output_layer].reshape(-1)
                emotion = EMOTIONS[np.argmax(emotion_prob)]
                
                # 결과값으로 박스 및 텍스트 그리기
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
                text = f"{gender}, {age_pred}y, {emotion}"
                cv2.putText(frame, text, (xmin, ymax - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
      

                state["human"]["gender"] = gender
                state["human"]["age"] = age_pred
                state["human"]["emotion"] = emotion
    return frame      


def visualize_segmentation(frame, masks, boxes, classes, scores, depths, class_names, alpha=0.5):
    global state
    overlay = frame.copy()

    state['boxes'] = []
    state["cnt_object"] = 0
    state["cnt_live"] = 0

    state["human"]["depth"] = ""
    state["human"]["position"] = ""

    for mask, box, cls_idx, score in zip(masks, boxes, classes, scores):
        class_name = class_names[cls_idx]

        is_living = class_name in LIVING_CLASSES
        color = (0, 0, 255) if is_living else (0, 255, 0)  # 빨강 vs 초록

        # 마스크 적용
        overlay[mask == 1] = (overlay[mask == 1] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        position = ""

        height, width, channels = frame.shape # 이 부분은 이미 있다고 가정합니다.

        # 3x3 그리드를 위한 cell 높이와 너비 계산
        # 정수 나누기를 사용하여 픽셀 단위로 구합니다.
        cell_h = height // 3
        cell_w = width // 3

        lastTime = time.time()

        # 중심 좌표 계산
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 위치 계산 (grid 3x3)
        if cy < cell_h:
            row = 'T'
        elif cy < 2 * cell_h:
            row = 'C'
        else:
            row = 'B'

        if cx < cell_w:
            col = 'L'
        elif cx < 2 * cell_w:
            col = 'C'
        else:
            col = 'R'

        position = row + col  # ex: "TC", "BR", etc.

        if is_living:  
            state["cnt_live"] += 1          
            #state["human"]["depth"] = depth
            state["human"]["position"] = position
        else:
            state["cnt_object"] += 1        

        state['boxes'].append({
            'class': class_name,
            'score': round(float(score), 2),
            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
            'position': position,  # 위치 정보 추가
            #'depth' : depth
        }) 

        label = f"{class_name}:{score:.2f}"
        cv2.putText(overlay, label, (x1, max(15, y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return overlay

def get_mask_depths(masks, depth_frame, low_percentile=5):
    depths = []
    depth_image = np.asanyarray(depth_frame.get_data())
    for mask in masks:
        if mask.sum() == 0:
            depths.append(0.0)
            continue
        depth_values = depth_image[mask == 1]
        valid = depth_values[depth_values > 0]
        if len(valid) > 0:
            low_thresh = np.percentile(valid, low_percentile)  # 예: 5번째 백분위수
            filtered = valid[valid >= low_thresh]
            if len(filtered) > 0:
                closest_depth_m = np.min(filtered) / 1000.0
            else:
                closest_depth_m = np.min(valid) / 1000.0
        else:
            closest_depth_m = 0.0
        depths.append(closest_depth_m)
    return depths

def processing_thread():
    global cnt_live, cnt_object, lastTime, state, cnt_image
    processing_times = collections.deque()

    print("============= processing....")

    while True:
        if not frame_queue.empty():
          frame = np.array(frame_queue.get())
          cnt_live = 0
          cnt_object = 0

          if cnt_image % 100 == 0:
            cv2.imwrite("capture.jpg", frame) #cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

          cnt_image = cnt_image + 1

          start_time = time.time()

          results = det_model(frame, device="intel:npu", imgsz=640, verbose=False, conf=0.3)  # 작을수록 빠름
          res = results[0]

          if hasattr(res, 'masks') and res.masks is not None:
              masks = res.masks.data.cpu().numpy().astype(np.uint8)
          else:
              masks = []

          boxes = res.boxes.xyxy.cpu().numpy()
          classes = res.boxes.cls.cpu().numpy().astype(int)
          scores = res.boxes.conf.cpu().numpy()

          # 시각화
          out = visualize_segmentation(frame, masks, boxes, classes, scores, None, class_names)

          # 얼굴 감지 모델 추론
          resized_frame = cv2.resize(frame, (face_det_width, face_det_height))
          input_tensor = np.expand_dims(resized_frame.transpose((2, 0, 1)), 0)
          face_det_results = face_det_compiled_model(input_tensor)[face_det_output_layer]

          out = visualize_face(out, face_det_results)

          # FPS 계산 및 표시
          curr_time = time.time()
          fps = 1.0 / (curr_time - start_time)
          cv2.putText(out, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

          if processed_frame_queue.full():
              processed_frame_queue.get()  # 가장 오래된 프레임 제거   

          processed_frame_queue.put(out)
        else:
            time.sleep(0.001)



@app.get("/")
def main():
  return { "result" : True, "data" : "AI-CPU-V2", "ip" : _IP, "port" : _PORT }      

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

@app.get("/prepare2")
async def prepare2():

  global hL
  global hR

  try:
      hL = HadnControler('/dev/ttyACM2') # L 컨트롤러 L동글 부터 연결
      hR = HadnControler('/dev/ttyACM3') # R 컨트롤러
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
    result = pipe_tts(inputs)
    print(f"Inference time: {t.time() - start_time:.4f} seconds")

    audio = list(result.values())[0].squeeze((0, 1))  

    print(t.time() - start)
    write(data=audio, rate=conf_tts.data.sampling_rate, filename=f"output/{filename}.wav")

    # 파일 열고 전송
    with open(f"output/{filename}.wav", "rb") as f:
        files = {"audio_file": (f"{filename}.wav", f, "audio/wav")} #10.42.0.1
        response = requests.post("http://192.168.12.128:59521/audio", files=files)

    return f"output/{filename}.wav"


import httpx  # httpx를 사용하여 비동기 HTTP 요청을 처리합니다.

# 원본 비디오 스트림 URL
SOURCE_VIDEO_URL = "http://192.168.12.128:59511/video_feed"
#SOURCE_VIDEO_URL = "http://192.168.12.117:59511/video_feed"

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
                                    frame = cv2.resize(frame, (640, 384)) 
                                    frame_queue.put_nowait(frame)
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
