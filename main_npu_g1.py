from fastapi.middleware.cors import CORSMiddleware
from serverinfo import si
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from huggingface_hub import snapshot_download, hf_hub_download
import time as t
from pydantic import BaseModel, Field
import numpy as np
import utils
from playsound import playsound
from scipy.io.wavfile import write
from text import text_to_sequence
import json
from pydub import AudioSegment
from serverinfo import si
from unitree_webrtc_connect.webrtc_audiohub import WebRTCAudioHub
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, VUI_COLOR, SPORT_CMD
import logging
from aiortc import MediaStreamTrack
from requests import get
import time
import cv2
from openvino import Core
from fastapi.staticfiles import StaticFiles
#from queue import Queue
from asyncio import Queue
from ultralytics import YOLO, FastSAM
import openvino as ov
#from playsound import playsound
from mandro import HadnControler
import threading
from fastapi.middleware.cors import CORSMiddleware
import hashlib
import asyncio
import requests
from pyapriltags import Detector
import httpx  # httpx를 사용하여 비동기 HTTP 요청을 처리합니다.


#optimum-cli export openvino --weight-format int4 --task text-generation-with-past --model growdle/HyperCLOVAX-SEED-Text-Instruct-1.5B ./CLOVAX-1.5B-ov-int4
#kakaocorp/kanana-1.5-2.1b-instruct-2505
#https://github.com/Unitree-Go2-Robot/go2_robot


def getHash(text):
  hash_func = hashlib.new('md5')
  hash_func.update(text.encode('utf-8'))
  return hash_func.hexdigest()

_IP = "192.168.12.117" #12.128"#"192.168.12.112" "192.168.21.19/g1 plus"

print(_IP)

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
ppe_model = YOLO('./models/yolo11n-helmet4_int8_openvino_model') #ppe_model = YOLO('./models/safety-11s_int8_openvino_model')
class_names = det_model.names
ppe_names = ppe_model.names

is_collecting = False

detector = Detector(families="tag36h11") # 사용하는 태그 패밀리에 맞춰 설정

# Enable logging for debugging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

_PORT = int(open("port.txt", 'r').read())

# 원본 비디오 스트림 URL
#SOURCE_VIDEO_URL = "http://10.42.0.1:59511/video_feed"
SOURCE_VIDEO_URL = f"http://{_IP}:59511/video_raw"

conn = None
audio_hub = None
track = None
lastColor = 'cyan'
state = { "charge" : 0, "temp" : 0, "voltage" : 0, "cnt_live" : 0, "cnt_object" : 0,  "boxes" : [], 
         "human" : { "age" : "", "gender" : "", "emotion" : "", "position" : ""}, "tag" : { "id" : None, "dist" : 0} }
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
depth_queue = Queue(maxsize=5)
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


config = {"PERFORMANCE_HINT": "LATENCY"}
#path_tts = snapshot_download(repo_id="rippertnt/on-vits2-multi-tts-v1", allow_patterns="*ov*")
pipe_tts = ov.compile_model(ov.read_model("./models/all_base_ov.xml"), device_name="CPU", config=config)
conf_tts = utils.get_hparams_from_file("./models/all_base_ov.json")

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

    for mask, box, cls_idx, score, depth in zip(masks, boxes, classes, scores, depths):
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
            state["human"]["depth"] = depth
            state["human"]["position"] = position
        else:
            state["cnt_object"] += 1        

        state['boxes'].append({
            'class': class_name,
            'score': round(float(score), 2),
            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
            'position': position,  # 위치 정보 추가
            'depth' : depth
        }) 

        label = f"{class_name}:{score:.2f} | {depth:.2f}m"
        cv2.putText(overlay, label, (x1, max(15, y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return overlay

def get_mask_depths(masks, depth_frame, low_percentile=5):
    depths = []
    depth_image = depth_frame
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


import aiohttp

# 데이터 공유를 위한 비동기 큐 (최신 1프레임만 유지하여 지연 시간 최소화)
raw_data_queue = asyncio.Queue(maxsize=1)

async def fetch_combined_frame(session):
    """
    서버로부터 합쳐진 바이너리 데이터(RGB + Depth)를 받아 분리함.
    네트워크 IO 및 단순 메모리 슬라이싱만 수행하여 속도를 극대화함.
    """
    # 원본 해상도 규격 (서버와 일치해야 함)
    W, H = 640, 480
    RGB_SIZE = W * H * 3          # 921,600 bytes
    DEPTH_SIZE = W * H * 2        # 614,400 bytes (16bit)
    TOTAL_SIZE = RGB_SIZE + DEPTH_SIZE

    try:
        # 1.0초 타임아웃으로 연결 유지 확인
        async with session.get(SOURCE_VIDEO_URL, timeout=aiohttp.ClientTimeout(total=1.0)) as response:
            if response.status == 200:
                # 스트림 전체를 한 번에 읽음
                data = await response.read()
                
                # 데이터 크기 검증 (데이터가 깨지거나 덜 왔을 경우 대비)
                if len(data) >= TOTAL_SIZE:
                    # 1. RGB 분리 (uint8)
                    # .copy()를 사용하지 않고 슬라이싱만 하여 메모리 효율 증대
                    frame = np.frombuffer(data[:RGB_SIZE], dtype=np.uint8).reshape(H, W, 3)
                    
                    # 2. Depth 분리 (uint16)
                    depth_frame = np.frombuffer(data[RGB_SIZE:TOTAL_SIZE], dtype=np.uint16).reshape(H, W)
                    
                    return frame, depth_frame
                else:
                    print(f"Warning: Data incomplete ({len(data)}/{TOTAL_SIZE} bytes)")
            else:
                print(f"Server Error: HTTP {response.status}")
                
    except asyncio.TimeoutError:
        print("Fetch Timeout: 서버 응답이 너무 늦습니다.")
    except Exception as e:
        print(f"Fetch Error: {e}")
        
    return None, None

async def receiver_loop():
    """[수신부] 서버에서 바이너리 데이터를 받아 큐에 넣음 (네트워크 전용)"""
    print("============= Receiver Loop Started")
    connector = aiohttp.TCPConnector(limit=None, keepalive_timeout=30)
    async with aiohttp.ClientSession(connector=connector) as session:
        while True:
            try:
                frame, depth = await fetch_combined_frame(session)
                if frame is not None:
                    # 큐가 가득 찼다면 기존 프레임을 버리고 최신 것을 넣음 (Latency 방지)
                    if raw_data_queue.full():
                        raw_data_queue.get_nowait()
                    await raw_data_queue.put((frame, depth))
                else:
                    await asyncio.sleep(0.001)
            except Exception as e:
                print(f"Receiver Error: {e}")
                await asyncio.sleep(0.1)

async def processing_loop():
    """[분석부] 큐에서 데이터를 꺼내 NPU 추론 및 시각화 수행 (연산 전용)"""
    global cnt_image
    print("============= Processing Loop Started")
    tag_size = 0.12
    fx, fy = 600, 600  # 예시 초점 거리

    while True:
        # 1. 수신부로부터 데이터 획득 (데이터가 올 때까지 await)
        frame, depth_frame = await raw_data_queue.get()
        start_time = time.time()

        # 2. 전처리 최적화: INTER_NEAREST 사용 (CPU 부하 감소)
        # AI 모델 입력 규격에 맞게 리사이즈
        frame_ai = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_NEAREST)
        depth_ai = cv2.resize(depth_frame, (640, 640), interpolation=cv2.INTER_NEAREST)

        # 3. NPU 추론
        res = det_model(frame_ai, device="intel:npu", verbose=False, conf=0.25)[0]
        #ppe_res = ppe_model(frame_ai, device="intel:npu", verbose=False, conf=0.25)[0]

        # 4. 후처리 및 시각화 로직
        if hasattr(res, 'masks') and res.masks is not None:
            masks = res.masks.data.cpu().numpy().astype(np.uint8)
        else:
            masks = []

        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy()

        # 시각화 (원본 크기 640x480으로 직접 생성하여 불필요한 리사이즈 제거)
        out = visualize_segmentation(frame_ai, masks, boxes, classes, scores,  get_mask_depths(masks, depth_ai), class_names)

        # 4. PPE 결과 시각화 추가 (파란색 박스)
        # 4. PPE 결과 시각화 추가 (파란색 박스 + 클래스 이름)

        """
        if ppe_res.boxes is not None:
          ppe_boxes = ppe_res.boxes.xyxy.cpu().numpy()
          ppe_scores = ppe_res.boxes.conf.cpu().numpy()
          ppe_classes = ppe_res.boxes.cls.cpu().numpy().astype(int)
          ppe_names = ppe_model.names  # 클래스 이름 딕셔너리 획득
          
          for i, box in enumerate(ppe_boxes):
              x1, y1, x2, y2 = map(int, box)
              conf = ppe_scores[i]
              cls_id = ppe_classes[i]
              
              # 클래스 이름 가져오기 (없을 경우 ID 표시)
              label_text = ppe_names.get(cls_id, str(cls_id))

              if 'helmet' in label_text or 'vest' in label_text:
                  x1, y1, x2, y2 = map(int, box)
                  conf = ppe_scores[i]
                  display_str = f"{label_text.capitalize()}: {conf:.2f}"
                  
                  # 1) 파란색 박스 그리기
                  cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
                  
                  # 2) 텍스트 배경 (파란색 바)
                  (w, h), _ = cv2.getTextSize(display_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                  # 텍스트가 화면 위로 나가지 않도록 y좌표 보정
                  text_y = max(y1, h + 10)
                  cv2.rectangle(out, (x1, text_y - h - 10), (x1 + w, text_y), (255, 0, 0), -1)
                  
                  # 3) 텍스트 쓰기 (흰색)
                  cv2.putText(out, display_str, (x1, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        """
        gray = cv2.cvtColor(frame_ai, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)

        if tags:
            # 픽셀 면적 기준으로 가장 큰 태그 선택
            best_tag = max(tags, key=lambda t: cv2.contourArea(t.corners.astype(np.float32)))
            
            # --- 투명 노란색 채우기 로직 ---
            overlay = out.copy()
            # 태그의 네 모서리 좌표 가져오기
            pts = best_tag.corners.reshape((-1, 1, 2)).astype(np.int32)
            # 노란색(BGR: 0, 255, 255)으로 다각형 채우기
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            
            # 투명도 적용 (0.2 opacity)
            # out * 0.8 + overlay * 0.2
            out = cv2.addWeighted(overlay, 0.2, out, 0.8, 0)
            
            # --- 우측 하단 정보 표시 ---
            tag_id = best_tag.tag_id
            cx, cy = int(best_tag.center[0]), int(best_tag.center[1])
            
            # Depth 데이터 활용 (범위 체크 포함)
            dist = 0.0
            if 0 <= cy < 640 and 0 <= cx < 640:
                dist = depth_ai[cy, cx] / 1000.0  # mm -> m

            info_str = f"ID: {tag_id} / Dist: {dist:.2f}m"
            state["tag"]["id"] = tag_id
            state["tag"]["dist"] = dist
            (w, h), baseline = cv2.getTextSize(info_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # 우측 하단 여백 설정 (20px)
            text_x = 640 - w - 20
            text_y = 640 - 20
            
            # 가독성을 위한 검정색 배경 바 (약간의 투명도 가능)
            cv2.rectangle(out, (text_x - 10, text_y - h - 10), (640, 640), (0, 0, 0), -1)
            # 노란색 글씨로 정보 출력
            cv2.putText(out, info_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        # 얼굴 감지 (추가 최적화: 필요한 경우에만 수행하거나 해상도 최소화)
        resized_face = cv2.resize(frame_ai, (face_det_width, face_det_height), interpolation=cv2.INTER_NEAREST)
        input_tensor = np.expand_dims(resized_face.transpose((2, 0, 1)), 0)
        face_det_results = face_det_compiled_model(input_tensor)[face_det_output_layer]
        
        out = visualize_face(out, face_det_results)

        # 5. FPS 계산 및 출력
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(out, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 6. 결과 스트리밍 큐에 전달
        if processed_frame_queue.full():
            try: processed_frame_queue.get_nowait()
            except: pass
        
        # 마지막 출력 해상도로 조정하여 put
        await processed_frame_queue.put(cv2.resize(out, (640, 480)))
        
        cnt_image += 1
        if cnt_image % 100 == 0:
            cv2.imwrite("capture.jpg", frame)

@app.get("/")
def main():
  return { "result" : True, "data" : "AI-CPU-V2", "ip" : _IP, "port" : _PORT }      

@app.get("/connect2")
async def connect2():
  global conn
  #global audio_hub
  conn =  UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP) #Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.0.101")
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
  return { "result" : True, "data" : True }      

@app.get("/prepare2")
async def prepare2():
  return { "result" : True, "data" : True }    
  
@app.get("/hand")
async def hand(cmd : str):
  requests.get(f"http://{_IP}:59521/hands?cmd={cmd}")
  return { "result" : True }    

@app.get("/heartbeat")
async def heartbeat():
  global state
  print(state)
  return { "result" : True, "data" : state }        

@app.get("/video_feed")
async def video_feed():
    async def generate():
        while True:
            # 1. 비동기 큐에서 프레임을 가져올 때까지 기다립니다 (await)
            # .get()은 데이터가 들어올 때까지 이벤트 루프를 넘겨주고 대기합니다.
            output = await processed_frame_queue.get()
            
            if output is not None:
                # 2. 인코딩 (이 부분은 CPU 연산이므로 루프를 살짝 점유하지만 OK)
                _, img_bytes = cv2.imencode('.jpg', output)
                frame = img_bytes.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            # 3. 큐 작업 완료 표시 (Optional)
            processed_frame_queue.task_done()

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
        files = {"audio_file": (f"{filename}.wav", f, "audio/wav")}
        response = requests.post(f"http://{_IP}:59521/audio", files=files)

    return f"output/{filename}.wav"

@app.get("/led")
async def led(r : int = 0,g : int = 0,b : int = 0,):
  print(f"http://{_IP}:59521/led?r={r}&g={g}&b={b}")
  response = requests.get(f"http://{_IP}:59521/led?r={r}&g={g}&b={b}")
  return { "result" : True }

import matplotlib.colors

@app.get("/color")
async def color(value : str = 'red'):
  print(value)
  colors = matplotlib.colors.to_rgb(value)
  arr = (np.array(colors) * 255).astype(int)
  print("color", arr)
  response = requests.get(f"http://{_IP}:59521/led?r={arr[0]}&g={arr[1]}&b={arr[2]}")
  return { "result" : True }


@app.get("/start_collection")
async def start_frame_collection():
    """수신 및 분석 루프 시작"""
    global is_collecting
    
    # 중복 실행 방지
    if is_collecting:
        return {"message": "이미 프레임 수집 및 분석이 진행 중입니다"}
    
    # 1. 기존 큐 초기화 (Raw 데이터 큐와 처리된 결과 큐 모두)
    for q in [raw_data_queue, processed_frame_queue]:
        while not q.empty():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                break
    
    print("모든 큐 초기화 완료")
    
    is_collecting = True
    
    # 2. 비동기 테스크로 수신부와 분석부를 각각 실행
    # 이제 이 두 함수는 백그라운드에서 서로 데이터를 주고받으며 독립적으로 작동합니다.
    asyncio.create_task(receiver_loop())   # 서버에서 데이터 가져오기 전담
    asyncio.create_task(processing_loop()) # NPU 분석 및 시각화 전담
    
    return {"message": "수신 및 분석 파이프라인이 시작되었습니다."}

print("NPU","2502010900")
