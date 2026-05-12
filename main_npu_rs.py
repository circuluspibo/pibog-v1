from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
import time
import time as t
import numpy as np
import subprocess
import collections
from pydub import AudioSegment
from serverinfo import si
import asyncio
import logging
from aiortc import MediaStreamTrack
import time
import cv2
from openvino import Core
from fastapi.staticfiles import StaticFiles
from queue import Queue
from ultralytics import YOLO
import threading
from fastapi.middleware.cors import CORSMiddleware
import hashlib
import asyncio
import pyrealsense2 as rs
import time
from collections import deque

is_collecting = False
collection_task = None

fps_buffer = deque()  # (timestamp, fps)
fps_lock = threading.Lock()

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
class_names = det_model.names


state = { "charge" : 0, "temp" : 0, "voltage" : 0, "cnt_live" : 0, "cnt_object" : 0,  "boxes" : [], 
         "human" : { "age" : "", "gender" : "", "emotion" : "", "position" : ""} }

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
    depth_image = np.asanyarray(depth_frame.get_data())  # (480, 640)
    
    for mask in masks:
        if mask.sum() == 0:
            depths.append(0.0)
            continue
        
        # mask 크기(640x640)에 맞게 depth_image 리사이즈
        mask_h, mask_w = mask.shape
        depth_resized = cv2.resize(
            depth_image, 
            (mask_w, mask_h), 
            interpolation=cv2.INTER_NEAREST  # depth는 nearest 보간 사용
        )
        
        depth_values = depth_resized[mask == 1]
        valid = depth_values[depth_values > 0]
        if len(valid) > 0:
            low_thresh = np.percentile(valid, low_percentile)
            filtered = valid[valid >= low_thresh]
            if len(filtered) > 0:
                closest_depth_m = np.min(filtered) / 1000.0
            else:
                closest_depth_m = np.min(valid) / 1000.0
        else:
            closest_depth_m = 0.0
        depths.append(closest_depth_m)
    return depths


# ===== 수정 2: processed_frame_queue 옆에 depth_frame_queue 추가 =====



# ===== 수정 3: processing_thread 안에서 depth 시각화 프레임 저장 =====
# out = visualize_face(out, face_det_results) 이후에 아래 코드 추가



# ===== 수정 4: depth_feed 엔드포인트 추가 =====

@app.get("/depth_feed")
async def depth_feed():
    def generate():
        while True:
            if not depth_frame_queue.empty():
                output = depth_frame_queue.get()
                _, img_bytes = cv2.imencode('.jpg', output)
                frame = img_bytes.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                time.sleep(0.01)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


ser = None

def getHash(text):
  hash_func = hashlib.new('md5')
  hash_func.update(text.encode('utf-8'))
  return hash_func.hexdigest()

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

_IP = "127.0.0.1" #si.getIP()
_PORT = int(open("port.txt", 'r').read())

processed_frame_queue = Queue(maxsize=5)
depth_frame_queue = Queue(maxsize=5)      # ← 추가

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

def processing_thread():
    global cnt_live, cnt_object, lastTime, state, cnt_image

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    print("============= processing....")  

    while True:
        cnt_live = 0
        cnt_object = 0
        boxes = []

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())        

        if cnt_image % 100 == 0:
            cv2.imwrite("capture.jpg", frame) #cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cnt_image = cnt_image + 1

        start_time = time.time()

        frame = cv2.resize(frame, (640, 640))
        res = det_model(frame, device="intel:npu", verbose=False, conf=0.25)[0] #, imgsz=640

        if hasattr(res, 'masks') and res.masks is not None:
            masks = res.masks.data.cpu().numpy().astype(np.uint8)
        else:
            masks = []

        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy()

        # 시각화
        out = visualize_segmentation(frame, masks, boxes, classes, scores, get_mask_depths(masks, depth_frame), class_names)

        # 얼굴 감지 모델 추론
        resized_frame = cv2.resize(frame, (face_det_width, face_det_height))
        input_tensor = np.expand_dims(resized_frame.transpose((2, 0, 1)), 0)
        face_det_results = face_det_compiled_model(input_tensor)[face_det_output_layer]

        out = visualize_face(out, face_det_results)

        # --- depth 컬러맵 프레임 생성 ---
        depth_image_raw = np.asanyarray(depth_frame.get_data())           # (480, 640)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image_raw, alpha=0.03),
            cv2.COLORMAP_JET
        )
        depth_colormap_resized = cv2.resize(depth_colormap, (640, 480))

        if depth_frame_queue.full():
            depth_frame_queue.get()
        depth_frame_queue.put(depth_colormap_resized)
             
             

        # FPS 계산 및 표시
        curr_time = time.time()
        fps = 1.0 / (curr_time - start_time)
        cv2.putText(out, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if processed_frame_queue.full():
            processed_frame_queue.get()  # 가장 오래된 프레임 제거   

        processed_frame_queue.put(cv2.resize(out, (640, 480)))

@app.get("/")
def main():
  return { "result" : True, "data" : "AI-NPU-MCR-V2", "ip" : _IP, "port" : _PORT }      


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

def fps_saver_thread():
    while True:
        time.sleep(30)  # 30초마다 실행
        
        with fps_lock:
            if len(fps_buffer) == 0:
                continue
            
            # 버퍼 복사
            data_to_save = list(fps_buffer)
            fps_buffer.clear()

        # 파일 저장 (버퍼는 잠금 없이! → 실시간 추론 안느려짐)
        filename = time.strftime("fps_log_%Y%m%d_%H%M%S.txt")
        with open(filename, "w") as f:
            for ts, fps in data_to_save:
                f.write(f"{ts},{fps}\n")

        print(f"[FPS Saver] {filename} saved with {len(data_to_save)} records")

def toFloat(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return value

@app.get("/monitor")
def monitor():
  return si.getAll()


@app.get("/start_collection")
async def start_frame_collection():
    """프레임 수집 시작"""
    global is_collecting, collection_task
    
    if is_collecting:
        return {"message": "이미 프레임 수집이 진행 중입니다"}

    
    is_collecting = True
    threading.Thread(target=fps_saver_thread, daemon=True).start()         
    threading.Thread(target=processing_thread, daemon=True).start()
         

    return {"message": "프레임 수집을 시작했습니다"}

@app.on_event("startup")
async def startup_event():
    global is_collecting
    is_collecting = True
    threading.Thread(target=fps_saver_thread, daemon=True).start()
    threading.Thread(target=processing_thread, daemon=True).start()
    print("✅ 프레임 수집 자동 시작")

print("Loading Complete","NPU")
