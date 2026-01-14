from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import time
import numpy as np
import cv2
from openvino import Core
from fastapi.staticfiles import StaticFiles
from queue import Queue
from ultralytics import YOLO
import threading
import csv
import os
from monitor import CPUPowerMonitor
import serial

# --- 전역 변수 초기화 (함수 외부에 위치) ---

#pw = CPUPowerMonitor(interval=1.0)
#pw.start()


# ------------------ 기본 설정 ---------------------
is_collecting = False
collection_task = None

DEVICE = "NPU"  # 일반 웹캠 사용이므로 CPU 권장
ov = Core()

try:
    conn = serial.Serial(port='/dev/ttyUSB0', baudrate=9600, timeout=1)
except serial.SerialException as e:
    raise ConnectionError(f"시리얼 포트 열기에 실패했습니다: {e}")

FACE_DETECTION_MODEL_XML = "./models/face-detection-retail-0005/FP16-INT8/face-detection-retail-0005.xml"
AGE_GENDER_MODEL_XML = "./models/age-gender-recognition-retail-0013/FP16-INT8/age-gender-recognition-retail-0013.xml"
EMOTION_MODEL_XML = "./models/emotions-recognition-retail-0003/FP16-INT8/emotions-recognition-retail-0003.xml"

LIVING_CLASSES = {'person', 'cat', 'dog', 'bird', 'teddy bear', 'cow', 'sheep', 'horse'}
EMOTIONS = ['neutral', 'happy', 'sad', 'surprise', 'anger']

# ------------------ 얼굴 탐지 ---------------------
face_det_model = ov.read_model(model=FACE_DETECTION_MODEL_XML)
face_det_compiled_model = ov.compile_model(model=face_det_model, device_name=DEVICE)
face_det_input_layer = face_det_compiled_model.input(0)
face_det_output_layer = face_det_compiled_model.output(0)
face_det_height, face_det_width = list(face_det_input_layer.shape)[2:]

# ------------------ 나이/성별 모델 -----------------
age_gender_model = ov.read_model(model=AGE_GENDER_MODEL_XML)
age_gender_compiled_model = ov.compile_model(age_gender_model, DEVICE)
age_gender_input_layer = age_gender_compiled_model.input(0)
age_output_layer = age_gender_compiled_model.output("age_conv3")
gender_output_layer = age_gender_compiled_model.output("prob")
age_gender_height, age_gender_width = list(age_gender_input_layer.shape)[2:]

# ------------------ 감정 모델 ---------------------
emotion_model = ov.read_model(model=EMOTION_MODEL_XML)
emotion_compiled_model = ov.compile_model(emotion_model, DEVICE)
emotion_input_layer = emotion_compiled_model.input(0)
emotion_output_layer = emotion_compiled_model.output(0)
emotion_height, emotion_width = list(emotion_input_layer.shape)[2:]

# ------------------ YOLO --------------------------
det_model = YOLO('./models/yolo11m-seg_int8_openvino_model')
class_names = det_model.names

# ------------------ 상태 값 ------------------------
state = {
    "charge": 0,
    "temp": 0,
    "voltage": 0,
    "cnt_live": 0,
    "cnt_object": 0,
    "boxes": [],
    "human": {"age": "", "gender": "", "emotion": "", "position": ""}
}

processed_frame_queue = Queue(maxsize=5)

# ------------------ 시각화 함수 ---------------------
def visualize_face(frame, face_det_results):
    global state
    h, w, _ = frame.shape

    for detection in face_det_results[0][0]:
        confidence = detection[2]
        if confidence > 0.5:
            xmin = int(detection[3] * w)
            ymin = int(detection[4] * h)
            xmax = int(detection[5] * w)
            ymax = int(detection[6] * h)

            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)

            face_img = frame[ymin:ymax, xmin:xmax]
            if face_img.size == 0:
                continue

            # Age/Gender
            resized_ag = cv2.resize(face_img, (age_gender_width, age_gender_height))
            ag_input = np.expand_dims(resized_ag.transpose(2, 0, 1), 0)
            ag_results = age_gender_compiled_model(ag_input)

            age_pred = int(ag_results[age_output_layer].reshape(1)[0] * 100)
            gender_idx = np.argmax(ag_results[gender_output_layer].reshape(-1))
            gender = "W" if gender_idx == 0 else "M"

            # Emotion
            resized_emotion = cv2.resize(face_img, (emotion_width, emotion_height))
            em_input = np.expand_dims(resized_emotion.transpose(2, 0, 1), 0)
            emotion_prob = emotion_compiled_model(em_input)[emotion_output_layer].reshape(-1)
            emotion = EMOTIONS[np.argmax(emotion_prob)]

            # Draw
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
            cv2.putText(frame, f"{gender},{age_pred},{emotion}", (xmin, ymax - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            state["human"]["gender"] = gender
            state["human"]["age"] = age_pred
            state["human"]["emotion"] = emotion

    return frame


# 전역 변수 설정
send_counter = 0 
SEND_INTERVAL = 20 
current_motor_angle = 0  # 현재 모터의 목표 각도 상태 저장

# 모터이동 포함
def visualize_segmentation(frame, masks, boxes, classes, scores, class_names, alpha=0.5):
    global state, send_counter, current_motor_angle, SEND_INTERVAL, conn
    overlay = frame.copy()

    state["boxes"] = []
    state["cnt_object"] = 0
    state["cnt_live"] = 0
    state["human"]["position"] = ""

    height, width, _ = frame.shape
    cell_h = height // 3
    cell_w = width // 3

    for mask, box, cls_idx, score in zip(masks, boxes, classes, scores):
        class_name = class_names[cls_idx]
        is_living = class_name in LIVING_CLASSES

        color = (0, 0, 255) if is_living else (0, 255, 0)

        if mask.sum() > 0:
            overlay[mask == 1] = (overlay[mask == 1] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # 중심 좌표 → 3x3 위치 계산
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        row = "T" if cy < cell_h else ("C" if cy < 2 * cell_h else "B")
        col = "L" if cx < cell_w else ("C" if cx < 2 * cell_w else "R")
        position = row + col
        send_counter += 1
        
        if is_living:
            state["cnt_live"] += 1
            state["human"]["position"] = position

            if send_counter >= SEND_INTERVAL:
                pos = state["human"]["position"]
                
                if pos != "":
                    # 1. 방향 판단 및 각도 증감
                    if "L" in pos:      # 사람이 왼쪽이면
                        current_motor_angle += 5
                    elif "R" in pos:    # 사람이 오른쪽이면
                        current_motor_angle -= 5
                    # "C"(중앙)인 경우 현재 각도 유지 (아무것도 안 함)

                    # 2. 최대/최소 범위 제한 (-50 ~ 50)
                    current_motor_angle = max(-50, min(50, current_motor_angle))

                    # 3. 명령어 전송
                    try:
                        cmd = f"#moter:{current_motor_angle}!"
                        conn.write(cmd.encode("utf-8"))
                        print(f"Tracking: {pos} -> Angle: {current_motor_angle}") 
                    except Exception as e:
                        print(f"Conn Error: {e}")

                send_counter = 0

        else:
            state["cnt_object"] += 1

        state["boxes"].append({
            "class": class_name,
            "score": float(score),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "position": position
        })

        cv2.putText(overlay, f"{class_name}:{score:.2f}", (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return overlay


# ------------------ 영상 처리 스레드 ---------------------
def processing_thread():
    global state

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    print("Webcam Processing Started... for senior mode")
    # FPS 측정을 위한 변수
    max_count = 0
    frame_count = 0

    # ---------------------------------------------

    # CSV 파일 헤더 작성 (최초 1회만 실행)
    log_file = "NPU_log.csv"
    new_file = not os.path.exists(log_file)

    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(['Timestamp', 'FPS','Watt'])

        start_time_sec = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            start_time = time.time()
            frame = cv2.resize(frame, (640, 640))

            # three point for calculate head x 2 arms
            res = det_model(frame, device="intel:npu", verbose=False, conf=0.25)[0] #, imgsz=640
            #res = det_model(frame, device="intel:npu", verbose=False, conf=0.25)[0] #, imgsz=640
            #res = det_model(frame, device="intel:npu", verbose=False, conf=0.25)[0] #, imgsz=640
            
            masks = res.masks.data.cpu().numpy().astype(np.uint8) if res.masks is not None else []
            boxes = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            scores = res.boxes.conf.cpu().numpy()

            out = visualize_segmentation(frame, masks, boxes, classes, scores, class_names)

            resized = cv2.resize(frame, (face_det_width, face_det_height))
            input_tensor = np.expand_dims(resized.transpose(2, 0, 1), 0)
            face_det_results = face_det_compiled_model(input_tensor)[face_det_output_layer]

            out = visualize_face(out, face_det_results)

            fps = 1.0 / (time.time() - start_time)
            cv2.putText(out, f"FPS: {fps:.2f}", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if processed_frame_queue.full():
                processed_frame_queue.get()

            processed_frame_queue.put(cv2.resize(out, (640, 480)))

            frame_count += 1
            
            # 현재 시간과 1초 전 측정 시작 시간 비교
            """
            if (time.time() - start_time_sec) >= 1.0 and max_count < 30:
                # 1초 동안의 평균 FPS 계산
                avg_fps = frame_count / (time.time() - start_time_sec)
                
                # 현재 시간 (타임스탬프)
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                writer.writerow([timestamp, f"{avg_fps:.2f}",pw.get_power()])
                f.flush()
                max_count += 1  
                print(f"[{timestamp}] AVG FPS: {avg_fps:.2f}를 CSV에 저장했습니다.")
                    
                # 변수 초기화: 다음 1초 측정을 위해
                frame_count = 0
                start_time_sec = time.time()        
            """

# ------------------ FastAPI ---------------------
app = FastAPI()

app.mount("/web", StaticFiles(directory="web"), name="web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def main():
    return {"result": True, "msg": "Webcam AI Server Running"}


@app.get("/heartbeat")
def heartbeat():
    return {"result": True, "data": state}


@app.get("/video_feed")
async def video_feed():

    def generate():
        while True:
            if not processed_frame_queue.empty():
                frame = processed_frame_queue.get()
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')
            else:
                time.sleep(0.01)

    return StreamingResponse(generate(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/start_collection")
def start_collection():
    global is_collecting

    if is_collecting:
        return {"message": "이미 실행 중"}

    is_collecting = True

    threading.Thread(target=processing_thread, daemon=True).start()

    return {"message": "웹캠 분석 시작"}


