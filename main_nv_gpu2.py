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
from nv_monitor import CPUPowerMonitor
# --- 전역 변수 초기화 (함수 외부에 위치) ---

os.environ["HF_HOME"] = '/home/circulus/git/HF_CACHE'

pw = CPUPowerMonitor(interval=1.0)
pw.start()

# ------------------ 기본 설정 ---------------------
is_collecting = False
collection_task = None

LIVING_CLASSES = {'person', 'cat', 'dog', 'bird', 'teddy bear', 'cow', 'sheep', 'horse'}

# ------------------ YOLO --------------------------
det_model = YOLO("./models/yolo11x-seg-int8.engine") #YOLO("yolo11x-seg.pt")
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


def visualize_segmentation(frame, masks, boxes, classes, scores, class_names, alpha=0.5):
    global state
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

        if is_living:
            state["cnt_live"] += 1
            state["human"]["position"] = position
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

    print("Webcam Processing Started...")
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
            res = det_model(frame, device=0, verbose=False, conf=0.25)[0] #, imgsz=640
            res = det_model(frame, device=0, verbose=False, conf=0.25)[0] #, imgsz=640
            res = det_model(frame, device=0, verbose=False, conf=0.25)[0] #, imgsz=640
            res = det_model(frame, device=0, verbose=False, conf=0.25)[0] #, imgsz=640
            
            masks = res.masks.data.cpu().numpy().astype(np.uint8) if res.masks is not None else []
            boxes = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            scores = res.boxes.conf.cpu().numpy()

            out = visualize_segmentation(frame, masks, boxes, classes, scores, class_names)
            fps = 1.0 / (time.time() - start_time)
            
            cv2.putText(out, f"FPS: {fps:.2f}", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if processed_frame_queue.full():
                processed_frame_queue.get()

            processed_frame_queue.put(cv2.resize(out, (640, 480)))

            frame_count += 1
            
            # 현재 시간과 1초 전 측정 시작 시간 비교
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


print("Loaded without RealSense. Ready!")
