from fastapi.middleware.cors import CORSMiddleware
from serverinfo import si
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
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
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCDataChannel, VideoStreamTrack
from requests import get
import time
import cv2
from openvino import Core
from fastapi.staticfiles import StaticFiles
from asyncio import Queue
from ultralytics import YOLO, FastSAM
import openvino as ov
from mandro import HadnControler
import threading
import hashlib
import asyncio
import requests
from pyapriltags import Detector
import httpx
import fractions
import aiohttp
from av import VideoFrame


def getHash(text):
    hash_func = hashlib.new('md5')
    hash_func.update(text.encode('utf-8'))
    return hash_func.hexdigest()

_IP = "192.168.21.9"

ov_core = Core()

FACE_DETECTION_MODEL_XML = "./models/face-detection-retail-0005/FP16-INT8/face-detection-retail-0005.xml"
AGE_GENDER_MODEL_XML = "./models/age-gender-recognition-retail-0013/FP16-INT8/age-gender-recognition-retail-0013.xml"
EMOTION_MODEL_XML = "./models/emotions-recognition-retail-0003/FP16-INT8/emotions-recognition-retail-0003.xml"

DEVICE = "NPU"
LIVING_CLASSES = {'person', 'cat', 'dog', 'bird', 'teddy bear', 'cow', 'sheep', 'horse'}
EMOTIONS = ['neutral', 'happy', 'sad', 'surprise', 'anger']

# 얼굴 탐지 모델
face_det_model = ov_core.read_model(model=FACE_DETECTION_MODEL_XML)
face_det_compiled_model = ov_core.compile_model(model=face_det_model, device_name=DEVICE)
face_det_input_layer = face_det_compiled_model.input(0)
face_det_output_layer = face_det_compiled_model.output(0)
face_det_height, face_det_width = list(face_det_input_layer.shape)[2:]

# 나이/성별 모델
age_gender_model = ov_core.read_model(model=AGE_GENDER_MODEL_XML)
age_gender_compiled_model = ov_core.compile_model(model=age_gender_model, device_name=DEVICE)
age_gender_input_layer = age_gender_compiled_model.input(0)
age_output_layer = age_gender_compiled_model.output("age_conv3")
gender_output_layer = age_gender_compiled_model.output("prob")
age_gender_height, age_gender_width = list(age_gender_input_layer.shape)[2:]

# 감정 모델
emotion_model = ov_core.read_model(model=EMOTION_MODEL_XML)
emotion_compiled_model = ov_core.compile_model(model=emotion_model, device_name=DEVICE)
emotion_input_layer = emotion_compiled_model.input(0)
emotion_output_layer = emotion_compiled_model.output(0)
emotion_height, emotion_width = list(emotion_input_layer.shape)[2:]

det_model = YOLO('./models/yolo11s-seg_int8_openvino_model')
ppe_model = YOLO('./models/yolo11n-helmet4_int8_openvino_model')
class_names = det_model.names
ppe_names = ppe_model.names

print(ppe_names)

is_collecting = False

detector = Detector(families="tag36h11")

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

_PORT = int(open("port.txt", 'r').read())

SOURCE_VIDEO_URL = f"http://{_IP}:59512/video_raw"

conn = None
audio_hub = None
track = None
lastColor = 'cyan'
state = {
    "charge": 0, "temp": 0, "voltage": 0,
    "cnt_live": 0, "cnt_object": 0, "boxes": [],
    "human": {"age": "", "gender": "", "emotion": "", "position": ""},
    "tag": {"id": None, "dist": 0}
}

G1_ARM = {
    "clamp": 17, "highFive": 18, "shakeHands_1": 27,
    "makeHeartBothHands": 20, "makeHeartSingleHands": 21,
    "blowKiss": 12, "hug": 19, "hightWave": 26, "lowWave": 25,
    "ultramanRay": 24, "bothHandsUp": 15, "singleHandsUp": 23,
    "Refuse": 22, "Release_Arm": 99,
}

G1_STATE = {
    "ZeroTorque": 0, "Damp": 1, "Preparation": 4, "Seating": 3,
    "Walk_G1": 500, "Walk2_G1": 501, "Run_G1": 801,
    "Squat_G1": 706, "SquatUp_G1": 706, "LieUp_G1": 702,
}

G1_BALANCE = {"Stand_G1": 0, "Step_G1": 1}

# ─────────────────────────────────────────────────────────────
# [변경] MJPEG 큐 제거 → WebRTC용 latest_frames 공유 메모리로 교체
# processed_frame_queue, frame_queue, depth_queue 는 아래로 대체됨
# ─────────────────────────────────────────────────────────────
raw_data_queue = Queue(maxsize=1)   # receiver_loop → processing_loop 용 (기존과 동일)

# WebRTC 트랙이 읽어갈 최신 프레임 캐시 (lock으로 보호)
latest_frames = {
    "main": None,   # 메인 세그멘테이션+PPE 합성 결과 (640x480 BGR numpy)
    "face": None,   # 얼굴 crop (BGR numpy, 가변 크기)
    "ppe":  None,   # PPE/헬멧 crop (BGR numpy, 가변 크기)
}
latest_frames_lock = threading.Lock()

cnt_live = 0
cnt_object = 0
lastTime = 0
cnt_image = 0

import os
import glob

config = {"PERFORMANCE_HINT": "LATENCY"}
pipe_tts = ov_core.compile_model(ov_core.read_model("./models/all_base_ov.xml"), device_name="CPU", config=config)
conf_tts = utils.get_hparams_from_file("./models/all_base_ov.json")

FACES_DIR = "faces"
os.makedirs(FACES_DIR, exist_ok=True)
last_face_saved_time = 0
latest_face_frame = None  # (하위 호환용, 이후 latest_frames["face"] 로 통일)

PPE_DIR = "ppe"
os.makedirs(PPE_DIR, exist_ok=True)
last_ppe_saved_time = 0
latest_ppe_frame = None   # (하위 호환용)


# ─────────────────────────────────────────────────────────────
# 파일 저장 헬퍼 (기존과 동일, latest_frames도 함께 업데이트)
# ─────────────────────────────────────────────────────────────

def save_ppe_async(crop_img, filename):
    global latest_ppe_frame
    try:
        _, img_encoded = cv2.imencode('.jpg', crop_img)
        latest_ppe_frame = img_encoded.tobytes()

        # [추가] WebRTC 트랙용 캐시 업데이트
        with latest_frames_lock:
            latest_frames["ppe"] = crop_img.copy()

        path = os.path.join(PPE_DIR, filename)
        cv2.imwrite(path, crop_img)

        files = sorted(glob.glob(os.path.join(PPE_DIR, "*.jpg")), key=os.path.getmtime)
        if len(files) > 20:
            for i in range(len(files) - 20):
                os.remove(files[i])
    except Exception as e:
        print(f"PPE Async Save Error: {e}")


def save_face_async(face_img, filename):
    global latest_face_frame
    try:
        _, img_encoded = cv2.imencode('.jpg', face_img)
        latest_face_frame = img_encoded.tobytes()

        # [추가] WebRTC 트랙용 캐시 업데이트
        with latest_frames_lock:
            latest_frames["face"] = face_img.copy()

        path = os.path.join(FACES_DIR, filename)
        cv2.imwrite(path, face_img)
    except Exception as e:
        print(f"Async Save Error: {e}")


# ─────────────────────────────────────────────────────────────
# AI 처리 함수들 (기존과 완전 동일)
# ─────────────────────────────────────────────────────────────

def visualize_face(frame, face_det_results):
    global state
    global last_face_saved_time
    global latest_face_frame
    current_time = time.time()
    h, w, _ = frame.shape

    state["human"]["gender"] = ""
    state["human"]["age"] = ""
    state["human"]["emotion"] = ""

    for detection in face_det_results[0][0]:
        confidence = detection[2]
        if confidence > 0.5:
            xmin = int(detection[3] * w)
            ymin = int(detection[4] * h)
            xmax = int(detection[5] * w)
            ymax = int(detection[6] * h)

            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            face_img = frame[ymin:ymax, xmin:xmax]

            if face_img.size > 0:
                if current_time - last_face_saved_time > 10.0:
                    print("face save...")
                    last_face_saved_time = current_time
                    face_filename = f"face_{int(current_time)}.jpg"
                    save_thread = threading.Thread(
                        target=save_face_async,
                        args=(face_img.copy(), face_filename)
                    )
                    save_thread.start()
                    print("save end...")

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)

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
        color = (0, 0, 255) if is_living else (0, 255, 0)

        overlay[mask == 1] = (overlay[mask == 1] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        height, width, channels = frame.shape
        cell_h = height // 3
        cell_w = width // 3

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

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

        position = row + col

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
            'position': position,
            'depth': depth
        })

        label = f"{class_name}:{score:.2f} | {depth:.2f}m"
        cv2.putText(overlay, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

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


# ─────────────────────────────────────────────────────────────
# 수신 루프 (기존과 동일)
# ─────────────────────────────────────────────────────────────

async def fetch_combined_frame(session):
    W, H = 640, 480
    RGB_SIZE = W * H * 3
    DEPTH_SIZE = W * H * 2
    TOTAL_SIZE = RGB_SIZE + DEPTH_SIZE

    try:
        async with session.get(SOURCE_VIDEO_URL, timeout=aiohttp.ClientTimeout(total=1.0)) as response:
            if response.status == 200:
                data = await response.read()
                if len(data) >= TOTAL_SIZE:
                    frame = np.frombuffer(data[:RGB_SIZE], dtype=np.uint8).reshape(H, W, 3)
                    depth_frame = np.frombuffer(data[RGB_SIZE:TOTAL_SIZE], dtype=np.uint16).reshape(H, W)
                    return frame, depth_frame
                else:
                    print(f"Warning: Data incomplete ({len(data)}/{TOTAL_SIZE} bytes)")
            else:
                print(f"Server Error: HTTP {response.status}")
    except asyncio.TimeoutError:
        print("Fetch Timeout")
    except Exception as e:
        print(f"Fetch Error: {e}")

    return None, None


async def receiver_loop():
    print("============= Receiver Loop Started")
    connector = aiohttp.TCPConnector(limit=None, keepalive_timeout=30)
    async with aiohttp.ClientSession(connector=connector) as session:
        while True:
            try:
                frame, depth = await fetch_combined_frame(session)
                if frame is not None:
                    if raw_data_queue.full():
                        raw_data_queue.get_nowait()
                    await raw_data_queue.put((frame, depth))
                else:
                    await asyncio.sleep(0.001)
            except Exception as e:
                print(f"Receiver Error: {e}")
                await asyncio.sleep(0.1)


# ─────────────────────────────────────────────────────────────
# 처리 루프 — AI 로직 전부 보존, 출력 부분만 WebRTC 캐시로 교체
# ─────────────────────────────────────────────────────────────

async def processing_loop():
    global cnt_image
    global last_ppe_saved_time
    global last_face_saved_time

    print("============= Processing Loop Started")

    while True:
        # 1. 수신부로부터 데이터 획득
        frame, depth_frame = await raw_data_queue.get()
        start_time = time.time()

        # 2. 전처리 (기존과 동일)
        frame_ai = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_NEAREST)
        depth_ai = cv2.resize(depth_frame, (640, 640), interpolation=cv2.INTER_NEAREST)

        # 3. NPU 추론 (기존과 동일)
        res = det_model(frame_ai, device="intel:npu", verbose=False, conf=0.25)[0]
        ppe_res = ppe_model(frame_ai, device="intel:npu", verbose=False, conf=0.25)[0]

        # 4. 후처리 (기존과 동일)
        if hasattr(res, 'masks') and res.masks is not None:
            masks = res.masks.data.cpu().numpy().astype(np.uint8)
        else:
            masks = []

        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy()

        # 5. 세그멘테이션 시각화 (기존과 동일)
        out = visualize_segmentation(
            frame_ai, masks, boxes, classes, scores,
            get_mask_depths(masks, depth_ai), class_names
        )

        # 6. PPE 탐지 + 시각화 (기존과 동일)
        if ppe_res.boxes is not None:
            ppe_boxes = ppe_res.boxes.xyxy.cpu().numpy()
            ppe_scores = ppe_res.boxes.conf.cpu().numpy()
            ppe_classes = ppe_res.boxes.cls.cpu().numpy().astype(int)
            ppe_names_local = ppe_model.names

            for i, box in enumerate(ppe_boxes):
                x1, y1, x2, y2 = map(int, box)
                conf = ppe_scores[i]
                cls_id = ppe_classes[i]
                label_text = ppe_names_local.get(cls_id, str(cls_id))

                if 'helmet' in label_text or 'face' in label_text:
                    display_str = f"{label_text.capitalize()}: {conf:.2f}"
                    current_time = time.time()

                    crop_h, crop_w = out.shape[:2]
                    y1_c, y2_c = max(0, y1), min(crop_h, y2)
                    x1_c, x2_c = max(0, x1), min(crop_w, x2)
                    ppe_crop = out[y1_c:y2_c, x1_c:x2_c]

                    if 'helmet' in label_text and current_time - last_ppe_saved_time > 10.0:
                        print("helmet save...")
                        last_ppe_saved_time = current_time
                        ppe_filename = f"ppe_{label_text}_{int(current_time)}.jpg"
                        threading.Thread(
                            target=save_ppe_async,
                            args=(ppe_crop.copy(), ppe_filename)
                        ).start()

                    elif 'face' in label_text and current_time - last_face_saved_time > 10.0:
                        print("face save...")
                        last_face_saved_time = current_time
                        face_filename = f"face_{int(current_time)}.jpg"
                        threading.Thread(
                            target=save_face_async,
                            args=(ppe_crop.copy(), face_filename)
                        ).start()

                    # PPE 박스 및 텍스트 그리기 (기존과 동일)
                    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    (w_t, h_t), _ = cv2.getTextSize(display_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    text_y = max(y1, h_t + 10)
                    cv2.rectangle(out, (x1, text_y - h_t - 10), (x1 + w_t, text_y), (255, 0, 0), -1)
                    cv2.putText(out, display_str, (x1, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 7. AprilTag 탐지 (기존과 동일)
        gray = cv2.cvtColor(frame_ai, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)

        if tags:
            best_tag = max(tags, key=lambda t: cv2.contourArea(t.corners.astype(np.float32)))

            overlay = out.copy()
            pts = best_tag.corners.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            out = cv2.addWeighted(overlay, 0.2, out, 0.8, 0)

            tag_id = best_tag.tag_id
            cx_t, cy_t = int(best_tag.center[0]), int(best_tag.center[1])

            dist = 0.0
            if 0 <= cy_t < 640 and 0 <= cx_t < 640:
                dist = depth_ai[cy_t, cx_t] / 1000.0

            info_str = f"ID: {tag_id} / Dist: {dist:.2f}m"
            state["tag"]["id"] = tag_id
            state["tag"]["dist"] = dist

            (w_tag, h_tag), baseline = cv2.getTextSize(info_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_x = 640 - w_tag - 20
            text_y = 640 - 20
            cv2.rectangle(out, (text_x - 10, text_y - h_tag - 10), (640, 640), (0, 0, 0), -1)
            cv2.putText(out, info_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 8. FPS 표시 (기존과 동일)
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(out, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # ─────────────────────────────────────────────────────
        # [변경] MJPEG 큐 put → latest_frames 캐시 업데이트
        # 기존: await processed_frame_queue.put(cv2.resize(out, (640, 480)))
        # 변경: WebRTC FrameProviderTrack이 이 값을 읽어감
        # ─────────────────────────────────────────────────────
        with latest_frames_lock:
            latest_frames["main"] = cv2.resize(out, (640, 480))

        cnt_image += 1
        if cnt_image % 100 == 0:
            cv2.imwrite("capture.jpg", frame)

        await asyncio.sleep(0)  # 이벤트 루프 양보


# ─────────────────────────────────────────────────────────────
# WebRTC: VideoStreamTrack 구현
# latest_frames 딕셔너리에서 BGR 프레임을 읽어 WebRTC 비디오로 송출
# ─────────────────────────────────────────────────────────────

class FrameProviderTrack(VideoStreamTrack):
    """
    latest_frames[frame_key] 의 BGR numpy 배열을 WebRTC VideoFrame으로 변환.

    - recv() 호출마다 await asyncio.sleep(1/fps) 으로 FPS 제어
    - 새 프레임 없으면 이전 프레임 재전송 (검정 화면 방지)
    - BGR → RGB 변환 후 av.VideoFrame.from_ndarray 사용
    """

    kind = "video"

    def __init__(self, frame_key: str, fps: int = 15):
        super().__init__()
        self.frame_key = frame_key
        self.fps = fps
        self._pts = 0
        self._time_base = fractions.Fraction(1, fps)
        self._blank = np.zeros((480, 640, 3), dtype=np.uint8)

    async def recv(self):
        await asyncio.sleep(1.0 / self.fps)

        with latest_frames_lock:
            bgr = latest_frames.get(self.frame_key)

        if bgr is None:
            bgr = self._blank

        # BGR → RGB (av 라이브러리 기준)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += 1

        return frame


# ─────────────────────────────────────────────────────────────
# WebRTC: 연결 관리자
# ─────────────────────────────────────────────────────────────

class WebRTCManager:
    """
    - 클라이언트 1개당 RTCPeerConnection 1개
    - 비디오 트랙 3개 (main 15fps / face 5fps / ppe 5fps) 단일 PC에 번들
    - DataChannel "state" 로 state JSON을 100ms 간격 diff 전송
      → 변경 없으면 전송 안 함 (대역폭 절약)
    - DataChannel ordered=False, maxRetransmits=0
      → UDP-like, 재전송 없이 최신 state 우선
    """

    def __init__(self):
        self.peer_connections: dict[str, RTCPeerConnection] = {}
        self.data_channels: dict[str, RTCDataChannel] = {}
        self._last_state_hash = None

    async def start_broadcast_loop(self, interval: float = 0.1):
        while True:
            await asyncio.sleep(interval)
            await self._broadcast_state()

    async def _broadcast_state(self):
        state_json = json.dumps(state, ensure_ascii=False)
        h = hash(state_json)
        if h == self._last_state_hash:
            return
        self._last_state_hash = h

        msg = state_json.encode("utf-8")
        dead = []
        for cid, dc in list(self.data_channels.items()):
            try:
                if dc.readyState == "open":
                    dc.send(msg)
                else:
                    dead.append(cid)
            except Exception as e:
                logger.warning(f"DataChannel send error [{cid}]: {e}")
                dead.append(cid)

        for cid in dead:
            await self.close(cid)

    async def create_answer(self, client_id: str, offer_sdp: str, offer_type: str) -> dict:
        pc = RTCPeerConnection()
        self.peer_connections[client_id] = pc

        # 비디오 트랙 3개 추가
        pc.addTrack(FrameProviderTrack("main", fps=15))
        pc.addTrack(FrameProviderTrack("face", fps=5))
        pc.addTrack(FrameProviderTrack("ppe",  fps=5))

        # DataChannel: UDP-like (ordered=False, maxRetransmits=0)
        dc = pc.createDataChannel("state", ordered=False, maxRetransmits=0)
        self.data_channels[client_id] = dc

        @dc.on("open")
        def on_dc_open():
            logger.info(f"DataChannel open [{client_id}]")

        @pc.on("connectionstatechange")
        async def on_conn_change():
            logger.info(f"PC [{client_id}] state: {pc.connectionState}")
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await self.close(client_id)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type=offer_type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    async def close(self, client_id: str):
        pc = self.peer_connections.pop(client_id, None)
        self.data_channels.pop(client_id, None)
        if pc:
            await pc.close()
        logger.info(f"PC closed [{client_id}]")

    async def close_all(self):
        for cid in list(self.peer_connections.keys()):
            await self.close(cid)


webrtc_manager = WebRTCManager()


# ─────────────────────────────────────────────────────────────
# FastAPI 앱
# ─────────────────────────────────────────────────────────────

app = FastAPI()

app.mount("/web", StaticFiles(directory="web"), name="web")
app.mount("/webfonts", StaticFiles(directory="webfonts"), name="webfonts")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(webrtc_manager.start_broadcast_loop(interval=0.1))
    logger.info("WebRTC broadcast loop started")


@app.on_event("shutdown")
async def on_shutdown():
    await webrtc_manager.close_all()


# ─────────────────────────────────────────────────────────────
# WebRTC 시그널링 엔드포인트
# ─────────────────────────────────────────────────────────────

class OfferRequest(BaseModel):
    sdp: str
    type: str
    client_id: str


@app.post("/webrtc/offer")
async def webrtc_offer(req: OfferRequest):
    """
    클라이언트 offer SDP → 서버 answer SDP 반환.
    단일 HTTP POST로 시그널링 완료.
    """
    answer = await webrtc_manager.create_answer(
        client_id=req.client_id,
        offer_sdp=req.sdp,
        offer_type=req.type,
    )
    return JSONResponse(answer)


@app.delete("/webrtc/{client_id}")
async def webrtc_disconnect(client_id: str):
    await webrtc_manager.close(client_id)
    return {"result": True}


# ─────────────────────────────────────────────────────────────
# 기존 엔드포인트 전부 보존
# ─────────────────────────────────────────────────────────────

@app.get("/")
def main():
    return {"result": True, "data": "AI-CPU-V2", "ip": _IP, "port": _PORT}


@app.get("/connect2")
async def connect2():
    global conn
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)
    await conn.connect()
    print(1)

    def lowstate_callback(message):
        msg = message['data']
        state["charge"] = msg['bms_state']['soc']
        state["temp"] = msg['temperature_ntc1']
        state["voltage"] = msg['power_v']

    conn.datachannel.pub_sub.subscribe(RTC_TOPIC['LOW_STATE'], lowstate_callback)
    return {"result": True, "data": True}


@app.get("/prepare")
async def prepare():
    return {"result": True, "data": True}


@app.get("/prepare2")
async def prepare2():
    return {"result": True, "data": True}


@app.get("/hand")
async def hand(cmd: str):
    requests.get(f"http://{_IP}:59521/hands?cmd={cmd}")
    return {"result": True}


@app.get("/heartbeat")
async def heartbeat():
    print(state)
    return {"result": True, "data": state}


@app.get("/start_collection")
async def start_frame_collection():
    global is_collecting
    if is_collecting:
        return {"message": "이미 프레임 수집 및 분석이 진행 중입니다"}

    while not raw_data_queue.empty():
        try:
            raw_data_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

    print("모든 큐 초기화 완료")
    is_collecting = True

    asyncio.create_task(receiver_loop())
    asyncio.create_task(processing_loop())

    return {"message": "수신 및 분석 파이프라인이 시작되었습니다."}


@app.get("/sport")
async def sport(cmd: str, x=0.0, y=0.0, z=0.0, data=None):
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
    elif data is not None:
        out = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"], {
                "api_id": SPORT_CMD[cmd],
                "parameter": {"data": float(data)}
            }
        )
    else:
        if lastCmd.get(cmd, False):
            lastCmd[cmd] = True
            out = await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], {
                    "api_id": SPORT_CMD[cmd],
                    "parameter": {"data": True}
                }
            )
        else:
            lastCmd[cmd] = False
            out = await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], {
                    "api_id": SPORT_CMD[cmd],
                    "parameter": {"data": False}
                }
            )
    print("response", out)
    return {"result": True, "data": out}


lastCmd = {}


def toFloat(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return value


@app.get("/manual")
async def manual(cmd: str, data: str):
    out = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"], {
            "api_id": int(cmd),
            "parameter": {"data": toFloat(data)}
        }
    )
    print("response", out)
    return {"result": True, "data": out}


@app.get("/arm")
async def arm(cmd="clamp"):
    global conn
    await conn.datachannel.pub_sub.publish_request_new(
        "rt/api/arm/request", {
            "api_id": 7106,
            "parameter": {"data": G1_ARM[cmd]}
        }
    )
    return {"result": True, "data": True}


@app.get("/walkG1")
async def walkG1(lx=0, ly=0, rx=0, ry=0):
    print("walking", f"L : {lx} {ly} | R : {rx} {ry}")
    global conn
    conn.datachannel.pub_sub.publish_without_callback(
        "rt/wirelesscontroller", {
            "lx": float(lx), "ly": float(ly), "rx": float(rx), "ry": float(ry)
        }
    )
    return {"result": True, "data": True}


@app.get("/stateG1")
async def stateG1(cmd="Walk_G1"):
    global conn
    await conn.datachannel.pub_sub.publish_request_new(
        "rt/api/sport/request", {
            "api_id": 7101,
            "parameter": {"data": G1_STATE[cmd]}
        }
    )
    return {"result": True, "data": True}


@app.get("/balanceG1")
async def balanceG1(cmd="Stand_G1"):
    global conn
    await conn.datachannel.pub_sub.publish_request_new(
        "rt/api/sport/request", {
            "api_id": 7102,
            "parameter": {"data": G1_BALANCE[cmd]}
        }
    )
    return {"result": True, "data": True}


@app.get("/speech")
async def speech(text: str, motion=None, voice=31, lang='ko'):
    print('speech', text)
    global audio_hub
    filename = getHash(text)
    if audio_hub is not None:
        response = await audio_hub.get_audio_list()
        if response and isinstance(response, dict):
            data_str = response.get('data', {}).get('data', '{}')
            audio_list = json.loads(data_str).get('audio_list', [])
            existing_audio = next((a for a in audio_list if a['CUSTOM_NAME'] == filename), None)
            if existing_audio:
                print(f"Audio file {filename} already exists, skipping upload")
                uuid = existing_audio['UNIQUE_ID']
            else:
                print(f"Audio file {filename} not found, proceeding with upload")
                audio_file_path = tts(text=text, voice=voice, lang=lang)
                logger.info(f"Using audio file: {audio_file_path}")
                response = await audio_hub.upload_audio_file(audio_file_path)
                uuid = None
                print(response)
                response = await audio_hub.get_audio_list()
                if response and isinstance(response, dict):
                    data_str = response.get('data', {}).get('data', '{}')
                    audio_list = json.loads(data_str).get('audio_list', [])
                existing_audio = next((a for a in audio_list if a['CUSTOM_NAME'] == filename), None)
                uuid = existing_audio['UNIQUE_ID']
        print(f"Starting audio playback of file: {uuid}")
        await audio_hub.play_by_uuid(uuid)
    return {"result": True, "data": True}


@app.get("/color")
async def color(value='purple', warn=0):
    global conn
    global lastColor
    if lastColor == value:
        return
    print(warn)
    if int(warn) > 0:
        await speech("저한테 접근하면 위험하니, 조심해 주세요.", 'Content', 0, 'ko')
    if conn is None:
        print('brightness', value)
    else:
        lastColor = value
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["VUI"],
            {"api_id": 1007, "parameter": {"color": value}}
        )
    return {"result": True, "data": True}


@app.get("/brightness")
async def brightness(value=10):
    global conn
    if conn is None:
        print('brightness', value)
    else:
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["VUI"],
            {"api_id": 1005, "parameter": {"brightness": int(value)}}
        )
    return {"result": True, "data": True}


@app.get("/mode")
async def mode(value='normal'):
    global conn
    if conn is None:
        print('mode', value)
    else:
        conn.datachannel.pub_sub.publish_without_callback(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1002, "parameter": {"name": value}}
        )
    return {"result": True, "data": True}


@app.get("/volume")
async def volume(value=10):
    global conn
    if conn is None:
        print('volume', value)
    else:
        conn.datachannel.pub_sub.publish_without_callback(
            RTC_TOPIC["VUI"],
            {"api_id": 1003, "parameter": {"volume": int(value)}}
        )
    return {"result": True, "data": True}


@app.get("/monitor")
def monitor():
    return si.getAll()


@app.get("/v1/tts", response_class=FileResponse, summary="입력한 문장으로 부터 음성을 생성합니다.")
def tts(text="", voice=31, lang='ko', static=0, isPlay=0):
    start = t.time()
    print(text, static)
    filename = getHash(text)

    phoneme_ids = text_to_sequence(text, [f'canvers_{lang}_cleaners'])
    text_np = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)

    inputs = {
        "input": text_np,
        "input_lengths": np.array([text_np.shape[1]], dtype=np.int64),
        "scales": np.array([0.667, 1.0, 0.8], dtype=np.float16),
        "sid": np.array([int(voice)], dtype=np.int64) if voice is not None else None
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
        audio_seg = AudioSegment.from_wav(f"output/{filename}.wav")
        audio_seg = audio_seg.set_frame_rate(22050)
        audio_seg = audio_seg.set_sample_width(2)
        audio_seg = audio_seg.set_channels(1)
        audio_seg.export(f"output/{filename}.wav", format='wav', codec="pcm_s16le")
        if int(isPlay) > 0:
            playsound(f"output/{filename}.wav")
        return f"output/{filename}.wav"


@app.get("/v2/tts", response_class=FileResponse, summary="음성 생성 후 로봇에서 재생")
def tts_v2(text="", voice=6, lang='ko', static=0, isPlay=0):
    start = t.time()
    print(text, static)
    filename = getHash(text)

    phoneme_ids = text_to_sequence(text, [f'canvers_{lang}_cleaners'])
    text_np = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)

    inputs = {
        "input": text_np,
        "input_lengths": np.array([text_np.shape[1]], dtype=np.int64),
        "scales": np.array([0.667, 1.0, 0.8], dtype=np.float16),
        "sid": np.array([int(voice)], dtype=np.int64) if voice is not None else None
    }

    start_time = t.time()
    result = pipe_tts(inputs)
    print(f"Inference time: {t.time() - start_time:.4f} seconds")

    audio = list(result.values())[0].squeeze((0, 1))
    print(t.time() - start)

    write(data=audio, rate=conf_tts.data.sampling_rate, filename=f"output/{filename}.wav")

    with open(f"output/{filename}.wav", "rb") as f:
        files = {"audio_file": (f"{filename}.wav", f, "audio/wav")}
        response = requests.post(f"http://{_IP}:59521/audio", files=files)

    return f"output/{filename}.wav"


@app.get("/led")
async def led(r: int = 0, g: int = 0, b: int = 0):
    print(f"http://{_IP}:59521/led?r={r}&g={g}&b={b}")
    requests.get(f"http://{_IP}:59521/led?r={r}&g={g}&b={b}")
    return {"result": True}


import matplotlib.colors


@app.get("/color_led")
async def color_led(value: str = 'red'):
    print(value)
    colors = matplotlib.colors.to_rgb(value)
    arr = (np.array(colors) * 255).astype(int)
    print("color", arr)
    requests.get(f"http://{_IP}:59521/led?r={arr[0]}&g={arr[1]}&b={arr[2]}")
    return {"result": True}


print("NPU", "2502010900")
