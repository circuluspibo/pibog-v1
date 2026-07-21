# ─────────────────────────────────────────────────────────────
# [최적화 1] 스레드 수 제한 — 반드시 numpy/torch import 보다 먼저 설정
#   NPU로 추론해도 YOLO 전/후처리(torch, OpenMP, BLAS)는 CPU 멀티코어를
#   전부 잡아먹어 700%가 나옴. 아래 설정으로 총 CPU%가 크게 떨어진다.
#   코어 수에 맞춰 2~4 사이에서 튜닝하면 된다. (FPS 손실은 거의 없음)
# ─────────────────────────────────────────────────────────────
import os
os.environ["OMP_NUM_THREADS"]      = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"]      = "2"
os.environ["NUMEXPR_NUM_THREADS"]  = "2"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import time
import json
import fractions
import asyncio
import threading
import hashlib
import logging
import queue

import numpy as np
import cv2
import torch
import pyrealsense2 as rs

from openvino import Core
from ultralytics import YOLO

from av import VideoFrame
from aiortc import (RTCPeerConnection, RTCSessionDescription, RTCDataChannel, VideoStreamTrack, RTCConfiguration)
from serverinfo import si

# OpenCV / torch 스레드도 소수로 제한
cv2.setNumThreads(2)
torch.set_num_threads(2)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
_IP   = "127.0.0.1"  # si.getIP()
_PORT = int(open("port.txt", 'r').read())

# 로컬 망 전용 → STUN/TURN 서버 없음. host candidate만 사용.
RTC_CONFIG = RTCConfiguration(iceServers=[])

DEVICE = "NPU"

# [최적화 2] 프레임 스로틀링 파라미터
#   - PROCESS_EVERY : N프레임마다 1번만 AI 처리 (30fps → 15fps)
#   - FACE_EVERY    : 얼굴/나이/성별/감정은 '처리된 프레임' 기준 N번마다 1번
#   - DEPTH_EVERY   : depth 컬러맵 생성도 N번마다 1번
DEVICE_RETRY_TRIES = 30
DEVICE_RETRY_DELAY = 2.0
PROCESS_EVERY = 2
FACE_EVERY    = 3
DEPTH_EVERY   = 2

FACE_DETECTION_MODEL_XML = "./models/face-detection-retail-0005/FP16-INT8/face-detection-retail-0005.xml"
AGE_GENDER_MODEL_XML     = "./models/age-gender-recognition-retail-0013/FP16-INT8/age-gender-recognition-retail-0013.xml"
EMOTION_MODEL_XML        = "./models/emotions-recognition-retail-0003/FP16-INT8/emotions-recognition-retail-0003.xml"

LIVING_CLASSES = {'person', 'cat', 'dog', 'bird', 'teddy bear', 'cow', 'sheep', 'horse'}
EMOTIONS       = ['neutral', 'happy', 'sad', 'surprise', 'anger']

# ─────────────────────────────────────────────────────────────
# 큐 (프레임 1개만 유지 → 항상 최신 프레임 송출)
# ─────────────────────────────────────────────────────────────
stream_q_main  = queue.Queue(maxsize=1)   # 원본처리(YOLO+얼굴) 채널
stream_q_depth = queue.Queue(maxsize=1)   # 깊이 채널

def q_put(q: queue.Queue, item):
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    q.put(item)

# ─────────────────────────────────────────────────────────────
# OpenVINO 모델 로드
#   [부팅 안정화] compile_model 을 재시도로 감싼다.
#   부팅 직후에는 NPU 드라이버(/dev/accel/accel0) 준비 전이라 compile이
#   실패할 수 있음 → 재시도로 장치가 올라올 때까지 대기.
# ─────────────────────────────────────────────────────────────
ov = Core()

def compile_with_retry(model, device, tries=DEVICE_RETRY_TRIES, delay=DEVICE_RETRY_DELAY):
    last_err = None
    for i in range(tries):
        try:
            return ov.compile_model(model=model, device_name=device)
        except Exception as e:
            last_err = e
            logger.error(f"[compile] {device} 실패 {i + 1}/{tries}: {e}")
            time.sleep(delay)
    raise RuntimeError(f"[compile] {device} 최종 실패: {last_err}")

face_det_model          = ov.read_model(model=FACE_DETECTION_MODEL_XML)
face_det_compiled_model = compile_with_retry(face_det_model, DEVICE)
face_det_input_layer    = face_det_compiled_model.input(0)
face_det_output_layer   = face_det_compiled_model.output(0)
face_det_height, face_det_width = list(face_det_input_layer.shape)[2:]

age_gender_model          = ov.read_model(model=AGE_GENDER_MODEL_XML)
age_gender_compiled_model = compile_with_retry(age_gender_model, DEVICE)
age_gender_input_layer    = age_gender_compiled_model.input(0)
age_output_layer          = age_gender_compiled_model.output("age_conv3")
gender_output_layer       = age_gender_compiled_model.output("prob")
age_gender_height, age_gender_width = list(age_gender_input_layer.shape)[2:]

emotion_model          = ov.read_model(model=EMOTION_MODEL_XML)
emotion_compiled_model = compile_with_retry(emotion_model, DEVICE)
emotion_input_layer    = emotion_compiled_model.input(0)
emotion_output_layer   = emotion_compiled_model.output(0)
emotion_height, emotion_width = list(emotion_input_layer.shape)[2:]

det_model   = YOLO('./models/yolo11s-seg_int8_openvino_model')
class_names = det_model.names

# ─────────────────────────────────────────────────────────────
# 전역 상태 (DataChannel로 전송)
# ─────────────────────────────────────────────────────────────
state = {
    "charge": 0, "temp": 0, "voltage": 0,
    "cnt_live": 0, "cnt_object": 0, "boxes": [],
    "human": {"age": "", "gender": "", "emotion": "", "position": "", "depth": ""},
}

is_collecting = False

# [최적화 2-b] 얼굴 분석 결과 캐시
#   FACE_EVERY 프레임마다만 얼굴 파이프라인(검출+나이/성별/감정)을 NPU로 돌리고,
#   그 사이 프레임은 마지막 결과(박스/라벨)를 그대로 그려서 부하를 줄인다.
last_face_annotations = []


def analyze_faces(frame, face_det_results):
    """얼굴 검출 + 나이/성별/감정 추론 → 결과를 캐시하고 state 갱신."""
    global state, last_face_annotations
    h, w, _ = frame.shape

    annotations = []
    state["human"]["gender"]  = ""
    state["human"]["age"]     = ""
    state["human"]["emotion"] = ""

    for detection in face_det_results[0][0]:
        confidence = detection[2]
        if confidence > 0.5:
            xmin = max(0, int(detection[3] * w))
            ymin = max(0, int(detection[4] * h))
            xmax = min(w, int(detection[5] * w))
            ymax = min(h, int(detection[6] * h))
            face_img = frame[ymin:ymax, xmin:xmax]

            if face_img.size > 0:
                resized_age_gender = cv2.resize(face_img, (age_gender_width, age_gender_height))
                ag_input_tensor = np.expand_dims(resized_age_gender.transpose((2, 0, 1)), 0)
                ag_results = age_gender_compiled_model(ag_input_tensor)

                age_pred    = int(ag_results[age_output_layer].reshape(1)[0] * 100)
                gender_prob = ag_results[gender_output_layer].reshape(-1)
                gender_idx  = np.argmax(gender_prob)
                gender      = "W" if gender_idx == 0 else "M"

                resized_emotion = cv2.resize(face_img, (emotion_width, emotion_height))
                emotion_input_tensor = np.expand_dims(resized_emotion.transpose((2, 0, 1)), 0)
                emotion_results = emotion_compiled_model(emotion_input_tensor)
                emotion_prob = emotion_results[emotion_output_layer].reshape(-1)
                emotion      = EMOTIONS[np.argmax(emotion_prob)]

                annotations.append({
                    "box": (xmin, ymin, xmax, ymax),
                    "text": f"{gender}, {age_pred}y, {emotion}",
                })

                state["human"]["gender"]  = gender
                state["human"]["age"]     = age_pred
                state["human"]["emotion"] = emotion

    last_face_annotations = annotations
    return annotations


def draw_faces(frame, annotations):
    """캐시된 얼굴 결과를 프레임에 그린다 (매 프레임 호출)."""
    for a in annotations:
        xmin, ymin, xmax, ymax = a["box"]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
        cv2.putText(frame, a["text"], (xmin, ymax - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return frame


def visualize_segmentation(frame, masks, boxes, classes, scores, depths, class_names, alpha=0.5):
    global state
    overlay = frame.copy()

    state['boxes']      = []
    state["cnt_object"] = 0
    state["cnt_live"]   = 0
    state["human"]["depth"]    = ""
    state["human"]["position"] = ""

    height, width, _ = frame.shape
    cell_h = height // 3
    cell_w = width // 3

    for mask, box, cls_idx, score, depth in zip(masks, boxes, classes, scores, depths):
        class_name = class_names[cls_idx]
        is_living  = class_name in LIVING_CLASSES
        color      = (0, 0, 255) if is_living else (0, 255, 0)

        overlay[mask == 1] = (overlay[mask == 1] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

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
            state["human"]["depth"]    = depth
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
        cv2.putText(overlay, label, (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return overlay


def get_mask_depths(masks, depth_frame, low_percentile=5):
    """[최적화 2-c] depth 이미지를 마스크마다 resize 하지 않고,
    마스크 해상도로 '한 번만' resize 한 뒤 재사용한다.
    (YOLO-seg 마스크는 모두 동일 해상도이므로 안전)"""
    depths = []
    if len(masks) == 0:
        return depths

    depth_image = np.asanyarray(depth_frame.get_data())

    # 마스크 해상도 기준 depth 리사이즈를 1회만 수행
    mask_h, mask_w = masks[0].shape
    depth_resized = cv2.resize(depth_image, (mask_w, mask_h),
                               interpolation=cv2.INTER_NEAREST)

    for mask in masks:
        if mask.sum() == 0:
            depths.append(0.0)
            continue

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


def getHash(text):
    hash_func = hashlib.new('md5')
    hash_func.update(text.encode('utf-8'))
    return hash_func.hexdigest()


# ─────────────────────────────────────────────────────────────
# RealSense 시작 (재시도)
#   부팅 직후 USB enumerate 전이면 실패하므로 장치가 올라올 때까지 재시도.
# ─────────────────────────────────────────────────────────────
def start_realsense(tries=DEVICE_RETRY_TRIES, delay=DEVICE_RETRY_DELAY):
    last_err = None
    for i in range(tries):
        try:
            pipeline = rs.pipeline()
            config   = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)
            print(f"[RealSense] start 성공 (시도 {i + 1})")
            return pipeline
        except Exception as e:
            last_err = e
            logger.error(f"[RealSense] start 실패 {i + 1}/{tries}: {e}")
            time.sleep(delay)
    raise RuntimeError(f"[RealSense] 최종 실패: {last_err}")


def warmup_yolo(tries=DEVICE_RETRY_TRIES, delay=DEVICE_RETRY_DELAY):
    """YOLO는 첫 추론 시 NPU 컴파일이 일어남 → 장치 미준비 시 재시도."""
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for i in range(tries):
        try:
            det_model(dummy, device="intel:npu", verbose=False, conf=0.25)
            print(f"[YOLO] NPU 워밍업 성공 (시도 {i + 1})")
            return
        except Exception as e:
            logger.error(f"[YOLO] NPU 워밍업 실패 {i + 1}/{tries}: {e}")
            time.sleep(delay)
    raise RuntimeError("[YOLO] NPU 워밍업 최종 실패")


# ─────────────────────────────────────────────────────────────
# Processing Thread : RealSense → AI → stream_q_main / stream_q_depth
# ─────────────────────────────────────────────────────────────
def processing_thread():
    global state, last_face_annotations

    warmup_yolo()
    pipeline = start_realsense()

    print("============= processing (RealSense → WebRTC)....")

    cnt_image  = 0
    frame_idx  = 0
    proc_idx   = 0
    last_out   = None  # 스킵 프레임에도 최신 결과 유지

    while True:
        try:
            frames      = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            frame_idx += 1

            # [최적화 2-a] N프레임마다 1번만 무거운 AI 처리
            if frame_idx % PROCESS_EVERY != 0:
                continue

            frame = np.asanyarray(color_frame.get_data())

            if cnt_image % 100 == 0:
                cv2.imwrite("capture.jpg", frame)
            cnt_image += 1

            start_time = time.time()

            frame = cv2.resize(frame, (640, 640))
            res   = det_model(frame, device="intel:npu", verbose=False, conf=0.25)[0]

            # ── 세그멘테이션 (시각화 유지) ──
            if hasattr(res, 'masks') and res.masks is not None:
                masks = res.masks.data.cpu().numpy().astype(np.uint8)
            else:
                masks = []

            boxes   = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            scores  = res.boxes.conf.cpu().numpy()

            out = visualize_segmentation(
                frame, masks, boxes, classes, scores,
                get_mask_depths(masks, depth_frame), class_names)

            # ── 얼굴: FACE_EVERY 처리프레임마다만 NPU 추론, 그 외엔 캐시 그리기 ──
            if proc_idx % FACE_EVERY == 0:
                resized_frame = cv2.resize(frame, (face_det_width, face_det_height))
                input_tensor  = np.expand_dims(resized_frame.transpose((2, 0, 1)), 0)
                face_det_results = face_det_compiled_model(input_tensor)[face_det_output_layer]
                analyze_faces(out, face_det_results)
            out = draw_faces(out, last_face_annotations)

            # ── depth 컬러맵 (DEPTH_EVERY 처리프레임마다만 생성) ──
            if proc_idx % DEPTH_EVERY == 0:
                depth_image_raw = np.asanyarray(depth_frame.get_data())
                depth_colormap  = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image_raw, alpha=0.03),
                    cv2.COLORMAP_JET)
                depth_colormap_resized = cv2.resize(depth_colormap, (640, 480))
                q_put(stream_q_depth, depth_colormap_resized)

            # FPS
            fps = 1.0 / max(1e-6, (time.time() - start_time))
            cv2.putText(out, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # ── 원본처리 채널 ──
            q_put(stream_q_main, cv2.resize(out, (640, 480)))

            proc_idx += 1

        except Exception as e:
            # 일시적 장치 오류로 스레드가 죽지 않도록 방어
            logger.error(f"[processing] 루프 오류: {e}")
            time.sleep(0.1)


# ─────────────────────────────────────────────────────────────
# WebRTC Tracks
# ─────────────────────────────────────────────────────────────
def _make_vf(bgr: np.ndarray, pts: int, tb) -> VideoFrame:
    vf = VideoFrame.from_ndarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy(), format="rgb24")
    vf.pts       = pts
    vf.time_base = tb
    return vf


class MainTrack(VideoStreamTrack):
    kind = "video"
    def __init__(self, fps=15):
        super().__init__()
        self._pts  = 0
        self._tb   = fractions.Fraction(1, 90000)
        self._step = 90000 // fps
        self._last = np.zeros((480, 640, 3), dtype=np.uint8)

    async def recv(self):
        bgr = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: stream_q_main.get(timeout=0.1) if not stream_q_main.empty() else None
        )
        if bgr is not None:
            self._last = bgr
        vf = _make_vf(self._last, self._pts, self._tb)
        self._pts += self._step
        return vf


class DepthTrack(VideoStreamTrack):
    kind = "video"
    # [최적화 3] depth 채널은 시각화용 → 인코딩 fps 를 낮춰 CPU 부담 감소
    def __init__(self, fps=8):
        super().__init__()
        self._pts  = 0
        self._tb   = fractions.Fraction(1, 90000)
        self._step = 90000 // fps
        self._last = np.zeros((480, 640, 3), dtype=np.uint8)

    async def recv(self):
        bgr = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: stream_q_depth.get(timeout=0.1) if not stream_q_depth.empty() else None
        )
        if bgr is not None:
            self._last = bgr
        vf = _make_vf(self._last, self._pts, self._tb)
        self._pts += self._step
        return vf


# ─────────────────────────────────────────────────────────────
# WebRTC Manager
# ─────────────────────────────────────────────────────────────
class WebRTCManager:
    def __init__(self):
        self._pcs: dict[str, RTCPeerConnection] = {}
        self._dcs: dict[str, RTCDataChannel]    = {}
        self._last_hash = None

    async def start_broadcast_loop(self, interval=0.1):
        while True:
            await asyncio.sleep(interval)
            open_dcs = [(cid, dc) for cid, dc in self._dcs.items()
                        if dc.readyState == "open"]
            dead = [cid for cid, dc in self._dcs.items()
                    if dc.readyState not in ("open", "connecting")]

            # state 변경 시에만 전송
            js = json.dumps(state, ensure_ascii=False)
            h  = hash(js)
            if h != self._last_hash:
                self._last_hash = h
                msg = js.encode()
                for cid, dc in open_dcs:
                    try:
                        dc.send(msg)
                    except Exception:
                        dead.append(cid)

            for cid in set(dead):
                await self.close(cid)

    async def create_offer(self, client_id: str) -> dict:
        pc = RTCPeerConnection(configuration=RTC_CONFIG)
        self._pcs[client_id] = pc

        pc.addTrack(MainTrack(fps=15))    # mid=0 원본처리
        pc.addTrack(DepthTrack(fps=8))    # mid=1 깊이

        dc = pc.createDataChannel("state", ordered=False, maxRetransmits=0)
        self._dcs[client_id] = dc

        @pc.on("connectionstatechange")
        async def _on_state():
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await self.close(client_id)

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        # 로컬 망: STUN 없이 host candidate만으로 gathering 완료 대기
        gather_done = asyncio.Event()

        @pc.on("icegatheringstatechange")
        def _on_gather():
            if pc.iceGatheringState == "complete":
                gather_done.set()

        if pc.iceGatheringState == "complete":
            gather_done.set()
        try:
            await asyncio.wait_for(gather_done.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"ICE gather timeout [{client_id}]")

        return {"sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "client_id": client_id}

    async def set_answer(self, client_id: str, sdp: str, typ: str):
        pc = self._pcs.get(client_id)
        if pc is None:
            raise ValueError(f"Unknown client_id: {client_id}")
        await pc.setRemoteDescription(RTCSessionDescription(sdp, typ))

    async def close(self, client_id):
        pc = self._pcs.pop(client_id, None)
        self._dcs.pop(client_id, None)
        if pc:
            await pc.close()

    async def close_all(self):
        for cid in list(self._pcs):
            await self.close(cid)


webrtc_manager = WebRTCManager()

# ─────────────────────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────────────────────
app = FastAPI()
app.mount("/web",      StaticFiles(directory="web"),      name="web")
app.mount("/webfonts", StaticFiles(directory="webfonts"), name="webfonts")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def strip_frame_options(request, call_next):
    response = await call_next(request)
    if "X-Frame-Options" in response.headers:
        del response.headers["X-Frame-Options"]
    return response


@app.on_event("startup")
async def _startup():
    global is_collecting
    is_collecting = True
    threading.Thread(target=processing_thread, daemon=True).start()
    asyncio.create_task(webrtc_manager.start_broadcast_loop())
    print("✅ RealSense processing + WebRTC broadcast 시작")


@app.on_event("shutdown")
async def _shutdown():
    await webrtc_manager.close_all()


class AnswerRequest(BaseModel):
    sdp: str
    type: str
    client_id: str


@app.get("/")
def main_route():
    return {"result": True, "data": "AI-NPU-MCR-V2-WEBRTC", "ip": _IP, "port": _PORT}


@app.get("/heartbeat")
async def heartbeat():
    return {"result": True, "data": state}


@app.get("/webrtc/offer")
async def webrtc_get_offer(client_id: str):
    return JSONResponse(await webrtc_manager.create_offer(client_id))


@app.post("/webrtc/answer")
async def webrtc_post_answer(req: AnswerRequest):
    await webrtc_manager.set_answer(req.client_id, req.sdp, req.type)
    return JSONResponse({"result": True})


@app.delete("/webrtc/{client_id}")
async def webrtc_disconnect(client_id: str):
    await webrtc_manager.close(client_id)
    return {"result": True}


@app.get("/monitor")
def monitor():
    return si.getAll()


@app.get("/start_collection")
async def start_frame_collection():
    global is_collecting
    if is_collecting:
        return {"message": "이미 프레임 수집이 진행 중입니다"}
    is_collecting = True
    threading.Thread(target=processing_thread, daemon=True).start()
    return {"message": "프레임 수집을 시작했습니다"}


print("Loading Complete", "NPU")