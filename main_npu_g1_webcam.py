"""
구조:
  Thread-1 receiver   : 웹캠(OpenCV) → raw_q
  Thread-mic          : 마이크 → WakeWord / ACLNet / VAD → STT → RAG
                        ACLNet 결과는 state["audio"] 에 직접 기록 → DataChannel 자동 전송
  Thread-2 processing : raw_q → YOLO + 얼굴분석 통합 시각화 → stream_q_main
                              → Depth Anything V2 추론 → stream_q_depth
                        depth min/max/center → state["depth"] → DataChannel 자동 전송
  asyncio  WebRTC     : stream_q_main  → MainTrack  (mid=0) → WebRTC 송출
                        stream_q_depth → DepthTrack (mid=1) → WebRTC 송출
  asyncio  FastAPI    : REST 엔드포인트
"""

import base64
import queue
import threading
import time
import hashlib
import asyncio
import json
import fractions
import os
import glob
import datetime
import logging
import requests
import numpy as np
import cv2
import matplotlib.colors
import pyaudio

from scipy.io.wavfile import write as write_wav
from scipy.io.wavfile import write as wav_write
import audioop
#from openwakeword.model import Model as WakeWordModel

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydub import AudioSegment
from playsound import playsound
from pyapriltags import Detector
from av import VideoFrame
from aiortc import (RTCPeerConnection, RTCSessionDescription,
                    RTCDataChannel, VideoStreamTrack, RTCConfiguration)

from openvino import Core
from ultralytics import YOLO

import utils
from text import text_to_sequence
from serverinfo import si

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
_IP          = "192.168.0.34"
_SERVER_PORT = 59530


DEPTH_SKIP_N = 5

WEBCAM_INDEX  = 6
WEBCAM_WIDTH  = 640
WEBCAM_HEIGHT = 480
WEBCAM_FPS    = 30

RTC_CONFIG = RTCConfiguration(iceServers=[])

# 오디오
WAKEWORD_MODEL_NAME      = "./models/alexa_v0.1.xml"
WAKEWORD_THRESHOLD       = 0.5
ACLNET_MODEL_XML         = "./models/aclnet.xml"
ACLNET_CLASSES_TXT       = "./models/aclnet_53cl.txt"
ACLNET_THRESHOLD         = 0.6          # ← threshold 이상 모든 클래스 반환
VAD_ACTIVATION_THRESHOLD = 500
SILENCE_DURATION         = 2
MAX_RECORDING_DURATION   = 15
RECORDING_OUTPUT_DIR     = "recordings_wakeword"
STT_API_URL              = "http://127.0.0.1:59532/v1/stt"
STT_LANG                 = "ko"
STT_IS_PLAY              = 0
RAG_URL                  = "http://127.0.0.1:59532/v1/rag/txt2chat"
AUDIO_FORMAT             = pyaudio.paInt16
AUDIO_CHANNELS           = 2 #1
AUDIO_RATE               = 44100 #16000
AUDIO_CHUNK              = 16000

os.makedirs(RECORDING_OUTPUT_DIR, exist_ok=True)
os.makedirs("output", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 큐
# ─────────────────────────────────────────────────────────────
raw_q         = queue.Queue(maxsize=1)
stream_q_main  = queue.Queue(maxsize=1)
stream_q_depth = queue.Queue(maxsize=1)   # Depth Anything V2 컬러맵
capture_q     = queue.Queue(maxsize=4)   # 얼굴 crop → DataChannel

def play(path):
    with open(path, "rb") as f:
        requests.post(f"http://{_IP}:59521/audio",
                      files={"audio_file": (f"{path}", f, "audio/mp3")})

def q_put(q: queue.Queue, item):
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    q.put(item)

# ─────────────────────────────────────────────────────────────
# OpenVINO 모델 로드
# ─────────────────────────────────────────────────────────────
DEVICE  = "NPU"
ov_core = Core()

face_det_compiled   = ov_core.compile_model(
    ov_core.read_model("./models/face-detection-retail-0005/FP16-INT8/face-detection-retail-0005.xml"), DEVICE)
age_gender_compiled = ov_core.compile_model(
    ov_core.read_model("./models/age-gender-recognition-retail-0013/FP16-INT8/age-gender-recognition-retail-0013.xml"), DEVICE)
emotion_compiled    = ov_core.compile_model(
    ov_core.read_model("./models/emotions-recognition-retail-0003/FP16-INT8/emotions-recognition-retail-0003.xml"), DEVICE)

face_det_h,   face_det_w   = list(face_det_compiled.input(0).shape)[2:]
age_gender_h, age_gender_w = list(age_gender_compiled.input(0).shape)[2:]
emotion_h,    emotion_w    = list(emotion_compiled.input(0).shape)[2:]

det_model   = YOLO("models/yolo11m-seg_int8_openvino_model")
class_names = det_model.names
detector    = Detector(families="tag36h11")

config_tts = {"PERFORMANCE_HINT": "LATENCY"}
pipe_tts   = ov_core.compile_model(ov_core.read_model("./models/all_base_ov.xml"), "CPU", config_tts)
conf_tts   = utils.get_hparams_from_file("./models/all_base_ov.json")

# ─────────────────────────────────────────────────────────────
# Depth Anything V2 (CPU — NPU 자원 분리)
# ─────────────────────────────────────────────────────────────
DEPTH_MODEL_XML  = "./models/depth_anything_v2_int8.xml"
DEPTH_INPUT_H    = 518   # 모델 권장 입력 크기
DEPTH_INPUT_W    = 518

depth_compiled   = None
depth_output     = None

try:
    _depth_model = ov_core.read_model(DEPTH_MODEL_XML)
    _depth_input = _depth_model.input(0)
    _depth_model.reshape({_depth_input.any_name: [1, 3, DEPTH_INPUT_H, DEPTH_INPUT_W]})
    depth_compiled = ov_core.compile_model(_depth_model, "CPU")
    depth_output   = depth_compiled.output(0)
    print(f"[DEPTH] Depth Anything V2 loaded ({DEPTH_INPUT_H}×{DEPTH_INPUT_W})")
except Exception as e:
    print(f"[ERROR] Depth model load failed: {e}")

def infer_depth(frame_bgr: np.ndarray):
    if depth_compiled is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        return blank, {"min": 0.0, "max": 0.0, "center": 0.0}

    rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp   = cv2.resize(rgb, (DEPTH_INPUT_W, DEPTH_INPUT_H)).astype(np.float32) / 255.0
    inp   = np.expand_dims(inp.transpose(2, 0, 1), 0)

    result = depth_compiled([inp])[depth_output]
    dmap   = result.squeeze()

    cy, cx  = DEPTH_INPUT_H // 2, DEPTH_INPUT_W // 2
    d_min   = float(dmap.min())
    d_max   = float(dmap.max())
    d_ctr   = float(dmap[cy, cx])
    depth_info = {
        "min":    round(d_min, 4),
        "max":    round(d_max, 4),
        "center": round(d_ctr, 4),
    }

    norm  = (dmap - d_min) / max(d_max - d_min, 1e-6)
    u8    = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA)
    colored = cv2.resize(colored, (640, 480))

    return colored, depth_info

# ACLNet (CPU)
audio_ov_config = {
    "SCHEDULING_CORE_TYPE": "ECORE_ONLY",
    "PERFORMANCE_HINT":     "LATENCY",
    "NUM_STREAMS":          "2",
    "INFERENCE_PRECISION_HINT": "f16",
    "CACHE_DIR":            "./ov_cache",
}
try:
    aclnet_compiled  = ov_core.compile_model(
        ov_core.read_model(ACLNET_MODEL_XML), "CPU", audio_ov_config)
    aclnet_output    = aclnet_compiled.output(0)
    aclnet_input_len = aclnet_compiled.input(0).shape[-1]
    with open(ACLNET_CLASSES_TXT) as f:
        ACLNET_CLASSES = [l.strip() for l in f]
    if aclnet_input_len != AUDIO_CHUNK:
        print(f"[WARN] ACLNet input {aclnet_input_len} ≠ CHUNK {AUDIO_CHUNK}")
except Exception as e:
    print(f"[ERROR] ACLNet load failed: {e}")
    aclnet_compiled = None
    ACLNET_CLASSES  = []

# ─────────────────────────────────────────────────────────────
# 전역 상태
# ─────────────────────────────────────────────────────────────
LIVING_CLASSES = {'person','cat','dog','bird','teddy bear','cow','sheep','horse'}
EMOTIONS       = ['neutral','happy','sad','surprise','anger']

state = {
    "charge": 0, "temp": 0, "voltage": 0,
    "cnt_live": 0, "cnt_object": 0, "boxes": [],
    "human": {"age": "", "gender": "", "emotion": "", "position": ""},
    "tag":   {"id": None, "dist": 0},
    # ↓ ACLNet: threshold 이상 모든 결과를 리스트로 저장
    # [{"cls": str, "prob": float}, ...]  — 비어있으면 []
    "audio": {"results": [], "ts": 0},
    "depth": {"min": 0.0, "max": 0.0, "center": 0.0},
}

is_collecting        = False
is_recording         = False
recorded_frames      = []
last_sound_time      = time.time()
recording_start_time = time.time()

_PORT = int(open("port.txt").read())

FACES_DIR = "faces"; os.makedirs(FACES_DIR, exist_ok=True)
last_face_saved_time = 0.0


# ─────────────────────────────────────────────────────────────
# 파일 저장
# ─────────────────────────────────────────────────────────────
def _save_face(img, filename):
    try:
        cv2.imwrite(os.path.join(FACES_DIR, filename), img)
        files = sorted(glob.glob(os.path.join(FACES_DIR, "*.jpg")), key=os.path.getmtime)
        for f in files[:-20]:
            os.remove(f)
    except Exception as e:
        print(f"face save error: {e}")

# ─────────────────────────────────────────────────────────────
# 통합 시각화 함수
# ─────────────────────────────────────────────────────────────
def visualize_all(frame_ai, masks, boxes, classes, scores, alpha=0.5):
    global state
    H, W = frame_ai.shape[:2]
    cell_h, cell_w = H // 3, W // 3

    state['boxes']            = []
    state["cnt_object"]       = 0
    state["cnt_live"]         = 0
    state["human"]["position"] = ""

    out = frame_ai.copy()

    for mask, box, cls_idx, score in zip(masks, boxes, classes, scores):
        cls_name  = class_names[cls_idx]
        is_living = cls_name in LIVING_CLASSES
        color     = (0, 0, 255) if is_living else (0, 255, 0)

        out[mask == 1] = (out[mask == 1] * (1 - alpha)
                          + np.array(color) * alpha).astype(np.uint8)

        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        row = 'T' if cy < cell_h else ('C' if cy < 2 * cell_h else 'B')
        col = 'L' if cx < cell_w else ('C' if cx < 2 * cell_w else 'R')
        pos = row + col

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{cls_name}:{score:.2f}",
                    (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        if is_living:
            state["cnt_live"] += 1
            state["human"]["position"] = pos
        else:
            state["cnt_object"] += 1

        state['boxes'].append({
            'class': cls_name, 'score': round(float(score), 2),
            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
            'position': pos,
        })

    state["human"]["gender"]  = ""
    state["human"]["age"]     = ""
    state["human"]["emotion"] = ""

    resized_fd = cv2.resize(frame_ai, (face_det_w, face_det_h),
                            interpolation=cv2.INTER_NEAREST)
    inp_fd = np.expand_dims(resized_fd.transpose(2, 0, 1), 0).astype(np.float32)
    dets   = face_det_compiled(inp_fd)[face_det_compiled.output(0)]

    face_crops = []

    for det in dets[0][0]:
        conf = float(det[2])
        if conf < 0.5:
            continue
        x1 = max(0, int(det[3] * W))
        y1 = max(0, int(det[4] * H))
        x2 = min(W, int(det[5] * W))
        y2 = min(H, int(det[6] * H))
        crop = frame_ai[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        ag_inp  = np.expand_dims(
            cv2.resize(crop, (age_gender_w, age_gender_h)).transpose(2, 0, 1), 0
        ).astype(np.float32)
        ag_out  = age_gender_compiled(ag_inp)
        age_val = int(ag_out[age_gender_compiled.output("age_conv3")].reshape(1)[0] * 100)
        gend_p  = ag_out[age_gender_compiled.output("prob")].reshape(-1)
        gender  = "W" if np.argmax(gend_p) == 0 else "M"

        em_inp  = np.expand_dims(
            cv2.resize(crop, (emotion_w, emotion_h)).transpose(2, 0, 1), 0
        ).astype(np.float32)
        em_prob = emotion_compiled(em_inp)[emotion_compiled.output(0)].reshape(-1)
        emotion = EMOTIONS[int(np.argmax(em_prob))]

        state["human"]["gender"]  = gender
        state["human"]["age"]     = age_val
        state["human"]["emotion"] = emotion

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"{gender} {age_val}y {emotion}"
        cv2.putText(out, label,
                    (x1, min(H - 4, y2 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        face_crops.append((crop.copy(), x1, y1, x2, y2))

    return out, face_crops


# ─────────────────────────────────────────────────────────────
# 오디오 헬퍼
# ─────────────────────────────────────────────────────────────
def _set_recording_active():
    global is_recording, recorded_frames, recording_start_time, last_sound_time
    is_recording         = True
    recorded_frames      = []
    recording_start_time = time.time()
    last_sound_time      = time.time()
    print("\n[REC] ▶️  Recording STARTED.")

def _stop_recording_and_save():
    global is_recording, recorded_frames
    if not is_recording:
        return
    is_recording = False
    if not recorded_frames:
        print("[REC] 📝 No audio captured.")
        return
    data     = np.concatenate(recorded_frames, axis=0)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(RECORDING_OUTPUT_DIR, f"wakeword_{ts}.wav")
    write_wav(filepath, AUDIO_RATE, data)
    print(f"[REC] 🛑 Saved → {filepath}")
    recorded_frames = []
    threading.Thread(target=_process_stt, args=(filepath,), daemon=True).start()

def _process_stt(filepath: str):
    print(f"[STT] 🚀 {os.path.basename(filepath)}")
    try:
        with open(filepath, 'rb') as f:
            resp = requests.post(
                STT_API_URL,
                params={'lang': STT_LANG, 'isPlay': STT_IS_PLAY},
                files={'file': (os.path.basename(filepath), f, 'audio/wav')},
                timeout=30)
        if resp.status_code == 200:
            text = resp.json().get('data', '').strip()
            if text:
                print(f"[STT] ✅ {text}")
                _call_rag(text)
        else:
            print(f"[STT] ❌ {resp.status_code}")
    except Exception as e:
        print(f"[STT] ❌ {e}")

def _call_rag(prompt: str):
    try:
        r = requests.get(RAG_URL, params={"prompt": prompt, "lang": "ko", "isPlay": 1}, timeout=60)
        print(f"[RAG] ✅ {r.json()}" if r.status_code == 200 else f"[RAG] ❌ {r.status_code}")
    except Exception as e:
        print(f"[RAG] ❌ {e}")

# ─────────────────────────────────────────────────────────────
# Thread-1: 웹캠 수신
# ─────────────────────────────────────────────────────────────
def receiver_thread():
    print("=== Receiver thread started (webcam)")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          WEBCAM_FPS)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam {WEBCAM_INDEX}")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        q_put(raw_q, frame)

# ─────────────────────────────────────────────────────────────
# Thread-mic: 마이크 → WakeWord / ACLNet / VAD
#   ACLNet: ACLNET_THRESHOLD 이상 모든 클래스를 results 리스트로 저장
# ─────────────────────────────────────────────────────────────
def mic_thread_func():
    global is_recording, recorded_frames, last_sound_time, recording_start_time
    print("=== Mic thread started")

    audio  = pyaudio.PyAudio()
    stream = audio.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS,
                        rate=AUDIO_RATE,input_device_index=0, input=True, frames_per_buffer=AUDIO_CHUNK)
    st = None
    try:
        CHUNK = 44100  # 1초 분량을 한 번에 읽음 (에러 방지용)
        while True:

# 읽기 부분
            data = stream.read(CHUNK, exception_on_overflow=False)
            mono = audioop.tomono(data, 2, 1, 1)
            converted, st = audioop.ratecv(mono, 2, 1, 44100, 16000, st)
          
            #chunk      = np.frombuffer(stream.read(AUDIO_CHUNK, exception_on_overflow=False), dtype=np.int16)
            chunk      = np.frombuffer(converted, dtype=np.int16)
            cur_time   = time.time()
            volume     = int(np.max(np.abs(chunk)))
            is_active  = volume > VAD_ACTIVATION_THRESHOLD

            if is_active:

                # ACLNet — 녹음 중이 아닐 때만
                # threshold 이상인 모든 클래스를 결과 리스트로 저장
                if not is_recording and aclnet_compiled:
                    inp   = chunk.astype(np.float32) / 32768.0
                    inp   = inp.reshape(1, 1, 1, -1)
                    probs = aclnet_compiled([inp])[aclnet_output].flatten()

                    results = []
                    for idx, prob in enumerate(probs):
                        if float(prob) >= ACLNET_THRESHOLD:
                            results.append({
                                "cls":  ACLNET_CLASSES[idx],
                                "prob": round(float(prob), 2),
                            })
                    # 확률 높은 순으로 정렬
                    results.sort(key=lambda x: x["prob"], reverse=True)

                    if results:
                        top = results[0]
                        print(f"[AEC] 🔥 {top['cls']} ({top['prob']:.2f})"

                        state["audio"] = {
                            "results": results,
                            "ts":      int(cur_time),
                        }
                    # 결과 없으면 state 초기화 (소리가 사라짐)
                    # 필요 시 주석 해제:
                    # else:
                    #     state["audio"] = {"results": [], "ts": int(cur_time)}

            # 녹음 처리
            if is_recording:
                recorded_frames.append(chunk)
                if cur_time - recording_start_time > MAX_RECORDING_DURATION:
                    _stop_recording_and_save()
                elif is_active:
                    last_sound_time = cur_time
                elif cur_time - last_sound_time > SILENCE_DURATION:
                    _stop_recording_and_save()

    except Exception as e:
        print(f"[MIC] Error: {e}")
    finally:
        stream.stop_stream(); stream.close(); audio.terminate()
        if is_recording:
            _stop_recording_and_save()

# ─────────────────────────────────────────────────────────────
# Thread-2: AI 처리
# ─────────────────────────────────────────────────────────────
def processing_thread():
    global last_face_saved_time
    cnt_image = 0
    depth_tick  = 0
    print("=== Processing thread started")

    while True:
        try:
            frame = raw_q.get(timeout=1.0)
        except queue.Empty:
            continue

        t0       = time.time()
        frame_ai = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_NEAREST)

        res     = det_model(frame_ai, device="intel:npu", verbose=False, conf=0.3)[0]
        masks   = res.masks.data.cpu().numpy().astype(np.uint8) if res.masks else []
        boxes   = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        scores  = res.boxes.conf.cpu().numpy()

        out, face_crops = visualize_all(frame_ai, masks, boxes, classes, scores)

        depth_tick += 1
        if depth_tick >= DEPTH_SKIP_N:
            depth_tick = 0
            depth_colored, depth_info = infer_depth(frame_ai)
            state["depth"] = depth_info
            q_put(stream_q_depth, depth_colored)

        cur_time = time.time()
        for i, (crop, x1, y1, x2, y2) in enumerate(face_crops):
            ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                msg = json.dumps({
                    "type":    "face",
                    "b64":     base64.b64encode(buf).decode(),
                    "age":     state["human"]["age"],
                    "gender":  state["human"]["gender"],
                    "emotion": state["human"]["emotion"],
                })
                try:
                    capture_q.put_nowait(msg)
                except queue.Full:
                    pass

            if i == 0 and cur_time - last_face_saved_time > 15.0:
                last_face_saved_time = cur_time
                threading.Thread(
                    target=_save_face,
                    args=(crop, f"face_{int(cur_time)}.jpg"),
                    daemon=True
                ).start()

        tags = detector.detect(cv2.cvtColor(frame_ai, cv2.COLOR_BGR2GRAY))
        if tags:
            best = max(tags, key=lambda t: cv2.contourArea(t.corners.astype(np.float32)))
            pts  = best.corners.reshape((-1, 1, 2)).astype(np.int32)
            ov2  = out.copy()
            cv2.fillPoly(ov2, [pts], (0, 255, 255))
            out  = cv2.addWeighted(ov2, 0.2, out, 0.8, 0)
            tid  = best.tag_id
            state["tag"]["id"]   = tid
            state["tag"]["dist"] = 0.0
            info = f"ID:{tid}"
            (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            tx, ty = 640 - tw - 20, 640 - 20
            cv2.rectangle(out, (tx - 10, ty - th - 10), (640, 640), (0, 0, 0), -1)
            cv2.putText(out, info, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        fps = 1.0 / (time.time() - t0)
        cv2.putText(out, f"FPS:{fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        q_put(stream_q_main, cv2.resize(out, (640, 480)))

        cnt_image += 1
        if cnt_image % 100 == 0:
            cv2.imwrite("capture.jpg", frame)

# ─────────────────────────────────────────────────────────────
# WebRTC
# ─────────────────────────────────────────────────────────────
def _make_vf(bgr: np.ndarray, pts: int, tb) -> VideoFrame:
    vf = VideoFrame.from_ndarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy(), format="rgb24")
    vf.pts       = pts
    vf.time_base = tb
    return vf


class MainTrack(VideoStreamTrack):
    kind = "video"
    def __init__(self):
        super().__init__()
        self._pts  = 0
        self._tb   = fractions.Fraction(1, 90000)
        self._step = 90000 // 15
        self._last = np.zeros((480, 640, 3), dtype=np.uint8)

    async def recv(self):
        bgr = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: stream_q_main.get(timeout=0.1) if not stream_q_main.empty() else None
        )
        if bgr is not None:
            self._last = bgr
        vf        = _make_vf(self._last, self._pts, self._tb)
        self._pts += self._step
        return vf


class DepthTrack(VideoStreamTrack):
    kind = "video"
    def __init__(self):
        super().__init__()
        self._pts  = 0
        self._tb   = fractions.Fraction(1, 90000)
        self._step = 90000 // 15
        self._last = np.zeros((480, 640, 3), dtype=np.uint8)

    async def recv(self):
        bgr = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: stream_q_depth.get(timeout=0.1) if not stream_q_depth.empty() else None
        )
        if bgr is not None:
            self._last = bgr
        vf        = _make_vf(self._last, self._pts, self._tb)
        self._pts += self._step
        return vf


class WebRTCManager:
    def __init__(self):
        self._pcs:  dict[str, RTCPeerConnection] = {}
        self._dcs:  dict[str, RTCDataChannel]    = {}
        self._last_hash = None

    async def start_broadcast_loop(self, interval=0.1):
        while True:
            await asyncio.sleep(interval)
            open_dcs = [(cid, dc) for cid, dc in self._dcs.items()
                        if dc.readyState == "open"]
            dead     = [cid for cid, dc in self._dcs.items()
                        if dc.readyState not in ("open", "connecting")]

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

            while not capture_q.empty():
                try:
                    cap_msg = capture_q.get_nowait()
                    for cid, dc in open_dcs:
                        try:
                            dc.send(cap_msg.encode())
                        except Exception:
                            pass
                except queue.Empty:
                    break

            for cid in set(dead):
                await self.close(cid)

    async def create_offer(self, client_id: str) -> dict:
        pc = RTCPeerConnection(configuration=RTC_CONFIG)
        self._pcs[client_id] = pc
        pc.addTrack(MainTrack())
        pc.addTrack(DepthTrack())
        dc = pc.createDataChannel("state", ordered=False, maxRetransmits=0)
        self._dcs[client_id] = dc

        @pc.on("connectionstatechange")
        async def _on_state():
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await self.close(client_id)

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def add_process_time_header(request, call_next):
    response = await call_next(request)
    # iframe 허용을 위해 X-Frame-Options 헤더 제거 또는 설정
    # 모든 도메인에서 허용하고 싶다면 헤더 자체를 삭제하거나 'ALLOWALL' (비권장) 처리
    if "X-Frame-Options" in response.headers:
        del response.headers["X-Frame-Options"]
    
    # 또는 특정 도메인만 허용하고 싶다면 (현대적인 브라우저 방식)
    # response.headers["Content-Security-Policy"] = "frame-ancestors 'self' http://parent-domain.com"
    
    return response

@app.on_event("startup")
async def _startup():
    asyncio.create_task(webrtc_manager.start_broadcast_loop())

@app.on_event("shutdown")
async def _shutdown():
    await webrtc_manager.close_all()


class AnswerRequest(BaseModel):
    sdp: str; type: str; client_id: str

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

@app.get("/")
def main_route():
    return {"result": True, "data": "AI-CPU-V2", "ip": _IP, "port": _PORT}

@app.get("/hand")
async def hand(cmd: str):
    requests.get(f"http://{_IP}:59521/hands?cmd={cmd}")
    return {"result": True}

@app.get("/heartbeat")
async def heartbeat():
    return {"result": True, "data": state}

@app.get("/rec/start")
def rec_start():
    if is_recording:
        return {"result": False, "message": "Already recording"}
    _set_recording_active()
    return {"result": True, "message": "Recording started"}

@app.get("/rec/stop")
def rec_stop():
    if not is_recording:
        return {"result": False, "message": "Not recording"}
    _stop_recording_and_save()
    return {"result": True, "message": "Recording stopped"}

@app.get("/start_collection")
def start_collection():
    global is_collecting
    if is_collecting:
        return {"message": "already running"}
    is_collecting = True
    threading.Thread(target=receiver_thread,   daemon=True).start()
    threading.Thread(target=processing_thread, daemon=True).start()
    threading.Thread(target=mic_thread_func,   daemon=True).start()
    return {"message": "started"}

@app.get("/monitor")
def monitor():
    return si.getAll()

def getHash(text):
    h = hashlib.new('md5'); h.update(text.encode()); return h.hexdigest()

# ─────────────────────────────────────────────────────────────
start_collection()
print("NPU", "2502010900")
