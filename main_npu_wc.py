# ─────────────────────────────────────────────────────────────
# [CPU 최적화] 스레드 캡 — numpy/torch import 보다 먼저.
#   이 파일은 Depth Anything(CPU 폴백)/ACLNet(CPU)/YOLO 후처리(torch)가
#   전부 CPU에서 돌아 300%가 나온다. setdefault라 pm2 env로 덮어쓸 수 있음.
# ─────────────────────────────────────────────────────────────
import os
os.environ.setdefault("OMP_NUM_THREADS",      "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS",      "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "4")

import base64
import queue
import threading
import time
import hashlib
import asyncio
import json
import fractions
import glob
import datetime
import logging
import subprocess
import requests
import numpy as np
import cv2
import torch
import matplotlib.colors
import pyaudio
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
import httpx

from scipy.io.wavfile import write as write_wav
from scipy.io.wavfile import write as wav_write
import audioop

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

cv2.setNumThreads(4)
torch.set_num_threads(4)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

_IP          = "192.168.0.34"
_SERVER_PORT = 59530

MOTION_CONTROL_URL = "http://127.0.0.1:3001"
RESUME_DELAY_SEC   = 10

# ── 워크로드 스로틀링 (300% → 완화) ──
DEVICE_RETRY_TRIES = 60
DEVICE_RETRY_DELAY = 2.0
PROCESS_EVERY = 2    # 원본 프레임 N개마다 1번 처리 (30fps → 15fps)
PPE_EVERY     = 2    # helmet 모델은 '처리된 프레임' N번마다 (안전동작 쿨다운 15s라 무방)
FACE_EVERY    = 3    # 얼굴/나이/성별/감정은 N번마다 (그 외엔 캐시 재사용)
TAG_EVERY     = 3    # AprilTag는 N번마다
DEPTH_SKIP_N  = 6    # Depth는 처리프레임 N번마다

WEBCAM_INDEX  = 6
WEBCAM_WIDTH  = 640
WEBCAM_HEIGHT = 480
WEBCAM_FPS    = 30

RTC_CONFIG = RTCConfiguration(iceServers=[])

WAKEWORD_MODEL_NAME      = "./models/alexa_v0.1.xml"
WAKEWORD_THRESHOLD       = 0.5
ACLNET_MODEL_XML         = "./models/aclnet.xml"
ACLNET_CLASSES_TXT       = "./models/aclnet_53cl.txt"
ACLNET_THRESHOLD         = 0.6
VAD_ACTIVATION_THRESHOLD = 500
SILENCE_DURATION         = 2
MAX_RECORDING_DURATION   = 15
RECORDING_OUTPUT_DIR     = "recordings_wakeword"
STT_API_URL              = "http://127.0.0.1:59532/v1/stt"
STT_LANG                 = "ko"
STT_IS_PLAY              = 0
RAG_URL                  = "http://127.0.0.1:59532/v1/rag/txt2chat"
AUDIO_FORMAT             = pyaudio.paInt16
AUDIO_CHANNELS           = 2
AUDIO_RATE               = 44100
AUDIO_CHUNK              = 16000

TARGET_STT_URL = "http://192.168.68.116:59532/v1/stt"

os.makedirs(RECORDING_OUTPUT_DIR, exist_ok=True)
os.makedirs("output", exist_ok=True)

raw_q          = queue.Queue(maxsize=1)
stream_q_main  = queue.Queue(maxsize=1)
stream_q_depth = queue.Queue(maxsize=1)
capture_q      = queue.Queue(maxsize=4)


# ─────────────────────────────────────────────────────────────
# 유선 인터페이스 — ★import 시점 크래시 방지 (부팅 시 랜 미준비 대비 재시도)
# ─────────────────────────────────────────────────────────────
def get_wired_interface(tries=DEVICE_RETRY_TRIES, delay=DEVICE_RETRY_DELAY):
    for i in range(tries):
        try:
            result = subprocess.run(
                ["ip", "-o", "link", "show", "up"],
                capture_output=True, text=True, check=True)
            for line in result.stdout.splitlines():
                name = line.split(":")[1].strip()
                if name.startswith(("enp", "eth")):
                    print(f"[NET] 유선 인터페이스: {name}")
                    return name
        except Exception as e:
            logger.error(f"[NET] ip 조회 실패: {e}")
        logger.error(f"[NET] 활성 유선 인터페이스 대기 {i + 1}/{tries}")
        time.sleep(delay)
    raise RuntimeError("활성화된 유선 인터페이스를 찾을 수 없습니다(최종).")


INTERFACE = get_wired_interface()


def play(path):
    subprocess.run(["./g1_audio", INTERFACE, path])
    if path == "alarm.wav":
        subprocess.run(["./g1_audio", INTERFACE, "safe.wav"])
    else:
        subprocess.run(["./g1_audio", INTERFACE, "unsafe.wav"])


def q_put(q: queue.Queue, item):
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    q.put(item)


# ─────────────────────────────────────────────────────────────
# OpenVINO 모델 로드 (NPU 재시도 컴파일)
# ─────────────────────────────────────────────────────────────
DEVICE  = "NPU"
ov_core = Core()


def compile_with_retry(xml, device, config=None, tries=20, delay=DEVICE_RETRY_DELAY):
    """부팅 직후 NPU 미준비 대비 재시도 컴파일.
    ★ read_model은 루프 밖에서 '1회'만 — 재시도마다 가중치를 재적재하며
      메모리가 늘던 문제를 제거. 최종 실패하면 예외를 던져 프로세스가 종료되고,
      pm2가 '새 프로세스'로 재시작하므로 메모리가 초기화된다(인프로세스 무한
      재시도로 누수를 안고 가지 않음)."""
    model = ov_core.read_model(xml)   # 1회만 적재
    last_err = None
    for i in range(tries):
        try:
            return ov_core.compile_model(model, device, config) if config \
                else ov_core.compile_model(model, device)
        except Exception as e:
            last_err = e
            logger.error(f"[compile] {device} 실패 {i + 1}/{tries}: {e}")
            time.sleep(delay)
    raise RuntimeError(f"[compile] {device} 최종 실패: {last_err}")


face_det_compiled   = compile_with_retry(
    "./models/face-detection-retail-0005/FP16-INT8/face-detection-retail-0005.xml", DEVICE)
age_gender_compiled = compile_with_retry(
    "./models/age-gender-recognition-retail-0013/FP16-INT8/age-gender-recognition-retail-0013.xml", DEVICE)
emotion_compiled    = compile_with_retry(
    "./models/emotions-recognition-retail-0003/FP16-INT8/emotions-recognition-retail-0003.xml", DEVICE)

face_det_h,   face_det_w   = list(face_det_compiled.input(0).shape)[2:]
age_gender_h, age_gender_w = list(age_gender_compiled.input(0).shape)[2:]
emotion_h,    emotion_w    = list(emotion_compiled.input(0).shape)[2:]

# 캐시 미리 참조 (매 프레임 output() 조회 비용 제거)
face_det_out = face_det_compiled.output(0)
age_out_l    = age_gender_compiled.output("age_conv3")
gender_out_l = age_gender_compiled.output("prob")
emotion_out  = emotion_compiled.output(0)

det_model   = YOLO("models/yolo11m-seg_int8_openvino_model")
ppe_model   = YOLO("models/helmet-11s_int8_openvino_model")
class_names = det_model.names
ppe_names   = ppe_model.names

# AprilTag: quad_decimate로 입력을 줄여 CPU 대폭 절감, nthreads 제한
detector    = Detector(families="tag36h11", nthreads=2, quad_decimate=2.0)

# ─────────────────────────────────────────────────────────────
# Depth Anything V2
#   ★ GPU 금지 → NPU 우선 시도, 안 되면 CPU(E코어 전용 + 스레드 4)로 폴백.
#   해상도(518)를 낮추면 CPU 부하가 크게 준다(예: 392=28*14). 품질 보고 조절.
# ─────────────────────────────────────────────────────────────
DEPTH_MODEL_XML   = "./models/depth_anything_v2_int8.xml"
DEPTH_INPUT_H     = 518
DEPTH_INPUT_W     = 518
DEPTH_DEVICE_PREFS = ["NPU", "CPU"]     # GPU 미사용
DEPTH_CPU_CONFIG = {
    "INFERENCE_NUM_THREADS": "4",
    "NUM_STREAMS":           "1",
    "PERFORMANCE_HINT":      "LATENCY",
    "SCHEDULING_CORE_TYPE":  "ECORE_ONLY",   # 추론을 E코어에만
    "ENABLE_HYPER_THREADING": "NO",
    "CACHE_DIR":             "./ov_cache",
}

depth_compiled = None
depth_output   = None
depth_device   = None


def _load_depth():
    global depth_compiled, depth_output, depth_device
    try:
        _m = ov_core.read_model(DEPTH_MODEL_XML)
        _in = _m.input(0)
        _m.reshape({_in.any_name: [1, 3, DEPTH_INPUT_H, DEPTH_INPUT_W]})
    except Exception as e:
        print(f"[DEPTH] 모델 읽기 실패: {e}")
        return

    for dev in DEPTH_DEVICE_PREFS:
        if dev == "CPU":
            # CPU는 풀옵션 → 실패 시 최소옵션 순으로 재시도(비하이브리드 등)
            for cfg in (DEPTH_CPU_CONFIG, {"CACHE_DIR": "./ov_cache"}):
                try:
                    depth_compiled = ov_core.compile_model(_m, "CPU", cfg)
                    depth_output   = depth_compiled.output(0)
                    depth_device   = "CPU"
                    print(f"[DEPTH] loaded on CPU {DEPTH_INPUT_H}x{DEPTH_INPUT_W} cfg={list(cfg)}")
                    return
                except Exception as e:
                    print(f"[DEPTH] CPU 컴파일 실패({list(cfg)}): {e}")
        else:
            try:
                depth_compiled = ov_core.compile_model(_m, dev, {"CACHE_DIR": "./ov_cache"})
                depth_output   = depth_compiled.output(0)
                depth_device   = dev
                print(f"[DEPTH] loaded on {dev} {DEPTH_INPUT_H}x{DEPTH_INPUT_W}")
                return
            except Exception as e:
                print(f"[DEPTH] {dev} 컴파일 실패(폴백): {e}")
    print("[DEPTH] 사용 불가 — depth 비활성화")


_load_depth()


def infer_depth(frame_bgr: np.ndarray):
    if depth_compiled is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        return blank, {"min": 0.0, "max": 0.0, "center": 0.0}

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(rgb, (DEPTH_INPUT_W, DEPTH_INPUT_H)).astype(np.float32) / 255.0
    inp = np.expand_dims(inp.transpose(2, 0, 1), 0)

    result = depth_compiled([inp])[depth_output]
    dmap   = result.squeeze()

    cy, cx = DEPTH_INPUT_H // 2, DEPTH_INPUT_W // 2
    d_min  = float(dmap.min())
    d_max  = float(dmap.max())
    d_ctr  = float(dmap[cy, cx])
    depth_info = {"min": round(d_min, 4), "max": round(d_max, 4), "center": round(d_ctr, 4)}

    norm    = (dmap - d_min) / max(d_max - d_min, 1e-6)
    u8      = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA)
    colored = cv2.resize(colored, (640, 480))
    return colored, depth_info


# ── ACLNet (오디오 분류) : CPU + E코어 ──
audio_ov_config = {
    "SCHEDULING_CORE_TYPE": "ECORE_ONLY",
    "PERFORMANCE_HINT":     "LATENCY",
    "NUM_STREAMS":          "1",
    "INFERENCE_PRECISION_HINT": "f16",
    "CACHE_DIR":            "./ov_cache",
}
aclnet_compiled = None
aclnet_output   = None
ACLNET_CLASSES  = []
try:
    try:
        aclnet_compiled = ov_core.compile_model(
            ov_core.read_model(ACLNET_MODEL_XML), "CPU", audio_ov_config)
    except Exception:
        # ECORE_ONLY 미지원 환경 폴백
        aclnet_compiled = ov_core.compile_model(
            ov_core.read_model(ACLNET_MODEL_XML), "CPU",
            {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": "./ov_cache"})
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

LIVING_CLASSES = {'person', 'cat', 'dog', 'bird', 'teddy bear', 'cow', 'sheep', 'horse'}
EMOTIONS       = ['neutral', 'happy', 'sad', 'surprise', 'anger']

state = {
    "charge": 0, "temp": 0, "voltage": 0,
    "cnt_live": 0, "cnt_object": 0, "boxes": [],
    "human": {"age": "", "gender": "", "emotion": "", "position": ""},
    "tag":   {"id": None, "dist": 0},
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
PPE_DIR   = "ppe";   os.makedirs(PPE_DIR,   exist_ok=True)
last_face_saved_time = 0.0
last_ppe_saved_time  = 0.0

motion_lock = threading.Lock()
last_face_annotations = []   # 얼굴 결과 캐시(스로틀 프레임 사이 재사용)


def find_webcam_mic_index(keywords=None):
    if keywords is None:
        keywords = ["webcam", "usb", "camera", "cam"]
    pa = pyaudio.PyAudio()
    found = None
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) <= 0:
                continue
            name = str(info.get("name", "")).lower()
            print(f"[MIC] idx={i} ch={info['maxInputChannels']} name={info.get('name')}")
            if any(k in name for k in keywords):
                found = i
                print(f"[MIC] ✅ matched → idx={i} ({info.get('name')})")
                break
        if found is None:
            try:
                default = pa.get_default_input_device_info()
                found = int(default["index"])
                print(f"[MIC] ⚠️  no match, using default → idx={found} ({default.get('name')})")
            except Exception:
                found = None
                print("[MIC] ❌ no input device found")
    finally:
        pa.terminate()
    return found


def pause_motion():
    try:
        requests.get(f"{MOTION_CONTROL_URL}/pause", timeout=3)
        print("[MOTION] ⏸️  paused")
    except Exception as e:
        print(f"[MOTION] pause error: {e}")


def resume_motion():
    try:
        requests.get(f"{MOTION_CONTROL_URL}/resume", timeout=3)
        print("[MOTION] ▶️  resumed")
    except Exception as e:
        print(f"[MOTION] resume error: {e}")


def run_safe_action(action_fn):
    if not motion_lock.acquire(blocking=False):
        print("[MOTION] action already in progress, skip")
        return
    try:
        pause_motion()
        try:
            action_fn()
        except Exception as e:
            print(f"[MOTION] action error: {e}")
        time.sleep(RESUME_DELAY_SEC)
    finally:
        resume_motion()
        motion_lock.release()


def _save_face(img, filename):
    try:
        cv2.imwrite(os.path.join(FACES_DIR, filename), img)
        files = sorted(glob.glob(os.path.join(FACES_DIR, "*.jpg")), key=os.path.getmtime)
        for f in files[:-20]:
            os.remove(f)
    except Exception as e:
        print(f"face save error: {e}")


def _save_ppe(img, filename):
    try:
        cv2.imwrite(os.path.join(PPE_DIR, filename), img)
        files = sorted(glob.glob(os.path.join(PPE_DIR, "*.jpg")), key=os.path.getmtime)
        for f in files[:-20]:
            os.remove(f)
    except Exception as e:
        print(f"ppe save error: {e}")


def draw_segmentation(frame_ai, masks, boxes, classes, scores, alpha=0.5):
    """세그멘테이션 오버레이 + 박스 + state(cnt/position/boxes) 갱신."""
    global state
    H, W = frame_ai.shape[:2]
    cell_h, cell_w = H // 3, W // 3

    state['boxes']             = []
    state["cnt_object"]        = 0
    state["cnt_live"]          = 0
    state["human"]["position"] = ""

    out = frame_ai.copy()

    for mask, box, cls_idx, score in zip(masks, boxes, classes, scores):
        cls_name  = class_names[cls_idx]
        is_living = cls_name in LIVING_CLASSES
        color     = (0, 0, 255) if is_living else (0, 255, 0)

        out[mask == 1] = (out[mask == 1] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)

        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        row = 'T' if cy < cell_h else ('C' if cy < 2 * cell_h else 'B')
        col = 'L' if cx < cell_w else ('C' if cx < 2 * cell_w else 'R')
        pos = row + col

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{cls_name}:{score:.2f}", (x1, max(15, y1 - 10)),
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
    return out


def analyze_faces(frame_ai):
    """얼굴 검출 + 나이/성별/감정 (NPU). 결과를 캐시하고 state.human 갱신."""
    global state, last_face_annotations
    H, W = frame_ai.shape[:2]

    state["human"]["gender"]  = ""
    state["human"]["age"]     = ""
    state["human"]["emotion"] = ""

    resized_fd = cv2.resize(frame_ai, (face_det_w, face_det_h), interpolation=cv2.INTER_NEAREST)
    inp_fd = np.expand_dims(resized_fd.transpose(2, 0, 1), 0).astype(np.float32)
    dets   = face_det_compiled(inp_fd)[face_det_out]

    anns = []
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
            cv2.resize(crop, (age_gender_w, age_gender_h)).transpose(2, 0, 1), 0).astype(np.float32)
        ag_out  = age_gender_compiled(ag_inp)
        age_val = int(ag_out[age_out_l].reshape(1)[0] * 100)
        gend_p  = ag_out[gender_out_l].reshape(-1)
        gender  = "W" if np.argmax(gend_p) == 0 else "M"

        em_inp  = np.expand_dims(
            cv2.resize(crop, (emotion_w, emotion_h)).transpose(2, 0, 1), 0).astype(np.float32)
        em_prob = emotion_compiled(em_inp)[emotion_out].reshape(-1)
        emotion = EMOTIONS[int(np.argmax(em_prob))]

        state["human"]["gender"]  = gender
        state["human"]["age"]     = age_val
        state["human"]["emotion"] = emotion

        anns.append({"box": (x1, y1, x2, y2), "text": f"{gender} {age_val}y {emotion}"})

    last_face_annotations = anns
    return anns


def draw_faces(out, anns):
    H = out.shape[0]
    for a in anns:
        x1, y1, x2, y2 = a["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(out, a["text"], (x1, min(H - 4, y2 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return out


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
# Receiver : 웹캠 → raw_q  (★열기 실패/끊김 시 재시도·재오픈)
# ─────────────────────────────────────────────────────────────
def receiver_thread():
    print("=== Receiver thread started (webcam)")
    dev_path = f"/dev/video{WEBCAM_INDEX}" if isinstance(WEBCAM_INDEX, int) else None
    fail = 0
    while True:
        # ★ 장치 노드가 없으면 VideoCapture를 만들지 않는다.
        #   (없는 장치를 반복 open하면 백엔드 핸들/버퍼가 새면서 메모리가 증가)
        if dev_path and not os.path.exists(dev_path):
            fail += 1
            logger.error(f"[WEBCAM] {dev_path} 없음 → 5초 후 재시도 ({fail})")
            if fail > 60:   # 5분 이상 부재 → 프로세스 종료(pm2가 새 프로세스로 재시작)
                logger.error("[WEBCAM] 장치 장기 부재 → 종료")
                os._exit(1)
            time.sleep(5.0)
            continue

        cap = cv2.VideoCapture(WEBCAM_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WEBCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          WEBCAM_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # ★ 드라이버 내부 버퍼 1장 → 항상 최신 프레임
        if not cap.isOpened():
            fail += 1
            logger.error(f"[WEBCAM] {WEBCAM_INDEX} 열기 실패 → 5초 후 재시도 ({fail})")
            cap.release()
            del cap                      # 핸들 즉시 해제
            if fail > 60:
                os._exit(1)
            time.sleep(5.0)
            continue

        fail = 0
        print(f"[WEBCAM] opened idx={WEBCAM_INDEX}")
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("[WEBCAM] frame read 실패 → 재오픈")
                break
            q_put(raw_q, frame)
        cap.release()
        del cap
        time.sleep(1.0)


def mic_thread_func():
    global is_recording, recorded_frames, last_sound_time, recording_start_time
    print("=== Mic thread started")

    # 부팅 직후 USB 오디오 미준비 대비: 장치 탐색 재시도
    mic_index = None
    for _ in range(DEVICE_RETRY_TRIES):
        mic_index = find_webcam_mic_index()
        if mic_index is not None:
            break
        time.sleep(DEVICE_RETRY_DELAY)
    if mic_index is None:
        print("[MIC] ❌ No usable input device. Mic thread aborted.")
        return

    audio = pyaudio.PyAudio()
    dev_info     = audio.get_device_info_by_index(mic_index)
    dev_channels = int(dev_info.get("maxInputChannels", 1))
    use_channels = 2 if dev_channels >= 2 else 1
    print(f"[MIC] using idx={mic_index} name={dev_info.get('name')} channels={use_channels}")

    try:
        stream = audio.open(format=AUDIO_FORMAT, channels=use_channels,
                            rate=AUDIO_RATE, input_device_index=mic_index,
                            input=True, frames_per_buffer=AUDIO_CHUNK)
    except Exception as e:
        print(f"[MIC] ❌ open failed on idx={mic_index}: {e}")
        audio.terminate()
        return

    st = None
    try:
        CHUNK = 44100
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            if use_channels == 2:
                mono = audioop.tomono(data, 2, 1, 1)
            else:
                mono = data
            converted, st = audioop.ratecv(mono, 2, 1, 44100, 16000, st)

            chunk     = np.frombuffer(converted, dtype=np.int16)
            cur_time  = time.time()
            volume    = int(np.max(np.abs(chunk)))
            is_active = volume > VAD_ACTIVATION_THRESHOLD

            if is_active:
                if not is_recording and aclnet_compiled:
                    inp   = chunk.astype(np.float32) / 32768.0
                    inp   = inp.reshape(1, 1, 1, -1)
                    probs = aclnet_compiled([inp])[aclnet_output].flatten()

                    results = []
                    for idx, prob in enumerate(probs):
                        if float(prob) >= ACLNET_THRESHOLD:
                            results.append({"cls": ACLNET_CLASSES[idx], "prob": round(float(prob), 2)})
                    results.sort(key=lambda x: x["prob"], reverse=True)

                    if results:
                        state["audio"] = {"results": results, "ts": int(cur_time)}

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


def warmup_yolo(model, name, tries=20, delay=DEVICE_RETRY_DELAY):
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for i in range(tries):
        try:
            model(dummy, device="intel:npu", verbose=False, conf=0.3)
            print(f"[YOLO] {name} NPU 워밍업 성공 (시도 {i + 1})")
            return
        except Exception as e:
            logger.error(f"[YOLO] {name} 워밍업 실패 {i + 1}/{tries}: {e}")
            time.sleep(delay)
    # ★ 끝내 실패하면 프레임마다 추론을 재시도하며 메모리가 늘 수 있으므로,
    #   틈새 재시도 대신 프로세스를 종료해 pm2가 새 프로세스로 재시작하게 한다.
    logger.error(f"[YOLO] {name} 워밍업 최종 실패 → 프로세스 종료(pm2 재시작)")
    os._exit(1)


def processing_thread():
    global last_face_saved_time, last_ppe_saved_time
    warmup_yolo(det_model, "det")
    warmup_yolo(ppe_model, "ppe")

    cnt_image = 0
    frame_idx = 0
    proc_idx  = 0
    consecutive_fail = 0
    print("=== Processing thread started")

    while True:
        try:
            frame = raw_q.get(timeout=1.0)
        except queue.Empty:
            continue

        frame_idx += 1
        if frame_idx % PROCESS_EVERY != 0:      # 프레임 스로틀
            continue

        try:
            t0       = time.time()
            frame_ai = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_NEAREST)

            res = det_model(frame_ai, device="intel:npu", verbose=False, conf=0.3)[0]
            masks   = res.masks.data.cpu().numpy().astype(np.uint8) if res.masks is not None else []
            boxes   = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            scores  = res.boxes.conf.cpu().numpy()

            out = draw_segmentation(frame_ai, masks, boxes, classes, scores)

            # ── PPE(helmet) : 스로틀 ──
            if proc_idx % PPE_EVERY == 0:
                cur_time = time.time()
                ppe_res  = ppe_model(frame_ai, device="intel:npu", verbose=False, conf=0.5)[0]
                if ppe_res.boxes is not None:
                    for i, box in enumerate(ppe_res.boxes.xyxy.cpu().numpy()):
                        x1, y1, x2, y2 = map(int, box)
                        conf   = float(ppe_res.boxes.conf.cpu().numpy()[i])
                        cls_id = int(ppe_res.boxes.cls.cpu().numpy()[i])
                        label  = ppe_names.get(cls_id, str(cls_id))

                        if 'helmet' in label or 'face' in label:
                            ch, cw   = frame_ai.shape[:2]
                            crop     = frame_ai[max(0, y1):min(ch, y2), max(0, x1):min(cw, x2)].copy()
                            cap_type = "ppe" if 'helmet' in label else "face"

                            ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 75])
                            if ok:
                                msg = json.dumps({
                                    "type":  cap_type,
                                    "b64":   base64.b64encode(buf).decode(),
                                    "label": label,
                                    "conf":  round(conf, 2),
                                })
                                try:
                                    capture_q.put_nowait(msg)
                                except queue.Full:
                                    pass

                            if cap_type == "ppe" and cur_time - last_ppe_saved_time > 15.0:
                                last_ppe_saved_time = cur_time

                                def _ppe_action(img, fn):
                                    def action():
                                        threading.Thread(target=_save_ppe, args=(img, fn), daemon=True).start()
                                        led("255", "255", "255")

                                        def send_ppe_request():
                                            try:
                                                httpx.get("http://127.0.0.1:59532/v2/img2chat",
                                                          params={"prompt": "다음과 같이 사람이 안전모를 쓴 상황에서 어떤 말을 하면 좋을까?"})
                                            except Exception as e:
                                                print(f"PPE Network error: {e}")
                                        threading.Thread(target=send_ppe_request, daemon=True).start()
                                        threading.Thread(target=play, args=('welcome.wav',), daemon=True).start()
                                        arm("lowWave")
                                        arm("Release_Arm")
                                    run_safe_action(action)

                                threading.Thread(
                                    target=_ppe_action,
                                    args=(crop.copy(), f"ppe_{label}_{int(cur_time)}.jpg"),
                                    daemon=True).start()

                            elif cap_type == "face" and cur_time - last_face_saved_time > 15.0:
                                last_face_saved_time = cur_time

                                def _face_action(img, fn):
                                    def action():
                                        led("255", "0", "0")
                                        threading.Thread(target=_save_face, args=(img, fn), daemon=True).start()

                                        def send_face_request():
                                            try:
                                                httpx.get("http://127.0.0.1:59532/v2/img2chat",
                                                          params={"prompt": "다음과 같이 사람이 안전모를 쓰지 않은 위험상황에 대해 이야기 해줘."})
                                            except Exception as e:
                                                print(f"Face Network error: {e}")
                                        threading.Thread(target=send_face_request, daemon=True).start()
                                        threading.Thread(target=play, args=('alarm.wav',), daemon=True).start()
                                        arm("Refuse")
                                        arm("Release_Arm")
                                    run_safe_action(action)

                                threading.Thread(
                                    target=_face_action,
                                    args=(crop.copy(), f"face_{int(cur_time)}.jpg"),
                                    daemon=True).start()

            # ── 얼굴(캐시) ──
            if proc_idx % FACE_EVERY == 0:
                analyze_faces(frame_ai)
            out = draw_faces(out, last_face_annotations)

            # ── Depth(스로틀) ──
            if proc_idx % DEPTH_SKIP_N == 0:
                depth_colored, depth_info = infer_depth(frame_ai)
                state["depth"] = depth_info
                q_put(stream_q_depth, depth_colored)

            # ── AprilTag(스로틀) ──
            if proc_idx % TAG_EVERY == 0:
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

            fps = 1.0 / max(1e-6, (time.time() - t0))
            cv2.putText(out, f"FPS:{fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            q_put(stream_q_main, cv2.resize(out, (640, 480)))

            proc_idx  += 1
            cnt_image += 1
            if cnt_image % 100 == 0:
                cv2.imwrite("capture2.jpg", frame)

            consecutive_fail = 0     # 성공 → 실패 카운터 리셋

        except Exception as e:
            # ★ 실패 시 50ms 고정 스핀 대신 백오프, 연속 실패가 과하면 프로세스 종료
            #   (실패를 안고 계속 도는 대신 pm2가 새 프로세스로 재시작 → 메모리 리셋)
            consecutive_fail += 1
            logger.error(f"[processing] 루프 오류({consecutive_fail}): {e}")
            if consecutive_fail >= 300:
                logger.error("[processing] 연속 실패 과다 → 프로세스 종료(pm2 재시작)")
                os._exit(1)
            time.sleep(min(2.0, 0.1 * consecutive_fail))


def _make_vf(bgr: np.ndarray, pts: int, tb) -> VideoFrame:
    vf = VideoFrame.from_ndarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy(), format="rgb24")
    vf.pts       = pts
    vf.time_base = tb
    return vf


def _drain_latest(q: queue.Queue):
    """큐에 쌓인 것 중 '가장 최신 1장'만 반환하고 나머지는 버린다(누적 방지).
    (큐 maxsize=1이라 사실상 최대 1장이지만, 방어적으로 끝까지 비운다.)"""
    bgr = None
    while True:
        try:
            bgr = q.get_nowait()
        except queue.Empty:
            break
    return bgr


class MainTrack(VideoStreamTrack):
    kind = "video"
    def __init__(self):
        super().__init__()
        self._last = np.zeros((480, 640, 3), dtype=np.uint8)

    async def recv(self):
        # ★ next_timestamp(): 실제 시간에 맞춰 프레임을 페이싱(sleep)하고 올바른 pts를 부여.
        #   → recv 폭주/타임스탬프 폭주로 인한 지연 누적 제거.
        pts, time_base = await self.next_timestamp()
        bgr = _drain_latest(stream_q_main)     # 최신 1장만
        if bgr is not None:
            self._last = bgr
        return _make_vf(self._last, pts, time_base)


class DepthTrack(VideoStreamTrack):
    kind = "video"
    def __init__(self):
        super().__init__()
        self._last = np.zeros((480, 640, 3), dtype=np.uint8)

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        bgr = _drain_latest(stream_q_depth)    # 최신 1장만
        if bgr is not None:
            self._last = bgr
        return _make_vf(self._last, pts, time_base)


# 소스 트랙은 프로세스당 1개만 생성 → MediaRelay로 공유 (접속마다 재생성 안 함)
main_source  = MainTrack()
depth_source = DepthTrack()
try:
    from aiortc.contrib.media import MediaRelay
    _relay = MediaRelay()
except Exception:
    _relay = None


class WebRTCManager:
    def __init__(self):
        self._pcs:  dict[str, RTCPeerConnection] = {}
        self._dcs:  dict[str, RTCDataChannel]    = {}
        self._last_hash = None

    async def start_broadcast_loop(self, interval=0.1):
        while True:
            await asyncio.sleep(interval)
            open_dcs = [(cid, dc) for cid, dc in self._dcs.items() if dc.readyState == "open"]
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
        if _relay is not None:
            pc.addTrack(_relay.subscribe(main_source))
            pc.addTrack(_relay.subscribe(depth_source))
        else:
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

app = FastAPI()
app.mount("/web",      StaticFiles(directory="web"),      name="web")
app.mount("/webfonts", StaticFiles(directory="webfonts"), name="webfonts")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def add_process_time_header(request, call_next):
    response = await call_next(request)
    if "X-Frame-Options" in response.headers:
        del response.headers["X-Frame-Options"]
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


G1_ACTION = {
    "clamp": 17, "highFive": 18, "shakeHands_1": 27, "makeHeartBothHands": 20,
    "makeHeartSingleHands": 21, "blowKiss": 12, "hug": 19, "hightWave": 26,
    "lowWave": 25, "ultramanRay": 24, "bothHandsUp": 15, "singleHandsUp": 23,
    "Refuse": 22, "Release_Arm": 99,
}


@app.get("/led")
def led(r: str = "255", g: str = "255", b: str = "255"):
    try:
        subprocess.run(["./g1_vui", INTERFACE, r, g, b], check=True)
        return {"result": True, "message": f"LED 색상 설정 완료: ({r}, {g}, {b})"}
    except subprocess.CalledProcessError as e:
        return {"result": False, "error": f"g1_vui 실행 실패: {e}"}


@app.get("/arm")
def arm(cmd: str = "lowWave"):
    if cmd not in G1_ACTION:
        return {"result": False, "error": f"Unknown command: {cmd}"}
    try:
        print(str(G1_ACTION[cmd]))
        subprocess.run(["./g1_action", INTERFACE, str(G1_ACTION[cmd])], check=True)
        return {"result": True, "message": f"Action 실행 완료: {cmd}"}
    except subprocess.CalledProcessError as e:
        return {"result": False, "error": f"g1_action 실행 실패: {e}"}


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


@app.post("/v1/stt")
async def proxy_stt(file: UploadFile = File(...), lang: str = Query("en"), isPlay: int = Query(0)):
    try:
        file_content = await file.read()
        params = {"lang": lang, "isPlay": isPlay}
        files = {"file": (file.filename, file_content, file.content_type)}
        async with httpx.AsyncClient() as client:
            response = await client.post(TARGET_STT_URL, params=params, files=files, timeout=60.0)
        return response.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=500, detail=f"Internal STT Server connection error: {exc}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()


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


start_collection()
print("NPU", "2502010900")