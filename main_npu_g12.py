"""
구조:
  Thread-1 receiver   : HTTP → raw_q (수신 전용)
  Thread-2 processing : raw_q → AI추론/시각화 → stream_q["main"/"depth"] (처리 전용)
  asyncio  WebRTC     : stream_q → FrameProviderTrack.recv() → WebRTC 송출
  asyncio  FastAPI    : 기존 REST 엔드포인트 그대로
"""

import queue
import threading
import time
import time as t
import hashlib
import asyncio
import json
import fractions
import os
import glob
import logging
import requests
import numpy as np
import cv2
import matplotlib.colors

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy.io.wavfile import write
from pydub import AudioSegment
from playsound import playsound
from pyapriltags import Detector
from av import VideoFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel, VideoStreamTrack, RTCConfiguration

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
_IP          = "192.168.21.19"
_SERVER_PORT = 59530
SOURCE_VIDEO_URL = f"http://{_IP}:59512/video_raw"

RTC_CONFIG = RTCConfiguration(iceServers=[])

# ─────────────────────────────────────────────────────────────
# 큐 선언 (threading.Queue, maxsize=1 → 최신 1장만 유지)
# ─────────────────────────────────────────────────────────────
raw_q          = queue.Queue(maxsize=1)
stream_q_main  = queue.Queue(maxsize=1)
stream_q_depth = queue.Queue(maxsize=1)
capture_q      = queue.Queue(maxsize=4)

def q_put(q: queue.Queue, item):
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    q.put(item)

# ─────────────────────────────────────────────────────────────
# OpenVINO 모델 로드
# ─────────────────────────────────────────────────────────────
DEVICE = "NPU"
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

det_model   = YOLO('./models/yolo11s-seg_int8_openvino_model')
ppe_model   = YOLO('./models/yolo11n-helmet4_int8_openvino_model')
class_names = det_model.names
ppe_names   = ppe_model.names
print(ppe_names)

detector = Detector(families="tag36h11")

config_tts = {"PERFORMANCE_HINT": "LATENCY"}
pipe_tts   = ov_core.compile_model(ov_core.read_model("./models/all_base_ov.xml"), "CPU", config_tts)
conf_tts   = utils.get_hparams_from_file("./models/all_base_ov.json")

# ─────────────────────────────────────────────────────────────
# 전역 상태
# ─────────────────────────────────────────────────────────────
LIVING_CLASSES = {'person','cat','dog','bird','teddy bear','cow','sheep','horse'}
EMOTIONS       = ['neutral','happy','sad','surprise','anger']

state = {
    "charge":0,"temp":0,"voltage":0,
    "cnt_live":0,"cnt_object":0,"boxes":[],
    "human":{"age":"","gender":"","emotion":"","position":""},
    "tag":{"id":None,"dist":0}
}

conn          = None
audio_hub     = None
lastColor     = 'cyan'
lastCmd       = {}
is_collecting = False

_PORT = int(open("port.txt").read())

FACES_DIR = "faces"; os.makedirs(FACES_DIR, exist_ok=True)
PPE_DIR   = "ppe";   os.makedirs(PPE_DIR,   exist_ok=True)
last_face_saved_time = 0.0
last_ppe_saved_time  = 0.0

# ─────────────────────────────────────────────────────────────
# 파일 저장
# ─────────────────────────────────────────────────────────────
def _save_face(img, filename):
    try:
        cv2.imwrite(os.path.join(FACES_DIR, filename), img)
    except Exception as e:
        print(f"face save error: {e}")

def _save_ppe(img, filename):
    try:
        cv2.imwrite(os.path.join(PPE_DIR, filename), img)
        files = sorted(glob.glob(os.path.join(PPE_DIR,"*.jpg")), key=os.path.getmtime)
        for f in files[:-20]:
            os.remove(f)
    except Exception as e:
        print(f"ppe save error: {e}")

# ─────────────────────────────────────────────────────────────
# AI 처리 헬퍼
# ─────────────────────────────────────────────────────────────
def visualize_segmentation(frame, masks, boxes, classes, scores, depths, alpha=0.5):
    global state
    overlay = frame.copy()
    state['boxes'] = []; state["cnt_object"] = 0; state["cnt_live"] = 0
    state["human"]["depth"] = ""; state["human"]["position"] = ""
    H, W = frame.shape[:2]
    cell_h, cell_w = H//3, W//3

    for mask, box, cls_idx, score, depth in zip(masks, boxes, classes, scores, depths):
        cls_name  = class_names[cls_idx]
        is_living = cls_name in LIVING_CLASSES
        color     = (0,0,255) if is_living else (0,255,0)
        overlay[mask==1] = (overlay[mask==1]*(1-alpha) + np.array(color)*alpha).astype(np.uint8)
        x1,y1,x2,y2 = map(int, box)
        cv2.rectangle(overlay,(x1,y1),(x2,y2),color,2)
        cx,cy = (x1+x2)//2, (y1+y2)//2
        row = 'T' if cy<cell_h else ('C' if cy<2*cell_h else 'B')
        col = 'L' if cx<cell_w else ('C' if cx<2*cell_w else 'R')
        pos = row+col
        if is_living:
            state["cnt_live"] += 1
            state["human"]["depth"]    = depth
            state["human"]["position"] = pos
        else:
            state["cnt_object"] += 1
        state['boxes'].append({'class':cls_name,'score':round(float(score),2),
                                'bbox':{'x1':x1,'y1':y1,'x2':x2,'y2':y2},'position':pos,'depth':depth})
        cv2.putText(overlay,f"{cls_name}:{score:.2f}|{depth:.2f}m",(x1,max(15,y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
    return overlay

def get_mask_depths(masks, depth_frame, low_pct=5):
    depths = []
    for mask in masks:
        if mask.sum()==0: depths.append(0.0); continue
        vals  = depth_frame[mask==1]; valid = vals[vals>0]
        if len(valid)>0:
            thr = np.percentile(valid, low_pct)
            f   = valid[valid>=thr]
            depths.append((np.min(f) if len(f)>0 else np.min(valid))/1000.0)
        else:
            depths.append(0.0)
    return depths

# ─────────────────────────────────────────────────────────────
# Thread-1: receiver
# ─────────────────────────────────────────────────────────────
def receiver_thread():
    W, H       = 640, 480
    RGB_SIZE   = W*H*3
    DEPTH_SIZE = W*H*2
    TOTAL      = RGB_SIZE + DEPTH_SIZE
    print("=== Receiver thread started")
    while True:
        try:
            resp = requests.get(SOURCE_VIDEO_URL, timeout=1.0)
            if resp.status_code == 200 and len(resp.content) >= TOTAL:
                raw   = resp.content
                frame = np.frombuffer(raw[:RGB_SIZE],      dtype=np.uint8 ).reshape(H,W,3).copy()
                depth = np.frombuffer(raw[RGB_SIZE:TOTAL], dtype=np.uint16).reshape(H,W  ).copy()
                q_put(raw_q, (frame, depth))
            else:
                time.sleep(0.01)
        except Exception as e:
            print(f"Receiver error: {e}")
            time.sleep(0.1)

# ─────────────────────────────────────────────────────────────
# Thread-2: processing
# ─────────────────────────────────────────────────────────────
def processing_thread():
    global last_face_saved_time, last_ppe_saved_time
    cnt_image = 0
    print("=== Processing thread started")

    while True:
        try:
            frame, depth_frame = raw_q.get(timeout=1.0)
        except queue.Empty:
            continue

        t0 = time.time()

        # 전처리
        frame_ai = cv2.resize(frame, (640,640), interpolation=cv2.INTER_NEAREST)
        depth_ai = cv2.resize(depth_frame, (640,640), interpolation=cv2.INTER_NEAREST)

        # ── Depth 컬러맵 → stream_q_depth ─────────────────────
        # [수정1] resize(640,480) — main 트랙과 크기 통일 필수
        # FrameProviderTrack._last_bgr 가 (480,640,3) 으로 초기화되어 있어
        # 크기가 다르면 VideoFrame 생성 시 오류 또는 검정화면 발생
        depth_vis     = np.clip(depth_ai, 0, 4000).astype(np.float32)
        depth_vis     = (depth_vis / 4000.0 * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
        q_put(stream_q_depth, cv2.resize(depth_colored, (640, 480)))

        # NPU 추론
        res     = det_model(frame_ai, device="intel:npu", verbose=False, conf=0.3)[0]
        ppe_res = ppe_model(frame_ai, device="intel:npu", verbose=False, conf=0.7)[0]

        # 세그멘테이션
        masks   = res.masks.data.cpu().numpy().astype(np.uint8) if res.masks else []
        boxes   = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        scores  = res.boxes.conf.cpu().numpy()

        # PPE 탐지
        cur_time = time.time()
        if ppe_res.boxes is not None:
            for i, box in enumerate(ppe_res.boxes.xyxy.cpu().numpy()):
                x1,y1,x2,y2 = map(int, box)
                conf   = float(ppe_res.boxes.conf.cpu().numpy()[i])
                cls_id = int(ppe_res.boxes.cls.cpu().numpy()[i])
                label  = ppe_names.get(cls_id, str(cls_id))

                if 'helmet' in label or 'face' in label:
                    ch,cw    = frame_ai.shape[:2]
                    crop     = frame_ai[max(0,y1):min(ch,y2), max(0,x1):min(cw,x2)].copy()
                    cap_type = "ppe" if 'helmet' in label else "face"

                    # DataChannel 캡처 전송
                    ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        import base64
                        b64 = base64.b64encode(buf).decode()
                        msg = json.dumps({"type":cap_type,"b64":b64,
                                          "label":label,"conf":round(conf,2)})
                        try:
                            capture_q.put_nowait(msg)
                        except queue.Full:
                            pass

                    # ── [수정2] LED/TTS/파일저장 → daemon 스레드로 분리 ──
                    # processing_thread 안에서 직접 호출하면 HTTP 요청(requests)이
                    # 수백ms~수초 블로킹 → 프레임 드롭 심화
                    if cap_type == "ppe" and cur_time - last_ppe_saved_time > 10.0:
                        last_ppe_saved_time = cur_time
                        def _ppe_action(img, fn):
                            led(255, 255, 255)
                            tts_v2("오늘도 좋은 하루입니다.", 31)
                            _save_ppe(img, fn)
                        threading.Thread(
                            target=_ppe_action,
                            args=(crop.copy(), f"ppe_{label}_{int(cur_time)}.jpg"),
                            daemon=True
                        ).start()

                    elif cap_type == "face" and cur_time - last_face_saved_time > 10.0:
                        last_face_saved_time = cur_time
                        def _face_action(img, fn):
                            led(255, 0, 0)
                            tts_v2("안전모를 착용해 주세요", 31)
                            _save_face(img, fn)
                        threading.Thread(
                            target=_face_action,
                            args=(crop.copy(), f"face_{int(cur_time)}.jpg"),
                            daemon=True
                        ).start()

        out = visualize_segmentation(frame_ai, masks, boxes, classes, scores,
                                     get_mask_depths(masks, depth_ai))

        # AprilTag
        tags = detector.detect(cv2.cvtColor(frame_ai, cv2.COLOR_BGR2GRAY))
        if tags:
            best  = max(tags, key=lambda t: cv2.contourArea(t.corners.astype(np.float32)))
            pts   = best.corners.reshape((-1,1,2)).astype(np.int32)
            ov2   = out.copy(); cv2.fillPoly(ov2,[pts],(0,255,255))
            out   = cv2.addWeighted(ov2,0.2,out,0.8,0)
            tid   = best.tag_id
            cx,cy = int(best.center[0]), int(best.center[1])
            dist  = depth_ai[cy,cx]/1000.0 if 0<=cy<640 and 0<=cx<640 else 0.0
            state["tag"]["id"] = tid; state["tag"]["dist"] = dist
            info  = f"ID:{tid} / {dist:.2f}m"
            (tw,th),_ = cv2.getTextSize(info,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
            tx = 640-tw-20; ty = 640-20
            cv2.rectangle(out,(tx-10,ty-th-10),(640,640),(0,0,0),-1)
            cv2.putText(out,info,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

        # FPS
        fps = 1.0/(time.time()-t0)
        cv2.putText(out,f"FPS:{fps:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

        # 메인 스트리밍 큐 push
        q_put(stream_q_main, cv2.resize(out,(640,480)))

        cnt_image += 1
        if cnt_image % 100 == 0:
            cv2.imwrite("capture.jpg", frame)

# ─────────────────────────────────────────────────────────────
# WebRTC VideoStreamTrack
# ─────────────────────────────────────────────────────────────
class FrameProviderTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, q: queue.Queue, fps: int = 15):
        super().__init__()
        self._q        = q
        self._pts      = 0
        self._tb       = fractions.Fraction(1, 90000)
        self._step     = 90000 // fps
        self._last_bgr = np.zeros((480,640,3), dtype=np.uint8)
        self._last_vf  = None

    async def recv(self):
        loop = asyncio.get_event_loop()

        def _get():
            try:
                return self._q.get(timeout=0.1)
            except queue.Empty:
                return None

        bgr = await loop.run_in_executor(None, _get)

        if bgr is not None:
            self._last_bgr = bgr
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._last_vf  = VideoFrame.from_ndarray(rgb, format="rgb24")
        elif self._last_vf is not None:
            vf           = self._last_vf
            vf.pts       = self._pts
            vf.time_base = self._tb
            self._pts   += self._step
            return vf
        else:
            rgb           = cv2.cvtColor(self._last_bgr, cv2.COLOR_BGR2RGB)
            self._last_vf = VideoFrame.from_ndarray(rgb, format="rgb24")

        vf           = self._last_vf
        vf.pts       = self._pts
        vf.time_base = self._tb
        self._pts   += self._step
        return vf

# ─────────────────────────────────────────────────────────────
# WebRTC 연결 관리
# ─────────────────────────────────────────────────────────────
class WebRTCManager:
    def __init__(self):
        self._pcs:  dict[str, RTCPeerConnection] = {}
        self._dcs:  dict[str, RTCDataChannel]    = {}
        self._last_hash = None

    async def start_broadcast_loop(self, interval=0.1):
        while True:
            await asyncio.sleep(interval)

            open_dcs = [(cid,dc) for cid,dc in self._dcs.items() if dc.readyState=="open"]
            dead     = [cid for cid,dc in self._dcs.items()
                        if dc.readyState not in ("open","connecting")]

            js = json.dumps(state, ensure_ascii=False)
            h  = hash(js)
            if h != self._last_hash:
                self._last_hash = h
                msg = js.encode()
                for cid,dc in open_dcs:
                    try: dc.send(msg)
                    except Exception: dead.append(cid)

            while not capture_q.empty():
                try:
                    cap_msg = capture_q.get_nowait()
                    for cid,dc in open_dcs:
                        try: dc.send(cap_msg.encode())
                        except Exception: pass
                except queue.Empty:
                    break

            for cid in set(dead):
                await self.close(cid)

    async def create_offer(self, client_id: str) -> dict:
        pc = RTCPeerConnection(configuration=RTC_CONFIG)
        self._pcs[client_id] = pc

        # mid=0: main, mid=1: depth  (addTrack 순서 = SDP mid 순서)
        pc.addTrack(FrameProviderTrack(stream_q_main,  fps=15))
        pc.addTrack(FrameProviderTrack(stream_q_depth, fps=15))

        dc = pc.createDataChannel("state", ordered=False, maxRetransmits=0)
        self._dcs[client_id] = dc

        @pc.on("connectionstatechange")
        async def _on_state():
            logger.info(f"PC [{client_id}] → {pc.connectionState}")
            if pc.connectionState in ("failed","closed","disconnected"):
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

        sdp   = pc.localDescription.sdp
        cands = [l for l in sdp.splitlines() if l.startswith("a=candidate")]
        print(f"[ICE] server candidates ({len(cands)}):")
        for c in cands:
            print("  ", c)

        return {"sdp":sdp,"type":pc.localDescription.type,"client_id":client_id}

    async def set_answer(self, client_id: str, answer_sdp: str, answer_type: str):
        pc = self._pcs.get(client_id)
        if pc is None:
            raise ValueError(f"Unknown client_id: {client_id}")
        await pc.setRemoteDescription(RTCSessionDescription(answer_sdp, answer_type))

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
    offer = await webrtc_manager.create_offer(client_id)
    return JSONResponse(offer)

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
    return {"result": True, "data": state}

@app.get("/start_collection")
async def start_collection():
    global is_collecting
    if is_collecting:
        return {"message": "already running"}
    is_collecting = True
    threading.Thread(target=receiver_thread,   daemon=True).start()
    threading.Thread(target=processing_thread, daemon=True).start()
    return {"message": "started"}

def toFloat(v):
    try: return float(v)
    except: return v

@app.get("/monitor")
def monitor():
    return si.getAll()

def getHash(text):
    h = hashlib.new('md5'); h.update(text.encode()); return h.hexdigest()

@app.get("/v1/tts", response_class=FileResponse)
def tts(text="", voice=31, lang='ko', static=0, isPlay=0):
    filename = getHash(text)
    ids      = text_to_sequence(text, [f'canvers_{lang}_cleaners'])
    inp      = np.expand_dims(np.array(ids, dtype=np.int64), 0)
    inputs   = {"input":inp,"input_lengths":np.array([inp.shape[1]],dtype=np.int64),
                "scales":np.array([0.667,1.0,0.8],dtype=np.float16),
                "sid":np.array([int(voice)],dtype=np.int64)}
    audio    = list(pipe_tts(inputs).values())[0].squeeze((0,1))
    if int(static) > 0:
        write(data=audio, rate=conf_tts.data.sampling_rate, filename="output/human.wav")
        return "output/human.wav"
    path = f"output/{filename}.wav"
    write(data=audio, rate=conf_tts.data.sampling_rate, filename=path)
    seg = AudioSegment.from_wav(path).set_frame_rate(22050).set_sample_width(2).set_channels(1)
    seg.export(path, format='wav', codec="pcm_s16le")
    if int(isPlay) > 0: playsound(path)
    return path

@app.get("/v2/tts", response_class=FileResponse)
def tts_v2(text="", voice=6, lang='ko'):
    filename = getHash(text)
    ids      = text_to_sequence(text, [f'canvers_{lang}_cleaners'])
    inp      = np.expand_dims(np.array(ids, dtype=np.int64), 0)
    inputs   = {"input":inp,"input_lengths":np.array([inp.shape[1]],dtype=np.int64),
                "scales":np.array([0.667,1.0,0.8],dtype=np.float16),
                "sid":np.array([int(voice)],dtype=np.int64)}
    audio    = list(pipe_tts(inputs).values())[0].squeeze((0,1))
    path     = f"output/{filename}.wav"
    write(data=audio, rate=conf_tts.data.sampling_rate, filename=path)
    with open(path,"rb") as f:
        requests.post(f"http://{_IP}:59521/audio",
                      files={"audio_file":(f"{filename}.wav",f,"audio/wav")})
    return path

@app.get("/led")
def led(r:int=0, g:int=0, b:int=0):
    requests.get(f"http://{_IP}:59521/led?r={r}&g={g}&b={b}")
    return {"result": True}

@app.get("/color")
def color(value:str='red'):
    rgb = (np.array(matplotlib.colors.to_rgb(value))*255).astype(int)
    requests.get(f"http://{_IP}:59521/led?r={rgb[0]}&g={rgb[1]}&b={rgb[2]}")
    return {"result": True}

print("NPU", "2502010900")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_webrtc_new:app", host="0.0.0.0", port=_SERVER_PORT, reload=False)
