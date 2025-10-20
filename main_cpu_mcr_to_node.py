from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
import serial
import time
from huggingface_hub import snapshot_download, hf_hub_download
import time as t
from serverinfo import si
import logging
import asyncio
from requests import get
import time
from fastapi.staticfiles import StaticFiles
import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
#from playsound import playsound
from mandro import HadnControler
import threading
from fastapi.middleware.cors import CORSMiddleware
import hashlib
import asyncio
import os
ser = None

def getHash(text):
  hash_func = hashlib.new('md5')
  hash_func.update(text.encode('utf-8'))
  return hash_func.hexdigest()

hL = None
hR = None

# Enable logging for debugging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

_IP = "127.0.0.1" #si.getIP()
_PORT = int(open("port.txt", 'r').read())

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

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import serial
import time
import json
from pathlib import Path
from types import SimpleNamespace

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

app = FastAPI(title="Robot Motion Control API")

app.mount("/web", StaticFiles(directory="web"), name="web")
app.mount("/webfonts", StaticFiles(directory="webfonts"), name="webfonts")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,  # 쿠키나 자격 증명 허용
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # 허용할 HTTP 메소드
    allow_headers=["*"],  # 모든 헤더 허용
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 시리얼 포트 설정 (전역 변수)
ser = None
SERIAL_PORT = "/dev/ttyACM0"  # 필요시 환경변수로 변경
BAUDRATE = 115200

# 모션 데이터 저장 경로
MOTION_DATA_PATH = Path("motion_data.json")

class RobotMotion(BaseModel):
    arm: int  # 0xFD: 왼팔, 0xFE: 오른팔, 0xFF: 양팔, 0xFC (머리)
    head_tilt : int      # 0-40도 
    head_pan : int       # 10-170 도 
    shoulder_front: int  # 0-160도
    shoulder_side: int   # 0-100도
    elbow_front: int     # 0-115도
    elbow_side: int      # 0-160도 (80이 중앙)
    finger: int          # 0-6 (3진법)
    duration: float      # 초 단위

class MotionSequence(BaseModel):
    name: str
    motions: List[RobotMotion]

def init_serial():
    """시리얼 포트 초기화"""
    global ser
    try:
        if ser is None or not ser.is_open:
            ser = serial.Serial(
                port=SERIAL_PORT,
                baudrate=BAUDRATE,
                timeout=1
            )
            time.sleep(2)  # 포트 안정화
        return True
    except Exception as e:
        print(f"시리얼 포트 초기화 실패: {e}")
        return False

def create_command(motion: RobotMotion) -> bytearray:
    """모션 데이터를 바이트 명령으로 변환"""
    return bytearray([
        motion.arm,
        motion.shoulder_front,
        motion.shoulder_side,
        motion.elbow_front,
        motion.elbow_side,
        motion.finger
    ])


## 머리용
def create_head_command(motion: RobotMotion) -> bytearray:
    """모션 데이터를 바이트 명령으로 변환"""
    return bytearray([
        252,
        motion.head_tilt,
        motion.head_pan,
        0,
        0,
        0
    ])

@app.on_event("startup")
async def startup_event():
    """서버 시작시 시리얼 포트 초기화"""
    init_serial()
    
    # 모션 데이터 파일이 없으면 생성
    if not MOTION_DATA_PATH.exists():
        MOTION_DATA_PATH.write_text(json.dumps({}))

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료시 시리얼 포트 닫기"""
    global ser
    if ser and ser.is_open:
        ser.close()

@app.get("/")
async def root():
    return {"message": "Robot Motion Control API", "status": "running"}

@app.post("/motion/execute")
async def execute_motion(motion: RobotMotion):
    """단일 모션 실행"""
    if not init_serial():
        raise HTTPException(status_code=500, detail="시리얼 포트 연결 실패")
    
    try:
        command = create_command(motion)
        ser.write(command)
        print(command)
        if motion.head_tilt or motion.head_pan:
            print('head or tilt')
            time.sleep(0.1)
            print(motion)
            command = create_head_command(motion)
            print(command)
            ser.write(command)
        time.sleep(motion.duration)
        return {"status": "success", "message": "모션 실행 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모션 실행 실패: {str(e)}")

@app.get("/sequence/play/{name}")
async def play_sequence(name : str):
    """모션 시퀀스 실행"""
    if not init_serial():
        raise HTTPException(status_code=500, detail="시리얼 포트 연결 실패")
    
    data = await get_sequence(name) 
    print(data)
    sequence = dict_to_namespace(data)
    print(sequence)
    try:
        for motion in sequence.motions:
            command = create_command(motion)
            ser.write(command)
            time.sleep(motion.duration)
        
        return {"status": "success", "message": f"시퀀스 '{sequence.name}' 실행 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시퀀스 실행 실패: {str(e)}")


@app.post("/sequence/execute")
async def execute_sequence(sequence: MotionSequence):
    """모션 시퀀스 실행"""
    if not init_serial():
        raise HTTPException(status_code=500, detail="시리얼 포트 연결 실패")
    
    try:
        for motion in sequence.motions:
            command = create_command(motion)
            ser.write(command)
            time.sleep(motion.duration)
        
        return {"status": "success", "message": f"시퀀스 '{sequence.name}' 실행 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시퀀스 실행 실패: {str(e)}")

@app.post("/sequence/save")
async def save_sequence(sequence: MotionSequence):
    """모션 시퀀스 저장"""
    try:
        data = json.loads(MOTION_DATA_PATH.read_text())
        data[sequence.name] = sequence.dict()
        MOTION_DATA_PATH.write_text(json.dumps(data, indent=2))
        return {"status": "success", "message": f"시퀀스 '{sequence.name}' 저장 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"저장 실패: {str(e)}")

@app.get("/sequence/list")
async def list_sequences():
    """저장된 시퀀스 목록 조회"""
    try:
        data = json.loads(MOTION_DATA_PATH.read_text())
        return {"sequences": list(data.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"목록 조회 실패: {str(e)}")

@app.get("/sequence/{name}")
async def get_sequence(name: str):
    """특정 시퀀스 조회"""
    try:
        data = json.loads(MOTION_DATA_PATH.read_text())
        if name not in data:
            raise HTTPException(status_code=404, detail=f"{name} 시퀀스를 찾을 수 없습니다")
        return data[name]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"조회 실패: {str(e)}")

@app.delete("/sequence/{name}")
async def delete_sequence(name: str):
    """시퀀스 삭제"""
    try:
        data = json.loads(MOTION_DATA_PATH.read_text())
        if name not in data:
            raise HTTPException(status_code=404, detail="시퀀스를 찾을 수 없습니다")
        
        del data[name]
        MOTION_DATA_PATH.write_text(json.dumps(data, indent=2))
        return {"status": "success", "message": f"시퀀스 '{name}' 삭제 완료"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"삭제 실패: {str(e)}")

@app.get("/serial/status")
async def serial_status():
    """시리얼 연결 상태 확인"""
    global ser
    is_connected = ser is not None and ser.is_open
    return {
        "connected": is_connected,
        "port": SERIAL_PORT if is_connected else None,
        "baudrate": BAUDRATE if is_connected else None
    }

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

@app.get("/init_mcr")
async def init_mcr():
  global ser
  ser = serial.Serial(
    port='/dev/ttyACM0',           # 연결된 장치 포트명  -> 리눅스는 /dev/ttyACM0 또는 /dev/ttyUSBxxx
    baudrate=115200,       # 장치에 맞게 설정
    timeout=1              # 읽기 타임아웃(초)
  )

  # 6바이트 바이너리는 아래와 같은 구조임
  # 1st: 0xFD 는 왼팔 구동, 0xFE 는 오른팔 구동, 0xFF 는 양팔 동시 구동
  # 2nd: 어깨 관절 전방 구동 각도 (0도는 아래로 내린 각도, 90도는 앞으로 나란히 각도, 최대 160도 까지 가능함)
  # 3rd: 어깨 관절 측방 구동 각도 (0도는 몸에 밀착한 각도, 90도는 좌우로 나란히 각도, 최대 100도 까지 가능함)
  # 4th: 팔꿈치 관절 전방 구동 각도 (0도는 팔꿈치를 완전히 편 각도, 90도는 팔꿈치를 90도로 구부린 각도, 최대 115도 까지 가능함)
  # 5th: 팔꿈치 관절 측방 구동 각도 (80도는 팔꿈치 관절이 전방, 0도는 팔꿈치 관절이 몸쪽으로 80도 안으로 들어옴, 160도는 팔꿈치 관절이 몸 밖으로 80도 회전)
  # 6th: 손가락 구동 명령 (3진법) - 0: 대기, 1: 엄지 잡기, 2:엄지 놓기, 3: 4손가락 잡기, 6: 4손가락 펴기 

  return {"result": True }
  
@app.get("/motion")
async def motion(name = None):    
    global ser

    if name is not None:
        await play_sequence(name)
    else:
        data_lower = bytearray([0xFF, 0x20, 0x00, 0x00, 0x00, 0x00])
        data_front = bytearray([0xFF, 0x5A, 0x00, 0x00, 0x50, 0x00])
        data_hello = bytearray([0xFE, 0x5A, 0x00, 0x50, 0x50, 0x00])
        data_hello_swing_left = bytearray([0xFE, 0x5A, 0x00, 0x50, 0x30, 0x00])
        data_hello_swing_right = bytearray([0xFE, 0x5A, 0x00, 0x50, 0x60, 0x00])

        data_server = bytearray([0xFF, 0x30, 0x15, 0x40, 0x30, 0x00])
        data_server_left_high = bytearray([0xFD, 0x30, 0x15, 0x50, 0x30, 0x00])
        data_server_left_low = bytearray([0xFD, 0x30, 0x15, 0x30, 0x40, 0x00])
        data_server_right_high = bytearray([0xFE, 0x30, 0x15, 0x50, 0x30, 0x00])
        data_server_right_low = bytearray([0xFE, 0x30, 0x15, 0x30, 0x40, 0x00])

        data_middle = bytearray([0xFF, 0x3A, 0x20, 0x30, 0x30, 0x00])  
            # data_lower 는 팔 내리고 있기
        ser.write(data_lower)
        time.sleep(1.5)

        # 인사 제스처로 팔 들기
        ser.write(data_hello)
        time.sleep(1.5)

        #오른팔 왼쪽 스윙 
        ser.write(data_hello_swing_left)
        time.sleep(0.5)

        #오른팔 오른쪽 스윙 
        ser.write(data_hello_swing_right)
        time.sleep(0.8)

        #오른팔 왼쪽 스윙 
        ser.write(data_hello_swing_left)
        time.sleep(0.5)

        #오른팔 오른쪽 스윙 
        ser.write(data_hello_swing_right)
        time.sleep(1.8)

        #오른팔 왼쪽 스윙 
        ser.write(data_hello_swing_left)
        time.sleep(0.5)

        #오른팔 오른쪽 스윙 
        ser.write(data_hello_swing_right)
        time.sleep(0.8)

        #오른팔 왼쪽 스윙 
        ser.write(data_hello_swing_left)
        time.sleep(0.5)

        #오른팔 오른쪽 스윙 
        ser.write(data_hello_swing_right)
        time.sleep(0.8)

        # 팔 내리기
        ser.write(data_lower)
        time.sleep(1.5)

    return { "result" : True }

def move(data, delay=1.0):
  global ser
  ser.write(data)
  time.sleep(delay)

# 💋 한 손 키스 날리기 (오른손 입 근처 + 엄지, 4손가락 움직임)
@app.get("/kiss_one_hand")
def kiss_one_hand():
    move(bytearray([0xFE, 0x60, 0x30, 0x50, 0x50, 0x01]), 0.8)  # 엄지 잡기
    move(bytearray([0xFE, 0x60, 0x30, 0x50, 0x50, 0x03]), 0.8)  # 손가락 잡기
    move(bytearray([0xFE, 0x60, 0x30, 0x50, 0x50, 0x06]), 1.0)  # 손 펴기

# 📦 한 손으로 짐 나르기 (앞으로 뻗기)
@app.get("/carry_with_one_hand")
def carry_with_one_hand():
    move(bytearray([0xFE, 0x70, 0x10, 0x40, 0x50, 0x03]), 1.5)

# 📥 양손 박스 운반 (양손 앞으로)
@app.get("/carry_box_both_hands")
def carry_box_both_hands():
    move(bytearray([0xFF, 0x70, 0x10, 0x50, 0x50, 0x03]), 1.5)

# 💖 하트 만들기 (팔 올려서 머리 앞 하트 자세)
@app.get("/make_heart")
def make_heart():
    move(bytearray([0xFF, 0x90, 0x30, 0x60, 0x40, 0x06]), 2.0)

# 👋 얼굴 앞 손 흔들기 (한 손으로 반복 좌우 움직임)
@app.get("/wave_in_front_of_face")
def wave_in_front_of_face():
    base = bytearray([0xFE, 0x60, 0x10, 0x40, 0x40, 0x00])
    left = bytearray([0xFE, 0x60, 0x10, 0x40, 0x30, 0x00])
    right = bytearray([0xFE, 0x60, 0x10, 0x40, 0x50, 0x00])
    move(base)
    for _ in range(10):
        move(left, 0.4)
        move(right, 0.4)

# 🙆 머리 위 손 흔들기
@app.get("/wave_above_head")
def wave_above_head():
    base = bytearray([0xFE, 0xA0, 0x30, 0x50, 0x40, 0x00])
    left = bytearray([0xFE, 0xA0, 0x30, 0x50, 0x30, 0x00])
    right = bytearray([0xFE, 0xA0, 0x30, 0x50, 0x50, 0x00])
    move(base)
    for _ in range(5):
        move(left, 1)
        move(right, 1)

# 🙋‍♀️ 왼손 들기
@app.get("/raise_left_hand")
def raise_left_hand():
    move(bytearray([0xFD, 0x90, 0x00, 0x50, 0x30, 0x00]), 1.5)

# 🙋‍♂️ 오른손 들기
@app.get("/raise_right_hand")
def raise_right_hand():
    move(bytearray([0xFE, 0x90, 0x00, 0x50, 0x30, 0x00]), 1.5)

# ❌ 팔 X자 모양 만들기 (팔 교차)
@app.get("/make_x_pose")
def make_x_pose():
    move(bytearray([0xFF, 0x80, 0x40, 0x50, 0x20, 0x00]), 2.0)

# 👏 박수치기 (양손 가까이)
@app.get("/clap")
def clap():
    close = bytearray([0xFF, 0x60, 0x10, 0x60, 0x30, 0x03])
    open_ = bytearray([0xFF, 0x60, 0x10, 0x60, 0x60, 0x06])
    for _ in range(5):
        move(close, 1)
        move(open_, 1)

# 👐 팔 벌리기
@app.get("/spread_arms")
def spread_arms():
    move(bytearray([0xFF, 0x40, 0x80, 0x30, 0x30, 0x00]), 1.5)

# 🙌 양손 위로 들기
@app.get("/raise_both_arms")
def raise_both_arms():
    move(bytearray([0xFF, 0xA0, 0x10, 0x50, 0x40, 0x00]), 2.0)

# 🧍 기본자세 (팔 내림)
@app.get("/default_pose")
def default_pose():
    move(bytearray([0xFF, 0x20, 0x00, 0x00, 0x00, 0x00]), 1.5)

# 🔀 왼팔 펴고 오른팔 위로
@app.get("/left_arm_out_right_up")
def left_arm_out_right_up(ser):
    left = bytearray([0xFD, 0x50, 0x70, 0x30, 0x40, 0x00])
    right = bytearray([0xFE, 0xA0, 0x10, 0x50, 0x30, 0x00])
    move(left, 0.5)
    move(right, 1.5)

# 🔁 오른팔 펴고 왼팔 위로
@app.get("/right_arm_out_left_up")
def right_arm_out_left_up():
    right = bytearray([0xFE, 0x50, 0x70, 0x30, 0x40, 0x00])
    left = bytearray([0xFD, 0xA0, 0x10, 0x50, 0x30, 0x00])
    move(right, 0.5)
    move(left, 1.5)

@app.get("/stop")
def stop():
    move(bytearray([0xFF, 0x00, 0x00, 0x00, 0x0, 0x00]), 1)  # 엄지 잡기

# WebSocket 연결 관리 (1 client 정도)
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    async def send_json_to_one(self, websocket: WebSocket, data):
        await websocket.send_json(data)
    async def broadcast(self, data):
        for ws in self.active_connections:
            await ws.send_json(data)

manager = ConnectionManager()

# 클라이언트와 통신: 즉각 제어 명령 수신
@app.websocket("/ws/control")
async def websocket_control(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            msg = await ws.receive_json()
            # 기대 포맷: { "frame": [b0, b1, b2, b3, b4, b5] }
            if "frame" in msg and isinstance(msg["frame"], list):
                frame = msg["frame"]
                # 시리얼로 즉시 전송
                if ser and ser.is_open:
                    ba = bytearray(frame)
                    ser.write(ba)
                # 옵션: 클라이언트에게 수신 확인 응답
                await manager.send_json_to_one(ws, {"status": "ok", "sent": frame})
    except WebSocketDisconnect:
        manager.disconnect(ws)

# REST API로 모션 저장 / 로드 / 플레이 기능 제공 (선택)
from fastapi import Body

@app.post("/api/save_motion")
async def save_motion(frames: list[list[int]] = Body(...)):
    with open("motion_data.json", "w") as f:
        json.dump(frames, f, indent=2)
    return {"status": "saved", "count": len(frames)}

@app.post("/api/play_motion")
async def play_motion():
    if not os.path.exists("motion_data.json"):
        return {"error": "no saved motion"}
    with open("motion_data.json", "r") as f:
        frames = json.load(f)
    for frame in frames:
        if ser and ser.is_open:
            ser.write(bytearray(frame))
            # 여기 딜레이 정보가 있다면 프론트에서 같이 보내야 함
            time.sleep(0.1)
    return {"status": "played", "frames": len(frames)}

print("Loading Complete","CPU")
