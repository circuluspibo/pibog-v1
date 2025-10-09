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
    arm: int  # 0xFD: 왼팔, 0xFE: 오른팔, 0xFF: 양팔
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
        time.sleep(motion.duration)
        return {"status": "success", "message": "모션 실행 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모션 실행 실패: {str(e)}")

@app.get("/sequence/play/{name}")
async def execute_sequence(name : str):
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)