from fastapi import FastAPI
from fastapi.responses import JSONResponse
import httpx
import asyncio
import queue
import threading
import numpy as np
import cv2
from PIL import Image
import io
import re

app = FastAPI()

# 원본 비디오 스트림 URL
SOURCE_VIDEO_URL = "http://127.0.0.1:59521/video_feed"

# 프레임 큐 생성 (최대 5개 프레임)
frame_queue = queue.Queue(maxsize=5)

# 프레임 수집 상태
is_collecting = False
collection_task = None

def parse_mjpeg_boundary(buffer):
    """boundary=frame 형식의 MJPEG 스트림 파싱"""
    frames = []
    remaining_buffer = buffer
    
    # boundary 패턴들
    boundary_patterns = [
        b'--frame\r\n',
        b'--frame\n', 
        b'\r\n--frame\r\n',
        b'\n--frame\n'
    ]
    
    while True:
        # boundary 찾기
        boundary_pos = -1
        next_boundary_pos = -1
        
        for pattern in boundary_patterns:
            pos = remaining_buffer.find(pattern)
            if pos != -1:
                boundary_pos = pos
                # 다음 boundary 찾기
                next_pos = remaining_buffer.find(pattern, pos + len(pattern))
                if next_pos != -1:
                    next_boundary_pos = next_pos
                    break
        
        if boundary_pos == -1 or next_boundary_pos == -1:
            break
            
        # 현재 프레임 데이터 추출
        frame_data = remaining_buffer[boundary_pos:next_boundary_pos]
        
        # Content-Type과 Content-Length 헤더 건너뛰기
        header_end = frame_data.find(b'\r\n\r\n')
        if header_end == -1:
            header_end = frame_data.find(b'\n\n')
        
        if header_end != -1:
            jpeg_data = frame_data[header_end + 4:]  # 헤더 이후 데이터
            
            # JPEG 시작 마커 확인
            jpeg_start = jpeg_data.find(b'\xff\xd8')
            if jpeg_start != -1:
                jpeg_data = jpeg_data[jpeg_start:]
                
                # JPEG 끝 마커 찾기
                jpeg_end = jpeg_data.find(b'\xff\xd9')
                if jpeg_end != -1:
                    complete_jpeg = jpeg_data[:jpeg_end + 2]
                    
                    try:
                        # OpenCV로 디코딩
                        nparr = np.frombuffer(complete_jpeg, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            # BGR을 RGB로 변환
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame)
                            print(f"✓ 프레임 추출 성공: {frame.shape}")
                        else:
                            print("프레임 디코딩 실패")
                            
                    except Exception as e:
                        print(f"프레임 디코딩 오류: {e}")
        
        # 처리된 부분 제거
        remaining_buffer = remaining_buffer[next_boundary_pos:]
    
    return frames, remaining_buffer

async def collect_frames():
    """원본 서버에서 프레임을 수집하여 큐에 저장"""
    global is_collecting
    
    buffer = b""
    frame_count = 0
    chunk_count = 0
    
    try:
        print("비디오 스트림 연결 시작...")
        
        # 더 긴 타임아웃 설정
        timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
        
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            print(f"연결 시도: {SOURCE_VIDEO_URL}")
            
            async with client.stream("GET", SOURCE_VIDEO_URL) as response:
                print(f"응답 상태: {response.status_code}")
                print(f"응답 헤더: {dict(response.headers)}")
                
                if response.status_code != 200:
                    print(f"HTTP 오류: {response.status_code}")
                    return
                
                response.raise_for_status()
                
                print("스트림 읽기 시작...")
                
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    if not is_collecting:
                        print("수집 중지 요청")
                        break
                    
                    chunk_count += 1
                    buffer += chunk
                    
                    # 5초마다 상태 출력
                    if chunk_count % 100 == 0:
                        print(f"청크 {chunk_count}개 수신, 버퍼 크기: {len(buffer)} bytes")
                        
                        # 버퍼에서 boundary 패턴 확인 (디버깅용)
                        if b'--frame' in buffer:
                            boundary_count = buffer.count(b'--frame')
                            print(f"발견된 boundary 개수: {boundary_count}")
                        
                        # 버퍼의 처음 200바이트 출력 (디버깅용)
                        if len(buffer) > 200:
                            sample = buffer[:200]
                            print(f"버퍼 샘플: {sample[:100]}")
                            if b'Content-Type' in sample:
                                print("Content-Type 헤더 발견")
                    
                    # 버퍼가 너무 커지지 않도록 제한
                    if len(buffer) > 2 * 1024 * 1024:  # 2MB 제한
                        print("버퍼 크기 제한, 일부 제거")
                        # 버퍼의 앞쪽 절반 제거
                        buffer = buffer[len(buffer)//2:]
                    
                    # 최소 버퍼 크기에 도달했을 때만 파싱 시도
                    if len(buffer) > 1000:  # 최소 1KB
                        try:
                            frames, buffer = parse_mjpeg_boundary(buffer)
                            
                            for frame in frames:
                                frame_count += 1
                                
                                # 큐가 꽉 찬 경우 오래된 프레임 제거
                                while frame_queue.full():
                                    try:
                                        dropped = frame_queue.get_nowait()
                                        print("오래된 프레임 드랍")
                                    except queue.Empty:
                                        break
                                
                                try:
                                    frame_queue.put_nowait(frame)
                                    print(f"✓ 프레임 #{frame_count} 큐에 추가 (크기: {frame_queue.qsize()}/{frame_queue.maxsize})")
                                except queue.Full:
                                    print("큐 풀 - 프레임 스킵")
                        
                        except Exception as e:
                            print(f"프레임 파싱 오류: {e}")
                            # 파싱 오류시 버퍼 일부 제거
                            if len(buffer) > 5000:
                                buffer = buffer[1000:]
                    
                    # CPU 사용률 조절
                    if chunk_count % 50 == 0:
                        await asyncio.sleep(0.01)
                        
                print("스트림 종료")
                    
    except httpx.TimeoutException as e:
        print(f"타임아웃 오류: {e}")
    except httpx.ConnectError as e:
        print(f"연결 오류: {e}")
    except httpx.RequestError as e:
        print(f"요청 오류: {e}")
    except Exception as e:
        print(f"예상치 못한 오류: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        is_collecting = False
        print(f"프레임 수집 종료. 총 {frame_count}개 프레임 처리됨")

@app.post("/start_collection")
async def start_frame_collection():
    """프레임 수집 시작"""
    global is_collecting, collection_task
    
    if is_collecting:
        return {"message": "이미 프레임 수집이 진행 중입니다"}
    
    # 기존 큐 비우기
    cleared = 0
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
            cleared += 1
        except queue.Empty:
            break
    
    print(f"큐 초기화: {cleared}개 프레임 제거")
    
    is_collecting = True
    collection_task = asyncio.create_task(collect_frames())
    
    return {"message": "프레임 수집을 시작했습니다"}

@app.post("/stop_collection")
async def stop_frame_collection():
    """프레임 수집 중지"""
    global is_collecting, collection_task
    
    print("프레임 수집 중지 요청")
    is_collecting = False
    
    if collection_task and not collection_task.done():
        collection_task.cancel()
        try:
            await asyncio.wait_for(collection_task, timeout=3.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            print("수집 작업 강제 종료됨")
    
    return {"message": "프레임 수집을 중지했습니다"}

@app.get("/queue_status")
async def get_queue_status():
    """큐 상태 확인"""
    return {
        "queue_size": frame_queue.qsize(),
        "queue_full": frame_queue.full(),
        "queue_empty": frame_queue.empty(),
        "is_collecting": is_collecting,
        "max_size": frame_queue.maxsize
    }

@app.get("/get_frame")
async def get_frame():
    """큐에서 프레임 가져오기 (테스트용)"""
    try:
        if frame_queue.empty():
            return {"message": "큐가 비어있습니다"}
        
        frame = frame_queue.get_nowait()
        return {
            "message": "프레임을 가져왔습니다",
            "frame_shape": frame.shape if frame is not None else None,
            "frame_dtype": str(frame.dtype) if frame is not None else None,
            "remaining_frames": frame_queue.qsize()
        }
    except queue.Empty:
        return {"message": "큐가 비어있습니다"}

@app.get("/clear_queue")
async def clear_queue():
    """큐 비우기"""
    cleared_count = 0
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
            cleared_count += 1
        except queue.Empty:
            break
    
    return {"message": f"{cleared_count}개의 프레임을 제거했습니다"}

@app.get("/test_connection")
async def test_connection():
    """원본 서버 연결 테스트"""
    try:
        print(f"연결 테스트: {SOURCE_VIDEO_URL}")
        
        timeout = httpx.Timeout(10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            # 먼저 HEAD 요청으로 테스트
            try:
                head_response = await client.head(SOURCE_VIDEO_URL)
                print(f"HEAD 응답: {head_response.status_code}")
            except:
                print("HEAD 요청 실패, GET으로 시도")
            
            # 실제 스트림 일부 읽기
            async with client.stream("GET", SOURCE_VIDEO_URL) as response:
                print(f"GET 스트림 응답: {response.status_code}")
                
                # 처음 몇 바이트만 읽어보기
                first_chunk = None
                chunk_count = 0
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    chunk_count += 1
                    if chunk_count == 1:
                        first_chunk = chunk
                    if chunk_count >= 3:  # 3개 청크만 테스트
                        break
                
                return {
                    "status": "success",
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "content_type": response.headers.get('content-type'),
                    "first_chunk_size": len(first_chunk) if first_chunk else 0,
                    "first_chunk_preview": first_chunk[:100].hex() if first_chunk else "",
                    "chunks_received": chunk_count
                }
    except Exception as e:
        print(f"연결 테스트 오류: {e}")
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

@app.get("/")
async def root():
    """서버 상태 확인용 엔드포인트"""
    return {"message": "프레임 수집 서버가 실행 중입니다"}

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "frame_queue_size": frame_queue.qsize(),
        "is_collecting": is_collecting
    }

# 프레임을 가져오는 함수들
def get_frame_from_queue():
    """큐에서 프레임을 가져오는 함수 (블로킹)"""
    try:
        return frame_queue.get(timeout=1.0)
    except queue.Empty:
        return None

def get_frame_from_queue_nowait():
    """큐에서 프레임을 가져오는 함수 (논블로킹)"""
    try:
        return frame_queue.get_nowait()
    except queue.Empty:
        return None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)