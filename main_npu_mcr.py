from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
import time
import time as t
import numpy as np
import subprocess
import collections
from pydub import AudioSegment
from serverinfo import si
import asyncio
import logging
from aiortc import MediaStreamTrack
import time
import cv2
from openvino import Core
from fastapi.staticfiles import StaticFiles
from queue import Queue
from ultralytics import YOLO
import threading
from fastapi.middleware.cors import CORSMiddleware
import hashlib
import asyncio
import pyrealsense2 as rs


ov = Core()

FACE_DETECTION_MODEL_XML = "./models/face-detection-retail-0005/FP16-INT8/face-detection-retail-0005.xml"
AGE_GENDER_MODEL_XML = "./models/age-gender-recognition-retail-0013/FP16-INT8/age-gender-recognition-retail-0013.xml"
EMOTION_MODEL_XML = "./models/emotions-recognition-retail-0003/FP16-INT8/emotions-recognition-retail-0003.xml"

DEVICE = "NPU"
# 생명체로 간주할 클래스명 (클래스 이름은 모델에 따라 다를 수 있음!)
LIVING_CLASSES = {'person', 'cat', 'dog', 'bird', 'teddy bear', 'cow', 'sheep', 'horse'}
EMOTIONS = ['neutral', 'happy', 'sad', 'surprise', 'anger']

# 얼굴 탐지 모델
face_det_model = ov.read_model(model=FACE_DETECTION_MODEL_XML)
face_det_compiled_model = ov.compile_model(model=face_det_model, device_name=DEVICE)
face_det_input_layer = face_det_compiled_model.input(0)
face_det_output_layer = face_det_compiled_model.output(0)
face_det_height, face_det_width = list(face_det_input_layer.shape)[2:]

# 나이/성별 모델
age_gender_model = ov.read_model(model=AGE_GENDER_MODEL_XML)
age_gender_compiled_model = ov.compile_model(model=age_gender_model, device_name=DEVICE)
age_gender_input_layer = age_gender_compiled_model.input(0)
age_output_layer = age_gender_compiled_model.output("age_conv3")
gender_output_layer = age_gender_compiled_model.output("prob")
age_gender_height, age_gender_width = list(age_gender_input_layer.shape)[2:]

# 감정 모델
emotion_model = ov.read_model(model=EMOTION_MODEL_XML)
emotion_compiled_model = ov.compile_model(model=emotion_model, device_name=DEVICE)
emotion_input_layer = emotion_compiled_model.input(0)
emotion_output_layer = emotion_compiled_model.output(0)
emotion_height, emotion_width = list(emotion_input_layer.shape)[2:]

det_model = YOLO('./models/yolo11s-seg_int8_openvino_model')
det_model("capture.jpg", device='intel:cpu', imgsz=640) # error 방지용
class_names = det_model.names



def visualize_face(frame,face_det_results):
    h, w, _ = frame.shape

    for detection in face_det_results[0][0]:  # OpenVINO 출력 형식에 맞게 인덱싱
        confidence = detection[2]  # 신뢰도
        if confidence > 0.5:  # 신뢰도 임계값
            xmin = int(detection[3] * w)
            ymin = int(detection[4] * h)
            xmax = int(detection[5] * w)
            ymax = int(detection[6] * h)
            
            # 얼굴 이미지 자르기 (유효한 범위 내에서)
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            face_img = frame[ymin:ymax, xmin:xmax]
            
            if face_img.size > 0:
                # 나이/성별 모델 추론
                resized_age_gender = cv2.resize(face_img, (age_gender_width, age_gender_height))
                # 입력 형태: [1, 3, H, W]
                ag_input_tensor = np.expand_dims(resized_age_gender.transpose((2, 0, 1)), 0)
                ag_results = age_gender_compiled_model(ag_input_tensor)
                
                # 결과 파싱
                age_pred = int(ag_results[age_output_layer].reshape(1)[0] * 100) # OpenVINO age-gender 모델은 나이값을 100으로 나눈 값으로 출력.
                gender_prob = ag_results[gender_output_layer].reshape(-1) # [female_prob, male_prob] 형태로 변환
                
                # OpenVINO documentation: prob output across 2 type classes [0 - female, 1 - male].
                gender_idx = np.argmax(gender_prob)
                gender = "W" if gender_idx == 0 else "M"
                
                # 감정 모델 추론
                resized_emotion = cv2.resize(face_img, (emotion_width, emotion_height))
                emotion_input_tensor = np.expand_dims(resized_emotion.transpose((2, 0, 1)), 0)
                emotion_results = emotion_compiled_model(emotion_input_tensor)
                
                # 결과 파싱
                emotion_prob = emotion_results[emotion_output_layer].reshape(-1)
                emotion = EMOTIONS[np.argmax(emotion_prob)]
                
                # 결과값으로 박스 및 텍스트 그리기
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
                text = f"{gender}, {age_pred}y, {emotion}"
                cv2.putText(frame, text, (xmin, ymax - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
      
    return frame      


def visualize_segmentation(frame, masks, boxes, classes, scores, depths, class_names, alpha=0.5):
    overlay = frame.copy()
    for mask, box, cls_idx, score, depth in zip(masks, boxes, classes, scores, depths):
        class_name = class_names[cls_idx]

        is_living = class_name in LIVING_CLASSES
        color = (0, 0, 255) if is_living else (0, 255, 0)  # 빨강 vs 초록

        """
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        position = ""

        if class_name == "person":
            rgb_color = (0, 0, 255)
            cnt_live += 1
            lastTime = time.time()

            # 중심 좌표 계산
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # 위치 계산 (grid 3x3)
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

            position = row + col  # ex: "TC", "BR", etc.

        else:
            rgb_color = (255, 255, 0)
            cnt_object += 1        

        boxes.append({
            'class': cls_name,
            'confidence': round(conf, 2),
            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
            'position': position  # 위치 정보 추가
        }) 
        """

        # 마스크 적용
        overlay[mask == 1] = (overlay[mask == 1] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name}:{score:.2f} | {depth:.2f}m"
        cv2.putText(overlay, label, (x1, max(15, y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return overlay

def get_mask_depths(masks, depth_frame, low_percentile=5):
    depths = []
    depth_image = np.asanyarray(depth_frame.get_data())
    for mask in masks:
        if mask.sum() == 0:
            depths.append(0.0)
            continue
        depth_values = depth_image[mask == 1]
        valid = depth_values[depth_values > 0]
        if len(valid) > 0:
            low_thresh = np.percentile(valid, low_percentile)  # 예: 5번째 백분위수
            filtered = valid[valid >= low_thresh]
            if len(filtered) > 0:
                closest_depth_m = np.min(filtered) / 1000.0
            else:
                closest_depth_m = np.min(valid) / 1000.0
        else:
            closest_depth_m = 0.0
        depths.append(closest_depth_m)
    return depths


ser = None

def getHash(text):
  hash_func = hashlib.new('md5')
  hash_func.update(text.encode('utf-8'))
  return hash_func.hexdigest()

hL = None
hR = None

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

_IP = "127.0.0.1" #si.getIP()
_PORT = int(open("port.txt", 'r').read())

state = { "charge" : 0, "temp" : 0, "voltage" : 0, "cnt_live" : 0, "cnt_object" : 0,  "boxes" : []}

processed_frame_queue = Queue(maxsize=5)

cnt_live = 0
cnt_object = 0
lastTime = 0
cnt_image = 0

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

def processing_thread():
    global cnt_live, cnt_object, lastTime, state, cnt_image

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    print("============= processing....")  

    while True:
        cnt_live = 0
        cnt_object = 0
        boxes = []

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())        

        if cnt_image % 100 == 0:
            cv2.imwrite("capture.jpg", frame) #cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cnt_image = cnt_image + 1

        start_time = time.time()

        results = det_model(frame, device="intel:npu", imgsz=640, verbose=False, conf=0.3)  # 작을수록 빠름
        res = results[0]

        if hasattr(res, 'masks') and res.masks is not None:
            masks = res.masks.data.cpu().numpy().astype(np.uint8)
        else:
            masks = []

        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy()

        # 시각화
        out = visualize_segmentation(frame, masks, boxes, classes, scores, get_mask_depths(masks, depth_frame), class_names)

        # 얼굴 감지 모델 추론
        resized_frame = cv2.resize(frame, (face_det_width, face_det_height))
        input_tensor = np.expand_dims(resized_frame.transpose((2, 0, 1)), 0)
        face_det_results = face_det_compiled_model(input_tensor)[face_det_output_layer]

        out = visualize_face(out, face_det_results)

        # FPS 계산 및 표시
        curr_time = time.time()
        fps = 1.0 / (curr_time - start_time)
        cv2.putText(out, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        state["cnt_live"] = cnt_live
        state["cnt_object"] = cnt_object
        state['boxes'] = boxes
        
        if processed_frame_queue.full():
            processed_frame_queue.get()  # 가장 오래된 프레임 제거   

        processed_frame_queue.put(out)



@app.get("/")
def main():
  return { "result" : True, "data" : "AI-CPU-V2", "ip" : _IP, "port" : _PORT }      

# Async function to receive video frames and put them in the queue
async def recv_camera_stream(track: MediaStreamTrack):
    while True:
        frame = await track.recv()
        img = frame.to_ndarray(format="bgr24")

        if frame_queue.full():
            frame_queue.get()  # 가장 오래된 프레임 제거        

        frame_queue.put(img)

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

@app.get("/video_feed")
async def video_feed():
  def generate():
    while True:
        if not processed_frame_queue.empty():
            output = processed_frame_queue.get()
            _, img_bytes = cv2.imencode('.jpg', output)
            frame = img_bytes.tobytes()

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            time.sleep(0.01)

  return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

lastCmd = {} 

def toFloat(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return value

@app.get("/monitor")
def monitor():
  return si.getAll()

@app.get("/v1/tts", response_class=FileResponse, summary="입력한 문장으로 부터 음성을 생성합니다.")
def tts(text = "", voice=31, lang='ko', static=0, isPlay=0):
    #org_text = parse.quote(text, safe='', encoding="cp949")
    start = t.time()
    print(text, static)
    filename = getHash(text)

    #phoneme_ids = text_to_sequence(text, conf_tts.data.text_cleaners)
    phoneme_ids = text_to_sequence(text, [f'canvers_{lang}_cleaners'])
    text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)

    inputs = {
        "input": text,
        "input_lengths":  np.array([text.shape[1]], dtype=np.int64),
        "scales": np.array([0.667, 1.0, 0.8], dtype=np.float16),
        "sid" : np.array([int(voice)], dtype=np.int64) if voice is not None else None
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
      audio = AudioSegment.from_wav(f"output/{filename}.wav")
      # Set specific audio parameters for compatibility
      audio = audio.set_frame_rate(22050)  # Standard sample rate
      audio = audio.set_sample_width(2)
      audio = audio.set_channels(1)
      #wav_file_path = audiofile_path.replace('.mp3', '.wav')
      audio.export(f"output/{filename}.wav", format='wav', codec="pcm_s16le" )#parameters=["-ar", "44100"])
      if int(isPlay) > 0 :
        subprocess.Popen(["play", f"output/{filename}.wav"]) # async
        #playsound(f"output/{filename}.wav")

      return f"output/{filename}.wav"
    
    # 31 korean
@app.get("/v2/tts", response_class=FileResponse, summary="음성 생성 후 로봇에서 재생")
def tts(text = "", voice=6, lang='ko', static=0, isPlay=0):
    #org_text = parse.quote(text, safe='', encoding="cp949")
    start = t.time()
    print(text, static)
    filename = getHash(text)

    #phoneme_ids = text_to_sequence(text, conf_tts.data.text_cleaners)
    phoneme_ids = text_to_sequence(text, [f'canvers_{lang}_cleaners'])
    text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)

    inputs = {
        "input": text,
        "input_lengths":  np.array([text.shape[1]], dtype=np.int64),
        "scales": np.array([0.667, 1.0, 0.8], dtype=np.float16),
        "sid" : np.array([int(voice)], dtype=np.int64) if voice is not None else None
    }

    start_time = t.time()
    result = pipe_tts(inputs)
    print(f"Inference time: {t.time() - start_time:.4f} seconds")

    audio = list(result.values())[0].squeeze((0, 1))  

    print(t.time() - start)
    write(data=audio, rate=conf_tts.data.sampling_rate, filename=f"output/{filename}.wav")

    # 파일 열고 전송
    #with open(f"output/{filename}.wav", "rb") as f:
    #    files = {"audio_file": (f"{filename}.wav", f, "audio/wav")}
    #    response = requests.post("http://192.168.12.117:59521/audio", files=files)

    subprocess.Popen(["play", f"output/{filename}.wav"]) # async

    return f"output/{filename}.wav"


import httpx  # httpx를 사용하여 비동기 HTTP 요청을 처리합니다.

# 원본 비디오 스트림 URL
SOURCE_VIDEO_URL = "http://127.0.0.1:59511/video_feed"
#SOURCE_VIDEO_URL = "http://192.168.12.117:59511/video_feed"

async def proxy_video_stream():
    """원본 서버에서 비디오 스트림을 받아서 다시 스트리밍"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            async with client.stream("GET", SOURCE_VIDEO_URL) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    yield chunk
        except httpx.RequestError as e:
            print(f"비디오 스트림 연결 오류: {e}")
            # 연결 오류 시 빈 프레임이나 오류 메시지를 반환할 수 있습니다
            yield b""
        except Exception as e:
            print(f"예상치 못한 오류: {e}")
            yield b""

@app.get("/video_feed2")
async def video_feed():
    """비디오 스트림을 프록시하여 제공"""
    return StreamingResponse(
        proxy_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
# 서버 B에서 비디오 스트리밍 시작

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
                            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame)
                            #print(f"✓ 프레임 추출 성공: {frame.shape}")
                        else:
                            print("프레임 디코딩 실패")
                            
                    except Exception as e:
                        print(f"프레임 디코딩 오류: {e}")
        
        # 처리된 부분 제거
        remaining_buffer = remaining_buffer[next_boundary_pos:]
    
    return frames, remaining_buffer


@app.get("/start_collection")
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
        except frame_queue.Empty:
            break
    
    print(f"큐 초기화: {cleared}개 프레임 제거")
    
    is_collecting = True
    threading.Thread(target=processing_thread, daemon=True).start()

    return {"message": "프레임 수집을 시작했습니다"}

print("Loading Complete","NPU")
