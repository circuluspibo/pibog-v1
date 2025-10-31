from fastapi.middleware.cors import CORSMiddleware
import librosa
from fastapi import FastAPI, File, UploadFile
from transformers import AutoTokenizer
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download, hf_hub_download
import time as t
from transformers import AutoTokenizer
from pydantic import BaseModel, Field
import numpy as np
import openvino_genai as ov_genai
from scipy.io.wavfile import write
from text import text_to_sequence
from serverinfo import si
import logging
from requests import get
from queue import Queue
from ultralytics import YOLO, FastSAM
import openvino as ov
from threading import Event, Thread
from transformers import AutoTokenizer
from pydantic import BaseModel, Field
from iterator import IterableStreamer
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from openvino import Tensor
from pathlib import Path
from openvino_genai import GenerationConfig

#optimum-cli export openvino --weight-format int4 --task text-generation-with-past --model growdle/HyperCLOVAX-SEED-Text-Instruct-1.5B ./CLOVAX-1.5B-ov-int4
#kakaocorp/kanana-1.5-2.1b-instruct-2505
#https://github.com/Unitree-Go2-Robot/go2_robot


# 너는 파이온이라는 휴머노이드 로봇으로 사람들을 지키기 위해 태어났어. 대화체로 사람처럼 대답하되, 다음과 같은 동작이 가능하니, 적절한 동작을 먼저 출력하고 대답을 이야기 해줘. - clamp, highFive, shakeHands_1, blowKiss, hug, hightWave, lowWave, ultramanRay, bothHandsUp, singleHandsUp, Refuse

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

_IP = "127.0.0.1" #si.getIP()
_PORT = int(open("port.txt", 'r').read())

app = FastAPI()

# 모든 도메인 허용 (allow_origins에 '*' 설정)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,  # 쿠키나 자격 증명 허용
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # 허용할 HTTP 메소드
    allow_headers=["*"],  # 모든 헤더 허용
)

class Param (BaseModel):
  text : str
  hash : str = Field(default='')
  voice : str = Field(default='main') 
  lang : str = Field(default='ko')
  type : str = Field(default='mp3')
  pitch : str = Field(default='medium')
  rate : str = Field(default='medium')
  volume : str = Field(default='medium')

"""
너는 파이온이라는 휴머노이드 로봇으로 사람들을 지키기 위해 태어났어.  
 
다음과 같은 동작중 하나를 선택할 수 있어
clamp, highFive, shakeHands_1, blowKiss, hug, hightWave, lowWave, ultramanRay, bothHandsUp, singleHandsUp, Refuse

응답시 가장 적절한 동작을 하나만 선택하고, 대화체로 마크업 없이 사람처럼 대답 해줘.
예시 : [동작] 응답어
"""
_SYSTEM = "당신은 서큘러스에서 만든 파이봇 이라고 하는 MIT 박사 수준의 로봇 인공지능 입니다. 젊은 톤의 대화체로 입력된 언어로 사람 같이 짧게 응답하세요."

def read_image(path: str) -> Tensor:
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic)
    return Tensor(image_data)

def read_images(path: str) -> list[Tensor]:
    entry = Path(path)
    if entry.is_dir():
        return [read_image(str(file)) for file in sorted(entry.iterdir())]
    return [read_image(path)]

class Chat(BaseModel):
  prompt : str = ''
  lang : str = 'auto'
  type : str =  _SYSTEM #" "당신은 데이비드라고 하는 10살 남자아이 성향의 유쾌하고 즐거운 인공지능입니다. 이모티콘도 잘 활용해서 젊은 말투로 대답하세요."
  rag :  str = ''  
  temp : float = 0.5
  top_p : float = 0.92
  top_k : int = 50
  max : int = 256 #16384

model_txt = snapshot_download(repo_id='Echo9Zulu/gemma-3-4b-it-qat-int4_asym-ov') # circulus/gemma-3-4b-it-ov-awq-sym
model_stt = snapshot_download(repo_id='circulus/whisper-large-v3-turbo-ov')

token_txt = AutoTokenizer.from_pretrained(model_txt)
pipe_txt = ov_genai.VLMPipeline(model_txt, device="GPU", config={"PERFORMANCE_HINT": "LATENCY"})
pipe_stt = ov_genai.WhisperPipeline(model_stt,device="GPU", config={"PERFORMANCE_HINT": "LATENCY"})

# for genai
async def process_stream(streamer, isStream=True, isPlay=0, lang='en'):
  cnt = 0
  sentence = ""
  print("streaming start...")
  for new_token in streamer:
    if "assistant" in new_token:
      cnt += 1
      if cnt == 1:
          continue  # Skip the fi'../models/stt'rst one (don't yield)
      elif cnt == 2:
          print("Forcing exit...")
          break  # Stop stream (don't yield)
        
    if isStream:
      #print(new_token)덷
      yield new_token
      #await asyncio.sleep(0) 
    elif "." in new_token or "\n" in new_token:
      sentence = sentence + new_token
      if len(sentence) > 3:      

        sentence = sentence.strip()

        if int(isPlay) > 0:
          get("http://127.0.0.1:59531/v2/tts", params={"text": sentence, "lang" : lang, "voice" : 31})         #korean

        print(sentence)
        yield sentence
        #await asyncio.sleep(0.01) # thread 처리
        sentence = ""
    else:
      sentence = sentence + new_token
  if len(sentence) > 3:
    yield sentence

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],#origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def main():
  return { "result" : True, "data" : "AI-CPU-V2", "ip" : _IP, "port" : _PORT }           


@app.get("/monitor")
def monitor():
  return si.getAll()

@app.get("/v1/txt2chat", summary="문장 기반의 chatgpt 스타일 구현")
def txt2chat(prompt : str ,system = _SYSTEM, isPlay = 0, lang='en'): # gen or med
  streamer = IterableStreamer(pipe_txt.get_tokenizer())

  messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": prompt}
  ] 
  """
  prompt = token_txt.apply_chat_template(
    messages,
    tokenize=False,
    enable_thinking=True,
    add_generation_prompt=True
  )
"""
  pipe_txt.start_chat(system_message=system)

  print(prompt)

  config = GenerationConfig(
      max_new_tokens=256,
      temperature=0.5,
      beam_size=1,
      do_sample=False, #fast for beam-search
      speculative_decoding=True,
      repetition_penalty=1.1,
      #top_k=50,
      #top_p=0.9,
  )

  generate_kwargs = dict(
      prompt = prompt,
      config = config,
      streamer=streamer, # !do_sample || top_k > 0
  )

  """
  generate_kwargs = dict(
      inputs = prompt,
      max_new_tokens= 256,
      temperature= 0.5,
      #do_sample=True,
      repetition_penalty=1.1,
      top_k=50,
      top_p=0.9,
      streamer=streamer, # !do_sample || top_k > 0
  )
  """

  t1 = Thread(target=pipe_txt.generate, kwargs=generate_kwargs)
  t1.start()

  out = process_stream(streamer, False, isPlay,lang)
  return StreamingResponse(out, media_type='text/event-stream')


@app.get("/v2/img2chat", summary="문장 기반의 chatgpt 스타일 구현")
@app.post("/v2/img2chat", summary="문장 기반의 chatgpt 스타일 구현")
def img2chat2(prompt = "" ,system = _SYSTEM, isPlay = 0, lang='en'): # gen or med
  streamer = IterableStreamer(pipe_txt.get_tokenizer())


  messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": prompt}
  ] 
  """
  prompt = token_txt.apply_chat_template(
    messages,
    tokenize=False,
    enable_thinking=True,
    add_generation_prompt=True
  )
  """

  pipe_txt.start_chat(system_message=system)
    
  print(prompt)

  config = GenerationConfig(
      max_new_tokens=256,
      temperature=0.5,
      beam_size=1,
      do_sample=False, #fast for beam-search
      speculative_decoding=True,
      repetition_penalty=1.1,
      #top_k=50,
      #top_p=0.9,
  )

  generate_kwargs = dict(
      prompt = prompt,
      images=read_image("capture.jpg"),
      config=config,
      streamer=streamer, # !do_sample || top_k > 0
  )

  t1 = Thread(target=pipe_txt.generate, kwargs=generate_kwargs)
  t1.start()

  out = process_stream(streamer, False, isPlay,lang)
  return StreamingResponse(out, media_type='text/event-stream')

@app.post("/v1/img2chat", summary="문장 기반의 chatgpt 스타일 구현")
def img2chat(file : UploadFile = File(...), prompt = "" ,system = _SYSTEM, isPlay = 0, lang='en'): # gen or med
  streamer = IterableStreamer(pipe_txt.get_tokenizer())

  messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": prompt}
  ] 
  """
  prompt = token_txt.apply_chat_template(
    messages,
    tokenize=False,
    enable_thinking=True,
    add_generation_prompt=True
  )
  """

  print(prompt)

  config = GenerationConfig(
      max_new_tokens=256,
      temperature=0.5,
      beam_size=1,
      do_sample=False, #fast for beam-search
      speculative_decoding=True,
      #repetition_penalty=1.1,
      #top_k=50,
      #top_p=0.9,
      
  )

  generate_kwargs = dict(
      prompt = prompt,
      images=read_image(file.file),
      config=config,
      streamer=streamer, # !do_sample || top_k > 0
  )

  t1 = Thread(target=pipe_txt.generate, kwargs=generate_kwargs)
  t1.start()

  out = process_stream(streamer, False, isPlay, lang)
  return StreamingResponse(out, media_type='text/event-stream')


@app.post("/v1/stt", summary="음성을 인식합니다.")
def stt(file : UploadFile = File(...), lang="ko", isPlay=0):
  start = t.time()
  location = f"uploads/{file.filename}"

  with open(location,"wb+") as file_object:
    file_object.write(file.file.read())
  
  raw_speech, samplerate = librosa.load(location, sr=16000)
  print('length',librosa.get_duration(y=raw_speech, sr=samplerate))
  raw =  raw_speech.tolist()

  out = pipe_stt.generate(
    raw,
    max_new_tokens=100,
    # 'task' and 'language' parameters are supported for multilingual models only
    language=f"<|{lang}|>",
    task="transcribe",
    #return_timestamps=True
    #streamer=streamer,
  )

  print(t.time()-start, str(out))


  #chat = Chat()
  #chat.prompt = str(out)

  return { "result" : True, "data" : str(out) } #txt2chat(chat, isPlay)

print("Loading Complete","GPU")
