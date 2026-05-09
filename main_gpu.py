#optimum-cli export openvino --task text-classification --weight-format int4 --ratio 0.8 --model Qwen/Qwen3-Reranker-0.6B models/Qwen3-Reranker-0.6B-ov
#optimum-cli export openvino --task text-classification --weight-format int4 --ratio 0.8 --model Qwen/Qwen3-Reranker-4B models/Qwen3-Reranker-4B-ov

#optimum-cli export openvino --task feature-extraction --weight-format int4 --ratio 0.8 --model Qwen/Qwen3-Embedding-0.6B models/Qwen3-Embedding-0.6B-ov
#optimum-cli export openvino --task feature-extraction --weight-format int4 --ratio 0.8 --model Qwen/Qwen3-Embedding-4B models/Qwen3-Embedding-4B-ov


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
import subprocess
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
import time
import csv
import os
from datetime import datetime
from monitor import CPUPowerMonitor
import asyncio

# support rag
import pandas as pd
from langchain_community.embeddings import OpenVINOBgeEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openvino_genai import TextRerankPipeline
import glob

RAG_DB_DIR = "./rag_db"


#OpenVINO/Qwen3-Reranker-0.6B-fp16-ov
#OpenVINO/Qwen3-Embedding-0.6B-int8-ov
#Echo9Zulu/gemma-3-4b-it-qat-int4_asym-ov
#circulus/whisper-large-v3-turbo-ov

rag_embedding = OpenVINOBgeEmbeddings(
    model_name_or_path="./models/Qwen3-Embedding-0.6B-int8-ov",
    model_kwargs={"device": "GPU"},
)

rag_db = Chroma(
    collection_name="arcos",
    persist_directory=RAG_DB_DIR,
    embedding_function=rag_embedding,
)

config = TextRerankPipeline.Config()
config.top_n = 5

# ===============================
# 🔥 RERANKER
# ===============================
#model_rerank = snapshot_download(repo_id="OpenVINO/Qwen3-Reranker-0.6B-fp16-ov")

reranker = TextRerankPipeline("./models/Qwen3-Reranker-0.6B-fp16-ov","GPU",config) #./models/Qwen3-Reranker-0.6B-ov


#optimum-cli export openvino --weight-format int4 --task text-generation-with-past --model growdle/HyperCLOVAX-SEED-Text-Instruct-1.5B ./CLOVAX-1.5B-ov-int4
#kakaocorp/kanana-1.5-2.1b-instruct-2505
#https://github.com/Unitree-Go2-Robot/go2_robot


# 너는 파이온이라는 휴머노이드 로봇으로 사람들을 지키기 위해 태어났어. 대화체로 사람처럼 대답하되, 다음과 같은 동작이 가능하니, 적절한 동작을 먼저 출력하고 대답을 이야기 해줘. - clamp, highFive, shakeHands_1, blowKiss, hug, hightWave, lowWave, ultramanRay, bothHandsUp, singleHandsUp, Refuse

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

_IP = "127.0.0.1" #si.getIP()
_PORT = int(open("port.txt", 'r').read())

app = FastAPI()

#pw = CPUPowerMonitor(interval=1.0)
#pw.start()

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

model_txt = "./models/Qwen3-VL-8B-it-ov-awq" #gemma-3-4b-it-ov-awq" #snapshot_download(repo_id='Echo9Zulu/gemma-3-4b-it-qat-int4_asym-ov') # circulus/gemma-3-4b-it-ov-awq-sym helenai/Qwen2.5-VL-3B-Instruct-ov-int4
model_stt = "./models/whisper-large-v3-turbo-ov-int4"#snapshot_download(repo_id='circulus/whisper-large-v3-turbo-ov')

config = {
    # 1. 지연시간 최소화 및 자원 집중
    "PERFORMANCE_HINT": "LATENCY",
    "EXECUTION_MODE_HINT" : "PERFORMANCE", # 추가 햇는데 나을지는
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
    "MODEL_PRIORITY": "HIGH",
    "NUM_STREAMS": "1",

    # 2. 메모리 및 정밀도 최적화
    "KV_CACHE_PRECISION": "u8",# KV 캐시 압축 (TPS 향상 핵심)
    "INFERENCE_PRECISION_HINT": "f16", # GPU 연산 가속

    # 3. 실행 우선순위 극대화
    "GPU_QUEUE_PRIORITY": "HIGH",
    "GPU_HOST_TASK_PRIORITY": "HIGH",
    "GPU_QUEUE_THROTTLE": "LOW",

    "CACHE_DIR": "./ov_cache",
    #"ENABLE_CPU_PINNING": "YES",# CPU-GPU 협업 효율 증대 -CPU 로 추론할때 전용으로 묶는것.. 
}
pipe_stt = ov_genai.WhisperPipeline(model_stt,device="GPU", config={"PERFORMANCE_HINT": "LATENCY"})

token_txt = AutoTokenizer.from_pretrained(model_txt)
pipe_txt = ov_genai.VLMPipeline(model_txt, device="GPU", config={"PERFORMANCE_HINT": "LATENCY"})

def get_rag_context(
    query: str,
    search_k: int = 20,
    rerank_k: int = 5,
) -> str:
    # 1️⃣ Embedding 기반 1차 검색
    docs = rag_db.similarity_search(query, k=search_k)
    if not docs:
        return ""

    candidates = [doc.page_content for doc in docs]
    print(len(candidates), candidates)

    # 2️⃣ Reranker (Cross-Encoder)
    rerank_results = reranker.rerank(query,candidates)

    # [(idx, score), ...] → 상위 문서
    top_docs = [
        candidates[idx] for idx, score in rerank_results[:rerank_k]
    ]

    # 3️⃣ Prompt Context 생성
    context = "\n".join([f"- {doc}" for doc in top_docs])
    return context    


# for genai
async def process_stream(streamer, isStream=True, isPlay=0, lang='en'):
    cnt = 0
    latency = 0
    isStart = False
    sentence = ""
    full_txt = ""
    print("streaming start...")

    # ---------------------------------
    # 🔥 token/s 측정용 변수
    # ---------------------------------
    start_time = time.time()
    total_tokens = 0


    for new_token in streamer:
        full_txt = full_txt + new_token
        if isStart is False:
          isStart = True
          latency =  time.time() - start_time 

        # token count 증가

        if "assistant" in new_token:
            cnt += 1
            if cnt == 1:
                continue  # skip
            elif cnt == 2:
                print("Forcing exit...")
                break

        # ---------------------
        # 🔥 Stream 모드 처리
        # ---------------------
        if isStream:
            yield new_token

        # ---------------------
        # 🔥 Sentence 모드 처리
        # ---------------------
        elif "." in new_token or "\n" in new_token or "?" in new_token or "!" in new_token: 
            sentence += new_token
            if len(sentence) > 3:

                sentence = sentence.strip()

                if int(isPlay) > 0:
                    get(
                      "http://127.0.0.1:59530/v1/tts",
                      params={"text": sentence, "lang": lang, "voice": 6}
                    )

                print(sentence)
                yield sentence
                await asyncio.sleep(0)
                sentence = ""

        else:
            sentence += new_token

    # 마지막 문장 처리
    if len(sentence) > 3:
        if int(isPlay) > 0:
            get(
                "http://127.0.0.1:59531/v2/tts",
                params={"text": sentence, "lang": lang, "voice": 31}
            )        
        yield sentence
        await asyncio.sleep(0)

    # ---------------------------------
    # 🔥 token/s 계산
    # ---------------------------------
    duration = time.time() - start_time
    total_tokens = len(token_txt(full_txt)['input_ids'])
    tokens_per_sec = total_tokens / duration if duration > 0 else 0

    print(f"Total tokens: {total_tokens}")
    print(f"Duration: {duration:.4f} sec")
    print(f"Tokens/s: {tokens_per_sec:.4f}")

    # ---------------------------------
    # 🔥 CSV 로그 저장
    # ---------------------------------
    log_file = "GPU_log.csv"
    new_file = not os.path.exists(log_file)

    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 헤더가 없으면 추가
        if new_file:
            writer.writerow([
                "timestamp", "Total_Tokens", "TTFT", "Duration",
                "Tokens/s", "Watt"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            total_tokens,
            latency,
            round(duration, 6),
            round(tokens_per_sec, 6),
            #pw.get_power()
        ])

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

@app.post("/v1/rag/upload_csv")
async def upload_csv_rag(file: UploadFile = File(...)):
    # Read CSV file into a pandas dataframe
    df = pd.read_csv(file.file)

    # Process each row to generate text data
    texts = []
    for _, row in df.iterrows():
        texts.append(" | ".join(map(str, row.values)))

    # Initialize the text splitter (if not done already)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    # Split the texts into smaller chunks for easier indexing
    docs = splitter.create_documents(texts)

    # Add documents to the RAG database (Chroma or similar)
    rag_db.add_documents(docs)

    # If Chroma supports `commit()` or `save()`, call that to persist changes.
    # For example, if it's a Chroma client object, use commit().
    #try:
    #    rag_db.commit()   # commit() error or rag_db.save(), depending on the library
    #except AttributeError:
    #    return {"result": False, "error": "Unable to persist the documents. Please check your database setup."}

    # Return a response indicating the result
    return {
        "result": True,
        "rows": len(df),
        "chunks": len(docs)
    }


@app.post("/v1/rag/upload_all_csv")
def upload_all_csv():
    # 우분투 시스템 사용자명을 자동으로 가져옵니다
    user_name = os.getenv("USER")  # 시스템 환경 변수에서 사용자명을 가져옴

    # 파일 경로 설정: /media/{시스템 사용자명}
    directory = f"/media/{user_name}"

    # 지정된 폴더 내의 모든 CSV 파일을 glob을 사용하여 하위 디렉토리까지 검색
    # '**'는 하위 디렉토리까지 포함하겠다는 의미입니다.
    csv_files = glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)

    print(user_name,csv_files)

    if not csv_files:
        return {"result": False, "error": "No CSV files found in the specified directory."}

    # 기존 rag_db 초기화
    rag_db.reset_collection()  # rag_db 초기화 함수 호출 (실제 사용 중인 라이브러리에서 초기화 방법에 맞게 수정 필요)

    # CSV 파일을 읽고 RAG DB에 추가하는 작업 수행
    total_rows = 0
    total_chunks = 0

    # Initialize the text splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for file_path in csv_files:
        # Read the CSV file into a pandas dataframe
        df = pd.read_csv(file_path)

        # Process each row to generate text data
        texts = []
        for _, row in df.iterrows():
            texts.append(" | ".join(map(str, row.values)))

        # Split the texts into smaller chunks for easier indexing
        docs = splitter.create_documents(texts)

        # Add documents to the RAG database (Chroma or similar)
        rag_db.add_documents(docs)

        # Track row and chunk counts for the response
        total_rows += len(df)
        total_chunks += len(docs)

    # If Chroma supports `commit()` or `save()`, call that to persist changes.
    # If necessary, you can uncomment the next lines to commit/save.
    # try:
    #     rag_db.commit()  # commit() or rag_db.save() depending on the library
    # except AttributeError:
    #     return {"result": False, "error": "Unable to persist the documents. Please check your database setup."}
    print("RAG > ",total_rows)
    # Return a response indicating the result
    return {
        "result": True,
        "total_files": len(csv_files),
        "total_rows": total_rows,
        "total_chunks": total_chunks
    }

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
      top_k=50,
      top_p=0.9,
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
  return StreamingResponse(out, media_type='text/plain; charset=utf-8', headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Content-Type-Options": "nosniff"})

@app.get("/v1/rag/txt2chat", summary="문장 기반의 chatgpt 스타일 구현")
def txt2rag(prompt : str ,system = _SYSTEM, isPlay = 0, lang='en'): # gen or med
  streamer = IterableStreamer(pipe_txt.get_tokenizer())

  rag_context = get_rag_context(prompt)
  if rag_context:
        prompt = f"""
다음은 참고 지식이다. 반드시 이 내용을 우선 참고하여 url 은 빼고 답변해주세요.

[지식]
{rag_context}

[질문]
{prompt}
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
      top_k=50,
      top_p=0.9,
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
  #return StreamingResponse(out, media_type='text/event-stream')
  return StreamingResponse(out, media_type='text/plain; charset=utf-8', headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Content-Type-Options": "nosniff"})


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
  #return StreamingResponse(out, media_type='text/event-stream')
  return StreamingResponse(out, media_type='text/plain; charset=utf-8', headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Content-Type-Options": "nosniff"})

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
  #return StreamingResponse(out, media_type='text/event-stream')
  return StreamingResponse(out, media_type='text/plain; charset=utf-8', headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Content-Type-Options": "nosniff"})    

@app.get("/v1/rag/img2chat", summary="RAG + Image Chat")
@app.post("/v1/rag/img2chat", summary="RAG + Image Chat")
def img2rag( prompt="", system=_SYSTEM, isPlay=0, lang='en',):
    streamer = IterableStreamer(pipe_txt.get_tokenizer())

    # ===============================
    # 🔥 RAG CONTEXT
    # ===============================
   
    rag_context = get_rag_context(prompt)
    if rag_context:
        prompt = f"""
다음은 참고 지식이다. 반드시 이 내용을 우선 참고하여 url 은 빼고 답변해주세요.

[지식]
{rag_context}

[질문]
{prompt}
"""

    pipe_txt.start_chat(system_message=system)

    config = GenerationConfig(
        max_new_tokens=256,
        do_sample=False,
        speculative_decoding=True,
        repetition_penalty=1.1,
    )

    generate_kwargs = dict(
        prompt=prompt,
        images=read_image("capture.jpg"),
        config=config,
        streamer=streamer,
    )

    t1 = Thread(target=pipe_txt.generate, kwargs=generate_kwargs)
    t1.start()

    # 🔥 기존 streaming 유지
    out = process_stream(streamer, False, isPlay, lang)
    #return StreamingResponse(out, media_type="text/event-stream")
    return StreamingResponse(out, media_type='text/plain; charset=utf-8', headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Content-Type-Options": "nosniff"})

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
#subprocess.Popen(["play", 'intel_inside.mp3']) # async

upload_all_csv()

# Kiosk 모드로 띄울 URL
url = "http://127.0.0.1:59531/web/pion2.html"

# Kiosk 모드로 Chromium 실행
"""
subprocess.Popen(['chromium', '--kiosk', url],
    stdin=subprocess.DEVNULL,       # Discard stdin from the child process
    stdout=subprocess.DEVNULL,      # Discard stdout from the child process
    stderr=subprocess.DEVNULL,      # Discard stderr from the child process
    start_new_session=True          # Start the process in a new session (POSIX only)
)
"""
