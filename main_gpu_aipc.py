# pip install -U diffusers
# pip install git+https://github.com/openvino-dev-samples/optimum-intel.git@2f62e5aee74b4acba3836e1f26678c0db0a09c00
# pip install --pre -U openvino-genai openvino openvino-tokenizers --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

from fastapi.middleware.cors import CORSMiddleware
import librosa
from fastapi import FastAPI, File, UploadFile
from transformers import AutoTokenizer
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import time as t
from transformers import AutoTokenizer
from pydantic import BaseModel, Field
import numpy as np
import openvino_genai as ov_genai
import subprocess
from serverinfo import si
import logging
from requests import get
import openvino as ov
from threading import Event, Thread
from transformers import AutoTokenizer
from pydantic import BaseModel, Field
from iterator import IterableStreamer
from fastapi.middleware.cors import CORSMiddleware
import torchvision.transforms.functional as F
from PIL import Image
from openvino import Tensor
from pathlib import Path
from openvino_genai import GenerationConfig
import time
from transformers import AutoConfig, AutoTokenizer, pipeline
from optimum.intel.openvino import OVModelForSeq2SeqLM
from fastapi import FastAPI, UploadFile, File, Form
import random
import torch
from ov_faceswap import FaceSwapOpenVINO 
import cv2
import os
import numpy as np
import torch
from scipy.io.wavfile import write
from optimum.intel import OVZImagePipeline

from speaker_encoder import audio
from speaker_encoder.hparams import sampling_rate, mel_window_step, partials_n_frames


# 변환된 모델 로드
#compiled_cmodel = ov.compile_model("models/FreeVC_ov/cmodel_ir.xml")
#compiled_smodel = ov.compile_model("models/FreeVC_ov/smodelir.xml")
#compiled_net_g = ov.compile_model("models/FreeVC_ov/net_gir.xml")


### 2. 보조 함수 (Speaker Embedding 추출 전용)
def compute_partial_slices(n_samples: int, rate=1.3, min_coverage=0.75):
    samples_per_frame = int((sampling_rate * mel_window_step / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = int(np.round((sampling_rate / rate) / samples_per_frame))
    
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partials_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partials_n_frames])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    return wav_slices, mel_slices

def embed_utterance(wav: np.ndarray, smodel_compiled):
    wav_slices, mel_slices = compute_partial_slices(len(wav))
    max_wave_length = wav_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

    mel = audio.wav_to_mel_spectrogram(wav)
    mels = np.array([mel[s] for s in mel_slices])
    
    # OpenVINO 추론
    output_layer = smodel_compiled.output(0)
    partial_embeds = smodel_compiled(mels)[output_layer]

    raw_embed = np.mean(partial_embeds, axis=0)
    return raw_embed / np.linalg.norm(raw_embed, 2)


### 3. 메인 추론 함수
def synthesize_audio(src_path, tgt_path):
    # Target Speaker Embedding 추출
    wav_tgt, _ = librosa.load(tgt_path, sr=sampling_rate)
    wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
    g_tgt = embed_utterance(wav_tgt, compiled_smodel)
    g_tgt = np.expand_dims(g_tgt, 0).astype(np.float32)

    # Source Content 추출
    wav_src, _ = librosa.load(src_path, sr=sampling_rate)
    wav_src = np.expand_dims(wav_src, axis=0).astype(np.float32)
    
    # cmodel 추론
    c_output = compiled_cmodel(wav_src)[compiled_cmodel.output(0)]
    c = c_output.transpose((0, 2, 1)) # [1, channel, time]

    # net_g(Synthesizer) 최종 추론
    # 입력 순서와 타입 주의
    tgt_audio = compiled_net_g([c, g_tgt])[compiled_net_g.output(0)]
    
    return tgt_audio[0][0]


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

model_txt = snapshot_download(repo_id='circulus/Qwen3-VL-8B-it-ov-awq') # helenai/Qwen3-VL-4B-Instruct-int4 Echo9Zulu/gemma-3-4b-it-qat-int4_asym-ov circulus/gemma-3-4b-it-ov-awq-sym helenai/Qwen2.5-VL-3B-Instruct-ov-int4
model_stt = snapshot_download(repo_id='circulus/whisper-large-v3-turbo-ov-int4') # translate not working, only general model
model_img = snapshot_download(repo_id='circulus/Z-Image-Turbo-ov-int4')
model_t2t = snapshot_download(repo_id='rippertnt/ko2en-ov-int4')

#swapper = FaceSwapOpenVINO(device_name="GPU") 

token_txt = AutoTokenizer.from_pretrained(model_txt)
#token_img = AutoTokenizer.from_pretrained(model_img)

conf = AutoConfig.from_pretrained(model_t2t, trust_remote_code=True)
token_t2t = AutoTokenizer.from_pretrained(model_t2t, trust_remote_code=True)
model_t2t = OVModelForSeq2SeqLM.from_pretrained(model_t2t, device='GPU', ov_config={"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}, config=conf, low_cpu_mem_usage=True, trust_remote_code=True) #"KV_CACHE_PRECISION": "u8", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32"
pipe_t2t = pipeline("text2text-generation", model=model_t2t, tokenizer=token_t2t)

pipe_stt = ov_genai.WhisperPipeline(model_stt,device="GPU", config={"PERFORMANCE_HINT": "LATENCY"})
pipe_txt = ov_genai.VLMPipeline(model_txt, device="GPU", config={"PERFORMANCE_HINT": "LATENCY"})


ov_pipe = OVZImagePipeline.from_pretrained(model_img, device="GPU")


def load_model(vlm=True):
  print('model',vlm)

# Output directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        elif "." in new_token or "\n" in new_token:
            sentence += new_token
            if len(sentence) > 3:

                sentence = sentence.strip()

                if int(isPlay) > 0:
                    get(
                      "http://127.0.0.1:59531/v2/tts",
                      params={"text": sentence, "lang": lang, "voice": 31}
                    )

                print(sentence)
                yield sentence
                sentence = ""

        else:
            sentence += new_token

    # 마지막 문장 처리
    if len(sentence) > 3:
        yield sentence

    # ---------------------------------
    # 🔥 token/s 계산
    # ---------------------------------
    duration = time.time() - start_time
    total_tokens = len(token_txt(full_txt)['input_ids'])
    tokens_per_sec = total_tokens / duration if duration > 0 else 0

    print(f"Total tokens: {total_tokens}")
    print(f"Duration: {duration:.4f} sec")
    print(f"Tokens/s: {tokens_per_sec:.4f}")


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


@app.get("/prepare")
def preprare(mode : int = 0):
  
  if mode == 0:
    load_model(vlm=True)
  else:
    load_model(vlm=False)
  return { "result" : mode }


@app.get("/v1/txt2chat", summary="문장 기반의 chatgpt 스타일 구현")
def txt2chat(prompt : str ,system = _SYSTEM, isPlay = 0, lang='en'): # gen or med
  load_model(True)
  streamer = IterableStreamer(pipe_txt.get_tokenizer())

  pipe_txt.start_chat(system_message=system)

  print(prompt)

  config = GenerationConfig(
      max_new_tokens=1024,
      temperature=0.5,
      #beam_size=1,
      #do_sample=False, #fast for beam-search
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


  t1 = Thread(target=pipe_txt.generate, kwargs=generate_kwargs)
  t1.start()

  out = process_stream(streamer, False, isPlay,lang)
  return StreamingResponse(out, media_type='text/event-stream')


@app.get("/v2/img2chat", summary="문장 기반의 chatgpt 스타일 구현")
@app.post("/v2/img2chat", summary="문장 기반의 chatgpt 스타일 구현")
def img2chat2(prompt = "" ,system = _SYSTEM, isPlay = 0, lang='en'): # gen or med
  load_model(True)
  streamer = IterableStreamer(pipe_txt.get_tokenizer())

  messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": prompt}
  ] 

  pipe_txt.start_chat(system_message=system)
    
  print(prompt)

  config = GenerationConfig(
      max_new_tokens=1024,
      temperature=0.5,
      #beam_size=1,
      #do_sample=False, #fast for beam-search
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
def img2chat(file : UploadFile = File(...), prompt = "recognize this image as OCR, response only json like this, { 'result' : 'value'}" ,system = "You are the AI OCR engine for recognizing hand-writen korean charactors", isPlay = 0, lang='en'): # gen or med
  load_model(True)
  streamer = IterableStreamer(pipe_txt.get_tokenizer())

  print(prompt)

  config = GenerationConfig(
      max_new_tokens=256,
      temperature=0.0,
      #beam_size=1,
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

  return { "result" : True, "data" : str(out) } #txt2chat(chat, isPlay)


@app.post("/v1/translate", summary="한글을 영어로 번역합니다")
def translate(prompt : str):
  out = pipe_t2t(prompt, max_new_tokens=512)[0]['generated_text']

  return { "result" : True, "data" : str(out) } #txt2chat(chat, isPlay)


@app.post("/v1/txt2img", response_class=FileResponse)
async def txt2img(prompt: str = "", seed: int = None, w : int = 512, h : int = 512, lang : str = 'ko'):
  load_model(False)

  if seed is None:
      seed = random.randint(0, 2**32 - 1)
  torch.manual_seed(seed)

  image = ov_pipe(
      prompt=prompt,
      height=h,
      width=w,
      num_inference_steps=9,  # This actually results in 8 DiT forwards
      guidance_scale=0.0,  # Guidance should be 0 for the Turbo models
      generator=torch.Generator("cpu").manual_seed(42),
  ).images[0]

  image.save(f"outputs/out_image_{seed}.png")

  return f"outputs/out_image_{seed}.png"


@app.post("/voice2wav", response_class=FileResponse)
def voice2wav(src : UploadFile = File(...), tgt : UploadFile = File(...)):
  src_file = f"uploads/{src.filename}"
  tgt_file = f"uploads/{tgt.filename}"

  with open(src_file,"wb+") as file_object:
    file_object.write(src.file.read())

  with open(tgt_file,"wb+") as file_object:
    file_object.write(tgt.file.read())    

  ### 4. 실행 및 결과 저장
  print('추론 시작...')
  start_time = time.time()

  #src_file = f"{gender}.wav"

  output_audio = synthesize_audio(src_file, tgt_file)

  # 결과 저장
  timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
  result_path = os.path.join("output", f"result_{timestamp}.wav")

  write(result_path, sampling_rate, output_audio)

  print(f"완료! 소요 시간: {time.time() - start_time:.2f}초")
  print(f"저장 위치: {result_path}")

  return result_path

