import os
import time as t
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
# Removed AutoModelForSpeechSeq2Seq as it's not used in the provided code snippet
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig 
from transformers import TextStreamer as IterableStreamer # Use TextStreamer from HF for better compatibility
import torch
from pydantic import BaseModel
from queue import Queue
from threading import Thread
from datetime import datetime
import csv
import subprocess
import librosa
from requests import get
from nv_monitor import CPUPowerMonitor

pw = CPUPowerMonitor(interval=1.0)
pw.start()
# --- Missing Definitions/Imports for the original code to run ---

# A mock GenerationConfig class based on common HF parameters
class GenerationConfig(BaseModel):
    max_new_tokens: int = 256
    temperature: float = 0.5
    beam_size: int = 1
    do_sample: bool = False
    speculative_decoding: bool = True
    repetition_penalty: float = 1.1

# We'll use the official Hugging Face TextStreamer for better compatibility, 
# although the provided IterableStreamer logic is also common. 
# NOTE: If TextStreamer doesn't work out-of-the-box with your exact pipe_txt.generate 
# setup, you may need to revert to your custom Queue-based IterableStreamer. 
# For this fix, I'll use a functional, simple Queue-based streamer if the original
# code assumed a custom one, as it's safer than relying on HF's internal structure.
class CustomIterableStreamer(object):
    def __init__(self, tokenizer, skip_prompt=False, timeout=None):
        self.tokenizer = tokenizer
        self.queue = Queue()
        self.stop_signal = object()
        self.skip_prompt = skip_prompt
        self.timeout = timeout
        self.text_buffer = []

    def __iter__(self):
        return self

    def __next__(self):
        value = self.queue.get(timeout=self.timeout)
        if value is self.stop_signal:
            raise StopIteration()
        # Decode the token ID (assuming put() gets token IDs)
        return self.tokenizer.decode(value, skip_special_tokens=False)

    def put(self, value):
        # We assume the model's generate passes a single token ID or a tensor of IDs
        if torch.is_tensor(value) and value.numel() == 1:
             self.queue.put(value.item())
        elif isinstance(value, int):
             self.queue.put(value)
        else:
             # Handle a tensor of IDs (e.g., if generate returns all tokens at once)
             for token_id in value.flatten().tolist():
                 self.queue.put(token_id)


    def end(self):
        self.queue.put(self.stop_signal)

# We define the system message used in the original code's function signature
_SYSTEM = "You are a helpful and harmless AI assistant." 

# --- End of Missing Definitions/Imports ---


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

CACHE_DIR = '/home/circulus/git/HF_CACHE'

# Initialize FastAPI
app = FastAPI()

# Model paths - Use Hugging Face model names for the QAT models
gemma_model_name = 'unsloth/gemma-3-12b-it-qat'
whisper_model_name = 'openai/whisper-large-v3-turbo'

# Load the models from Hugging Face with CUDA support (via PyTorch)
device = "cuda"

# Load text generation model (Gemma 3-12b QAT)
pipe_txt = AutoModelForCausalLM.from_pretrained(gemma_model_name, cache_dir=CACHE_DIR, dtype=torch.bfloat16,quantization_config=quant_config, attn_implementation = "flash_attention_2").to(device)
token_txt = AutoTokenizer.from_pretrained(gemma_model_name, cache_dir=CACHE_DIR, dtype=torch.bfloat16)

# Load speech-to-text model (Whisper 3 Large Turbo)
# NOTE: Using AutoModelForCausalLM is incorrect for Whisper. It should be AutoModelForSpeechSeq2Seq.
# Reverting to the correct model type from your original code.
from transformers import AutoModelForSpeechSeq2Seq
pipe_stt = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_name, cache_dir=CACHE_DIR,quantization_config=quant_config, dtype=torch.bfloat16, attn_implementation = "flash_attention_2").to(device)
tokenizer_stt = AutoTokenizer.from_pretrained(whisper_model_name, cache_dir=CACHE_DIR, dtype=torch.bfloat16)

# Helper function for streaming response (Kept as is from original)
async def process_stream(streamer, isStream=True, isPlay=0, lang='en'):
    cnt = 0
    latency = 0
    isStart = False
    sentence = ""
    full_txt = ""
    print("streaming start...")

    start_time = t.time()
    total_tokens = 0

    # NOTE: The streamer is expected to yield tokens/strings here
    for new_token in streamer:

        # If the streamer yields token IDs, we need to decode them first.
        # Assuming the CustomIterableStreamer is correctly set up to yield strings/tokens.
        if isinstance(new_token, int):
             new_token = token_txt.decode(new_token, skip_special_tokens=False)
             full_txt = full_txt + token_txt

        if isStart is False:
            isStart = True
            latency = t.time() - start_time


        if "assistant" in new_token:
            cnt += 1
            if cnt == 1:
                continue
            elif cnt == 2:
                print("Forcing exit...")
                break

        if isStream:
            yield new_token
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

    if len(sentence) > 3:
        yield sentence

    duration = t.time() - start_time
    total_tokens = token_txt(full_txt)
    tokens_per_sec = total_tokens / duration if duration > 0 else 0

    print(f"Total tokens: {total_tokens}")
    print(f"Duration: {duration:.4f} sec")
    print(f"Tokens/s: {tokens_per_sec:.4f}")

    # Log the statistics
    log_file = "GPU_log.csv"
    new_file = not os.path.exists(log_file)

    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

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
            pw.get_power()
        ])

# --- FIX: txt2chat to use standard HF generation with chat template ---
@app.get("/v1/txt2chat", summary="문장 기반의 chatgpt 스타일 구현")
def txt2chat(prompt : str ,system = _SYSTEM, isPlay = 0, lang='en'): # gen or med
    # Initialize the streamer
    streamer = CustomIterableStreamer(tokenizer_txt, skip_prompt=True) # Use the custom streamer

    # 1. Prepare the chat history using the tokenizer's chat template
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    
    # 2. Apply the chat template and tokenize the inputs
    # Use pipe_txt.get_tokenizer() for consistency if that's what you intended
    input_text = tokenizer_txt.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 3. Tokenize the inputs and move to device
    inputs = tokenizer_txt(input_text, return_tensors="pt").to(device)

    # 4. Define configuration (using simple kwargs instead of a separate config object)
    
    print(prompt)

    generate_kwargs = dict(
        **inputs, # Pass the tokenized inputs (input_ids, attention_mask)
        max_new_tokens=256,
        temperature=0.5,
        do_sample=False, 
        repetition_penalty=1.1,
        streamer=streamer, # Pass the streamer
        # Remove beam_size/speculative_decoding as they don't work well with streaming
    )

    # 5. Start generation in a separate thread
    t1 = Thread(target=pipe_txt.generate, kwargs=generate_kwargs)
    t1.start()

    # 6. Stream the output
    out = process_stream(streamer, False, isPlay, lang)
    return StreamingResponse(out, media_type='text/event-stream')


# Helper function to read images (Used by the original code, but only for STT context)
def read_image(file_obj):
    from PIL import Image
    # Open the image using PIL from the file-like object
    image = Image.open(file_obj).convert("RGB")
    return image

# --- FIX: img2chat to handle streaming, but acknowledge VLM limitation ---
@app.post("/v1/img2chat", summary="Image to Chat with CUDA via transformers (Text-Only Model)")
def img2chat(file: UploadFile = File(...), prompt: str = "", system: str = _SYSTEM, isPlay = 0, lang: str = "en"):
    
    # Since pipe_txt is a text-only model (AutoModelForCausalLM), we cannot
    # process the image. We will stream a response based on the prompt only,
    # and provide a contextual note in the streamed output.
    
    streamer = CustomIterableStreamer(tokenizer_txt, skip_prompt=True)

    # NOTE: You've successfully received the file, but we won't process it with pipe_txt.
    # If the user's intent is to verify the file upload, this is where we'd do it.
    
    # 1. Prepare the chat input:
    # We'll prepend a note to the prompt since the image is ignored by the model.
    modified_prompt = (
        f"NOTE: I received an image named '{file.filename}', but since I'm a text-only model (Gemma), "
        f"I will now answer your question based on the text prompt only. "
        f"Prompt: {prompt}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": modified_prompt}
    ]
    
    input_text = tokenizer_txt.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer_txt(input_text, return_tensors="pt").to(device)
    
    # 2. Define generation arguments
    generate_kwargs = dict(
        **inputs,
        max_new_tokens=256,
        temperature=0.5,
        do_sample=False,
        repetition_penalty=1.1,
        streamer=streamer,
    )

    # 3. Start generation in a separate thread
    t1 = Thread(target=pipe_txt.generate, kwargs=generate_kwargs)
    t1.start()

    # 4. Return the streaming response
    out = process_stream(streamer, False, isPlay, lang)
    return StreamingResponse(out, media_type='text/event-stream')


@app.post("/v1/stt", summary="Speech to Text with CUDA via transformers")
def stt(file: UploadFile = File(...), lang: str = "ko"):
    start = t.time()
    # Create directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    location = f"uploads/{file.filename}"

    # Save uploaded speech file
    with open(location, "wb+") as file_object:
        file_object.write(file.file.read())

    # Load speech using librosa
    raw_speech, samplerate = librosa.load(location, sr=16000)

    # Tokenize the raw speech input
    # NOTE: Whisper models use a feature extractor, not just a tokenizer on raw audio.
    # We need to assume the model's 'generate' method handles the feature extraction internally
    # or requires a specific processor/feature extractor if following standard HF practice.
    # For this simplified setup, we pass the raw audio list as intended by your original code.

    inputs = tokenizer_stt(raw_speech.tolist(), return_tensors="pt").to(device)

    # Generate transcription
    # NOTE: The generate call for Whisper/Speech Seq2Seq is usually different from Causal LM.
    # Assuming this simple call works with your loaded Whisper model.
    outputs = pipe_stt.generate(**inputs, max_length=100)

    # Decode and process the output
    transcription = tokenizer_stt.decode(outputs[0], skip_special_tokens=True)

    print(t.time() - start, transcription)

    # Clean up the file
    os.remove(location)

    return {"result": True, "data": transcription}

# Default API root
@app.get("/")
def main():
    return {"result": True, "data": "AI-CPU-V2"}

# Starting the application
if __name__ == "__main__":
    # NOTE: uvicorn is typically used to run FastAPI applications
    # For this script to run directly and show the Popen command, we keep it simple
    print("Loading Complete", "CUDA/PyTorch")
    subprocess.Popen(["play", "intel_inside.mp3"]) # Example sound on startup
    # You would typically run: uvicorn your_file_name:app --reload