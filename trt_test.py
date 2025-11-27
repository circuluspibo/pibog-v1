

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer
from tensorrt_llm.runtime import LLM, GenerationConfig

app = FastAPI()

# Load engine & tokenizer
engine_dir = "pytorch/gemma-3-12b-it-QAT-INT4"
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b")
llm = LLM(engine_dir)

class Request(BaseModel):
    prompt: str
    max_tokens: int = 128

@app.post("/generate")
async def generate_text(req: Request):

    async def stream():
        inputs = tokenizer(req.prompt, return_tensors="pt")["input_ids"]
        for token in llm.stream_generate(
            inputs,
            GenerationConfig(max_new_tokens=req.max_tokens)
        ):
            yield tokenizer.decode([token], skip_special_tokens=True)

    return StreamingResponse(stream(), media_type="text/plain")