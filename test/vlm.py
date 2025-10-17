import argparse

import numpy as np
import openvino_genai
from PIL import Image
from openvino import Tensor
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

def streamer(subword: str) -> bool:
    '''

    Args:
        subword: sub-word of the generated text.

    Returns: Return flag corresponds whether generation should be stopped.

    '''
    print(subword, end='', flush=True)

    # No value is returned as in this example we don't want to stop the generation in this method.
    # "return None" will be treated the same as "return openvino_genai.StreamingStatus.RUNNING".


def read_image(path: str) -> Tensor:
    '''

    Args:
        path: The path to the image.

    Returns: the ov.Tensor containing the image.

    '''
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic)
    return Tensor(image_data)


def read_images(path: str) -> list[Tensor]:
    entry = Path(path)
    if entry.is_dir():
        return [read_image(str(file)) for file in sorted(entry.iterdir())]
    return [read_image(path)]


def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument('model_dir', help="Path to the model directory", default='./gemma-3-4b-it-ov')
    #parser.add_argument('image_dir', help="Image file or dir with images", default='three.jpg')
    #parser.add_argument('device', nargs='?', default='GPU', help="Device to run the model on (default: CPU)")
    #args = parser.parse_args()

    rgbs = read_images('suji.jpg')

    # GPU and NPU can be used as well.
    # Note: If NPU is selected, only the language model will be run on the NPU.
    enable_compile_cache = dict()
    #if args.device == "GPU":
    # Cache compiled models on disk for GPU to save time on the next run.
    # It's not beneficial for CPU.
    enable_compile_cache["CACHE_DIR"] = "vlm_cache"
    enable_compile_cache["PERFORMANCE_HINT"] = "LATENCY"
    #optimum-cli export openvino --model unsloth/gemma-3-4b-it --weight-format nf4 --quant-mode nf4_f8e4m3 --awq gemma-3-4b-it-ov-nf4
    #model_path = "./Phi-4-mm-it-ov-int4"
    model_path = snapshot_download(repo_id='circulus/gemma-3-4b-it-ov-awq-sym') #./gemma-3-4b-it-ov-int4'
    pipe = openvino_genai.VLMPipeline(model_path, 'GPU', **enable_compile_cache)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 1024

    
    #pipe.start_chat()
    pipe.start_chat(system_message="너는 데이빗이라는 소년 챗봇으로 친절하고 상냥하게 대답해줘. 대화체를 이용하도록 해.")

    prompt = input('question:\n')
    pipe.generate(prompt, images=rgbs, generation_config=config, streamer=streamer)

    while True:
        try:
            prompt = input("\n----------\n"
                "question:\n")
        except EOFError:
            break
        pipe.generate(prompt, generation_config=config, streamer=streamer)
    pipe.finish_chat()


if '__main__' == __name__:
    main()