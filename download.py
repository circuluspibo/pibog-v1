import os
from huggingface_hub import snapshot_download

# 1. 모델 리스트 정의 (Hugging Face Repo ID : 폴더명)
model_list = {
    #"OpenVINO/Qwen3-Reranker-0.6B-int8-ov": "Qwen3-Reranker-0.6B-int8-ov",   # 미사용
    #"OpenVINO/Qwen3-Embedding-0.6B-int8-ov": "Qwen3-Embedding-0.6B-int8-ov", # 미사용
    #"OpenVINO/Qwen3.5-9B-int4-ov": "Qwen3.5-9B-int4-ov",  #Echo9Zulu/gemma-3-4b-it-qat-int4_asym-ov": "gemma-3-4b-it-qat-int4_asym-ov",
    #"circulus/yolo-11s-safety-ov": "safety-11s_int8_openvino_model",         # 미사용 (안전 감지)
    #"circulus/yolo26s-helmet-ov": "yolo26s-helmet-ov",                       # 미사용
    #"circulus/yolo26s-seg-ov": "yolo26s-seg-ov",                             # 미사용 (NPU는 레포 내 yolo11m-seg 사용)
    #"HarmenWessels/gemma-4-12B-it-qat-int4-ov" : "gemma-4-12B-it-qat-int4-ov",
    #"OpenVINO/gemma-4-26b-a4b-it-int4-ov" : "gemma-4-26b-a4b-it-int4-ov",
    #"circulus/Qwen3.5-4B-eagle3-ov-int4" : "Qwen3.5-4B-eagle3-ov-int4",
    #"OpenVINO/Qwen3.5-4B-int4-ov" : "Qwen3.5-4b-int4-ov",                    # 미사용
    #"OpenVINO/Qwen3.5-2B-int4-ov" : "Qwen3.5-2B-int4-ov",                    # 구 model_txt, gemma-4-E2B로 교체됨
    #"OpenVINO/gemma-4-E4B-it-int4-ov" : "gemma-4-E4B-it-int4-ov",            # main_gpu.py 전용
    "circulus/whisper-large-v3-turbo-ov-int4": "whisper-large-v3-turbo-ov-int4",  # GPU STT (main_gpu_sr.py model_stt)
    "OpenVINO/gemma-4-E2B-it-int4-ov" : "gemma-4-E2B-it-int4-ov",  # GPU LLM (main_gpu_sr.py model_txt)
    "rippertnt/ko2en-ov-int4" : "ko2en-ov-int4",                   # GPU 번역 (model_t2t)
    "circulus/on-canvers-real-v3.9.1-int8" : "on-canvers-real-v3.9.1-int8"  # GPU 이미지 생성 (model_img)
}


base_dir = "./models"
os.makedirs(base_dir, exist_ok=True)

def download_models():
    """인터넷이 연결된 상태에서 모델을 ./models 폴더로 다운로드"""
    print("--- 모델 다운로드를 시작합니다 ---")
    for repo_id, folder_name in model_list.items():
        local_dir = os.path.join(base_dir, folder_name)
        print(f"\n[Downloading] {repo_id} -> {local_dir}")

        # snapshot_download 설정
        # local_dir_use_symlinks=False: 실제 파일을 복사하여 저장 (오프라인 환경 이전용)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True  # 중단 시 다시 시작 가능
        )

    snapshot_download(
        repo_id="rippertnt/on-vits2-multi-tts-v1",
        local_dir="./models",
        local_dir_use_symlinks=False,
        resume_download=True,  # 중단 시 다시 시작 가능
        allow_patterns="*ov*")


    print("\n모든 모델 다운로드 완료.")

def load_offline_model(repo_id):
    """네트워크 재시도 없이 로컬 파일에서만 모델 로드"""
    # 1. 환경 변수 설정 (Hugging Face 라이브러리의 네트워크 접근 원천 차단)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    # 2. 로컬 경로 설정
    folder_name = model_list.get(repo_id)
    local_model_path = os.path.abspath(os.path.join(base_dir, folder_name))

    print(f"오프라인 경로에서 모델을 로드합니다: {local_model_path}")

    # 실제 모델 로드 예시 (OpenVINO용 optimum 라이브러리 사용 가정)
    # from optimum.intel.openvino import OVModelForCausalLM
    # model = OVModelForCausalLM.from_pretrained(local_model_path, local_files_only=True)
    return local_model_path

# --- 실행 부분 ---
if __name__ == "__main__":
    # 최초 1회 실행하여 모델 다운로드 (인터넷 연결 필요)
    download_models()

    # 이후 네트워크가 없는 상황에서 로드 테스트
    # test_path = load_offline_model("OpenVINO/Qwen3-Reranker-0.6B-fp16-ov")
