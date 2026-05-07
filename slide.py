"""
slide.py — /slide endpoint
--------------------------
GET /slide?page=1&text=Hello&motion=greet

Flow:
  1. TTS 생성 (기존 pipe_tts 재사용)
  2. motion POST 요청 (async, fire-and-forget)
  3. 음성 재생 (blocking — 완료까지 대기)
  4. 3초 후 응답 반환
"""

import asyncio
import hashlib
import os
import time as t

import httpx
import numpy as np
from fastapi import APIRouter
from scipy.io.wavfile import write

from text import text_to_sequence

# ── 상수 ────────────────────────────────────────────────────────
MOTION_SERVER = "http://127.0.0.1:50003"
OUTPUT_DIR    = "output"
POST_PLAY_WAIT = 3.0   # 재생 완료 후 응답까지 대기 (초)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _hash(text: str) -> str:
    h = hashlib.md5()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _generate_audio(text: str, pipe_tts, conf_tts, voice: int = 31, lang: str = "en") -> str:
    """TTS 추론 → WAV 저장 → 파일 경로 반환"""
    filename = _hash(text)
    out_path = f"{OUTPUT_DIR}/{filename}.wav"

    if os.path.exists(out_path):          # 동일 텍스트 캐시 활용
        return out_path

    phoneme_ids = text_to_sequence(text, [f"canvers_{lang}_cleaners"])
    text_arr    = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)

    inputs = {
        "input":         text_arr,
        "input_lengths": np.array([text_arr.shape[1]], dtype=np.int64),
        "scales":        np.array([0.667, 1.0, 0.8],  dtype=np.float16),
        "sid":           np.array([int(voice)],        dtype=np.int64),
    }

    result = pipe_tts(inputs)
    audio  = list(result.values())[0].squeeze((0, 1))
    write(data=audio, rate=conf_tts.data.sampling_rate, filename=out_path)
    return out_path


async def _fire_motion(motion: str) -> None:
    """모션 서버에 비동기 POST — 실패해도 slide 흐름에 영향 없음"""
    url  = f"{MOTION_SERVER}/motions/run"
    body = {"filename": f"{motion}.json"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
    except Exception as e:
        # 모션 실패는 로그만 남기고 무시
        print(f"[slide] motion '{motion}' failed: {e}")


async def _play_audio(file_path: str) -> None:
    """비동기 서브프로세스로 음성 재생 — 완료까지 await"""
    proc = await asyncio.create_subprocess_exec(
        "play", file_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()


def create_slide_router(pipe_tts, conf_tts) -> APIRouter:
    """
    main.py 에서 호출:
        from slide import create_slide_router
        app.include_router(create_slide_router(pipe_tts, conf_tts))
    """
    router = APIRouter()

    @router.get("/slide", summary="슬라이드 페이지 알림 + TTS 재생 + 모션 실행")
    async def slide(
        page:   int  = 1,
        text:   str  = "",
        motion: str  = "",
        voice:  int  = 31,
        lang:   str  = "en",
    ):
        start = t.time()

        # ── 1. TTS 생성 (CPU 작업 → 별도 스레드) ───────────────
        if text.strip():
            audio_path = await asyncio.to_thread(
                _generate_audio, text, pipe_tts, conf_tts, voice, lang
            )
        else:
            audio_path = None

        # ── 2. 모션 fire-and-forget (TTS 재생과 병렬) ───────────
        if motion.strip():
            asyncio.ensure_future(_fire_motion(motion.strip()))

        # ── 3. 음성 재생 (완료까지 대기) ────────────────────────
        if audio_path:
            await _play_audio(audio_path)

        # ── 4. 재생 완료 후 3초 대기 ────────────────────────────
        await asyncio.sleep(POST_PLAY_WAIT)

        elapsed = round(t.time() - start, 3)
        return {
            "result":  True,
            "page":    page,
            "motion":  motion or None,
            "elapsed": elapsed,
        }

    return router
