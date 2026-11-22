"""
문제 확인용 샘플 만들기

오디오/영상 파일의 앞부분(기본 3분)을 작은 MP3(모노 16kHz)로 저장합니다.
음성 인식은 원래 모노 16kHz로 처리하므로, 줄여도 자막 결과는 똑같습니다.

사용법:
    python make_sample.py 파일.mp3 [분]
"""

import os
import sys

import av
import numpy as np
from faster_whisper import decode_audio

SAMPLE_RATE = 16000


def write_audio(audio: np.ndarray, out_path: str, codec: str) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with av.open(out_path, "w") as container:
        stream = container.add_stream(codec, rate=SAMPLE_RATE, layout="mono")
        stream.bit_rate = 48000
        frame = av.AudioFrame.from_ndarray(pcm[np.newaxis, :], format="s16", layout="mono")
        frame.sample_rate = SAMPLE_RATE
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def make_sample(path: str, minutes: float) -> str:
    with open(path, "rb") as f:  # 파일 객체로 넘기면 한글 경로도 안전
        audio = decode_audio(f, sampling_rate=SAMPLE_RATE)
    audio = audio[: int(minutes * 60 * SAMPLE_RATE)]
    base = os.path.splitext(path)[0]
    last_error = None
    for codec, ext in (("libmp3lame", ".mp3"), ("aac", ".m4a")):
        out_path = f"{base}_sample{ext}"
        try:
            write_audio(audio, out_path, codec)
            return out_path
        except Exception as e:  # noqa: BLE001 - 다음 형식으로 시도
            last_error = e
            if os.path.exists(out_path):
                os.remove(out_path)
    raise RuntimeError(f"샘플을 만들지 못했습니다: {last_error}")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("오디오 파일을 '3-make-sample.bat' 위에 끌어다 놓아 주세요.")
        return 1
    minutes = 3.0
    if len(args) > 1:
        try:
            minutes = float(args[-1])
            args = args[:-1]
        except ValueError:
            pass
    ok = True
    for path in args:
        if not os.path.isfile(path):
            print(f"파일을 찾을 수 없습니다: {path}")
            ok = False
            continue
        print(f"{os.path.basename(path)} → 앞 {minutes:g}분을 작은 파일로 만드는 중...")
        try:
            out = make_sample(path, minutes)
            size_mb = os.path.getsize(out) / 1024 / 1024
            print(f"  완료: {out} ({size_mb:.1f}MB)")
        except Exception as e:  # noqa: BLE001
            print(f"  ! 실패: {e}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
