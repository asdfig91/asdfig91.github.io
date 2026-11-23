"""
CapCut용 고정확도 자막(SRT) 생성기

영상/음성 파일을 Whisper large-v3 (faster-whisper)로 받아쓰고,
읽기 좋은 길이로 나눠서 같은 폴더에 .srt 파일로 저장합니다.

사용법:
    python make_subtitles.py 영상1.mp4 [영상2.mp4 ...]
    python make_subtitles.py --check      (모델 다운로드 + GPU 확인만)
"""

import argparse
import glob
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
CONFIG_PATH = os.path.join(HERE, "config.txt")
WORDS_PATH = os.path.join(HERE, "words.txt")

DEFAULT_CONFIG = {
    "model": "large-v3",
    "language": "ko",
    "max_chars": "20",
    "max_seconds": "6",
    "remove_period": "yes",
    "device": "auto",
    "sync_fix": "yes",
    "remove_words": "어, 어어, 음, 으음",
    "end_pad": "-0.1",
}

# Whisper가 조용한 구간에서 지어내는 경우가 많은 문장들 (유튜브 학습 데이터 영향)
HALLUCINATION_PHRASES = [
    "시청해주셔서 감사합니다",
    "시청해 주셔서 감사합니다",
    "구독과 좋아요",
    "구독 좋아요",
    "다음 영상에서 만나요",
    "MBC 뉴스",
    "KBS 뉴스",
    "자막 제공",
    "자막 by",
]


# ---------------------------------------------------------------------------
# 설정 / 단어장 읽기
# ---------------------------------------------------------------------------

def is_yes(value: str) -> bool:
    return value.strip().lower() in ("yes", "y", "true", "1", "예", "네")


def read_config(path: str = CONFIG_PATH) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if not os.path.exists(path):
        return cfg
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            cfg[key.strip().lower()] = value.split("#", 1)[0].strip()
    return cfg


def read_words(path: str = WORDS_PATH):
    """단어장: 그냥 적은 줄은 인식 힌트, 'A -> B' 줄은 자동 바꾸기."""
    hints, replacements = [], []
    if not os.path.exists(path):
        return hints, replacements
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "->" in line:
                wrong, right = line.split("->", 1)
                wrong, right = wrong.strip(), right.strip()
                if wrong:
                    replacements.append((wrong, right))
                    # 맞는말이 한 단어일 때만 인식 힌트로도 쓴다. "산행을 갔고" 같은 구절을 힌트로 주면
                    # Whisper가 다른 곳까지 다르게 받아써서, 맞던 자막이 바뀔 수 있다.
                    if right and " " not in right and right not in hints:
                        hints.append(right)
            elif line not in hints:
                hints.append(line)
    return hints, replacements


# ---------------------------------------------------------------------------
# 자막 나누기 (단어 단위 타임스탬프 → 읽기 좋은 자막 줄)
# ---------------------------------------------------------------------------

@dataclass
class Word:
    start: float
    end: float
    text: str  # 앞 공백 포함 (Whisper 원본 형태)


@dataclass
class Cue:
    start: float
    end: float
    text: str


SENTENCE_END = re.compile(r"[.?!…。？！]$")
# Whisper가 마침표를 빠뜨려도 알아볼 수 있는 한국어 문장 끝맺음 (예: 갑시다, 있네요, 했다)
KOREAN_SENTENCE_END = re.compile(r"(니다|시다|세요|어요|아요|해요|에요|예요|네요|군요|죠|구나|었다|았다|했다|겠다)$")
# 줄을 나누기 좋은 자리: 쉼표, 그리고 말이 이어지는 연결 어미 (예: 있고, 갔는데, 했으니까)
SOFT_BREAK = re.compile(r"([,，、]|[가-힣]고|는데|니까|면서|지만|어서|아서|해서)$")


def ends_sentence(text: str) -> bool:
    text = text.strip()
    return bool(SENTENCE_END.search(text) or KOREAN_SENTENCE_END.search(text.rstrip(",，、")))


def join_words(words: List[Word]) -> str:
    return "".join(w.text for w in words).strip()


def fix_first_word(words: List[Word], max_gap: float = 0.6) -> None:
    """Whisper는 구간 첫 단어의 시작을 앞쪽 무음에 잘못 찍는 경우가 많다.
    1) 첫 단어가 둘째 단어와 멀리 떨어져 있으면 (예: 22.8초에 나오는 "한"을 19.6초로)
       첫 단어만 앞 문장 끝에 따로 뜨므로, 둘째 단어 바로 앞으로 옮긴다.
    2) 첫 단어가 앞 무음을 흡수해 길게 잡히면 (예: 두 글자 "랜덤"이 1.5초)
       자막이 1초 넘게 일찍 뜨므로, 글자 수에 맞는 길이로 줄인다."""
    if not words:
        return
    first = words[0]
    letters = len(re.sub(r"\W", "", first.text))
    max_duration = 0.25 + 0.15 * max(letters, 1)  # 보통 말 속도보다 넉넉하게
    if len(words) > 1:
        second = words[1]
        if second.start - first.end > max_gap and not SENTENCE_END.search(first.text.strip()):
            first.end = second.start
    if first.end - first.start > max_duration:
        first.start = first.end - max_duration


def replace_in_words(words: List[Word], replacements) -> List[Word]:
    """단어장의 '틀린말 -> 맞는말'을 단어 단위로 적용한다.
    "제로 마이야"처럼 여러 단어에 걸친 표기는 한 단어로 합쳐서 바꾼다 (시간은 첫 단어 시작~마지막 단어 끝)."""
    for wrong, right in replacements:
        if not wrong:
            continue
        search_from = 0
        while True:
            starts, text = [], ""
            for w in words:
                starts.append(len(text))
                text += w.text
            pos = text.find(wrong, search_from)
            if pos < 0:
                break
            end = pos + len(wrong)
            a = max(i for i, st in enumerate(starts) if st <= pos)
            b = max(i for i, st in enumerate(starts) if st < end)
            merged_text = text[starts[a]:pos] + right + text[end:starts[b] + len(words[b].text)]
            merged = Word(words[a].start, words[b].end, merged_text)
            words = words[:a] + ([merged] if merged_text.strip() else []) + words[b + 1:]
            search_from = starts[a] + len(merged_text)
    return words


def remove_filler_words(words: List[Word], fillers) -> List[Word]:
    """'어', '음' 처럼 따로 떨어진 말버릇을 뺀다. '어?' '어!' 같은 감탄은 남긴다."""
    if not fillers:
        return words
    return [w for w in words if w.text.strip().rstrip(".,…~") not in fillers
            or w.text.strip()[-1:] in "?!"]


def split_into_phrases(segments: List[List[Word]], max_gap: float = 0.8) -> List[List[Word]]:
    """문장 끝, Whisper 구간 끝, 긴 쉼에서 반드시 끊는다."""
    phrases = []
    for seg_words in segments:
        cur: List[Word] = []
        for w in seg_words:
            if cur and (w.start - cur[-1].end > max_gap or ends_sentence(cur[-1].text)):
                phrases.append(cur)
                cur = []
            cur.append(w)
        if cur:
            phrases.append(cur)

    # Whisper가 "일단." 처럼 짧은 단어 뒤에 마침표를 잘못 찍으면 한 단어짜리 자막이 생긴다.
    # 한국어 문장 끝맺음이 아닌 짧은 단어 하나에 마침표만 있고 바로 말이 이어지면 다음 문장에 붙인다.
    merged: List[List[Word]] = []
    for k, phrase in enumerate(phrases):
        text = phrase[0].text.strip()
        nxt = phrases[k + 1] if k + 1 < len(phrases) else None
        if (nxt is not None and len(phrase) == 1 and text.endswith(".") and not text.endswith("..")
                and len(text.rstrip(".")) <= 4 and not KOREAN_SENTENCE_END.search(text.rstrip("."))
                and nxt[0].start - phrase[0].end <= 0.5):
            phrase[0].text = phrase[0].text.rstrip(".")
            phrases[k + 1] = phrase + nxt
            continue
        merged.append(phrase)
    return merged


def balance_phrase(words: List[Word], max_chars: int, max_seconds: float) -> List[List[Word]]:
    """한 문장을 글자 수가 고르게, 쉼표 위치를 우선해서 여러 줄로 나눈다 (DP)."""
    n = len(words)
    if len(join_words(words)) <= max_chars and words[-1].end - words[0].start <= max_seconds:
        return [words]

    INF = float("inf")
    best = [INF] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        for i in range(j - 1, -1, -1):
            chunk = words[i:j]
            length = len(join_words(chunk))
            if length > max_chars and j - i > 1:
                break  # 더 앞에서 시작하면 더 길어질 뿐
            cost = (max_chars - length) ** 2 + 30  # 줄 수가 늘어나는 것에 대한 벌점
            if chunk[-1].end - chunk[0].start > max_seconds and j - i > 1:
                cost += 400
            if j < n:
                if SOFT_BREAK.search(chunk[-1].text.strip()):
                    cost -= 60  # 쉼표 뒤에서 끊기 선호
                gap = words[j].start - chunk[-1].end
                cost -= min(gap, 0.6) * 100  # 잠깐 쉬는 곳에서 끊기 선호
            if best[i] + cost < best[j]:
                best[j] = best[i] + cost
                back[j] = i

    chunks = []
    j = n
    while j > 0:
        i = back[j]
        chunks.append(words[i:j])
        j = i
    chunks.reverse()
    return chunks


def apply_text_rules(text: str, replacements, remove_period: bool) -> str:
    for wrong, right in replacements:
        text = text.replace(wrong, right)
    text = re.sub(r"\s+", " ", text).strip()
    if remove_period:
        text = re.sub(r"(?<![.])[.,。，]$", "", text).strip()
    return text


def build_cues(segments: List[List[Word]], max_chars: int, max_seconds: float,
               replacements=(), remove_period: bool = True, lead: float = 0.0,
               end_pad: float = 0.25, bridge_gap: float = 0.3, min_duration: float = 0.7) -> List[Cue]:
    cues: List[Cue] = []
    for phrase in split_into_phrases(segments):
        for chunk in balance_phrase(phrase, max_chars, max_seconds):
            text = apply_text_rules(join_words(chunk), replacements, remove_period)
            if text:
                cues.append(Cue(chunk[0].start, chunk[-1].end, text))

    # 자막은 늦게 뜨는 것보다 아주 살짝 먼저 뜨는 게 자연스럽다 (앞 자막의 말이 끝나기 전으로는 당기지 않음)
    if lead > 0:
        for k, cue in enumerate(cues):
            floor = cues[k - 1].end if k > 0 else 0.0
            cue.start = max(min(cue.start, max(cue.start - lead, floor)), 0.0)

    # 타이밍 다듬기: 말 끝난 뒤 조금 더 보여주고, 짧은 빈틈은 메워서 깜빡임 방지
    for k, cue in enumerate(cues):
        next_start = cues[k + 1].start if k + 1 < len(cues) else None
        end = cue.end + end_pad
        end = max(end, cue.start + min_duration)  # 너무 짧게 깜빡이지 않게
        if next_start is not None:
            if next_start - end < bridge_gap:
                end = next_start
            end = min(end, next_start)
        cue.end = max(end, cue.start + 0.05)
    return cues


def format_time(t: float) -> str:
    ms = int(round(max(t, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: List[Cue]) -> str:
    blocks = []
    for i, c in enumerate(cues, 1):
        blocks.append(f"{i}\n{format_time(c.start)} --> {format_time(c.end)}\n{c.text}\n")
    return "\n".join(blocks)


def unique_output_path(media_path: str) -> str:
    base = os.path.splitext(media_path)[0]
    path = base + ".srt"
    n = 2
    while os.path.exists(path):
        path = f"{base} ({n}).srt"
        n += 1
    return path


# ---------------------------------------------------------------------------
# 음성 인식
# ---------------------------------------------------------------------------

def add_nvidia_dll_paths():
    """pip로 설치한 CUDA 라이브러리(cuBLAS, cuDNN)를 Windows가 찾을 수 있게 한다."""
    if os.name != "nt":
        return
    import site
    roots = site.getsitepackages() + [site.getusersitepackages()]
    for root in roots:
        for bin_dir in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(bin_dir)
            except (OSError, AttributeError):
                pass


def ensure_model(model_name: str) -> str:
    """모델을 (처음 한 번만) 내려받고 폴더 경로를 돌려준다. 이미 받았으면 인터넷 없이 바로 사용."""
    if os.path.isdir(model_name):
        return model_name
    import huggingface_hub
    from faster_whisper.utils import _MODELS

    repo_id = model_name if "/" in model_name else _MODELS.get(model_name)
    if repo_id is None:
        raise RuntimeError(f"config.txt 의 model 값이 올바르지 않습니다: {model_name} "
                           "(large-v3 또는 large-v3-turbo 를 쓰세요)")
    patterns = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]
    os.makedirs(MODELS_DIR, exist_ok=True)
    try:
        path = huggingface_hub.snapshot_download(repo_id, allow_patterns=patterns,
                                                 cache_dir=MODELS_DIR, local_files_only=True)
        if os.path.isfile(os.path.join(path, "model.bin")):
            return path
    except Exception:  # noqa: BLE001 - 아직 안 받았으면 아래에서 내려받는다
        pass
    print("  처음 한 번 모델을 내려받습니다 (large-v3 는 약 3GB). 끝날 때까지 기다려 주세요...")
    try:
        return huggingface_hub.snapshot_download(repo_id, allow_patterns=patterns, cache_dir=MODELS_DIR)
    except Exception as e:  # noqa: BLE001
        first_line = (str(e).splitlines() or [type(e).__name__])[0][:200]
        raise RuntimeError("모델을 내려받지 못했습니다. 인터넷 연결과 디스크 공간(약 3GB)을 확인하고 "
                           f"다시 실행해 주세요.\n  ({first_line})") from e


def load_model(model_name: str, device_pref: str):
    import numpy as np
    from faster_whisper import WhisperModel

    model_path = ensure_model(model_name)  # 다운로드 문제와 GPU 문제를 따로 구분하기 위해 먼저 받는다

    attempts = []
    if device_pref in ("auto", "gpu", "cuda"):
        attempts += [("cuda", "float16"), ("cuda", "int8_float16")]
    if device_pref in ("auto", "cpu"):
        attempts += [("cpu", "int8")]

    last_error = None
    for device, compute_type in attempts:
        try:
            model = WhisperModel(model_path, device=device, compute_type=compute_type,
                                 cpu_threads=(os.cpu_count() or 4) if device == "cpu" else 0)
            # 실제로 한 번 돌려봐야 GPU 라이브러리 문제를 알 수 있다
            segs, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language="ko",
                                       beam_size=5, vad_filter=False)
            list(segs)
            label = "GPU (빠름)" if device == "cuda" else "CPU (GPU보다 느림)"
            print(f"  → {label} 로 실행합니다. [{device}/{compute_type}]")
            return model
        except Exception as e:  # noqa: BLE001 - 다음 방법으로 넘어간다
            last_error = e
            if device == "cuda" and compute_type == "float16":
                print(f"  (GPU 사용 불가: {(str(e).splitlines() or [''])[0][:120]})")
    raise RuntimeError(f"모델을 불러오지 못했습니다: {last_error}")


def looks_like_hallucination(seg) -> bool:
    text = seg.text.strip()
    if not text:
        return True
    for phrase in HALLUCINATION_PHRASES:
        if phrase in text and (seg.no_speech_prob > 0.3 or seg.avg_logprob < -0.8):
            return True
    return False


def transcribe_words(model, audio, cfg: dict, hints: List[str]) -> List[List[Word]]:
    """Whisper로 받아쓰고, Whisper 구간마다 단어 목록을 돌려준다."""
    language = cfg["language"].strip().lower()
    language = None if language in ("", "auto") else language
    duration = len(audio) / 16000

    segments, info = model.transcribe(
        audio,
        language=language,
        task="transcribe",
        beam_size=5,
        best_of=5,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        condition_on_previous_text=False,  # 같은 문장이 반복되는 오류 방지
        vad_filter=True,                   # 무음 구간 건너뛰기 → 지어내기 방지
        word_timestamps=True,              # 단어 단위 타이밍 → 자막 싱크 정확
        hallucination_silence_threshold=2.0,
        hotwords=", ".join(hints) if hints else None,
    )
    if language is None:
        print(f"  감지된 언어: {info.language} ({info.language_probability:.0%})")

    all_words: List[List[Word]] = []
    prev_text = None
    for seg in segments:
        pct = min(seg.end / duration, 1.0) * 100 if duration else 100
        print(f"  [{pct:5.1f}%] {format_time(seg.start)[:8]}  {seg.text.strip()}")
        if looks_like_hallucination(seg) or seg.text.strip() == prev_text:
            continue
        prev_text = seg.text.strip()
        words = [Word(w.start, w.end, w.word) for w in (seg.words or []) if w.word.strip()]
        fix_first_word(words)
        if words:
            all_words.append(words)
    return all_words


def transcribe_file(model, path: str, cfg: dict, hints: List[str], replacements,
                    aligner=None) -> Optional[str]:
    from faster_whisper import decode_audio

    max_chars = int(cfg["max_chars"])
    max_seconds = float(cfg["max_seconds"])
    remove_period = is_yes(cfg["remove_period"])

    print("  오디오를 읽는 중...")
    with open(path, "rb") as f:  # 파일 객체로 넘기면 한글 경로도 안전
        audio = decode_audio(f, sampling_rate=16000)
    duration = len(audio) / 16000
    if duration < 0.1:
        print("  ! 소리가 없는 파일입니다.")
        return None
    print(f"  오디오 길이: {format_time(duration)[:8]}")

    started = time.time()
    all_words = transcribe_words(model, audio, cfg, hints)
    report_missing_speech(all_words, duration)
    timing = {}
    if aligner is not None and all_words:
        from aligner import align_all
        print("  싱크 보정 중 (글자 단위로 음성에 다시 맞추기)...")
        try:
            fixed = align_all(aligner, audio, all_words)
            print(f"  싱크 보정: {fixed}/{len(all_words)}개 구간을 정밀하게 맞췄어요")
            # 보정된 시간은 정확하므로 여유를 거의 두지 않는다 (테스트: 끝 오차 평균 0.03~0.09초)
            # 자막은 말보다 0.1초 먼저 뜨고, 보정 모델이 단어 끝을 살짝 늦게 잡는 만큼(0.1초) 일찍 내린다
            try:
                end_pad = float(cfg.get("end_pad", "-0.1"))
            except ValueError:
                end_pad = -0.1
            timing = dict(lead=0.1, end_pad=end_pad, bridge_gap=0.1, min_duration=0.5)
        except Exception as e:  # noqa: BLE001 - 보정이 실패해도 자막은 만든다
            print(f"  (싱크 보정을 건너뜁니다: {e})")

    fillers = {x.strip() for x in cfg.get("remove_words", "").split(",") if x.strip()}
    all_words = [remove_filler_words(replace_in_words(ws, replacements), fillers) for ws in all_words]
    all_words = [ws for ws in all_words if ws]

    elapsed = time.time() - started
    cues = build_cues(all_words, max_chars, max_seconds, replacements, remove_period, **timing)
    if not cues:
        print("  ! 인식된 말이 없습니다.")
        return None
    out_path = unique_output_path(path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(to_srt(cues))
    print(f"  완료: 자막 {len(cues)}개, {elapsed:.0f}초 걸림")
    return out_path


def report_missing_speech(segments: List[List[Word]], duration: float, min_gap: float = 120.0) -> None:
    """목소리를 하나도 찾지 못한 긴 구간(기본 2분 이상)이 있으면 알려준다.
    예: 목소리만 뽑은 트랙이 영상 중간에 끝나서, 그 뒤가 음악뿐이거나 조용한 경우."""
    times = [(ws[0].start, ws[-1].end) for ws in segments if ws]
    edges = [0.0] + [t for pair in times for t in pair] + [duration]
    gaps = [(edges[k], edges[k + 1]) for k in range(0, len(edges) - 1, 2) if edges[k + 1] - edges[k] >= min_gap]
    for a, b in gaps:
        print(f"  ! {format_time(a)[:8]} ~ {format_time(b)[:8]} 사이에서는 목소리를 찾지 못했어요.")
        print("    이 부분의 오디오에 목소리가 들어 있는지 확인해 주세요 (트랙이 끝났거나 음소거됐을 수 있어요).")


def pick_files_with_dialog() -> List[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        files = filedialog.askopenfilenames(
            title="자막을 만들 영상(또는 음성) 파일을 고르세요",
            filetypes=[("영상/음성", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.mp3 *.wav *.m4a *.aac *.flac *.ogg"),
                       ("모든 파일", "*.*")],
        )
        root.destroy()
        return list(files)
    except Exception:  # noqa: BLE001
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description="CapCut용 고정확도 자막 생성기")
    parser.add_argument("files", nargs="*", help="영상 또는 음성 파일")
    parser.add_argument("--check", action="store_true", help="모델 다운로드와 GPU 확인만 한다")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    cfg = read_config()
    hints, replacements = read_words()

    files = args.files
    if not files and not args.check:
        files = pick_files_with_dialog()
        if not files:
            print("영상 파일을 '2-make-subtitles.bat' 위에 끌어다 놓아 주세요.")
            return 1

    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        for f in missing:
            print(f"파일을 찾을 수 없습니다: {f}")
        return 1

    add_nvidia_dll_paths()
    print(f"모델 준비 중: {cfg['model']}")
    try:
        model = load_model(cfg["model"], cfg["device"].strip().lower())
    except RuntimeError as e:
        print(f"\n! {e}")
        return 1

    aligner = None
    if is_yes(cfg.get("sync_fix", "yes")):
        try:
            from aligner import Aligner, ensure_align_model
            aligner = Aligner(ensure_align_model(MODELS_DIR))
        except Exception as e:  # noqa: BLE001 - 보정 없이도 자막은 만들 수 있다
            print(f"  (싱크 보정 모델을 준비하지 못해서 보정 없이 진행합니다: {(str(e).splitlines() or [''])[0][:150]})")

    if args.check:
        print("준비 완료!")
        return 0

    if hints:
        print(f"단어장 힌트 {len(hints)}개, 자동 바꾸기 {len(replacements)}개 적용")

    ok = 0
    for i, path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {os.path.basename(path)}")
        try:
            out = transcribe_file(model, path, cfg, hints, replacements, aligner)
            if out:
                print(f"  저장: {out}")
                ok += 1
        except Exception as e:  # noqa: BLE001 - 다음 파일은 계속 처리
            print(f"  ! 실패: {e}")

    print(f"\n끝! {ok}/{len(files)}개 파일의 자막을 만들었어요.")
    print("CapCut(PC)에서 '텍스트 → 로컬 자막 → 가져오기'로 .srt 파일을 불러오세요.")
    return 0 if ok == len(files) else 1


if __name__ == "__main__":
    sys.exit(main())
