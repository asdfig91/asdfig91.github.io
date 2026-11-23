"""
자막 싱크 보정 (강제 정렬)

Whisper가 받아쓴 문장을 한국어 음성 모델(wav2vec2, 글자 단위 CTC)로
음성에 글자 하나하나 다시 맞춰서, 단어의 시작/끝 시간을 정확하게 고친다.
Whisper 자체의 단어 타이밍은 말이 빠르거나 배경음이 있으면 흐트러지기 쉽다.

모델: kresnik/wav2vec2-large-xlsr-korean (Apache-2.0) 의 ONNX 변환본
"""

import copy
import json
import os
import re
from typing import List, Optional, Sequence

import numpy as np

REPO_ID = "XDcobra/wav2vec2-large-xlsr-53-korean-ONNX"
REVISION = "a6162331bac7e12c609ffe34c0a641b3efa6df62"  # 검증한 버전으로 고정
FILES = ["vocab.json", "onnx/model_int8.onnx"]
SAMPLE_RATE = 16000
FRAME_SEC = 0.02  # 모델 출력 한 칸 = 20ms


_SINO = "영일이삼사오육칠팔구"
_NATIVE_ONES = ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]
_NATIVE_TENS = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]
# 이 단위 앞의 1~99는 고유어로 읽는다 (두 시, 세 명, 한 번)
_NATIVE_COUNTERS = ("시간", "시", "개", "명", "살", "번", "마리", "잔", "대", "권", "장", "달", "가지",
                    "판", "병", "채", "벌", "그릇", "곳", "사람")


def _sino_under_10000(n: int) -> str:
    out = ""
    for value, name in ((1000, "천"), (100, "백"), (10, "십")):
        d, n = divmod(n, value)
        if d:
            out += ("" if d == 1 else _SINO[d]) + name
    return out + (_SINO[n] if n else "")


def _sino(n: int) -> str:
    if n == 0:
        return "영"
    if n >= 10 ** 8:
        return "".join(_SINO[int(c)] for c in str(n))
    man, rest = divmod(n, 10000)
    return (("" if man == 1 else _sino_under_10000(man)) + "만" if man else "") + _sino_under_10000(rest)


def _native(n: int) -> str:
    tens, ones = divmod(n, 10)
    return ("스무" if tens == 2 and ones == 0 else _NATIVE_TENS[tens]) + _NATIVE_ONES[ones]


def spoken_form(text: str) -> str:
    """정렬용 읽기: 숫자를 한글로 바꾼다 (15분 → 십오분, 2시가 → 두시가).
    정렬 모델은 한글만 알기 때문에, 숫자를 그대로 두면 그만큼 자막이 늦게 뜬다."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)

    def read(m):
        n = int(m.group())
        after = text[m.end():]
        if 1 <= n <= 99 and after.startswith(_NATIVE_COUNTERS):
            return _native(n)
        return _sino(n)
    return re.sub(r"\d+", read, text)


def ensure_align_model(models_dir: str) -> str:
    """정렬 모델을 (처음 한 번만) 내려받고 폴더 경로를 돌려준다."""
    import huggingface_hub

    try:
        path = huggingface_hub.snapshot_download(REPO_ID, revision=REVISION, allow_patterns=FILES,
                                                 cache_dir=models_dir, local_files_only=True)
        if all(os.path.isfile(os.path.join(path, f)) for f in FILES):
            return path
    except Exception:  # noqa: BLE001 - 아직 안 받았으면 아래에서 내려받는다
        pass
    print("  싱크 보정 모델을 처음 한 번 내려받습니다 (약 360MB)...")
    return huggingface_hub.snapshot_download(REPO_ID, revision=REVISION, allow_patterns=FILES,
                                             cache_dir=models_dir)


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def ctc_viterbi(logp: np.ndarray, tokens: Sequence[int], blank: int):
    """CTC 강제 정렬. 각 토큰이 차지하는 (시작 프레임, 끝 프레임) 과 평균 점수를 돌려준다."""
    T = logp.shape[0]
    L = len(tokens)
    S = 2 * L + 1
    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = tokens
    # s-2 에서 건너뛸 수 있는 자리: 토큰 자리이면서 앞 토큰과 다른 글자
    can_skip = np.zeros(S, dtype=bool)
    can_skip[3::2] = np.asarray(tokens[1:]) != np.asarray(tokens[:-1])

    NEG = -1e30
    alpha = np.full(S, NEG)
    alpha[0] = logp[0, ext[0]]
    if S > 1:
        alpha[1] = logp[0, ext[1]]
    back = np.zeros((T, S), dtype=np.int8)  # 0: 그대로, 1: 한 칸, 2: 두 칸
    for t in range(1, T):
        stay = alpha
        step1 = np.concatenate(([NEG], alpha[:-1]))
        step2 = np.where(can_skip, np.concatenate(([NEG, NEG], alpha[:-2])), NEG)
        stacked = np.stack([stay, step1, step2])
        choice = stacked.argmax(axis=0)
        alpha = stacked[choice, np.arange(S)] + logp[t, ext]
        back[t] = choice

    s = S - 1 if S == 1 or alpha[S - 1] >= alpha[S - 2] else S - 2
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])

    spans = [[-1, -1] for _ in range(L)]
    scores = []
    for t, s in enumerate(path):
        if s % 2 == 1:
            k = s // 2
            if spans[k][0] < 0:
                spans[k][0] = t
            spans[k][1] = t + 1
            scores.append(logp[t, ext[s]])
    mean_score = float(np.mean(scores)) if scores else -99.0
    return spans, mean_score


class Aligner:
    def __init__(self, model_dir: str):
        import onnxruntime as ort

        with open(os.path.join(model_dir, "vocab.json"), encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.blank = self.vocab["[PAD]"]
        self.sep = self.vocab.get("|")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(os.path.join(model_dir, "onnx", "model_int8.onnx"),
                                            sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def emissions(self, audio: np.ndarray) -> np.ndarray:
        x = audio.astype(np.float32)
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        logits = self.session.run(None, {self.input_name: x[np.newaxis, :]})[0][0]
        return _log_softmax(logits.astype(np.float32))

    def align_segment(self, audio: np.ndarray, words, win_start: float, win_end: float,
                      min_score: float = -4.0, end_cap: float = 0.3, keep_last_end: bool = True) -> bool:
        """한 구간의 단어들(Word 객체, start/end/text)을 제자리에서 고친다. 성공하면 True."""
        a = int(max(win_start, 0.0) * SAMPLE_RATE)
        b = int(min(win_end, len(audio) / SAMPLE_RATE) * SAMPLE_RATE)
        if b - a < SAMPLE_RATE // 4:
            return False

        tokens: List[int] = []
        owner: List[int] = []  # 각 토큰이 몇 번째 단어의 글자인지
        for wi, w in enumerate(words):
            chars = [self.vocab[c] for c in spoken_form(w.text.strip()) if c in self.vocab and c != "|"]
            if not chars:
                continue
            if tokens and self.sep is not None:
                tokens.append(self.sep)
                owner.append(-1)
            tokens.extend(chars)
            owner.extend([wi] * len(chars))
        if not tokens:
            return False

        logp = self.emissions(audio[a:b])
        if logp.shape[0] < len(tokens) + 1:
            return False
        spans, score = ctc_viterbi(logp, tokens, self.blank)
        if score < min_score:
            return False  # 정렬이 불확실하면 원래 타이밍을 유지

        # CTC는 글자가 '시작되는 순간'만 짧게 잡는다. 글자 끝은 다음 글자 시작까지 늘리되,
        # 쉬는 구간까지 먹지 않도록 end_cap 초까지만 늘린다.
        n_frames = logp.shape[0]
        cap = int(round(end_cap / FRAME_SEC))
        for k in range(len(spans)):
            nxt = spans[k + 1][0] if k + 1 < len(spans) else n_frames
            spans[k][1] = max(spans[k][1], min(nxt, spans[k][1] + cap))

        offset = a / SAMPLE_RATE
        original_last_end = words[-1].end
        new_times: List[Optional[List[float]]] = [None] * len(words)
        for (s, e), wi in zip(spans, owner):
            if wi < 0 or s < 0:
                continue
            st, en = offset + s * FRAME_SEC, offset + e * FRAME_SEC
            if new_times[wi] is None:
                new_times[wi] = [st, en]
            else:
                new_times[wi][1] = en

        # 정렬할 글자가 없는 단어(숫자, 영어 등)는 앞뒤 단어 사이에 끼워 넣는다
        known = [i for i, t in enumerate(new_times) if t is not None]
        if not known:
            return False
        for i in range(len(words)):
            if new_times[i] is not None:
                continue
            prev_i = max((k for k in known if k < i), default=None)
            next_i = min((k for k in known if k > i), default=None)
            lo = new_times[prev_i][1] if prev_i is not None else max(new_times[next_i][0] - 0.4, offset)
            hi = new_times[next_i][0] if next_i is not None else lo + 0.4
            missing = [k for k in range(prev_i + 1 if prev_i is not None else 0,
                                        next_i if next_i is not None else len(words))]
            step = max(hi - lo, 0.0) / len(missing)
            for j, k in enumerate(missing):
                new_times[k] = [lo + j * step, lo + (j + 1) * step]
            known = sorted(set(known) | set(missing))

        # 구간의 마지막 단어 끝은 Whisper 쪽이 더 정확했다 (테스트 기준). 너무 동떨어지지 않으면 그쪽을 쓴다.
        if keep_last_end and new_times[-1] is not None:
            last = new_times[-1]
            if last[1] < original_last_end <= last[1] + 1.0:
                last[1] = original_last_end

        # 구간 마지막 단어는 뒤따르는 글자가 없어서, 끝이 뒤쪽 배경음까지 늘어나기 쉽다.
        # 글자 수에 비해 너무 길면 자른다 (자막이 말보다 오래 남는 것을 막는다).
        if new_times[-1] is not None:
            last = new_times[-1]
            letters = len(re.sub(r"\W", "", spoken_form(words[-1].text)))
            last[1] = min(last[1], last[0] + 0.3 + 0.2 * max(letters, 1))

        for w, (st, en) in zip(words, new_times):
            w.start, w.end = st, max(en, st + 0.02)
        return True


def voice_level(audio: np.ndarray, t0: float, t1: float, peak: bool = False) -> float:
    """목소리 대역(300~3400Hz)의 세기(dB). 평균, 또는 peak=True 이면 가장 큰 값.
    녹음 크기와 상관없이 두 위치를 비교하는 데 쓴다."""
    t1 = max(t1, t0 + 0.1)
    levels = []
    for t in np.arange(t0, t1, 0.05):
        x = audio[int(t * SAMPLE_RATE):int((t + 0.1) * SAMPLE_RATE)]
        if len(x) < 64:
            continue
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        freqs = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
        levels.append(10 * np.log10(spec[(freqs > 300) & (freqs < 3400)].sum() + 1e-9))
    if not levels:
        return -99.0
    return float(np.max(levels) if peak else np.mean(levels))


def align_all(aligner: "Aligner", audio: np.ndarray, segments: List[list], pad: float = 0.5,
              shift_failed: bool = False, max_search: float = 6.0, wide_min_score: float = -15.0) -> int:
    """모든 Whisper 구간을 정렬한다. 앞뒤 구간과 겹치지 않는 범위에서 여유를 둔다. 고친 구간 수를 돌려준다.

    정렬에 실패한 구간(배경음이 크거나 받아쓰기가 틀린 곳)은 Whisper 타이밍을 그대로 둔다.
    shift_failed=True 이면 성공한 구간들이 옮겨진 만큼 같이 옮기지만, 실제 영상 확인에서
    이렇게 옮긴 자막이 오히려 늦게 뜨는 경우가 많아서 기본으로는 끈다."""
    total = len(audio) / SAMPLE_RATE
    ok = [False] * len(segments)
    shifts = []
    for i, words in enumerate(segments):
        if not words:
            continue
        prev_end = segments[i - 1][-1].end if i > 0 and segments[i - 1] else 0.0
        next_start = segments[i + 1][0].start if i + 1 < len(segments) and segments[i + 1] else total
        win_start = max(words[0].start - pad, prev_end, 0.0)
        win_end = min(words[-1].end + pad, max(next_start, words[-1].end), total)
        before = words[0].start
        if aligner.align_segment(audio, words, win_start, win_end):
            ok[i] = True
            shifts.append(words[0].start - before)

    # 2차: 실패한 구간은 Whisper가 시간을 몇 초씩 잘못 찍은 경우가 많다 (예: 짧은 말을 조용한 곳에).
    # 앞뒤 구간 사이의 빈 곳 전체(Whisper 위치에서 최대 max_search 초)에서 그 말을 다시 찾는다.
    for i, words in enumerate(segments):
        if ok[i] or not words:
            continue
        lo = segments[i - 1][-1].end if i > 0 and segments[i - 1] else 0.0
        hi = segments[i + 1][0].start if i + 1 < len(segments) and segments[i + 1] else total
        lo = max(lo, words[0].start - max_search, 0.0)
        hi = min(hi, words[-1].end + max_search, total)
        if hi - lo <= (words[-1].end - words[0].start) + 2 * pad + 0.2:
            continue  # 1차와 거의 같은 범위라 다시 찾을 필요가 없다
        trial = [copy.copy(w) for w in words]
        # 배경음에 묻힌 짧은 말은 점수가 낮게 나오므로 기준을 낮추는 대신,
        # 옮긴 자리에 실제로 목소리가 있는지(원래 자리보다 확실히 작지 않은지) 확인한다.
        if not aligner.align_segment(audio, trial, lo, hi, min_score=wide_min_score):
            continue
        # 말이 시작하는 부분(처음 0.4초)에 목소리가 있는지, 그 안의 가장 큰 소리로 비교한다.
        # 평균을 쓰면 짧은 말 뒤의 무음이나 늘어난 끝 단어 때문에 맞는 위치를 놓친다.
        before = voice_level(audio, words[0].start, words[0].start + 0.4, peak=True)
        after = voice_level(audio, trial[0].start, trial[0].start + 0.4, peak=True)
        if after < before - 6.0:
            continue  # 옮긴 자리가 원래 자리보다 확실히 조용하면 믿지 않는다
        orig_span = words[-1].end - words[0].start
        trial_span = trial[-1].end - trial[0].start
        if trial_span <= orig_span * 1.5 + 0.8:
            for w, t in zip(words, trial):
                w.start, w.end = t.start, t.end
        else:
            # 넓은 범위에서는 뒤쪽 단어가 엉뚱하게 멀리 붙을 수 있다 (예: "노크 소리"의 "소리"가 5초 뒤로).
            # 이럴 때는 첫 단어 위치만 믿고, 구간 전체를 Whisper의 단어 간격 그대로 옮긴다.
            shift = trial[0].start - words[0].start
            shift = min(shift, hi - words[-1].end) if shift > 0 else max(shift, lo - words[0].start)
            for w in words:
                w.start += shift
                w.end += shift
        ok[i] = True

    if shift_failed and len(shifts) >= 3:
        shift = float(np.median(shifts))
        if 0.05 <= abs(shift) <= 0.6:
            for i, words in enumerate(segments):
                if ok[i] or not words:
                    continue
                lo = segments[i - 1][-1].end if i > 0 and segments[i - 1] else 0.0
                hi = segments[i + 1][0].start if i + 1 < len(segments) and segments[i + 1] else total
                s = min(max(shift, lo - words[0].start), hi - words[-1].end) if shift > 0 else \
                    max(shift, lo - words[0].start)
                for w in words:
                    w.start += s
                    w.end += s
    return sum(ok)
