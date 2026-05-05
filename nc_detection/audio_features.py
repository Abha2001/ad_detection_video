"""Audio feature extraction: ASR (faster-whisper) + RMS + spectral flatness.

ffmpeg is required to extract a 16kHz mono WAV from arbitrary video.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, List

import numpy as np

from .segment import EnrichedSegment

SR = 16000


def extract_wav(video_path: Path, wav_path: Path) -> None:
    """Extract mono 16kHz WAV from a video. Idempotent if wav_path already exists."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    if wav_path.exists():
        return
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", str(SR), "-vn",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def transcribe(wav_path: Path, model_size: str = "base") -> List[dict]:
    """Run faster-whisper on the wav with word-level timestamps.

    Each item: {"start": float, "end": float, "text": str, "words": [{"start", "end", "word"}]}.
    Uses GPU if available (cuda + float16), else CPU (int8).
    """
    from faster_whisper import WhisperModel

    try:
        import torch
        on_gpu = torch.cuda.is_available()
    except Exception:
        on_gpu = False

    device = "cuda" if on_gpu else "cpu"
    compute_type = "float16" if on_gpu else "int8"

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(
        str(wav_path), vad_filter=True, word_timestamps=True
    )
    out = []
    for s in segments:
        words = []
        if s.words:
            for w in s.words:
                words.append({
                    "start": float(w.start) if w.start is not None else float(s.start),
                    "end": float(w.end) if w.end is not None else float(s.end),
                    "word": w.word,
                })
        out.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text.strip(),
            "words": words,
        })
    return out


def compute_audio_stats(
    wav_path: Path,
    segments: Iterable[EnrichedSegment],
    silence_db: float = -40.0,
    hop_length: int = 512,
) -> None:
    """Fill rms_mean, rms_silence_ratio, spectral_flatness_mean in place."""
    import librosa

    y, sr = librosa.load(str(wav_path), sr=SR, mono=True)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-10, ref=np.max)
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    for seg in segments:
        mask = (times >= seg.start_sec) & (times < seg.end_sec)
        if not mask.any():
            continue
        seg.rms_mean = float(rms[mask].mean())
        seg.rms_silence_ratio = float((rms_db[mask] < silence_db).mean())
        seg.spectral_flatness_mean = float(flatness[mask].mean())


def assign_asr_to_segments(
    asr_segments: List[dict], segments: List[EnrichedSegment]
) -> None:
    """Assign ASR words to segments using per-word timestamps when available.

    Falls back to whole-segment overlap if word timestamps are missing — same
    behavior as before. With word timestamps a silent intro/bumper does NOT
    inherit speech text from a later Whisper segment that overlapped with it.
    """
    for asr in asr_segments:
        words = asr.get("words")
        if words:
            for w in words:
                w_mid = 0.5 * (w["start"] + w["end"])
                for seg in segments:
                    if seg.start_sec <= w_mid < seg.end_sec:
                        seg.asr_text = (seg.asr_text + " " + w["word"]).strip()
                        break
        else:
            a_s, a_e = asr["start"], asr["end"]
            text = asr["text"]
            for seg in segments:
                if a_e <= seg.start_sec or a_s >= seg.end_sec:
                    continue
                seg.asr_text = (seg.asr_text + " " + text).strip()
