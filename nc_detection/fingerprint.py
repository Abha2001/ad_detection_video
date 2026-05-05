"""Audio fingerprinting for repeat / recap detection.

Builds a compact MFCC-based perceptual hash per segment, then cross-matches
segments against a database of fingerprints from other videos. Pure Python /
librosa — no chromaprint / acoustid C dependency.

A "match" sets EnrichedSegment.fingerprint_hit = True, which the LLM fuser
then sees as evidence for the recap class. Same mechanic catches reused
sponsor-read music beds across episodes (Brilliant outro, LTT WAN intro etc.).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .segment import EnrichedSegment

SR = 16000
HOP = 1024            # 16 ms hops; ~62 frames/sec
N_MFCC = 13
MIN_SEGMENT_SEC = 2.0  # ignore very short segments
MATCH_THRESHOLD = 0.85
WINDOW_SEC = 5.0       # sliding window length used for cross-correlation


def _mfcc(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    if y.size == 0:
        return np.zeros((N_MFCC, 0), dtype=np.float32)
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP).astype(np.float32)
    # Per-coefficient z-norm makes features invariant to overall loudness.
    mu = m.mean(axis=1, keepdims=True)
    sd = m.std(axis=1, keepdims=True) + 1e-6
    return (m - mu) / sd


def fingerprint_video(
    wav_path: Path, segments: List[EnrichedSegment]
) -> Dict[Tuple[float, float], np.ndarray]:
    """Compute a per-segment MFCC fingerprint. Keyed by (start_sec, end_sec)."""
    import librosa

    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    out: Dict[Tuple[float, float], np.ndarray] = {}
    for seg in segments:
        if seg.duration < MIN_SEGMENT_SEC:
            continue
        s = max(0, int(seg.start_sec * SR))
        e = min(len(y), int(seg.end_sec * SR))
        if e <= s:
            continue
        out[(seg.start_sec, seg.end_sec)] = _mfcc(y[s:e], SR)
    return out


def _windowed_match_score(target: np.ndarray, ref: np.ndarray) -> float:
    """Best-aligned cross-correlation score between two MFCC sequences.

    Slides a `WINDOW_SEC` window over the target and finds the best matching
    position in the reference. Returns a [0,1] similarity score.
    """
    if target.size == 0 or ref.size == 0:
        return 0.0

    win_frames = int(WINDOW_SEC * SR / HOP)
    if target.shape[1] < win_frames or ref.shape[1] < win_frames:
        # Either side too short for a full window — fall back to whatever fits.
        win_frames = min(target.shape[1], ref.shape[1])
        if win_frames < 16:
            return 0.0

    # Normalize the target window once.
    tw = target[:, :win_frames]
    tw_flat = tw.flatten()
    tw_norm = np.linalg.norm(tw_flat) + 1e-6

    best = 0.0
    # Slide over the reference.
    step = max(1, win_frames // 4)
    for i in range(0, ref.shape[1] - win_frames + 1, step):
        rw = ref[:, i:i + win_frames].flatten()
        score = float(np.dot(tw_flat, rw) / (tw_norm * (np.linalg.norm(rw) + 1e-6)))
        if score > best:
            best = score
    return best


class FingerprintDB:
    """A flat collection of (label, mfcc) entries. Label is for debugging only;
    we treat any match as a 'recap / reused content' hit."""

    def __init__(self) -> None:
        self.entries: List[Tuple[str, np.ndarray]] = []

    def add(self, label: str, mfcc: np.ndarray) -> None:
        self.entries.append((label, mfcc))

    def add_video(
        self, video_label: str, fingerprints: Dict[Tuple[float, float], np.ndarray]
    ) -> None:
        for (s, e), m in fingerprints.items():
            self.entries.append((f"{video_label}@{s:.1f}-{e:.1f}", m))

    def best_match(self, target: np.ndarray, exclude_label_prefix: str = "") -> Tuple[float, str]:
        best_score = 0.0
        best_label = ""
        for label, ref in self.entries:
            if exclude_label_prefix and label.startswith(exclude_label_prefix):
                continue
            score = _windowed_match_score(target, ref)
            if score > best_score:
                best_score = score
                best_label = label
        return best_score, best_label

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.entries, f)

    @classmethod
    def load(cls, path: Path) -> "FingerprintDB":
        db = cls()
        with open(path, "rb") as f:
            db.entries = pickle.load(f)
        return db


def mark_fingerprint_hits(
    segments: List[EnrichedSegment],
    fingerprints: Dict[Tuple[float, float], np.ndarray],
    db: FingerprintDB,
    own_label: str,
    threshold: float = MATCH_THRESHOLD,
) -> None:
    """Set seg.fingerprint_hit on segments whose MFCC matches any DB entry
    that didn't come from the same video (own_label is excluded by prefix)."""
    for seg in segments:
        m = fingerprints.get((seg.start_sec, seg.end_sec))
        if m is None:
            continue
        score, _ = db.best_match(m, exclude_label_prefix=own_label)
        if score >= threshold:
            seg.fingerprint_hit = True
