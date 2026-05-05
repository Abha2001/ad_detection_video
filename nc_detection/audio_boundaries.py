"""Audio-based segmentation cuts.

Visual frame-difference boundary detection misses transitions that are
visually smooth but audibly distinct — synthetic ad insertions with crossfades,
podcast topic shifts, music-bed entries on sponsor reads. Detecting silent
intervals in the audio (RMS dips) gives us those boundaries.

Returns the *midpoints* of silent intervals as candidate segment boundaries.
Caller merges these with visual boundaries to get the final cut list.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

SR = 16000


def detect_silence_boundaries(
    wav_path: Path,
    min_silence_ms: int = 300,
    silence_db: float = -35.0,
    hop_length: int = 512,
) -> List[float]:
    """Return list of timestamps (seconds) at the midpoint of each silent
    interval >= min_silence_ms. Each is a candidate segment boundary."""
    import librosa

    y, sr = librosa.load(str(wav_path), sr=SR, mono=True)
    if y.size == 0:
        return []

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-10, ref=np.max)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    is_silent = rms_db < silence_db
    min_silence_frames = int((min_silence_ms / 1000) * sr / hop_length)

    boundaries: List[float] = []
    in_silence = False
    start_idx = 0
    for i, s in enumerate(is_silent):
        if s and not in_silence:
            in_silence = True
            start_idx = i
        elif not s and in_silence:
            in_silence = False
            length = i - start_idx
            if length >= min_silence_frames:
                mid = times[(start_idx + i) // 2]
                boundaries.append(float(mid))
    if in_silence:
        length = len(is_silent) - start_idx
        if length >= min_silence_frames:
            mid = times[(start_idx + len(is_silent) - 1) // 2]
            boundaries.append(float(mid))

    return boundaries


def merge_boundaries(
    visual: List[float], audio: List[float], merge_window_sec: float = 0.5
) -> List[float]:
    """Combine visual and audio boundaries, deduplicating cuts within
    merge_window_sec of each other (visual takes precedence)."""
    out = sorted(set(visual))
    visual_set_sorted = sorted(visual)
    j = 0
    for a in sorted(audio):
        # Check if any visual boundary already exists near this audio boundary
        while j < len(visual_set_sorted) and visual_set_sorted[j] < a - merge_window_sec:
            j += 1
        nearby = (
            j < len(visual_set_sorted)
            and abs(visual_set_sorted[j] - a) <= merge_window_sec
        )
        if not nearby:
            out.append(a)
    return sorted(out)
