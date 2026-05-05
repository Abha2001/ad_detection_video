"""Visual feature extraction: motion energy + per-segment shot counts.

Optical flow is computed on a downsampled grayscale stream sampled at sample_fps,
which keeps a 1-hour video tractable on CPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .segment import EnrichedSegment


def compute_motion_energy(
    video_path: Path,
    segments: List[EnrichedSegment],
    sample_fps: float = 2.0,
    down_size: tuple = (160, 90),
) -> None:
    """Fill motion_energy in place — mean optical-flow magnitude per segment."""
    if not segments:
        return
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))

    buckets: Dict[int, List[float]] = {i: [] for i in range(len(segments))}

    def seg_idx_for(t: float, hint: int) -> int:
        i = hint
        while i < len(segments) - 1 and t >= segments[i].end_sec:
            i += 1
        return i

    prev_gray = None
    frame_idx = 0
    seg_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            t = frame_idx / fps
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), down_size)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 2, 15, 2, 5, 1.1, 0
                )
                mag = float(np.linalg.norm(flow, axis=2).mean())
                seg_i = seg_idx_for(t, seg_i)
                if segments[seg_i].start_sec <= t < segments[seg_i].end_sec:
                    buckets[seg_i].append(mag)
            prev_gray = gray
        frame_idx += 1
    cap.release()

    for i, seg in enumerate(segments):
        if buckets[i]:
            seg.motion_energy = float(np.mean(buckets[i]))


def assign_shot_counts(
    fine_boundaries_sec: List[float], segments: List[EnrichedSegment]
) -> None:
    """Count fine-grained shot boundaries falling within each (possibly merged) segment."""
    for seg in segments:
        seg.shot_count = sum(
            1 for b in fine_boundaries_sec if seg.start_sec <= b < seg.end_sec
        )
