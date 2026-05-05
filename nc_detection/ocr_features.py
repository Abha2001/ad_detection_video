"""OCR for on-screen branding text.

Samples a couple of frames per segment, runs EasyOCR on each, and stores the
detected text in seg.ocr_text. Catches signals that don't surface in ASR:

- Sponsor URLs and brand names (e.g. "brilliant.org/veritasium", "use code XYZ")
- Channel/show watermarks and title cards
- "Subscribe" / "Patreon" CTAs in self-promo segments
- "Starting Soon" overlays on stream pre-rolls
- Lower-thirds with speaker info

Runs on GPU when available. Resizes frames to 720p before OCR to keep it
fast — branding text is large enough to survive that downscale.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from .segment import EnrichedSegment

_READER = None


def _get_reader():
    global _READER
    if _READER is not None:
        return _READER
    import easyocr
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        gpu = False
    _READER = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    return _READER


def _sample_times(seg: EnrichedSegment, max_per_seg: int = 2) -> List[float]:
    """Pick representative timestamps inside a segment to OCR."""
    if seg.duration < 1.5:
        return [(seg.start_sec + seg.end_sec) / 2]
    if seg.duration < 5 or max_per_seg == 1:
        return [seg.start_sec + 0.5, (seg.start_sec + seg.end_sec) / 2][:max_per_seg]
    return [
        seg.start_sec + 0.5,
        (seg.start_sec + seg.end_sec) / 2,
        seg.end_sec - 0.5,
    ][:max_per_seg]


def _ocr_text(reader, frame: np.ndarray) -> str:
    """Run OCR on a frame, return space-joined detected text."""
    h, w = frame.shape[:2]
    if w > 720:
        scale = 720 / w
        frame = cv2.resize(frame, (720, int(h * scale)))
    try:
        results = reader.readtext(frame, detail=0, paragraph=False)
    except Exception:
        return ""
    return " ".join(r.strip() for r in results if r and r.strip())


def compute_ocr_text(
    video_path: Path, segments: List[EnrichedSegment], max_per_seg: int = 2
) -> None:
    """Fill seg.ocr_text in place by sampling and OCR'ing frames per segment."""
    if not segments:
        return
    reader = _get_reader()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Build the sampling plan in time order so we sweep the file once.
    plan = []
    for i, seg in enumerate(segments):
        for t in _sample_times(seg, max_per_seg):
            plan.append((t, i))
    plan.sort()

    next_idx = 0
    seg_text: dict[int, list[str]] = {i: [] for i in range(len(segments))}

    for target_t, seg_idx in plan:
        target_frame = int(target_t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            continue
        text = _ocr_text(reader, frame)
        if text:
            seg_text[seg_idx].append(text)
    cap.release()

    for i, seg in enumerate(segments):
        # Deduplicate identical OCR strings across frames in same segment.
        unique = list(dict.fromkeys(seg_text[i]))
        seg.ocr_text = " | ".join(unique)
