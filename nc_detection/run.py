"""End-to-end driver: video -> EnrichedSegments JSON.

Usage:
    python -m nc_detection.run path/to/video.mp4 \
        --work-dir ./work \
        --boundaries cuts.json \
        --whisper-model base
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import cv2

from .audio_features import (
    assign_asr_to_segments,
    compute_audio_stats,
    extract_wav,
    transcribe,
)
from .segment import EnrichedSegment, segments_from_boundaries
from .visual_features import assign_shot_counts, compute_motion_energy


def get_video_duration(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return (n / fps) if fps and fps > 0 else 0.0


def run(
    video_path: Path,
    work_dir: Path,
    boundaries_sec: Optional[List[float]] = None,
    whisper_model: str = "base",
    classify: bool = False,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    duration = get_video_duration(video_path)
    boundaries = boundaries_sec or []

    segments = segments_from_boundaries(boundaries, duration)

    wav_path = work_dir / (video_path.stem + ".wav")
    extract_wav(video_path, wav_path)

    asr = transcribe(wav_path, model_size=whisper_model)
    assign_asr_to_segments(asr, segments)
    compute_audio_stats(wav_path, segments)
    compute_motion_energy(video_path, segments)
    assign_shot_counts(boundaries, segments)

    # OCR for on-screen branding (URLs, sponsor logos, "Subscribe", etc.).
    # Only run inline if a GPU is available — on CPU EasyOCR is slow enough
    # that we'd rather skip and let a separate GPU sbatch (add_ocr.sbatch)
    # populate ocr_text afterwards.
    try:
        import torch
        if torch.cuda.is_available():
            from .ocr_features import compute_ocr_text
            compute_ocr_text(video_path, segments)
    except ImportError:
        pass

    # Topic drift: catches off-topic inserts (synthetic ads that are content-
    # shaped but topically unrelated to the host video).
    try:
        from .topic import compute_topic_drift
        compute_topic_drift(segments)
    except ImportError:
        pass  # sentence-transformers not installed — skip silently

    if classify:
        from .classify import classify_segments
        classify_segments(segments)

    out_path = work_dir / (video_path.stem + ".segments.json")
    out_path.write_text(json.dumps([s.to_dict() for s in segments], indent=2))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--work-dir", type=Path, default=Path("./work"))
    ap.add_argument(
        "--boundaries",
        type=Path,
        help="JSON file containing a list of cut timestamps in seconds",
    )
    ap.add_argument("--whisper-model", default="base")
    ap.add_argument("--classify", action="store_true",
                    help="Run LLM classifier (requires ANTHROPIC_API_KEY)")
    args = ap.parse_args()

    boundaries = (
        json.loads(args.boundaries.read_text()) if args.boundaries else None
    )
    out = run(args.video, args.work_dir, boundaries, args.whisper_model,
              classify=args.classify)
    print(out)


if __name__ == "__main__":
    main()
