"""Run the full pipeline on the corpus and print binary ad-detection metrics.

Usage:
    python -m nc_detection.bench \
        --corpus-dir /scratch1/abhajha/csci576/sample_videos/csci576 \
        --work-dir   /scratch1/abhajha/csci576/work \
        --whisper-model base \
        [--classify]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .boundary import detect_boundaries_sec
from .eval import binary_metrics, load_gt_ad_intervals, load_gt_duration
from .run import run as run_pipeline
from .segment import EnrichedSegment, Taxonomy


def label_segments_by_overlap(segments, gt_ads) -> None:
    """Fallback labeling when --classify is off: mark a segment as filler if
    it overlaps any GT ad interval. Useful as an upper-bound sanity check."""
    for seg in segments:
        for s, e in gt_ads:
            if seg.end_sec > s and seg.start_sec < e:
                seg.label = Taxonomy.FILLER
                break
        else:
            seg.label = Taxonomy.CORE_CONTENT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--whisper-model", default="base")
    ap.add_argument("--classify", action="store_true")
    ap.add_argument(
        "--detector",
        default="frame_diff",
        help="Detector key (frame_diff | histogram | edge | black_frame | transnet_clip)",
    )
    ap.add_argument("--oracle", action="store_true",
                    help="Skip classifier; label segments by GT-overlap (upper bound)")
    args = ap.parse_args()

    videos_dir = args.corpus_dir / "videos_with_ads"
    info_dir = args.corpus_dir / "video_info"
    args.work_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'video':<12} {'P':>6} {'R':>6} {'F1':>6} {'tp':>6} {'fp':>6} {'fn':>6}")

    f1_total = 0.0
    n = 0
    for video_path in sorted(videos_dir.glob("*.mp4")):
        info_path = info_dir / (video_path.stem + ".json")
        if not info_path.exists():
            print(f"{video_path.stem}: no GT, skipping")
            continue

        boundaries = detect_boundaries_sec(video_path, args.detector)
        out_path = run_pipeline(
            video_path,
            args.work_dir,
            boundaries_sec=boundaries,
            whisper_model=args.whisper_model,
            classify=args.classify and not args.oracle,
        )

        segments_data = json.loads(out_path.read_text())
        segments = [
            EnrichedSegment(
                start_sec=s["start_sec"],
                end_sec=s["end_sec"],
                label=Taxonomy(s["label"]),
            )
            for s in segments_data
        ]

        gt_ads = load_gt_ad_intervals(info_path)
        duration = load_gt_duration(info_path)

        if args.oracle:
            label_segments_by_overlap(segments, gt_ads)

        m = binary_metrics(segments, gt_ads, duration)
        print(
            f"{video_path.stem:<12} "
            f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} "
            f"{m['tp_sec']:>6} {m['fp_sec']:>6} {m['fn_sec']:>6}"
        )
        f1_total += m["f1"]
        n += 1

    if n > 0:
        print(f"{'mean':<12} {'':>6} {'':>6} {f1_total/n:>6.3f}")


if __name__ == "__main__":
    main()
