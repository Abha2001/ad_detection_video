"""Run LLM classifier across the entire corpus and print binary metrics."""
import argparse
import json
import time
from pathlib import Path

from nc_detection.classify import classify_segments
from nc_detection.eval import binary_metrics, load_gt_ad_intervals, load_gt_duration
from nc_detection.segment import EnrichedSegment, Taxonomy


def load_segments(path: Path):
    out = []
    for s in json.loads(path.read_text()):
        seg = EnrichedSegment(
            start_sec=s["start_sec"],
            end_sec=s["end_sec"],
            asr_text=s.get("asr_text", ""),
            ocr_text=s.get("ocr_text", ""),
            rms_mean=s.get("rms_mean", 0.0),
            rms_silence_ratio=s.get("rms_silence_ratio", 0.0),
            spectral_flatness_mean=s.get("spectral_flatness_mean", 0.0),
            motion_energy=s.get("motion_energy", 0.0),
            shot_count=s.get("shot_count", 0),
            fingerprint_hit=s.get("fingerprint_hit", False),
            position_norm=s.get("position_norm", 0.0),
        )
        out.append(seg)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=Path("work"))
    ap.add_argument("--info-dir", type=Path,
                    default=Path("sample_videos/csci576/video_info"))
    args = ap.parse_args()

    print(f"{'video':<14} {'segs':>5} {'fp_hits':>7} {'P':>6} {'R':>6} {'F1':>6} {'time':>7}")
    f1_total = 0.0
    n = 0
    for segs_path in sorted(args.work_dir.glob("test_*.segments.json")):
        stem = segs_path.name.replace(".segments.json", "")
        info_path = args.info_dir / f"{stem}.json"
        if not info_path.exists():
            continue

        segments = load_segments(segs_path)
        fp_hits = sum(1 for s in segments if s.fingerprint_hit)

        t0 = time.time()
        classify_segments(segments)
        elapsed = time.time() - t0

        # Save classified
        out_path = args.work_dir / f"{stem}.segments.classified.json"
        out_path.write_text(json.dumps([s.to_dict() for s in segments], indent=2))

        # Eval
        gt_ads = load_gt_ad_intervals(info_path)
        duration = load_gt_duration(info_path)
        m = binary_metrics(segments, gt_ads, duration)

        print(f"{stem:<14} {len(segments):>5} {fp_hits:>7} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} "
              f"{elapsed:>6.1f}s")
        f1_total += m["f1"]
        n += 1

    if n:
        print(f"{'mean':<14} {'':>5} {'':>7} {'':>6} {'':>6} {f1_total/n:>6.3f}")


if __name__ == "__main__":
    main()
