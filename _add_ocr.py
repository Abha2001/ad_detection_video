"""Add OCR text + topic_drift to an existing segments JSON without re-running
the whole pipeline. Pairs with classify_one.sbatch which then re-classifies."""
import json
import sys
import time
from pathlib import Path

from nc_detection.segment import EnrichedSegment
from nc_detection.ocr_features import compute_ocr_text
from nc_detection.topic import compute_topic_drift


def main() -> None:
    video_path = Path(sys.argv[1])
    segs_path = Path(sys.argv[2])

    data = json.loads(segs_path.read_text())
    segments = []
    for s in data:
        segments.append(EnrichedSegment(
            start_sec=s["start_sec"], end_sec=s["end_sec"],
            asr_text=s.get("asr_text", ""),
            ocr_text=s.get("ocr_text", ""),
            rms_mean=s.get("rms_mean", 0.0),
            rms_silence_ratio=s.get("rms_silence_ratio", 0.0),
            spectral_flatness_mean=s.get("spectral_flatness_mean", 0.0),
            motion_energy=s.get("motion_energy", 0.0),
            shot_count=s.get("shot_count", 0),
            fingerprint_hit=s.get("fingerprint_hit", False),
            position_norm=s.get("position_norm", 0.0),
            topic_drift=s.get("topic_drift", 0.0),
        ))

    print(f"Computing OCR on {len(segments)} segments...")
    t0 = time.time()
    compute_ocr_text(video_path, segments, max_per_seg=2)
    print(f"  OCR done in {time.time()-t0:.1f}s")

    # Recompute topic drift since ASR + OCR could affect topic salience
    if not any(s.topic_drift for s in segments):
        print("Computing topic drift...")
        t0 = time.time()
        compute_topic_drift(segments)
        print(f"  drift done in {time.time()-t0:.1f}s")

    segs_path.write_text(json.dumps([s.to_dict() for s in segments], indent=2))
    print(f"Wrote {segs_path}")

    # Show the segments with OCR text caught
    nonempty = [s for s in segments if s.ocr_text]
    print(f"\n{len(nonempty)} segments have OCR text. First 8:")
    for s in nonempty[:8]:
        print(f"  {s.start_sec:7.1f}-{s.end_sec:7.1f}s  ocr={s.ocr_text[:120]!r}")


if __name__ == "__main__":
    main()
