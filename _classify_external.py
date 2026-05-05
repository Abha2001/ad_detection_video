"""Classify external (non-corpus) videos. No ground truth, just dump label distribution."""
import json
import time
from collections import Counter
from pathlib import Path

from nc_detection.classify import classify_segments
from nc_detection.segment import EnrichedSegment, Taxonomy

WORK = Path("work_external")


def load_segments(path):
    out = []
    for s in json.loads(path.read_text()):
        out.append(EnrichedSegment(
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
        ))
    return out


def main():
    for segs_path in sorted(WORK.glob("*.segments.json")):
        stem = segs_path.name.replace(".segments.json", "")
        segments = load_segments(segs_path)
        fp_hits = sum(1 for s in segments if s.fingerprint_hit)

        t0 = time.time()
        classify_segments(segments)
        elapsed = time.time() - t0

        out_path = WORK / f"{stem}.segments.classified.json"
        out_path.write_text(json.dumps([s.to_dict() for s in segments], indent=2))

        dist = Counter(s.label.value for s in segments)
        total = sum((s.end_sec - s.start_sec) for s in segments)
        nc_sec = sum((s.end_sec - s.start_sec) for s in segments
                     if s.label not in (Taxonomy.CORE_CONTENT, Taxonomy.UNKNOWN))
        print(f"\n{stem}: {len(segments)} segs, {fp_hits} fp_hits, {elapsed:.1f}s")
        print(f"  total {total:.0f}s, non-content {nc_sec:.0f}s ({100*nc_sec/total:.1f}%)")
        print(f"  distribution: {dict(dist)}")


if __name__ == "__main__":
    main()
