"""Load cached segments, classify with the LLM, evaluate vs ground truth."""
import json
import sys
import time
from pathlib import Path

from nc_detection.classify import classify_segments
from nc_detection.eval import binary_metrics, load_gt_ad_intervals, load_gt_duration
from nc_detection.segment import EnrichedSegment, Taxonomy

video_stem = sys.argv[1] if len(sys.argv) > 1 else "test_001"

work = Path("work")
segs_path = work / f"{video_stem}.segments.json"
info = Path("sample_videos/csci576/video_info") / f"{video_stem}.json"

segments = []
for s in json.loads(segs_path.read_text()):
    segments.append(
        EnrichedSegment(
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
    )

print(f"Loaded {len(segments)} segments from {segs_path}")

t0 = time.time()
classify_segments(segments)
elapsed = time.time() - t0
print(f"Classified in {elapsed:.1f}s ({elapsed/len(segments):.1f}s/seg)")

# Distribution
from collections import Counter
dist = Counter(s.label.value for s in segments)
print("Label distribution:", dict(dist))

# Save with classifications
out_path = work / f"{video_stem}.segments.classified.json"
out_path.write_text(json.dumps([s.to_dict() for s in segments], indent=2))
print(f"Wrote {out_path}")

# Eval
gt_ads = load_gt_ad_intervals(info)
duration = load_gt_duration(info)
m = binary_metrics(segments, gt_ads, duration)
print(
    f"\nLLM metrics: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  "
    f"tp={m['tp_sec']} fp={m['fp_sec']} fn={m['fn_sec']} ad_sec={m['support_ad_sec']}"
)
