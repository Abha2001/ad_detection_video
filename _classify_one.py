"""Classify a single video's segments. Uses the (now updated) classify.py."""
import json
import sys
import time
from collections import Counter
from pathlib import Path

from nc_detection.classify import classify_segments
from nc_detection.eval import binary_metrics, load_gt_ad_intervals, load_gt_duration
from nc_detection.segment import EnrichedSegment, Taxonomy

stem = sys.argv[1] if len(sys.argv) > 1 else "test_001"
work = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("work")
info_dir = Path("sample_videos/csci576/video_info")

segs_path = work / f"{stem}.segments.json"
print(f"loading {segs_path}")
data = json.loads(segs_path.read_text())
segments = []
for s in data:
    seg = EnrichedSegment(
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
    )
    segments.append(seg)

print(f"{len(segments)} segments")
t0 = time.time()
classify_segments(segments, batch_size=8)
print(f"classified in {time.time()-t0:.1f}s")

out_path = work / f"{stem}.segments.classified.json"
out_path.write_text(json.dumps([s.to_dict() for s in segments], indent=2))

dist = Counter(s.label.value for s in segments)
print(f"distribution: {dict(dist)}")

info_path = info_dir / f"{stem}.json"
if info_path.exists():
    gt = load_gt_ad_intervals(info_path)
    dur = load_gt_duration(info_path)
    m = binary_metrics(segments, gt, dur)
    print(f"binary metrics: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

print()
print("Non-core_content segments:")
for s in segments:
    if s.label != Taxonomy.CORE_CONTENT:
        print(f"  {s.start_sec:7.1f}-{s.end_sec:7.1f}s  {s.label.value:<13} c={s.confidence:.2f}")
        print(f"     asr: {s.asr_text[:90]!r}")
        print(f"     why: {s.rationale[:120]}")
