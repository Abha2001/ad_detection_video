"""End-to-end demo script: video -> classified segments JSON in one go.

Runs the full pipeline on a GPU node (Whisper + LLM both on GPU). Designed to
fit comfortably within a 10-minute live-demo window for a fresh 20-30 min video.
"""
import argparse
import json
import time
from pathlib import Path

from nc_detection.audio_boundaries import detect_silence_boundaries, merge_boundaries
from nc_detection.audio_features import extract_wav
from nc_detection.boundary import detect_boundaries_sec
from nc_detection.classify import classify_segments
from nc_detection.run import run as run_pipeline
from nc_detection.segment import EnrichedSegment, Taxonomy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--work-dir", type=Path, default=Path("work_demo"))
    ap.add_argument("--whisper-model", default="base")
    ap.add_argument("--adaptive-k", type=float, default=1.0)
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    print(f"[+0.0s] starting demo on {args.video.name}")
    t = time.time()
    visual_b = detect_boundaries_sec(
        args.video, detector_key="frame_diff", sample_every=5, adaptive_k=args.adaptive_k
    )
    wav_path = args.work_dir / (args.video.stem + ".wav")
    extract_wav(args.video, wav_path)
    audio_b = detect_silence_boundaries(wav_path)
    boundaries = merge_boundaries(visual_b, audio_b)
    print(f"[+{time.time()-start:5.1f}s] boundaries: {len(visual_b)} visual + {len(audio_b)} audio -> {len(boundaries)} merged ({time.time()-t:.1f}s)")

    t = time.time()
    out = run_pipeline(
        args.video,
        args.work_dir,
        boundaries_sec=boundaries,
        whisper_model=args.whisper_model,
        classify=False,  # we'll classify here so we can time it separately
    )
    print(f"[+{time.time()-start:5.1f}s] features extracted ({time.time()-t:.1f}s)")

    segs_data = json.loads(out.read_text())
    segments = [
        EnrichedSegment(
            start_sec=s["start_sec"], end_sec=s["end_sec"],
            asr_text=s.get("asr_text", ""), ocr_text=s.get("ocr_text", ""),
            rms_mean=s.get("rms_mean", 0.0),
            rms_silence_ratio=s.get("rms_silence_ratio", 0.0),
            spectral_flatness_mean=s.get("spectral_flatness_mean", 0.0),
            motion_energy=s.get("motion_energy", 0.0),
            shot_count=s.get("shot_count", 0),
            fingerprint_hit=s.get("fingerprint_hit", False),
            position_norm=s.get("position_norm", 0.0),
        )
        for s in segs_data
    ]

    t = time.time()
    classify_segments(segments, batch_size=8)
    print(f"[+{time.time()-start:5.1f}s] classified {len(segments)} segments ({time.time()-t:.1f}s)")

    out_path = args.work_dir / (args.video.stem + ".segments.classified.json")
    out_path.write_text(json.dumps([s.to_dict() for s in segments], indent=2))
    print(f"[+{time.time()-start:5.1f}s] wrote {out_path}")

    # Post-process to enforce position-based rules the LLM sometimes misses.
    import subprocess
    subprocess.run(
        ["python", "_postprocess_labels.py", str(out_path)],
        check=False, cwd=str(Path.cwd()),
    )
    print(f"[+{time.time()-start:5.1f}s] post-processed")

    from collections import Counter
    dist = Counter(s.label.value for s in segments)
    total = sum(s.duration for s in segments)
    nc_sec = sum(s.duration for s in segments if s.label not in (Taxonomy.CORE_CONTENT, Taxonomy.UNKNOWN))
    print()
    print(f"=== {args.video.stem} ===")
    print(f"  total {total:.0f}s, non-content {nc_sec:.0f}s ({100*nc_sec/total:.1f}%)")
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        secs = sum(s.duration for s in segments if s.label.value == label)
        print(f"    {label:<14} {count:>3} segs   {secs:>6.1f}s  ({100*secs/total:5.1f}%)")
    print()
    print(f"open: http://$(hostname):8000/player/?v={args.video.stem}")


if __name__ == "__main__":
    main()
