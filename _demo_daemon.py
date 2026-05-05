"""Pre-warmed demo daemon.

Runs on a GPU node with all models loaded in memory ahead of time. Watches a
filesystem inbox for new videos, processes them through the full pipeline,
writes classified segment JSON, then moves the video to a 'done' folder.

Designed for tight live-demo timing: when the user drops a video, the daemon
starts work immediately without the 30-90s of model-load latency.

Usage:
    sbatch demo_daemon.sbatch
    # then on demo day, drop videos into demo_inbox/
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Eagerly import all heavy modules so models are loaded once at startup.
print("[daemon] importing models...", flush=True)
import cv2  # noqa
import librosa  # noqa

from nc_detection.audio_boundaries import (
    detect_silence_boundaries,
    merge_boundaries,
)
from nc_detection.audio_features import (
    assign_asr_to_segments,
    compute_audio_stats,
    extract_wav,
    transcribe,
)
from nc_detection.boundary import detect_boundaries_sec
from nc_detection.classify import classify_segments, _get_pipeline
from nc_detection.ocr_features import compute_ocr_text, _get_reader
from nc_detection.run import get_video_duration
from nc_detection.segment import EnrichedSegment, Taxonomy, segments_from_boundaries
from nc_detection.topic import compute_topic_drift, _get_model as _get_st
from nc_detection.visual_features import assign_shot_counts


def _warmup() -> None:
    """Force lazy initializers to actually load the models."""
    print("[daemon] warming Llama...", flush=True)
    _get_pipeline()
    print("[daemon] warming EasyOCR...", flush=True)
    _get_reader()
    print("[daemon] warming sentence-transformers...", flush=True)
    _get_st()
    print("[daemon] warmup complete", flush=True)


def process_one(video_path: Path, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    print(f"[demo] start {video_path.name}", flush=True)
    duration = get_video_duration(video_path)

    t = time.time()
    visual_b = detect_boundaries_sec(
        video_path, detector_key="frame_diff", sample_every=5, adaptive_k=1.0
    )
    wav_path = work_dir / (video_path.stem + ".wav")
    extract_wav(video_path, wav_path)
    audio_b = detect_silence_boundaries(wav_path)
    boundaries = merge_boundaries(visual_b, audio_b)
    print(f"[demo] +{time.time()-start:5.1f}s  boundaries: {len(visual_b)} visual + {len(audio_b)} audio -> {len(boundaries)} ({time.time()-t:.1f}s)", flush=True)

    segments = segments_from_boundaries(boundaries, duration)
    assign_shot_counts(boundaries, segments)

    t = time.time()
    asr = transcribe(wav_path, model_size="base")
    assign_asr_to_segments(asr, segments)
    print(f"[demo] +{time.time()-start:5.1f}s  whisper ({time.time()-t:.1f}s)", flush=True)

    t = time.time()
    compute_audio_stats(wav_path, segments)
    print(f"[demo] +{time.time()-start:5.1f}s  audio stats ({time.time()-t:.1f}s)", flush=True)

    # SKIPPING motion_energy: CPU-bound, marginal signal — saves 2-3 min.
    # Segments will have motion_energy=0; classifier still works on other signals.

    t = time.time()
    compute_ocr_text(video_path, segments, max_per_seg=1)  # 1 frame/seg for speed
    print(f"[demo] +{time.time()-start:5.1f}s  ocr ({time.time()-t:.1f}s)", flush=True)

    t = time.time()
    compute_topic_drift(segments)
    print(f"[demo] +{time.time()-start:5.1f}s  topic drift ({time.time()-t:.1f}s)", flush=True)

    t = time.time()
    classify_segments(segments, batch_size=16)
    print(f"[demo] +{time.time()-start:5.1f}s  classified {len(segments)} segs ({time.time()-t:.1f}s)", flush=True)

    out_path = work_dir / (video_path.stem + ".segments.classified.json")
    out_path.write_text(json.dumps([s.to_dict() for s in segments], indent=2))

    # Post-process for position-rule sanity.
    subprocess.run(
        ["python", "_postprocess_labels.py", str(out_path)],
        check=False,
        cwd=str(Path.cwd()),
    )
    print(f"[demo] +{time.time()-start:5.1f}s  TOTAL — wrote {out_path}", flush=True)

    # Brief summary
    from collections import Counter
    dist = Counter(s.label.value for s in segments)
    nc = sum(s.duration for s in segments if s.label not in (Taxonomy.CORE_CONTENT, Taxonomy.UNKNOWN))
    print(f"[demo]   distribution: {dict(dist)}", flush=True)
    print(f"[demo]   non-content: {nc:.0f}s of {duration:.0f}s ({100*nc/duration:.1f}%)", flush=True)
    return out_path


def main() -> None:
    inbox = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo_inbox")
    outbox = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("work_demo")
    done = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("demo_done")
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    done.mkdir(parents=True, exist_ok=True)

    print(f"[daemon] inbox={inbox} outbox={outbox} done={done}", flush=True)
    _warmup()
    print(f"[daemon] ready — drop *.mp4 into {inbox}", flush=True)

    while True:
        videos = sorted(inbox.glob("*.mp4")) + sorted(inbox.glob("*.mkv")) + sorted(inbox.glob("*.webm"))
        if not videos:
            time.sleep(2)
            continue
        for v in videos:
            try:
                process_one(v, outbox)
                shutil.move(str(v), str(done / v.name))
            except Exception as e:
                print(f"[daemon] FAILED on {v.name}: {e}", flush=True)
                # Move it out so we don't retry forever
                shutil.move(str(v), str(done / (v.name + ".failed")))


if __name__ == "__main__":
    main()
