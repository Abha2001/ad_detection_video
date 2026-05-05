"""End-to-end: video -> classified segments JSON. No ground-truth required."""
import argparse
import time
from pathlib import Path

from nc_detection.audio_boundaries import detect_silence_boundaries, merge_boundaries
from nc_detection.audio_features import extract_wav
from nc_detection.boundary import detect_boundaries_sec
from nc_detection.run import run as run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--work-dir", type=Path, default=Path("work"))
    ap.add_argument("--whisper-model", default="base")
    ap.add_argument("--adaptive-k", type=float, default=1.0)
    ap.add_argument("--sample-every", type=int, default=5)
    ap.add_argument("--classify", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    visual_boundaries = detect_boundaries_sec(
        args.video,
        detector_key="frame_diff",
        sample_every=args.sample_every,
        adaptive_k=args.adaptive_k,
    )
    print(f"[{time.time()-t0:6.1f}s] Visual boundaries: {len(visual_boundaries)} cuts")

    # Audio-based boundaries fill in transitions where the visual signal is
    # smooth (synthetic ad crossfades, podcast topic shifts, sponsor music-bed
    # entries). Extract WAV first since the regular pipeline needs it anyway.
    t0 = time.time()
    wav_path = args.work_dir / (args.video.stem + ".wav")
    extract_wav(args.video, wav_path)
    audio_boundaries = detect_silence_boundaries(wav_path)
    boundaries = merge_boundaries(visual_boundaries, audio_boundaries)
    print(f"[{time.time()-t0:6.1f}s] Audio boundaries: {len(audio_boundaries)} silent-interval cuts; merged to {len(boundaries)} total")

    t0 = time.time()
    out = run_pipeline(
        args.video,
        args.work_dir,
        boundaries_sec=boundaries,
        whisper_model=args.whisper_model,
        classify=args.classify,
    )
    print(f"[{time.time()-t0:6.1f}s] Wrote {out}")


if __name__ == "__main__":
    main()
