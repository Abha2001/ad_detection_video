"""Cross-video fingerprint pass.

For every video that has a segments JSON in a given work-dir, computes MFCC
fingerprints from its WAV and adds them to a shared FingerprintDB. Then
matches each segment against the DB *excluding* the segment's own video,
sets fingerprint_hit=True on segments that match anything else, and rewrites
the segments JSON in place.

After this pass, segments that share audio with another episode are flagged —
which is exactly the signal the LLM fuser needs for the `recap` class.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from nc_detection.fingerprint import (
    FingerprintDB,
    fingerprint_video,
    mark_fingerprint_hits,
)
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
        try:
            seg.label = Taxonomy(s.get("label", "unknown"))
        except ValueError:
            seg.label = Taxonomy.UNKNOWN
        seg.confidence = float(s.get("confidence", 0.0))
        seg.rationale = s.get("rationale", "")
        out.append(seg)
    return out


def write_segments(path: Path, segments) -> None:
    path.write_text(json.dumps([s.to_dict() for s in segments], indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dirs", nargs="+", type=Path,
                    help="Directories containing *.segments.json + *.wav pairs")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--db-path", type=Path, default=Path("work/fingerprints.pkl"))
    args = ap.parse_args()

    db = FingerprintDB()
    fingerprints_by_video = {}

    for wd in args.work_dirs:
        for segs_path in sorted(wd.glob("*.segments.json")):
            stem = segs_path.name.replace(".segments.json", "")
            wav_path = wd / f"{stem}.wav"
            if not wav_path.exists():
                print(f"skip {stem}: no wav", file=sys.stderr)
                continue
            segments = load_segments(segs_path)
            t0 = time.time()
            fps = fingerprint_video(wav_path, segments)
            print(f"[{time.time()-t0:5.1f}s] {stem}: {len(fps)} fingerprints")
            fingerprints_by_video[(wd, stem)] = (segments, fps, segs_path)
            db.add_video(stem, fps)

    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    db.save(args.db_path)
    print(f"DB saved: {len(db.entries)} entries -> {args.db_path}")

    print("\nCross-matching ...")
    for (wd, stem), (segments, fps, segs_path) in fingerprints_by_video.items():
        t0 = time.time()
        mark_fingerprint_hits(segments, fps, db, own_label=stem,
                              threshold=args.threshold)
        hits = sum(1 for s in segments if s.fingerprint_hit)
        print(f"[{time.time()-t0:5.1f}s] {stem}: {hits} hits")
        write_segments(segs_path, segments)


if __name__ == "__main__":
    main()
