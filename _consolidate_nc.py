"""Consolidate consecutive non-content segments into longer 'sponsorship' blocks.

The audio-silence boundary detector splits long ads into many small pieces
(every silence in the ad's voiceover becomes a cut). The LLM labels each
sub-piece as transition / dead_air / etc. — semantically reasonable per-
segment, but visually fragmented in the player and unhelpful for skip-block
behavior.

This pass walks the classified segments and merges any run of >=2 adjacent
non-content segments whose total duration is >= MIN_BLOCK seconds, re-labelling
the merged block as 'sponsorship'. Core_content segments are never touched.
"""
import argparse
import json
from pathlib import Path

NON_CONTENT = {"intro", "outro", "sponsorship", "self_promo", "recap",
               "transition", "dead_air", "waiting_room", "filler"}
MIN_BLOCK_SEC = 15.0


def consolidate(segments: list[dict]) -> tuple[int, list[dict]]:
    if not segments:
        return 0, segments
    out: list[dict] = []
    i = 0
    fixed = 0
    while i < len(segments):
        s = segments[i]
        if s["label"] not in NON_CONTENT:
            out.append(s)
            i += 1
            continue
        # Greedy: extend run while consecutive segments are non-content
        j = i
        while j + 1 < len(segments) and segments[j + 1]["label"] in NON_CONTENT:
            j += 1
        if j > i:
            run = segments[i : j + 1]
            total = run[-1]["end_sec"] - run[0]["start_sec"]
            if total >= MIN_BLOCK_SEC:
                # Pick the consolidated label:
                #   - run is 100% dead_air -> stay dead_air (don't promote
                #     pure silence to sponsorship)
                #   - any other mix (transition + dead_air, sponsorship +
                #     dead_air, etc.) -> sponsorship (the strongest NC
                #     signal — these runs straddle ad boundaries)
                labels_in_run = {s["label"] for s in run}
                if labels_in_run == {"dead_air"}:
                    dominant = "dead_air"
                else:
                    dominant = "sponsorship"
                merged = {
                    **run[0],
                    "end_sec": run[-1]["end_sec"],
                    "duration_sec": total,
                    "label": dominant,
                    "confidence": min(s.get("confidence", 0.5) for s in run),
                    "rationale": f"[consolidate] merged {len(run)} adjacent non-content segments ({total:.1f}s) -> {dominant}",
                    "asr_text": " ".join(s.get("asr_text", "") for s in run).strip(),
                    "ocr_text": " | ".join(filter(None, (s.get("ocr_text", "") for s in run))),
                }
                out.append(merged)
                fixed += len(run) - 1
                i = j + 1
                continue
        out.append(s)
        i += 1
    return fixed, out


def main() -> None:
    global MIN_BLOCK_SEC
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--min-block", type=float, default=MIN_BLOCK_SEC)
    args = ap.parse_args()
    MIN_BLOCK_SEC = args.min_block
    for p in args.paths:
        data = json.loads(p.read_text())
        n, data = consolidate(data)
        p.write_text(json.dumps(data, indent=2))
        print(f"{p}: consolidated {n} segments")


if __name__ == "__main__":
    main()
