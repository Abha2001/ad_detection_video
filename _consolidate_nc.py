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


def _trim_trailing_dead_air(
    run: list[dict], min_trim_sec: float = 5.0
) -> tuple[list[dict], list[dict]]:
    """If a run ends with substantial dead_air (>= min_trim_sec total),
    split it off so it remains a separate dead_air block. Short tail
    dead_air stays merged — those are typically brief silences inside
    a single ad and shouldn't break the bridge's same-label-border check."""
    j = len(run)
    while j > 0 and run[j - 1]["label"] == "dead_air":
        j -= 1
    if j == 0 or j == len(run):
        return run, []
    tail = run[j:]
    tail_dur = tail[-1]["end_sec"] - tail[0]["start_sec"]
    if tail_dur < min_trim_sec:
        return run, []  # tail is too short to be meaningful — keep merged
    return run[:j], tail


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
            full_run = segments[i : j + 1]
            # If run ends with pure dead_air, trim it off — keep it as a
            # separate dead_air block so the post-ad silence isn't
            # absorbed into the sponsorship label.
            run, tail_dead_air = _trim_trailing_dead_air(full_run)
            total = run[-1]["end_sec"] - run[0]["start_sec"] if run else 0
            if total >= MIN_BLOCK_SEC:
                # Pick the consolidated label:
                #   - pure dead_air -> dead_air
                #   - pure self_promo -> self_promo (preserve "subscribe" CTA)
                #   - pure outro -> outro
                #   - pure intro -> intro
                #   - any other mix (incl. just transitions, sponsorship +
                #     dead_air, etc.) -> sponsorship
                labels_in_run = {s["label"] for s in run}
                if labels_in_run == {"dead_air"}:
                    dominant = "dead_air"
                elif labels_in_run == {"self_promo"}:
                    dominant = "self_promo"
                elif labels_in_run == {"outro"}:
                    dominant = "outro"
                elif labels_in_run == {"intro"}:
                    dominant = "intro"
                # pure-with-dead_air variants (e.g. self_promo + dead_air at end)
                elif labels_in_run == {"self_promo", "dead_air"}:
                    dominant = "self_promo"
                elif labels_in_run == {"outro", "dead_air"}:
                    dominant = "outro"
                elif labels_in_run == {"intro", "dead_air"}:
                    dominant = "intro"
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
                # Re-emit any trailing dead_air segments untouched
                for da in tail_dead_air:
                    out.append(da)
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
