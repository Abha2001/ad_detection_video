"""Post-process LLM-classified segments to fix obvious sanity violations.

The 3B Llama model occasionally contradicts its own rationale — e.g. labelling
an end-of-video segment as `intro` while admitting "late position" in the
rationale. This script applies deterministic position-based rules that ALWAYS
hold, without re-running the LLM:

- A segment in the last 5% of the video can never be `intro` -> `outro` if it
  has a non-speech audio cue, else `core_content`.
- A segment in the first 5% can never be `outro` -> `intro` (or core_content).
- Anything labelled `unknown` after parse failures -> `core_content` so it's
  treated as default-play in the player.
"""
import argparse
import json
from pathlib import Path


_NC = {"intro", "outro", "sponsorship", "self_promo", "recap",
       "transition", "dead_air", "waiting_room", "filler"}


def bridge_core_content_in_nc(
    segments: list[dict], max_bridge_sec: float = 30.0
) -> int:
    """If a *run* of core_content segments is wedged between two non-content
    segments of the same class, and the run is short, the LLM likely
    under-fired on the middle of an ad — bridge by relabelling the run.

    Common case: synthetic content-shaped ad has speech that LLM thinks is
    real content because it's topically plausible. We detect it as ad at the
    edges (boundaries fire on the cut in/out) but miss the middle. If the
    gap is short enough to plausibly be inside one ad, fill it.
    """
    fixed = 0
    i = 0
    while i < len(segments):
        # Find a run of core_content segments
        if segments[i]["label"] != "core_content":
            i += 1
            continue
        run_start = i
        while i < len(segments) and segments[i]["label"] == "core_content":
            i += 1
        run_end = i  # exclusive
        # Need NC on both sides
        if run_start == 0 or run_end >= len(segments):
            continue
        prev_label = segments[run_start - 1]["label"]
        next_label = segments[run_end]["label"]
        # Only bridge across sponsorship/intro/outro/self_promo/recap. dead_air
        # and transition are short by nature; bridging through them would
        # convert real content to silence, which is wrong.
        BRIDGEABLE = {"sponsorship", "intro", "outro", "self_promo", "recap", "filler"}
        if prev_label not in BRIDGEABLE or next_label not in BRIDGEABLE:
            continue
        if prev_label != next_label:
            continue
        run_total = (
            segments[run_end - 1]["end_sec"] - segments[run_start]["start_sec"]
        )
        if run_total > max_bridge_sec:
            continue
        for k in range(run_start, run_end):
            segments[k]["label"] = prev_label
            segments[k]["confidence"] = 0.7
            segments[k]["rationale"] = (
                f"[bridge] core_content run sandwiched between {prev_label} blocks — "
                f"likely the middle of an under-fired non-content block"
            )
            fixed += 1
    return fixed


def drop_misplaced_intro_outro(segments: list[dict]) -> int:
    """Drop intro/outro labels that aren't at the actual start/end of video.

    The LLM fires outro on any segment with position > 0.95 + short ASR,
    even if it's just a brief bridging phrase mid-sentence followed by
    more host speech. Real outros are contiguous at the very end. Same
    for intros at the start. We require the segment to be the trailing
    (or leading) NC run — anything later (or earlier) reverts to
    core_content.
    """
    fixed = 0
    # Drop outros that have core_content (or non-outro NC) AFTER them.
    last_outro_run_end = len(segments)
    for i in range(len(segments) - 1, -1, -1):
        if segments[i]["label"] == "outro":
            last_outro_run_end = i
        elif segments[i]["label"] == "core_content":
            break
    # Drop any outro before last_outro_run_end that has a core_content gap after it
    in_run = False
    for i in range(len(segments) - 1, -1, -1):
        if segments[i]["label"] == "outro":
            if i < last_outro_run_end:
                # Check if any core_content segment lies between this outro
                # and the trailing outro run; if so, this is a misplaced outro
                misplaced = any(
                    segments[k]["label"] == "core_content"
                    for k in range(i + 1, len(segments))
                )
                if misplaced:
                    segments[i]["label"] = "core_content"
                    segments[i]["confidence"] = 0.7
                    segments[i]["rationale"] = "[postproc] outro relabeled — core_content follows"
                    fixed += 1

    # Drop intros that have core_content BEFORE them.
    for i in range(len(segments)):
        if segments[i]["label"] == "intro":
            misplaced = any(
                segments[k]["label"] == "core_content"
                for k in range(0, i)
            )
            if misplaced:
                segments[i]["label"] = "core_content"
                segments[i]["confidence"] = 0.7
                segments[i]["rationale"] = "[postproc] intro relabeled — core_content precedes"
                fixed += 1

    return fixed


def drop_short_sandwiched(segments: list[dict], max_short_sec: float = 10.0) -> int:
    """Drop short non-content segments sandwiched between core_content.

    A short pause in continuous host speech often gets cut by audio silence
    detection and labeled transition / dead_air. If neighbors on both sides
    are core_content, this is almost certainly a false-positive non-content
    fire and we should restore it to content.
    """
    fixed = 0
    for i in range(1, len(segments) - 1):
        s = segments[i]
        if s["label"] in _NC and (s["end_sec"] - s["start_sec"]) <= max_short_sec:
            prev_label = segments[i - 1]["label"]
            next_label = segments[i + 1]["label"]
            if prev_label == "core_content" and next_label == "core_content":
                s["label"] = "core_content"
                s["confidence"] = 0.7
                s["rationale"] = (
                    f"[sandwich] short {s.get('rationale', '')[:40]} between "
                    f"core_content -> core_content"
                )
                fixed += 1
    return fixed


def fix(segments: list[dict]) -> tuple[int, list[dict]]:
    if not segments:
        return 0, segments
    total = segments[-1]["end_sec"]
    fixed = 0
    for s in segments:
        pos = s.get("position_norm") or (s["start_sec"] / total if total else 0.0)
        label = s.get("label", "core_content")
        asr_len = len((s.get("asr_text") or "").strip())
        silence_ratio = s.get("rms_silence_ratio") or 0
        spec_flat = s.get("spectral_flatness_mean") or 0
        non_speech_cue = (
            asr_len < 25 or spec_flat < 0.2 or silence_ratio > 0.3
        )

        # Rule A: nothing past 95% of video can be intro
        if pos > 0.95 and label == "intro":
            s["label"] = "outro" if non_speech_cue else "core_content"
            s["rationale"] = f"[postproc] intro->{'outro' if non_speech_cue else 'core_content'} (pos={pos:.3f})"
            fixed += 1
            continue

        # Rule B: nothing in first 5% can be outro
        if pos < 0.05 and label == "outro":
            s["label"] = "intro" if non_speech_cue else "core_content"
            s["rationale"] = f"[postproc] outro->{'intro' if non_speech_cue else 'core_content'} (pos={pos:.3f})"
            fixed += 1
            continue

        # Rule C: parse failures default to core_content
        if label == "unknown":
            s["label"] = "core_content"
            s["rationale"] = "[postproc] unknown -> core_content"
            s["confidence"] = 0.5
            fixed += 1
            continue

        # Rule D (force): obvious silent-bumper intro at start that the LLM missed.
        if (
            label == "core_content"
            and pos < 0.04
            and asr_len < 10
            and (s["end_sec"] - s["start_sec"]) < 30
            and (silence_ratio > 0.2 or spec_flat < 0.25)
        ):
            s["label"] = "intro"
            s["confidence"] = 0.85
            s["rationale"] = f"[postproc] forced intro: pos={pos:.3f}, asr_len={asr_len}, silence={silence_ratio:.2f}, spec={spec_flat:.2f}"
            fixed += 1
            continue

        # Rule E (force): obvious silent-credit outro at end that the LLM missed.
        if (
            label == "core_content"
            and pos > 0.96
            and asr_len < 25
            and (s["end_sec"] - s["start_sec"]) < 60
            and (silence_ratio > 0.2 or spec_flat < 0.25)
        ):
            s["label"] = "outro"
            s["confidence"] = 0.85
            s["rationale"] = f"[postproc] forced outro: pos={pos:.3f}, asr_len={asr_len}, silence={silence_ratio:.2f}"
            fixed += 1
            continue

        # Rule F (force): empty ASR + sustained silence + decent duration -> dead_air.
        if (
            label == "core_content"
            and asr_len < 5
            and silence_ratio > 0.7
            and (s["end_sec"] - s["start_sec"]) > 1.5
        ):
            s["label"] = "dead_air"
            s["confidence"] = 0.9
            s["rationale"] = f"[postproc] forced dead_air: silence_ratio={silence_ratio:.2f}"
            fixed += 1
            continue

    return fixed, segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--sandwich-max-sec", type=float, default=10.0,
                    help="drop NC segments shorter than this when sandwiched in core_content")
    args = ap.parse_args()
    for p in args.paths:
        data = json.loads(p.read_text())
        n, data = fix(data)
        m = drop_misplaced_intro_outro(data)
        s = drop_short_sandwiched(data, max_short_sec=args.sandwich_max_sec)
        b = bridge_core_content_in_nc(data, max_bridge_sec=45.0)
        # consolidate again so the newly-bridged core_content -> NC blocks
        # merge cleanly with their neighbors
        try:
            from _consolidate_nc import consolidate
            _, data = consolidate(data)
        except Exception:
            pass
        p.write_text(json.dumps(data, indent=2))
        print(f"{p}: {n} rule-fixes, {m} misplaced-intro/outro, {s} sandwich-drops, {b} bridges")


if __name__ == "__main__":
    main()
