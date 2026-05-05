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
    args = ap.parse_args()
    for p in args.paths:
        data = json.loads(p.read_text())
        n, data = fix(data)
        p.write_text(json.dumps(data, indent=2))
        print(f"{p}: {n} fixes")


if __name__ == "__main__":
    main()
