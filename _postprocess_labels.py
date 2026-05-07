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


_ST_MODEL = None


def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is not None:
        return _ST_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    except Exception:
        _ST_MODEL = False  # marker for "tried and failed"
    return _ST_MODEL


def _asr_sim(a: str, b: str) -> float:
    """Cosine similarity between ASR strings via sentence-transformers.
    Returns 1.0 if either is empty (so audio-only check applies).
    """
    if not a or not b or len(a) < 5 or len(b) < 5:
        return 1.0
    m = _get_st_model()
    if not m:
        return 1.0
    import numpy as np
    e = m.encode([a, b], normalize_embeddings=True)
    return float(np.dot(e[0], e[1]))


def extend_nc_via_audio(
    segments: list[dict], max_extend_sec: float = 25.0, max_distance: float = 0.20,
    min_asr_sim: float = 0.45,
) -> int:
    """Extend NC blocks over adjacent core_content when BOTH the audio
    profile matches AND the ASR is topically similar (or empty). Pure
    audio similarity isn't enough — host wrap-up speech can sound the
    same as host CTA speech, so we also require the content to be on
    the same topic before merging."""
    """Extend non-content blocks backwards/forwards over adjacent
    core_content segments whose audio profile matches the NC block's.

    Catches the case where the LLM mis-classified the start of an ad as
    core_content because Whisper produced rich-looking (but gibberish)
    ASR over the ad's foreign-language audio. The audio character — low
    silence_ratio, higher motion — gives us a clean signal that the
    'content' segment is actually still inside the ad.
    """
    import math

    BRIDGEABLE = {"sponsorship", "intro", "outro", "self_promo", "recap", "filler"}

    def profile(seg: dict) -> tuple[float, float, float, float]:
        return (
            seg.get("spectral_flatness_mean", 0.0),
            seg.get("rms_mean", 0.0),
            seg.get("rms_silence_ratio", 0.0),
            min(seg.get("motion_energy", 0.0), 5.0) / 5.0,
        )

    def dist(a, b) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    fixed = 0
    for i in range(len(segments)):
        s = segments[i]
        if s["label"] not in BRIDGEABLE:
            continue
        nc_profile = profile(s)
        # intro / outro / self_promo are usually short — extending them
        # too far overgrows into normal host speech. Keep tight.
        ext_cap = (
            10.0 if s["label"] in ("intro", "outro", "self_promo") else max_extend_sec
        )

        # Extend backwards over adjacent core_content with matching audio
        # AND topically similar ASR (or empty ASR).
        extended_back = 0
        j = i - 1
        while j >= 0 and segments[j]["label"] == "core_content":
            d = dist(profile(segments[j]), nc_profile)
            seg_dur = segments[j]["end_sec"] - segments[j]["start_sec"]
            if extended_back + seg_dur > ext_cap:
                break
            if d > max_distance:
                break
            asr_a = segments[j].get("asr_text", "") or ""
            asr_b = s.get("asr_text", "") or ""
            sim = _asr_sim(asr_a, asr_b)
            if sim < min_asr_sim:
                break  # different topic — don't extend
            segments[j]["label"] = s["label"]
            segments[j]["confidence"] = 0.7
            segments[j]["rationale"] = (
                f"[extend-back] audio profile dist={d:.3f} from {s['label']}"
            )
            extended_back += seg_dur
            fixed += 1
            j -= 1

        # Extend forwards
        extended_fwd = 0
        j = i + 1
        while j < len(segments) and segments[j]["label"] == "core_content":
            d = dist(profile(segments[j]), nc_profile)
            seg_dur = segments[j]["end_sec"] - segments[j]["start_sec"]
            if extended_fwd + seg_dur > ext_cap:
                break
            if d > max_distance:
                break
            asr_a = segments[j].get("asr_text", "") or ""
            asr_b = s.get("asr_text", "") or ""
            sim = _asr_sim(asr_a, asr_b)
            if sim < min_asr_sim:
                break
            segments[j]["label"] = s["label"]
            segments[j]["confidence"] = 0.7
            segments[j]["rationale"] = (
                f"[extend-fwd] audio profile dist={d:.3f} from {s['label']}"
            )
            extended_fwd += seg_dur
            fixed += 1
            j += 1
    return fixed


def bridge_by_audio_features(
    segments: list[dict], max_bridge_sec: float = 90.0, max_distance: float = 0.30
) -> int:
    """Bridge core_content gaps whose AUDIO features look like the
    bordering non-content blocks (and don't look like the surrounding
    content). The signal that distinguishes a synthetic ad from talking-
    head content here is silence_ratio (~0 inside ad, >0.05 in content)
    and motion_energy (high inside ad). Comparing the gap's audio profile
    to the bordering NC profile is a much cleaner bridge signal than raw
    duration.
    """
    import math

    def profile(seg: dict) -> tuple[float, float, float, float]:
        return (
            seg.get("spectral_flatness_mean", 0.0),
            seg.get("rms_mean", 0.0),
            seg.get("rms_silence_ratio", 0.0),
            min(seg.get("motion_energy", 0.0), 5.0) / 5.0,  # normalize
        )

    def dist(a, b) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    fixed = 0
    BRIDGEABLE = {"sponsorship", "intro", "outro", "self_promo", "recap", "filler"}
    i = 0
    while i < len(segments):
        if segments[i]["label"] != "core_content":
            i += 1
            continue
        run_start = i
        while i < len(segments) and segments[i]["label"] == "core_content":
            i += 1
        run_end = i
        if run_start == 0 or run_end >= len(segments):
            continue
        prev_label = segments[run_start - 1]["label"]
        next_label = segments[run_end]["label"]
        if prev_label not in BRIDGEABLE or next_label not in BRIDGEABLE:
            continue
        if prev_label != next_label:
            continue
        run_total = (
            segments[run_end - 1]["end_sec"] - segments[run_start]["start_sec"]
        )
        if run_total > max_bridge_sec:
            continue

        # Average audio profile of the gap and of the bordering NC blocks.
        gap_segs = segments[run_start:run_end]
        gap_profile = tuple(
            sum(p) / len(gap_segs)
            for p in zip(*(profile(s) for s in gap_segs))
        )
        border_profile = tuple(
            (a + b) / 2
            for a, b in zip(
                profile(segments[run_start - 1]),
                profile(segments[run_end]),
            )
        )

        d = dist(gap_profile, border_profile)
        if d > max_distance:
            continue

        for k in range(run_start, run_end):
            segments[k]["label"] = prev_label
            segments[k]["confidence"] = 0.75
            segments[k]["rationale"] = (
                f"[audio-bridge] gap audio profile dist={d:.3f} from bordering "
                f"{prev_label} -> reclassify"
            )
            fixed += 1
    return fixed


def bridge_by_asr_similarity(
    segments: list[dict], min_similarity: float = 0.55, max_bridge_sec: float = 90.0
) -> int:
    """Bridge via ASR semantic similarity, not just duration.

    For each run of core_content with NC of the same label on both sides,
    embed (left_NC.asr + right_NC.asr) and (gap.asr) and compare cosines.
    If similar enough, the gap is likely continuing the same ad/content,
    so we relabel.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return 0
    import numpy as np

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

    fixed = 0
    BRIDGEABLE = {"sponsorship", "intro", "outro", "self_promo", "recap", "filler"}
    i = 0
    while i < len(segments):
        if segments[i]["label"] != "core_content":
            i += 1
            continue
        run_start = i
        while i < len(segments) and segments[i]["label"] == "core_content":
            i += 1
        run_end = i
        if run_start == 0 or run_end >= len(segments):
            continue
        prev_label = segments[run_start - 1]["label"]
        next_label = segments[run_end]["label"]
        if prev_label not in BRIDGEABLE or next_label not in BRIDGEABLE:
            continue
        if prev_label != next_label:
            continue
        run_total = (
            segments[run_end - 1]["end_sec"] - segments[run_start]["start_sec"]
        )
        if run_total > max_bridge_sec:
            continue

        # Embed and compare
        gap_text = " ".join(
            segments[k].get("asr_text", "") for k in range(run_start, run_end)
        ).strip()
        ref_text = (
            segments[run_start - 1].get("asr_text", "")
            + " "
            + segments[run_end].get("asr_text", "")
        ).strip()
        if not gap_text or not ref_text:
            continue
        emb = model.encode([gap_text, ref_text], normalize_embeddings=True)
        sim = float(np.dot(emb[0], emb[1]))
        if sim < min_similarity:
            continue

        for k in range(run_start, run_end):
            segments[k]["label"] = prev_label
            segments[k]["confidence"] = 0.7
            segments[k]["rationale"] = (
                f"[asr-bridge] gap ASR cos-sim={sim:.2f} to bordering {prev_label} "
                f"-> reclassify as {prev_label}"
            )
            fixed += 1
    return fixed


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


def label_by_graphic_ocr(segments: list[dict]) -> int:
    """Generic intro/outro detection via OCR graphic overlays.

    Heuristic: a segment with OCR text whose tokens don't overlap the
    segment's own ASR is showing a *graphic* (stat overlay, title card,
    channel watermark, lower-third) — not closed-caption text. At the
    start of the video this is intro material; at the end it's
    outro/self_promo. Position alone is too coarse, but position +
    graphic-OCR-evidence is reliable.
    """
    if not segments:
        return 0
    fixed = 0
    total_dur = segments[-1]["end_sec"]

    def ocr_is_graphic(asr: str, ocr: str) -> bool:
        if not ocr or len(ocr.strip()) < 4:
            return False
        # Tokenize lowercase alphabetic words >=3 chars from each.
        import re
        ocr_toks = set(t.lower() for t in re.findall(r"[A-Za-z]{3,}", ocr))
        if not ocr_toks:
            # all-numeric / symbolic OCR (e.g. "0.2%") — definitely graphic
            return bool(re.search(r"\d", ocr))
        asr_toks = set(t.lower() for t in re.findall(r"[A-Za-z]{3,}", asr or ""))
        # If <30% of OCR tokens appear in ASR, the OCR is graphic content
        # (overlay, watermark, brand) rather than burned-in captions.
        overlap = len(ocr_toks & asr_toks) / len(ocr_toks)
        return overlap < 0.3

    # Intro: scan all segments in first 5% — every one with graphic OCR
    # becomes intro, plus any leading core_content before the first hit.
    last_intro_idx = -1
    for i, s in enumerate(segments):
        pos = s.get("position_norm") or (s["start_sec"] / total_dur if total_dur else 0)
        if pos > 0.05:
            break
        if s["label"] != "core_content":
            continue
        if ocr_is_graphic(s.get("asr_text", ""), s.get("ocr_text", "")):
            last_intro_idx = i
    if last_intro_idx >= 0:
        for k in range(0, last_intro_idx + 1):
            if segments[k]["label"] == "core_content":
                segments[k]["label"] = "intro"
                segments[k]["confidence"] = 0.8
                segments[k]["rationale"] = (
                    f"[graphic-ocr] graphic-style OCR detected through pos "
                    f"{segments[last_intro_idx].get('position_norm', 0):.3f}"
                )
                fixed += 1

    # Outro: scan all segments in last 5% — every one with graphic OCR
    # becomes outro, plus any trailing core_content after the first hit.
    first_outro_idx = -1
    for i in range(len(segments) - 1, -1, -1):
        s = segments[i]
        pos = s.get("position_norm") or (s["start_sec"] / total_dur if total_dur else 0)
        if pos < 0.95:
            break
        if s["label"] not in ("core_content", "outro", "self_promo"):
            continue
        if ocr_is_graphic(s.get("asr_text", ""), s.get("ocr_text", "")):
            first_outro_idx = i
    if first_outro_idx >= 0:
        for k in range(first_outro_idx, len(segments)):
            if segments[k]["label"] == "core_content":
                segments[k]["label"] = "outro"
                segments[k]["confidence"] = 0.8
                segments[k]["rationale"] = (
                    f"[graphic-ocr] graphic-style OCR at position "
                    f"{segments[k].get('position_norm', 0):.3f}"
                )
                fixed += 1

    return fixed


def merge_consecutive_same_label(segments: list[dict]) -> int:
    """Merge consecutive segments that share the same label into a single
    segment so the player timeline shows one solid block per label run.
    Concatenates ASR/OCR; takes the min confidence."""
    if not segments:
        return 0
    out = [dict(segments[0])]
    merged = 0
    for s in segments[1:]:
        last = out[-1]
        if s["label"] == last["label"]:
            last["end_sec"] = s["end_sec"]
            last["duration_sec"] = last["end_sec"] - last["start_sec"]
            asr_a = (last.get("asr_text") or "").strip()
            asr_b = (s.get("asr_text") or "").strip()
            if asr_b:
                last["asr_text"] = (asr_a + " " + asr_b).strip() if asr_a else asr_b
            ocr_a = (last.get("ocr_text") or "").strip()
            ocr_b = (s.get("ocr_text") or "").strip()
            if ocr_b:
                last["ocr_text"] = (ocr_a + " | " + ocr_b).strip(" |") if ocr_a else ocr_b
            last["confidence"] = min(last.get("confidence", 1.0), s.get("confidence", 1.0))
            merged += 1
        else:
            out.append(dict(s))
    segments[:] = out
    return merged


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
        # Generic intro/outro detection via OCR graphic-overlay signal.
        g = label_by_graphic_ocr(data)
        m = drop_misplaced_intro_outro(data)
        s = drop_short_sandwiched(data, max_short_sec=args.sandwich_max_sec)
        # First: extend NC blocks via audio similarity over neighbors that
        # the LLM mis-classified as core_content (e.g., gibberish ASR over
        # non-English ad audio). This catches the late-start of ads.
        e = extend_nc_via_audio(data, max_extend_sec=25.0, max_distance=0.20)
        # Then: audio-feature bridge for gaps inside an NC run.
        b = bridge_by_audio_features(data, max_bridge_sec=90.0, max_distance=0.30)
        b += e
        # Merge consecutive same-label segments so the player shows one
        # solid block per label run instead of stripes. We don't run the
        # full consolidate again because its dominant-label logic would
        # re-collapse outro+self_promo+dead_air mixes back to sponsorship.
        mc = merge_consecutive_same_label(data)
        p.write_text(json.dumps(data, indent=2))
        print(f"{p}: {n} rule-fixes, {g} graphic-ocr, {m} misplaced-intro/outro, {s} sandwich-drops, {b} bridges, {mc} same-label merges")


if __name__ == "__main__":
    main()
