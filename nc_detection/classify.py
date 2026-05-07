"""LLM fuser using a local Llama via Hugging Face transformers.

Runs on GPU via `device_map="auto"` (offloads to whatever's visible) and falls
back to CPU when no GPU is present. Default model is unsloth's non-gated mirror
of Llama 3.2 3B Instruct — small enough to be quick, capable enough for
structured per-segment classification.
"""
from __future__ import annotations

import json
import os
from typing import Iterable

from .segment import EnrichedSegment, Taxonomy

DEFAULT_MODEL = os.environ.get(
    "NC_LLM_MODEL", "unsloth/Llama-3.2-3B-Instruct"
)

_LABELS = [t.value for t in Taxonomy if t != Taxonomy.UNKNOWN]

SYSTEM_PROMPT = """You classify a single segment of a long-form video into ONE taxonomy class.

Taxonomy:
- core_content: main subject of the video (host speaking on the topic, the demo)
- intro: opening title sequence, brand bumper, theme music + title card
- outro: closing credits, end cards, "thanks for watching" sign-off
- sponsorship: paid promotional read; product pitch (sponsor music bed common)
- self_promo: creator's own channel/patreon/merch/newsletter pitch
- recap: repeated content from another episode (fingerprint hit is the cue)
- transition: short interstitial / stinger between sections
- dead_air: silence + blank frame (no useful content)
- waiting_room: pre-stream "starting soon" / countdown / lobby
- filler: off-topic insert with no value

Per-segment signals (JSON):
- duration_sec, position_norm (0=start of video, 1=end)
- asr_text (speech transcript; may be empty)
- ocr_text (on-screen text; may be empty)
- rms_mean, rms_silence_ratio (0-1; fraction of segment under silence threshold)
- spectral_flatness_mean (~0=tone/music, ~0.3-0.5=speech, ~1=noise)
- motion_energy (low=static/holding; high=action)
- shot_count (cuts inside segment)
- fingerprint_hit (audio matches another episode)
- topic_drift (0..1; how off-topic this segment is relative to the rest of
  the video. >=0.4 means the segment's content diverges sharply from the
  host video's overall topic — typical of an ad insert or an unrelated clip)

DECISION RULES — apply in this order, stop at the first match:

1. fingerprint_hit == true -> recap. (Strongest signal. Confidence >= 0.9.)

1b. topic_drift >= 0.45 AND duration_sec > 5 AND asr_text length > 25
    -> sponsorship (off-topic insert). The segment talks about something
    unrelated to the rest of the video — typical of an injected ad clip.
    If drift is moderate (0.35-0.45) and duration is small (< 10s), prefer
    transition. Confidence reflects how far drift exceeds 0.45.

2. position_norm < 0.05 AND duration_sec < 60 AND
   (asr_text length < 25 OR spectral_flatness_mean < 0.2 OR rms_silence_ratio > 0.3
    OR ocr_text looks like a channel/show title card or speaker affiliation
       — e.g. "TEDx<City>", "<Channel Name>", "Episode N", "<Speaker Name>",
       host's logo or wordmark — *without* a sponsor URL or promo code)
   -> intro.
   This catches title-card / logo-bumper / opening music segments at video start.
   Channel branding at the start (TEDx, LTT, MKBHD) is intro, not sponsorship —
   it's the host advertising themselves, not a third-party paid promo.
   If ASR is rich speech AND OCR doesn't look like a title card, this is content
   even if position is low. Confidence 0.85.

3. position_norm > 0.95 AND duration_sec < 90 AND
   (asr_text length < 40 OR spectral_flatness_mean < 0.2 OR rms_silence_ratio > 0.3)
   -> outro.
   Same logic — late position alone doesn't make a segment outro. Require an
   accompanying audio cue (closing music, credits-card silence, or thin/no
   speech). Continuous host speech at the end is still core_content.

4. asr_text is empty or under ~30 chars AND rms_silence_ratio < 0.6 AND motion_energy > 0.3 -> NOT core_content.
   The segment has audio but no speech. Pick from {sponsorship, transition, filler} using:
     - duration_sec < 5 -> transition
     - otherwise -> sponsorship  (music-bed ad with little voiceover)

5. asr_text is empty AND rms_silence_ratio > 0.6 -> dead_air or waiting_room.
     - position_norm < 0.05 -> waiting_room
     - otherwise -> dead_air

6. asr_text OR ocr_text contains an EXPLICIT promotional phrase from the
   list below — sponsorship requires *promotional language*, not just a
   brand name. Brand names alone are not enough; the segment must SOUND
   like the host is selling something.
   Promotional cues:
     - "sponsored by", "today's video is brought to you by"
     - "promo code", "use code <X>", "get N% off", "first N people"
     - "use my link", "go to <url> to get", "thanks to <brand> for sponsoring"
     - "[sponsored]" / "[ad]" tag
     - sponsor-style channel URL: "<sponsor>.com/<channel>" or "<sponsor>.com/<host>"
   -> sponsorship.

   NOT sponsorship (very important — do NOT over-fire):
   - A brand name appearing in OCR/ASR because the brand IS the subject of
     the video. A phone review will show "iPhone 15 Pro" repeatedly; a
     software tutorial will show "Adobe Photoshop" throughout; a cooking
     video shows ingredient brands. None of these are sponsorship.
   - The host's own channel name, show title, speaker affiliation,
     speaker's organization (e.g. "TEDxAustin", "Linus Tech Tips",
     "Prof. X, MIT") — those are intro/self_promo (rules 2 and 7), NOT
     sponsorship.
   - A URL of a paper, dataset, source citation, GitHub repo, news source
     ("nytimes.com/article/..."). Citations are content.

   Heuristic for the gray area: if removing every promotional phrase
   from the segment's text would leave nothing salient, it's sponsorship.
   If removing the brand name would still leave normal explanatory speech,
   it's content.

7. asr_text OR ocr_text contains the host's own CTAs to themselves
   ("subscribe", "like and subscribe", "patreon", "merch", "channel members",
    "join my channel", "support the channel", "smash that like button",
    the host's own merch domain like "lttstore.com")
   -> self_promo.
   These are calls to action for the host's own platform, not a paid sponsor.

7b. ocr_text contains a "starting soon" / countdown / lobby overlay
    ("starting soon", "be right back", "stream begins", numeric countdown like "0:30")
    AND position_norm < 0.1 -> waiting_room.

8. None of the above -> core_content. Confidence reflects strength of the rich-ASR signal.

Output STRICT JSON only (no prose, no markdown):
{"label": <one of the taxonomy strings>, "confidence": <0..1>, "rationale": "<one short sentence citing the rule that matched>"}"""


_PIPELINE = None


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    # float16 is supported on every recent GPU (P100, V100, A100, A40, L40S);
    # bfloat16 isn't supported on P100/V100 and silently falls back to fp32,
    # which doubles VRAM and OOMs on 16GB cards.
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL,
        torch_dtype=dtype,
        device_map="auto",
    )
    _PIPELINE = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return _PIPELINE


def _evidence(seg: EnrichedSegment) -> dict:
    return {
        "duration_sec": round(seg.duration, 2),
        "position_norm": round(seg.position_norm, 3),
        "asr_text": seg.asr_text[:1500],
        "ocr_text": seg.ocr_text[:500],
        "rms_mean": round(seg.rms_mean, 4),
        "rms_silence_ratio": round(seg.rms_silence_ratio, 3),
        "spectral_flatness_mean": round(seg.spectral_flatness_mean, 3),
        "motion_energy": round(seg.motion_energy, 3),
        "shot_count": seg.shot_count,
        "fingerprint_hit": seg.fingerprint_hit,
        "topic_drift": round(seg.topic_drift, 3),
    }


def _extract_json(text: str) -> dict:
    """Pull the first valid JSON object out of the model's response."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])


def classify_segment(seg: EnrichedSegment) -> None:
    """Fill seg.label / confidence / rationale via a single LLM call."""
    pipe = _get_pipeline()
    user = (
        "Segment evidence:\n"
        + json.dumps(_evidence(seg), indent=2)
        + "\n\nClassify."
    )

    out = pipe(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        max_new_tokens=200,
        do_sample=False,
        return_full_text=False,
    )
    text = out[0]["generated_text"]
    if isinstance(text, list):
        text = text[-1].get("content", "")

    try:
        parsed = _extract_json(text)
        label = parsed.get("label", "")
        seg.label = Taxonomy(label) if label in _LABELS else Taxonomy.UNKNOWN
        seg.confidence = float(parsed.get("confidence", 0.0))
        seg.rationale = str(parsed.get("rationale", ""))
    except (ValueError, json.JSONDecodeError, KeyError):
        seg.label = Taxonomy.UNKNOWN
        seg.confidence = 0.0
        seg.rationale = f"parse_failure: {text[:120]}"


def classify_segments(segments: Iterable[EnrichedSegment], batch_size: int = 1) -> None:
    """Classify all segments in place. Batches GPU inference for throughput.

    The system prompt is ~3K tokens after we added topic-drift, OCR, and
    branding-disambiguation rules. Each call's KV cache is ~1.1GB on
    Llama 3.2 3B fp16; on a 16GB GPU we have ~8GB headroom after the model
    itself, so batch_size=1 is the safe default. Bigger GPUs can pass
    batch_size up to 8.
    """
    seg_list = list(segments)
    if not seg_list:
        return

    pipe = _get_pipeline()
    prompts = []
    for seg in seg_list:
        user = (
            "Segment evidence:\n"
            + json.dumps(_evidence(seg), indent=2)
            + "\n\nClassify."
        )
        prompts.append(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]
        )

    # Process in chunks so we can free CUDA cache between batches and avoid
    # fragmenting allocations on small GPUs.
    outputs = []
    try:
        import torch
    except ImportError:
        torch = None
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        out = pipe(
            batch,
            max_new_tokens=120,
            do_sample=False,
            return_full_text=False,
            batch_size=batch_size,
        )
        outputs.extend(out)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    for seg, out in zip(seg_list, outputs):
        item = out[0] if isinstance(out, list) else out
        text = item.get("generated_text", "")
        if isinstance(text, list):
            text = text[-1].get("content", "")
        try:
            parsed = _extract_json(text)
            label = parsed.get("label", "")
            seg.label = Taxonomy(label) if label in _LABELS else Taxonomy.UNKNOWN
            seg.confidence = float(parsed.get("confidence", 0.0))
            seg.rationale = str(parsed.get("rationale", ""))
        except (ValueError, json.JSONDecodeError, KeyError):
            seg.label = Taxonomy.UNKNOWN
            seg.confidence = 0.0
            seg.rationale = f"parse_failure: {str(text)[:120]}"
