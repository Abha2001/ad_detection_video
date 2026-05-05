"""LLM fuser: per-segment evidence -> Taxonomy label.

Uses Claude Opus 4.7 with prompt caching. The taxonomy explanation + JSON
schema sit in the cached system prompt; only the per-segment evidence
varies between calls, so 90% of input tokens are read from cache after
the first segment.
"""
from __future__ import annotations

import json
from typing import Iterable

from .segment import EnrichedSegment, Taxonomy

MODEL = "claude-opus-4-7"

_LABELS = [t.value for t in Taxonomy if t != Taxonomy.UNKNOWN]

SYSTEM_PROMPT = """You classify a segment of a long-form video as either core content or one of several non-content categories.

Taxonomy:
- core_content: main subject of the video (host speaking, demo, the actual topic)
- intro: opening title sequence, brand bumper, theme music with title card
- outro: closing credits, end cards, "thanks for watching" sign-off
- sponsorship: paid promotional read; host pitches a product/service
- self_promo: creator promoting their own channel/patreon/merch/newsletter
- recap: repeated boilerplate from a prior episode/segment
- transition: short interstitial or stinger between sections
- dead_air: silence, blank/static frame, no useful content
- waiting_room: pre-stream "starting soon" / countdown / lobby screen
- filler: irrelevant insert; off-topic tangent adding no value

You receive a JSON object with these signals per segment:
- duration_sec, position_norm (0=start, 1=end of video)
- asr_text: speech transcript (may be empty)
- ocr_text: text visible on screen (may be empty)
- rms_mean, rms_silence_ratio (0-1, fraction under silence threshold)
- spectral_flatness_mean (~0 = music/tone, ~0.5 = speech, ~1 = noise)
- motion_energy (mean optical flow magnitude; low = static/holding)
- shot_count (number of shot cuts in the segment)
- fingerprint_hit (true if audio matches another episode)

Heuristics:
- High silence_ratio + low motion + position_norm < 0.05 -> waiting_room or intro
- High silence_ratio + low motion + minimal asr/ocr -> dead_air
- "sponsored", "promo code", "use my link" in asr_text -> sponsorship
- "subscribe", "patreon", "merch" mid-video -> self_promo
- fingerprint_hit -> recap
- Low motion + short duration + between sections -> transition
- Default to core_content when signals are ambiguous and asr_text is rich

Pick the single most-likely label. Confidence is 0.0-1.0."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": _LABELS},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["label", "confidence", "rationale"],
    "additionalProperties": False,
}


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
    }


def classify_segment(client, seg: EnrichedSegment) -> None:
    """Fill seg.label / confidence / rationale via a single Claude call."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        output_config={
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}
        },
        messages=[
            {
                "role": "user",
                "content": (
                    "Segment evidence:\n"
                    + json.dumps(_evidence(seg), indent=2)
                    + "\n\nClassify."
                ),
            }
        ],
    )
    text = next(b.text for b in response.content if b.type == "text")
    parsed = json.loads(text)

    try:
        seg.label = Taxonomy(parsed["label"])
    except (ValueError, KeyError):
        seg.label = Taxonomy.UNKNOWN
    seg.confidence = float(parsed.get("confidence", 0.0))
    seg.rationale = str(parsed.get("rationale", ""))


def classify_segments(segments: Iterable[EnrichedSegment]) -> None:
    """Classify all segments in place. Reuses the client so the cached
    system prompt is shared across all calls."""
    import anthropic

    client = anthropic.Anthropic()
    for seg in segments:
        classify_segment(client, seg)
