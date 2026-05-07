"""ASR-aware segment merging.

Visual + audio cuts can over-split a continuous sentence — for example,
a camera angle change while the host is mid-explanation produces two
shot-bounded segments inside one breath. Walking the boundaries against
word-level Whisper timestamps lets us drop the cuts that fall inside
continuous speech, leaving boundaries only where there's a real pause
or scene change with audio break.

Operates after ASR is assigned but before audio/motion/OCR feature
computation, so the downstream features describe the merged units.
"""
from __future__ import annotations

from typing import Iterable, List

from .segment import EnrichedSegment


def merge_speech_continuous(
    segments: List[EnrichedSegment],
    asr_segments: Iterable[dict],
    gap_threshold: float = 0.4,
) -> List[EnrichedSegment]:
    """Drop segment boundaries that cut mid-speech.

    For each boundary between two adjacent segments, find the last word
    that ended before it and the first word that started after it. If
    the gap between them is < gap_threshold seconds, the boundary is
    inside continuous speech — merge the second segment into the first.
    Boundaries with no word evidence on one side are kept (no merge).
    """
    if len(segments) <= 1:
        return list(segments)

    # Flatten all word entries across asr segments, sorted by start time.
    all_words: List[dict] = []
    for asr in asr_segments:
        ws = asr.get("words") or []
        all_words.extend(ws)
    all_words.sort(key=lambda w: w["start"])

    if not all_words:
        return list(segments)

    out: List[EnrichedSegment] = [segments[0]]
    for seg in segments[1:]:
        boundary = seg.start_sec  # equals previous segment's end_sec

        word_before = None
        word_after = None
        for w in all_words:
            if w["end"] < boundary:
                if word_before is None or w["end"] > word_before["end"]:
                    word_before = w
            elif w["start"] > boundary:
                word_after = w
                break

        merge = False
        if word_before is not None and word_after is not None:
            gap = word_after["start"] - word_before["end"]
            if gap < gap_threshold:
                merge = True

        if merge:
            prev = out[-1]
            prev.end_sec = seg.end_sec
            joined = (prev.asr_text + " " + seg.asr_text).strip()
            prev.asr_text = joined
            # ocr_text concatenated; aggregate later when OCR runs.
            if seg.ocr_text:
                prev.ocr_text = (prev.ocr_text + " | " + seg.ocr_text).strip(" |")
        else:
            out.append(seg)

    return out
