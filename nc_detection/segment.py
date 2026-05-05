"""Segment data model for non-content detection."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class Taxonomy(str, Enum):
    CORE_CONTENT = "core_content"
    INTRO = "intro"
    OUTRO = "outro"
    SPONSORSHIP = "sponsorship"
    SELF_PROMO = "self_promo"
    RECAP = "recap"
    TRANSITION = "transition"
    DEAD_AIR = "dead_air"
    WAITING_ROOM = "waiting_room"
    FILLER = "filler"
    UNKNOWN = "unknown"


@dataclass
class EnrichedSegment:
    start_sec: float
    end_sec: float

    asr_text: str = ""
    ocr_text: str = ""
    rms_mean: float = 0.0
    rms_silence_ratio: float = 0.0
    spectral_flatness_mean: float = 0.0
    motion_energy: float = 0.0
    shot_count: int = 0
    fingerprint_hit: bool = False
    position_norm: float = 0.0
    topic_drift: float = 0.0  # 1 - cosine_sim to video-wide topic; higher = off-topic

    label: Taxonomy = Taxonomy.UNKNOWN
    confidence: float = 0.0
    rationale: str = ""

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    def to_dict(self) -> dict:
        return {
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "duration_sec": self.duration,
            "asr_text": self.asr_text,
            "ocr_text": self.ocr_text,
            "rms_mean": self.rms_mean,
            "rms_silence_ratio": self.rms_silence_ratio,
            "spectral_flatness_mean": self.spectral_flatness_mean,
            "motion_energy": self.motion_energy,
            "shot_count": self.shot_count,
            "fingerprint_hit": self.fingerprint_hit,
            "position_norm": self.position_norm,
            "topic_drift": self.topic_drift,
            "label": self.label.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


def segments_from_boundaries(
    boundaries_sec: List[float], total_duration: float
) -> List[EnrichedSegment]:
    """Build EnrichedSegments from a sorted list of cut timestamps (seconds)."""
    edges = [0.0] + sorted(b for b in boundaries_sec if 0 < b < total_duration) + [total_duration]
    out: List[EnrichedSegment] = []
    for s, e in zip(edges[:-1], edges[1:]):
        if e > s:
            out.append(
                EnrichedSegment(
                    start_sec=s,
                    end_sec=e,
                    position_norm=(s / total_duration) if total_duration > 0 else 0.0,
                )
            )
    return out
