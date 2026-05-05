"""Topic-drift signal: how off-topic a segment is relative to the whole video.

Embeds the concatenated ASR (the video's overall topic) and each segment's ASR
with a small sentence-transformer (all-MiniLM-L12-v2). Cosine similarity to
the centroid is rescaled to a [0,1] drift score where higher = more off-topic.

This is the load-bearing signal for catching ads that are content-shaped (e.g.
synthetic ad insertion of an unrelated educational clip into an educational
video — no sponsor language, but topically distinct).
"""
from __future__ import annotations

from typing import List

import numpy as np

from .segment import EnrichedSegment

_MODEL = None
_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
MIN_TEXT_LEN = 10


def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    from sentence_transformers import SentenceTransformer
    _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def compute_topic_drift(segments: List[EnrichedSegment]) -> None:
    """Fill seg.topic_drift in place. Drift is roughly 1 - cosine_sim
    between segment ASR embedding and the video's centroid embedding."""
    if not segments:
        return

    # Build the centroid from segments that actually have ASR. A segment
    # with no speech contributes nothing to the topic.
    indexed = [(i, s) for i, s in enumerate(segments) if len(s.asr_text) >= MIN_TEXT_LEN]
    if not indexed:
        return

    model = _get_model()
    texts = [s.asr_text for _, s in indexed]

    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    # Duration-weighted centroid so a 5-min segment counts more than a 5s one.
    weights = np.array([(s.end_sec - s.start_sec) for _, s in indexed], dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)
    centroid = (embs * weights[:, None]).sum(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

    for (idx, _), emb in zip(indexed, embs):
        sim = float(np.dot(centroid, emb))
        # Map cos sim from [-1, 1] to drift in [0, 1] where 1 = very off-topic.
        # Empirically: own-topic segments cluster at sim ~0.7+, off-topic at ~0.3-0.5.
        # We compress by clamping low sims to 1.0 drift.
        drift = max(0.0, 1.0 - sim)
        segments[idx].topic_drift = drift

    # Segments with no ASR get a neutral drift so the LLM doesn't over-interpret.
    for s in segments:
        if len(s.asr_text) < MIN_TEXT_LEN:
            s.topic_drift = 0.0
