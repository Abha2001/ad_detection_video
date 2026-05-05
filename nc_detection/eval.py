"""Binary evaluation against a video_info JSON (ad vs content).

Computes per-second confusion matrix and precision/recall/F1 between
predicted non-content segments and ground-truth ad intervals.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from .segment import EnrichedSegment, Taxonomy

Interval = Tuple[float, float]


def load_gt_ad_intervals(json_path: Path) -> List[Interval]:
    data = json.loads(json_path.read_text())
    return [
        (a["final_video_ad_start_seconds"], a["final_video_ad_end_seconds"])
        for a in data.get("inserted_ads", [])
    ]


def load_gt_duration(json_path: Path) -> float:
    data = json.loads(json_path.read_text())
    return float(
        data.get("output_duration_seconds")
        or data.get("original_video_duration_seconds")
        or 0.0
    )


def is_predicted_nc(seg: EnrichedSegment) -> bool:
    return seg.label not in (Taxonomy.CORE_CONTENT, Taxonomy.UNKNOWN)


def binary_metrics(
    segments: List[EnrichedSegment],
    gt_ads: List[Interval],
    total_duration: float,
    step: float = 1.0,
) -> dict:
    """Per-second binary confusion matrix between predicted non-content and GT ads."""
    n = max(1, int(total_duration / step) + 1)

    pred = [False] * n
    truth = [False] * n

    def fill(arr: List[bool], s: float, e: float) -> None:
        si = max(0, int(s / step))
        ei = min(n, int(e / step) + 1)
        for i in range(si, ei):
            arr[i] = True

    for seg in segments:
        if is_predicted_nc(seg):
            fill(pred, seg.start_sec, seg.end_sec)

    for s, e in gt_ads:
        fill(truth, s, e)

    tp = sum(1 for i in range(n) if pred[i] and truth[i])
    fp = sum(1 for i in range(n) if pred[i] and not truth[i])
    fn = sum(1 for i in range(n) if not pred[i] and truth[i])
    tn = sum(1 for i in range(n) if not pred[i] and not truth[i])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp_sec": tp,
        "fp_sec": fp,
        "fn_sec": fn,
        "tn_sec": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support_ad_sec": tp + fn,
        "support_content_sec": fp + tn,
    }
