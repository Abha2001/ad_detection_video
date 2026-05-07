"""Glue around the teammate's ad_detection_video shot detectors.

Produces shot boundaries as a list of timestamps in seconds, suitable for
feeding into segments_from_boundaries().
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import cv2

# Find shot_detection.py from teammate's repo. Two supported layouts:
#  1. dev: nc_detection/ sits next to a clone named "ad_detection_video/"
#  2. PR'd: nc_detection/ lives inside the ad_detection_video/ repo itself
def _find_shot_detection_dir():
    here = Path(__file__).resolve().parent  # nc_detection/
    # PR'd layout: shot_detection.py is in nc_detection's parent (repo root).
    if (here.parent / "shot_detection.py").exists():
        return here.parent
    # Dev layout: a sibling directory named "ad_detection_video" alongside
    # nc_detection (i.e. one level up from boundary.py).
    sibling = here.parent / "ad_detection_video"
    if (sibling / "shot_detection.py").exists():
        return sibling
    return None

_AD_REPO = _find_shot_detection_dir()
if _AD_REPO is not None and str(_AD_REPO) not in sys.path:
    sys.path.insert(0, str(_AD_REPO))


_DETECTOR_REGISTRY = {
    "frame_diff": "FrameDifferenceDetector",
    "histogram": "HistogramDifferenceDetector",
    "edge": "EdgeChangeRatioDetector",
    "black_frame": "BlackFrameTransitionDetector",
    "transnet_clip": "HybridTransNetCLIPDetector",  # requires torch + open_clip
}


def detect_boundaries_sec(
    video_path: Path,
    detector_key: str = "frame_diff",
    sample_every: int = 5,
    adaptive_k: float = 3.0,
) -> List[float]:
    """Run a teammate's shot boundary detector and return cut times in seconds."""
    import shot_detection  # type: ignore[import-not-found]

    cls_name = _DETECTOR_REGISTRY.get(detector_key)
    if cls_name is None:
        raise KeyError(
            f"Unknown detector_key {detector_key!r}; "
            f"options: {list(_DETECTOR_REGISTRY)}"
        )
    detector = getattr(shot_detection, cls_name)()
    result = detector.detect(
        video_path, sample_every=sample_every, adaptive_k=adaptive_k
    )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    return sorted({f / fps for f in result.detected_frames})
