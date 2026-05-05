# Multimodal Non-Content Segmentation

CSCI 576 final project — a multimodal pipeline that segments long-form video
into core content vs a 10-class non-content taxonomy (intro, outro,
sponsorship, self-promo, recap, transition, dead air, waiting room, filler).

## Architecture

```
video.mp4 -> boundary detection (visual frame-diff + audio silence) -> segments
  per segment:
    Whisper ASR (word-aligned)
    librosa: RMS, silence ratio, spectral flatness
    optical-flow motion energy
    EasyOCR (sponsor URLs, channel watermarks, "subscribe" CTAs)
    sentence-transformers topic-drift vs video centroid
    MFCC fingerprint (cross-episode recap detection)
  -> Llama 3.2 3B fuser with structured rule prompt -> {label, confidence, rationale}
  -> deterministic post-processor (position-rule sanity)
  -> player UI with timeline + per-class actions
```

## Modules (`nc_detection/`)

- `segment.py` — `EnrichedSegment` dataclass + `Taxonomy` enum
- `boundary.py` — wraps `ad_detection_video/shot_detection.py` for visual cuts
- `audio_boundaries.py` — silence-interval midpoints from RMS dips
- `audio_features.py` — ffmpeg WAV extract, faster-whisper ASR (word timestamps), librosa stats
- `visual_features.py` — optical-flow motion energy, shot-count aggregation
- `ocr_features.py` — EasyOCR on sampled frames per segment
- `topic.py` — sentence-transformers cosine-distance to video centroid
- `fingerprint.py` — MFCC cross-correlation for recap detection
- `classify.py` — Llama 3.2 3B classifier with structured-JSON output
- `classify_anthropic.py` — alternate Claude API backend (kept as reference)
- `eval.py` — binary precision/recall/F1 vs ground-truth ad intervals
- `run.py` — end-to-end driver
- `bench.py` — corpus-level benchmark walker

## Demo flow (live, ~5-7 min runtime budget)

```bash
# ~5 min before demo — pre-warm GPU + models
sbatch demo_daemon.sbatch
# wait for "[daemon] ready" in work_demo/daemon_*.out

# during demo — drop fresh video
cp /path/to/unseen.mp4 demo_inbox/
# daemon picks it up within 2s, processes ~3-5 min on A40
# open player at:
#   .../player/?video=../work_demo/unseen.mp4&segs=../work_demo/unseen.segments.classified.json
```

## Player UI (`player/index.html`)

Single-file HTML5 video player with:
- canvas-rendered timeline colored by class
- drag-to-scrub, click-to-seek, hover tooltips
- per-class action selector (play / fast-forward / skip)
- auto-skip toggle
- ad-hoc URL params: `?video=...&segs=...` for any pair
- keyboard: `j` prev seg / `l` next seg / `k` play-pause / `s` next non-content

Serve with cache-busting headers:
```bash
python player/serve_nocache.py 8000
```

## Built on

The `ad_detection_video/` directory is teammate's repo, kept as a
working dependency for shot-boundary detection (`shot_detection.py`).
