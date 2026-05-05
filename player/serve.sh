#!/bin/bash
# Serve the player + all video/segments artifacts at http://<host>:8000/player/
cd "$(dirname "$0")/.." || exit 1
exec python3 -m http.server 8000 --bind 0.0.0.0
