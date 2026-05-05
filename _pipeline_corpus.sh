#!/bin/bash
set -e
cd /scratch1/abhajha/csci576
source /apps/conda/miniforge3/24.11.3/etc/profile.d/conda.sh
conda activate csci576

for i in test_002 test_003 test_004 test_005; do
  echo "=== $i ==="
  python _process_one.py "sample_videos/csci576/videos_with_ads/${i}.mp4" \
    --whisper-model base --adaptive-k 1.0 --work-dir work
done
echo "=== fingerprint pass ==="
python _fingerprint_all.py work --threshold 0.85
