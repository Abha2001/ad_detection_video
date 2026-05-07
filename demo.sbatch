#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=nc_demo
#SBATCH --output=/scratch1/abhajha/csci576/work_demo/demo_%j.out
#SBATCH --error=/scratch1/abhajha/csci576/work_demo/demo_%j.err

set -e
cd /scratch1/abhajha/csci576
export HF_HOME=/scratch1/abhajha/.cache/huggingface
mkdir -p "$HF_HOME" work_demo
source /apps/conda/miniforge3/24.11.3/etc/profile.d/conda.sh
conda activate csci576

VIDEO="${VIDEO:-$1}"
if [ -z "$VIDEO" ]; then
  echo "usage: VIDEO=/path/to/file.mp4 sbatch demo.sbatch" >&2
  exit 1
fi

nvidia-smi | head -10
python _demo.py "$VIDEO" --work-dir work_demo
