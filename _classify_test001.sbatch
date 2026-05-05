#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=nc_classify
#SBATCH --output=/scratch1/abhajha/csci576/work/classify_%j.out
#SBATCH --error=/scratch1/abhajha/csci576/work/classify_%j.err

set -e
cd /scratch1/abhajha/csci576
export HF_HOME=/scratch1/abhajha/.cache/huggingface
mkdir -p "$HF_HOME"
source /apps/conda/miniforge3/24.11.3/etc/profile.d/conda.sh
conda activate csci576
nvidia-smi
python _classify_only.py "${VIDEO_STEM:-test_001}"
