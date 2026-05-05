#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=nc_demo_daemon
#SBATCH --output=/scratch1/abhajha/csci576/work_demo/daemon_%j.out
#SBATCH --error=/scratch1/abhajha/csci576/work_demo/daemon_%j.err

set -e
cd /scratch1/abhajha/csci576
export HF_HOME=/scratch1/abhajha/.cache/huggingface
mkdir -p "$HF_HOME" demo_inbox work_demo demo_done
source /apps/conda/miniforge3/24.11.3/etc/profile.d/conda.sh
conda activate csci576
nvidia-smi | head -10
python _demo_daemon.py demo_inbox work_demo demo_done
