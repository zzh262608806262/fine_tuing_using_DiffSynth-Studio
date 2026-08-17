#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-50
#SBATCH --gpus=1
#SBATCH -t 04:00:00
#SBATCH -J quant-save
#SBATCH -o slurm/logs/quant-save-%j.out
#SBATCH -e slurm/logs/quant-save-%j.err

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface

# 保存 nf4 量化 checkpoint（缺失的量化权重）
python scripts/quantize.py --mode save --method bitsandbytes_nf4 \
    --output_path ./models/quantized/Wan2.1-T2V-1.3B_nf4

echo "DONE quantize save"
ls -la ./models/quantized/
