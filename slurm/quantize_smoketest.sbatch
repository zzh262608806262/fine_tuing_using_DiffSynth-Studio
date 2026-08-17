#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-50
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -J quant-smoke
#SBATCH -o slurm/logs/quant-smoke-%j.out
#SBATCH -e slurm/logs/quant-smoke-%j.err

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true

# 冒烟测试: 加载已保存的 nf4 量化 checkpoint, 2 步推理验证可用性
python scripts/quantize.py --mode load --method bitsandbytes_nf4 \
    --quantized_path ./models/quantized/Wan2.1-T2V-1.3B_nf4.safetensors \
    --num_inference_steps 2 \
    --output_video ./models/quantized/smoke_test.mp4

echo "DONE quantize smoke test"
ls -la ./models/quantized/
