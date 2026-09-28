#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 0:30:00
#SBATCH -J exp019_quantize
#SBATCH -o outputs/exp019_quantize_%j.log
#SBATCH -e outputs/exp019_quantize_%j.err

# Exp019 Phase 9: 量化GradDiff擦除模型

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

echo "========================================"
echo "Exp019 Phase 9 - 模型量化"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "========================================"

# 激活环境
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

export PYTHONNOUSERSITE=1

INPUT_DIR="models/wan5b/exp019_graddiff_erased"
OUTPUT_DIR="models/wan5b/exp019_graddiff_erased_int8"

echo ""
echo "Configuration:"
echo "  Input: $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Quantization: int8"
echo "========================================"

python scripts/exp019_quantize.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --dtype int8

echo ""
echo "========================================"
echo "✓ 量化完成"
echo "输出目录: $OUTPUT_DIR"
echo "完成时间: $(date)"
echo "========================================"
