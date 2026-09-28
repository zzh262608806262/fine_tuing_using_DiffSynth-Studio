#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -J exp016-quant-MODELNAME
#SBATCH -o slurm/logs/exp016_quant_MODELNAME_%j.out
#SBATCH -e slurm/logs/exp016_quant_MODELNAME_%j.err

# Exp016 Stage 1: 量化模型为NF4
# 模板变量: MODELNAME

set -e

# 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface

echo "=================================================="
echo "Exp016 Stage 1: 量化 MODELNAME"
echo "=================================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "=================================================="
echo ""

cd /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio

python -u scripts/exp016_quantize_pure.py \
    --model MODELNAME \
    --output-dir "models/quantized/exp016_MODELNAME_nf4"

echo ""
echo "=================================================="
echo "✅ MODELNAME 量化完成"
echo "完成时间: $(date)"
echo "=================================================="
