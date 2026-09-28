#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -J exp016-regen-ARMNAME
#SBATCH -o slurm/logs/exp016_regen_ARMNAME_%j.out
#SBATCH -e slurm/logs/exp016_regen_ARMNAME_%j.err

# Exp016: 补充生成缺失视频
# 模板变量: ARMNAME

set -e

# 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface

echo "=================================================="
echo "Exp016: 补充生成缺失视频 - ARMNAME"
echo "=================================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "=================================================="

cd /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio

python -u scripts/exp016_generate_missing.py --arm ARMNAME

echo ""
echo "=================================================="
echo "✅ ARMNAME 补充生成完成"
echo "完成时间: $(date)"
echo "=================================================="
