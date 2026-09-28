#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 2:00:00
#SBATCH -J exp016-fp16-MODELNAME
#SBATCH -o slurm/logs/exp016_fp16_MODELNAME_%j.out
#SBATCH -e slurm/logs/exp016_fp16_MODELNAME_%j.err

# Exp016 Stage 0: FP16基线视频生成（纯模型，无LoRA）
# 模板变量: MODELNAME

set -e

# 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface

echo "=================================================="
echo "Exp016 Stage 0: 生成FP16基线 - MODELNAME"
echo "=================================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "=================================================="
echo ""
echo "生成配置:"
echo "  模型: MODELNAME"
echo "  提示词数: 99"
echo "  输出: outputs/exp016_baseline/MODELNAME"
echo ""

cd /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio

python -u scripts/exp016_generate_fp16.py \
    --model MODELNAME \
    --start 0 \
    --end 99 \
    --output-dir "outputs/exp016_baseline/MODELNAME"

echo ""
echo "=================================================="
echo "✅ MODELNAME FP16基线生成完成"
echo "完成时间: $(date)"
echo "=================================================="
