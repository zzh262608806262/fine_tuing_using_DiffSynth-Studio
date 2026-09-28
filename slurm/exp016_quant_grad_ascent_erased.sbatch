#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 2:00:00
#SBATCH -J exp016-quant-grad_ascent_erased
#SBATCH -o slurm/logs/exp016_quant_grad_ascent_erased_%j.out
#SBATCH -e slurm/logs/exp016_quant_grad_ascent_erased_%j.err

# Exp016: 量化单个臂
# 模板变量: grad_ascent_erased

set -e

echo "=================================================="
echo "Exp016: 量化 grad_ascent_erased"
echo "=================================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "=================================================="
echo ""

# 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

# 禁止 diffsynth 在线下载（使用本地模型）
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 输出目录
OUTPUT_DIR="models/quantized/exp016_grad_ascent_erased_nf4"

echo "量化配置:"
echo "  臂: grad_ascent_erased"
echo "  输出: $OUTPUT_DIR"
echo "  量化方法: NF4"
echo ""

# 运行量化
python scripts/exp016_quantize.py \
    --arm grad_ascent_erased \
    --output-dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "❌ 量化失败！"
    exit 1
fi

echo ""
echo "=================================================="
echo "✅ grad_ascent_erased 量化完成"
echo "完成时间: $(date)"
echo "=================================================="
