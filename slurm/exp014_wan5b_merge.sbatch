#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 02:00:00
#SBATCH -J exp014-merge
#SBATCH -o logs/exp014_merge_%j.out
#SBATCH -e logs/exp014_merge_%j.err

# Exp 014: 擦除LoRA合并+桥接到DiffSynth格式

set -e

echo "=================================================="
echo "Exp 014 - Merge Unlearning LoRA"
echo "=================================================="
echo "开始时间: $(date)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 配置
ADAPTER_PATH="models/unlearn/exp014_wan5b_nudity_600steps/adapter_final.pt"
BASE_DIR="models/Wan-AI/Wan2.2-TI2V-5B-Diffusers/transformer"
ERASED_OUT="models/unlearn/exp014_wan5b_nudity_600steps/merged"
BRIDGE_OUT="models/wan5b/exp014_erased"
BASE_BRIDGE_OUT="models/wan5b/exp014_base"

echo "📊 配置:"
echo "  - Adapter: $ADAPTER_PATH"
echo "  - Base dir: $BASE_DIR"
echo "  - Erased输出: $ERASED_OUT"
echo "  - Bridge输出: $BRIDGE_OUT"
echo ""

# 1. 合并擦除LoRA到基座并桥转为erased模型
echo "🔧 [1/2] 合并擦除LoRA并桥转..."
python scripts/merge_unlearn_lora_wan5b.py \
    --adapter "$ADAPTER_PATH" \
    --base-dir "$BASE_DIR" \
    --output-dir "$ERASED_OUT" \
    --bridge-output-dir "$BRIDGE_OUT" \
    --force

echo "✅ Erased模型完成"
echo ""

# 2. 桥转base模型（无擦除）
echo "🔧 [2/2] 桥转base模型..."
python scripts/convert_wan_5b_bridge.py \
    --input-dir "models/Wan-AI/Wan2.2-TI2V-5B-Diffusers/transformer" \
    --output-dir "$BASE_BRIDGE_OUT"

echo "✅ Base模型完成"
echo ""
echo "=================================================="
echo "✅ Exp 014合并阶段完成！"
echo "完成时间: $(date)"
echo ""
echo "产物："
echo "  - Base模型: $BASE_BRIDGE_OUT/"
echo "  - Erased模型: $BRIDGE_OUT/"
echo ""
echo "下一步: 运行微调脚本"
echo "  sbatch slurm/exp014_wan5b_finetune.sbatch"
echo "=================================================="
