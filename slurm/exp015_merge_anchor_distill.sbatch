#!/bin/bash
#SBATCH --job-name=exp015_merge_anchor_distill
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/exp015_merge_anchor_distill-%j.out
#SBATCH --error=slurm/logs/exp015_merge_anchor_distill-%j.err

# Exp 015 LoRA 合并脚本 - AnchorDistill
# 将擦除 LoRA 合并回 diffusers 基座，然后桥转换为 DiffSynth 格式

set -euo pipefail

METHOD="anchor_distill"

echo "========================================"
echo "Exp 015 LoRA 合并: $METHOD"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "========================================"

FT_ROOT="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"
VU_ROOT="/home/x_jiage/jiage/video-unlearning"
cd "$FT_ROOT"

# 激活环境（合并需要 diffusers）
echo "[ENV] 激活环境..."
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"

# VU .venv 有完整的 diffusers
if [ -f "$VU_ROOT/.venv/bin/activate" ]; then
    echo "  使用 VU .venv 环境"
    source "$VU_ROOT/.venv/bin/activate"
else
    echo "[ERROR] 未找到 VU .venv 环境"
    exit 1
fi

# 验证环境
python -c "import torch; print(f'  torch: {torch.__version__}')"
python -c "import diffusers; print(f'  diffusers: {diffusers.__version__}')"

# 路径配置
BASE_DIFFUSERS="models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
UNLEARN_LORA_DIR="models/unlearn/exp015_wan5b_nudity_${METHOD}"
OUTPUT_ERASED_DIFFUSERS="models/wan5b/exp015_${METHOD}_erased_diffusers"
OUTPUT_ERASED_DIFFSYNTH="models/wan5b/exp015_${METHOD}_erased"

echo "[CONFIG] 合并配置:"
echo "  方法: $METHOD"
echo "  基座: $BASE_DIFFUSERS"
echo "  擦除 LoRA: $UNLEARN_LORA_DIR"
echo "  输出 (diffusers): $OUTPUT_ERASED_DIFFUSERS"
echo "  输出 (DiffSynth): $OUTPUT_ERASED_DIFFSYNTH"

# 检查输入
if [ ! -d "$BASE_DIFFUSERS" ]; then
    echo "[ERROR] 基座不存在: $BASE_DIFFUSERS"
    exit 1
fi

if [ ! -d "$UNLEARN_LORA_DIR" ]; then
    echo "[ERROR] 擦除 LoRA 不存在: $UNLEARN_LORA_DIR"
    echo "  请先完成 Stage 1（擦除训练）"
    exit 1
fi

echo "  ✓ 输入文件检查通过"

# 创建输出目录
mkdir -p "$(dirname "$OUTPUT_ERASED_DIFFUSERS")"
mkdir -p "$(dirname "$OUTPUT_ERASED_DIFFSYNTH")"

# 执行合并（复用 Exp 012 的 merge_unlearn_lora_wan5b.py）
echo ""
echo "[MERGE] 合并擦除 LoRA..."
echo "========================================"

python3 scripts/merge_unlearn_lora_wan5b.py \
    --adapter "$UNLEARN_LORA_DIR/adapter_final.pt" \
    --base-dir "$BASE_DIFFUSERS/transformer" \
    --output-dir "$OUTPUT_ERASED_DIFFUSERS/transformer" \
    --bridge-output-dir "$OUTPUT_ERASED_DIFFSYNTH" \
    --force

if [ $? -ne 0 ]; then
    echo "[ERROR] LoRA 合并失败"
    exit 1
fi

echo "  ✓ LoRA 合并+桥转换完成"

# 创建 T5/VAE 软链接（使用绝对路径）
echo ""
echo "[LINK] 创建 T5/VAE 软链接..."
SHARED_DIR="models/DiffSynth-Studio/Wan-Series-Converted-Safetensors"
T5_FILE="models_t5_umt5-xxl-enc-bf16.safetensors"
VAE_FILE="Wan2.2_VAE.safetensors"

for f in "$T5_FILE" "$VAE_FILE"; do
    src="$PWD/$SHARED_DIR/$f"
    dst="$OUTPUT_ERASED_DIFFSYNTH/$f"
    if [ -f "$src" ]; then
        ln -sf "$src" "$dst"
        echo "  ✓ $f -> $(readlink -f "$dst")"
    else
        echo "  [ERROR] 共享文件不存在: $src"
        exit 1
    fi
done

echo ""
echo "========================================"
echo "[SUCCESS] $METHOD 合并完成"
echo "  Diffusers 格式: $OUTPUT_ERASED_DIFFUSERS"
echo "  DiffSynth 格式: $OUTPUT_ERASED_DIFFSYNTH"
echo "Time: $(date)"
echo "========================================"
