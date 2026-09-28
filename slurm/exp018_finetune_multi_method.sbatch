#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 3-00:00:00
#SBATCH -J exp018-ft-%x
#SBATCH -o slurm/logs/exp018_finetune_%x-%j.out
#SBATCH -e slurm/logs/exp018_finetune_%x-%j.err

# ============================================================================
# Exp018: 多方法擦除模型的高密度微调（repeat=25, epochs=20）
#
# 目的: 对 Exp015 的 4 种擦除方法进行细粒度微调，生成 20 个 checkpoints
#       用于分析不同擦除方法的安全回潮曲线
#
# 方法: esd, npo, grad_ascent, anchor_distill
# 基于: Exp014 v4 脚本（已解决环境和 tensorboard 问题）
# ============================================================================

set -euo pipefail

# 从环境变量或参数获取方法名
METHOD="${METHOD:-$1}"
if [ -z "$METHOD" ]; then
    echo "❌ 错误: 必须指定方法名"
    echo "用法: METHOD=<method> sbatch exp018_finetune_multi_method.sbatch"
    echo "方法: esd, npo, grad_ascent, anchor_distill"
    exit 1
fi

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# ---- 环境配置 ----
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

export LD_LIBRARY_PATH="/home/x_jiage/.conda/envs/diffsynth/lib:$LD_LIBRARY_PATH"
export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0

mkdir -p slurm/logs

echo "=================================================="
echo "Exp018 - ${METHOD} 擦除模型高密度微调"
echo "=================================================="
echo "节点: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo ""
echo "配置:"
echo "  方法: ${METHOD}"
echo "  Dataset repeat: 25"
echo "  Epochs: 20"
echo "  Checkpoints: 20 个（每 epoch 保存）"
echo "=================================================="
echo ""

# ---- 路径配置 ----
ERASED_MODEL="models/wan5b/exp015_${METHOD}_erased"
OUTPUT_PATH="models/finetune/exp018_${METHOD}_ft"

# ---- 前置检查 ----
echo "🔍 前置检查:"
echo "-----------------------------------"

if [ ! -d "$ERASED_MODEL" ]; then
    echo "❌ 错误: 擦除模型不存在: $ERASED_MODEL"
    exit 1
fi

# 检查必需的模型文件
required_files=(
    "diffusion_pytorch_model-00001-of-00003.safetensors"
    "diffusion_pytorch_model-00002-of-00003.safetensors"
    "diffusion_pytorch_model-00003-of-00003.safetensors"
    "models_t5_umt5-xxl-enc-bf16.safetensors"
    "Wan2.2_VAE.safetensors"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$ERASED_MODEL/$file" ]; then
        echo "❌ 错误: 缺少文件: $ERASED_MODEL/$file"
        exit 1
    fi
done

if [ ! -f "data/tiger_dataset/metadata_100.csv" ]; then
    echo "❌ 错误: 训练数据不存在: data/tiger_dataset/metadata_100.csv"
    exit 1
fi

echo "✅ 擦除模型: $ERASED_MODEL"
echo "✅ 训练数据: data/tiger_dataset/metadata_100.csv"
echo "✅ 输出路径: $OUTPUT_PATH"
echo ""

# ---- 执行微调 ----
echo "🚀 开始微调 ${METHOD}_erased"
echo "-----------------------------------"
echo "开始时间: $(date)"
echo ""

# 构造 model_id_with_origin_paths 参数
MODEL_ID="wan5b/exp015_${METHOD}_erased"
MODEL_PATHS="${MODEL_ID}:diffusion_pytorch_model*.safetensors,${MODEL_ID}:models_t5_umt5-xxl-enc-bf16.safetensors,${MODEL_ID}:Wan2.2_VAE.safetensors"

python -u scripts/finetune_wan5b.py \
    --arm base \
    --model_id_with_origin_paths "$MODEL_PATHS" \
    --dataset_repeat 25 \
    --num_epochs 20 \
    --output_path "$OUTPUT_PATH" \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --use_gradient_checkpointing \
    --no_tensorboard_log

echo ""
echo "-----------------------------------"
echo "完成时间: $(date)"
echo ""

# ---- 验证产物 ----
echo "🔍 验证产物:"
echo "-----------------------------------"

if [ -d "$OUTPUT_PATH" ]; then
    echo "✅ 输出目录存在: $OUTPUT_PATH"

    # 统计 checkpoint 数量
    checkpoint_count=$(find "$OUTPUT_PATH" -name "epoch-*.safetensors" | wc -l)
    echo "  Checkpoint 数量: $checkpoint_count (预期 20)"

    if [ "$checkpoint_count" -eq 20 ]; then
        echo "  ✅ Checkpoint 数量正确"
    else
        echo "  ⚠️  Checkpoint 数量不符（预期 20）"
    fi

    # 显示文件大小
    echo ""
    echo "  产物文件:"
    ls -lh "$OUTPUT_PATH"/*.safetensors 2>/dev/null | tail -5 || echo "  (无 safetensors 文件)"
else
    echo "❌ 输出目录不存在: $OUTPUT_PATH"
    exit 1
fi

echo ""
echo "=================================================="
echo "Exp018 ${METHOD} 微调完成"
echo "=================================================="
