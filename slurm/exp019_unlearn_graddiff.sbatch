#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 0:30:00
#SBATCH -J exp019_graddiff_unlearn
#SBATCH -o outputs/exp019_graddiff_unlearn_%j.log
#SBATCH -e outputs/exp019_graddiff_unlearn_%j.err

set -euo pipefail

# 项目根目录
FT_ROOT="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"
VU_ROOT="/home/x_jiage/jiage/video-unlearning"
cd "$FT_ROOT"

# 激活 VU 环境
echo "[ENV] 激活 VU 环境..."
if [ -f "$VU_ROOT/.venv/bin/activate" ]; then
    echo "  使用 VU .venv 环境"
    source "$VU_ROOT/.venv/bin/activate"
else
    echo "  错误: 找不到 VU 环境"
    exit 1
fi

# 训练参数（与 Exp015 一致）
METHOD=GradDiff
STEPS=600
RETAIN_WEIGHT=1.0
BATCH_SIZE=1
LR=1e-5

# 路径
MODEL_PATH=models/Wan-AI/Wan2.2-TI2V-5B-Diffusers  # 使用 Diffusers 版本
LATENT_CACHE_DIR=data/wan5b/unlearn_latents  # 使用预先缓存的 latents
TRAIN_DIR=models/unlearn/exp019_wan5b_nudity_graddiff

# GradDiff 需要单独的 retain latent manifest（使用绝对路径）
RETAIN_LATENT_MANIFEST=${FT_ROOT}/${LATENT_CACHE_DIR}/retain_only_latent_manifest.jsonl

echo "  模型: ${MODEL_PATH}"
echo "  Latent manifest: ${LATENT_CACHE_DIR}/latent_manifest.jsonl"
echo "  Retain latent manifest: ${RETAIN_LATENT_MANIFEST}"
echo "  输出目录: ${TRAIN_DIR}"
echo "  步数: ${STEPS}, 保存间隔: 100"
echo "=========================================="

# 运行训练（GradDiff 需要 retain-manifest 指向包含 latent 字段的文件）
python scripts/run_unlearn_wan5b.py \
    --train \
    --method ${METHOD} \
    --model-path ${MODEL_PATH} \
    --latent-manifest ${LATENT_CACHE_DIR}/latent_manifest.jsonl \
    --retain-manifest ${RETAIN_LATENT_MANIFEST} \
    --train-dir ${TRAIN_DIR} \
    --unlearn-steps ${STEPS} \
    --batch-size ${BATCH_SIZE} \
    --learning-rate ${LR} \
    --save-every 100 \
    --retain-weight ${RETAIN_WEIGHT}

echo "Exp019 GradDiff 擦除训练完成"
