#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --job-name=exp020a_unlearn_npo_masked
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/exp020a_unlearn-%j.out
#SBATCH --error=slurm/logs/exp020a_unlearn-%j.err

# Exp020a NPOMasked擦除训练
# 用法: sbatch slurm/exp020a_unlearn_npo_masked.sbatch

set -euo pipefail

echo "========================================"
echo "Exp020a NPOMasked擦除训练"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "========================================"

# 项目根目录
FT_ROOT="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"
VU_ROOT="/home/x_jiage/jiage/video-unlearning"
cd "$FT_ROOT"

# 激活VU环境（masked方法需要VU的依赖）
echo "[ENV] 激活VU环境..."
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"

# 检查VU是否有conda环境
if [ -f "$VU_ROOT/.venv/bin/activate" ]; then
    echo "  使用VU .venv环境"
    source "$VU_ROOT/.venv/bin/activate"
elif conda env list | grep -q "video-unlearning"; then
    echo "  使用conda video-unlearning环境"
    conda activate video-unlearning
else
    echo "  [WARNING] 未找到VU专用环境，使用diffsynth环境"
    conda activate diffsynth
fi

# 环境验证
echo "[ENV] 验证依赖..."
python -c "import torch; print(f'  torch: {torch.__version__}')"
python -c "import diffusers; print(f'  diffusers: {diffusers.__version__}')"

# 配置参数
MODEL_PATH="$FT_ROOT/models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
LATENT_CACHE_DIR="$FT_ROOT/data/wan5b/unlearn_latents"
OUTPUT_DIR="$FT_ROOT/models/unlearn/exp020a_wan5b_nudity_npo_masked"

# 训练参数（与Exp015对齐）
UNLEARN_STEPS=600
SAVE_EVERY=100
LEARNING_RATE=1e-5
BATCH_SIZE=1

echo "[CONFIG] 训练配置:"
echo "  方法: NPOMasked"
echo "  模型: $MODEL_PATH"
echo "  Latent缓存: $LATENT_CACHE_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  步数: $UNLEARN_STEPS, 保存间隔: $SAVE_EVERY"
echo "  学习率: $LEARNING_RATE, batch_size: $BATCH_SIZE"

# 检查latent缓存
if [ ! -d "$LATENT_CACHE_DIR" ]; then
    echo "[ERROR] Latent缓存不存在: $LATENT_CACHE_DIR"
    echo "  请先运行缓存阶段（复用Exp015的缓存）"
    exit 1
fi

LATENT_COUNT=$(find "$LATENT_CACHE_DIR" -name "*.pt" | wc -l)
echo "  ✓ Latent缓存已存在 ($LATENT_COUNT 个文件)"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 执行擦除训练
echo ""
echo "[TRAIN] 开始NPOMasked擦除训练..."
echo "========================================"

python3 scripts/run_unlearn_wan5b.py \
    --train \
    --method "NPOMasked" \
    --vu-root "$VU_ROOT" \
    --model-path "$MODEL_PATH" \
    --latent-manifest "$LATENT_CACHE_DIR/latent_manifest.jsonl" \
    --train-dir "$OUTPUT_DIR" \
    --erase-concept "breasts" \
    --unlearn-steps "$UNLEARN_STEPS" \
    --save-every "$SAVE_EVERY" \
    --learning-rate "$LEARNING_RATE" \
    --batch-size "$BATCH_SIZE"

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] NPOMasked擦除训练完成"
    echo "  输出目录: $OUTPUT_DIR"
    ls -lh "$OUTPUT_DIR"
else
    echo "[FAILED] NPOMasked擦除训练失败 (exit code: $EXIT_CODE)"
fi
echo "Time: $(date)"
echo "========================================"

exit $EXIT_CODE
