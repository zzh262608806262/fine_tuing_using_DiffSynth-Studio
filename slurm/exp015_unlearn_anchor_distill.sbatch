#!/bin/bash
#SBATCH --job-name=exp015_unlearn_anchor_distill
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/exp015_unlearn_anchor_distill-%j.out
#SBATCH --error=slurm/logs/exp015_unlearn_anchor_distill-%j.err

# Exp 015 AnchorDistill 擦除训练（使用空 anchor）

set -euo pipefail

METHOD="anchor_distill"

echo "========================================"
echo "Exp 015 擦除训练: $METHOD (blank distillation)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "========================================"

FT_ROOT="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"
VU_ROOT="/home/x_jiage/jiage/video-unlearning"
cd "$FT_ROOT"

# 激活 VU 环境
echo "[ENV] 激活 VU 环境..."
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ -f "$VU_ROOT/.venv/bin/activate" ]; then
    echo "  使用 VU .venv 环境"
    source "$VU_ROOT/.venv/bin/activate"
elif conda env list | grep -q "video-unlearning"; then
    echo "  使用 conda video-unlearning 环境"
    conda activate video-unlearning
else
    echo "  [WARNING] 未找到 VU 专用环境，使用 diffsynth 环境"
    conda activate diffsynth
fi

# 环境验证
echo "[ENV] 验证依赖..."
python -c "import torch; print(f'  torch: {torch.__version__}')"
python -c "import diffusers; print(f'  diffusers: {diffusers.__version__}')"
python -c "import hydra; print(f'  hydra: {hydra.__version__}')" || echo "  [WARNING] hydra 未安装"

# 路径配置
MODEL_PATH="models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
LATENT_CACHE_DIR="data/wan5b/unlearn_latents"
OUTPUT_DIR="models/unlearn/exp015_wan5b_nudity_${METHOD}"

# 训练参数（与其他方法对齐）
UNLEARN_STEPS=600
SAVE_EVERY=100
LEARNING_RATE=1e-5
BATCH_SIZE=1

echo "[CONFIG] 训练配置:"
echo "  方法: $METHOD"
echo "  Anchor: '' (blank distillation)"
echo "  基座: $MODEL_PATH"
echo "  Latent 缓存: $LATENT_CACHE_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  训练步数: $UNLEARN_STEPS"
echo "  学习率: $LEARNING_RATE"
echo "  Batch size: $BATCH_SIZE"
echo ""

# 检查输入
if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] 基座不存在: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$LATENT_CACHE_DIR/latent_manifest.jsonl" ]; then
    echo "[ERROR] Latent manifest 不存在: $LATENT_CACHE_DIR/latent_manifest.jsonl"
    exit 1
fi

echo "  ✓ 输入文件检查通过"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 调用 VU 训练脚本（使用空 anchor）
echo ""
echo "[TRAIN] 开始 AnchorDistill 擦除训练 (blank distillation)..."
echo "========================================"

python3 scripts/run_unlearn_wan5b.py \
    --train \
    --method "AnchorDistill" \
    --anchor "" \
    --vu-root "$VU_ROOT" \
    --model-path "$MODEL_PATH" \
    --latent-manifest "$LATENT_CACHE_DIR/latent_manifest.jsonl" \
    --train-dir "$OUTPUT_DIR" \
    --unlearn-steps "$UNLEARN_STEPS" \
    --save-every "$SAVE_EVERY" \
    --learning-rate "$LEARNING_RATE" \
    --batch-size "$BATCH_SIZE"

if [ $? -ne 0 ]; then
    echo "[ERROR] 训练失败"
    exit 1
fi

echo ""
echo "========================================"
echo "[SUCCESS] $METHOD 擦除训练完成"
echo "  输出目录: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
echo "Time: $(date)"
echo "========================================"
