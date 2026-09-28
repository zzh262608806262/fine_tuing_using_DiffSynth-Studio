#!/bin/bash
#SBATCH --job-name=exp015_unlearn_npo
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/exp015_unlearn_npo-%j.out
#SBATCH --error=slurm/logs/exp015_unlearn_npo-%j.err

# Exp 015 多方法擦除训练模板
# 用法: 复制并替换 npo 为具体方法名（grad_diff/esd/npo/anchor_distill）

set -euo pipefail

METHOD="npo"  # 默认方法，实际使用时会被替换

echo "========================================"
echo "Exp 015 擦除训练: $METHOD"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "========================================"

# 项目根目录
FT_ROOT="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"
VU_ROOT="/home/x_jiage/jiage/video-unlearning"
cd "$FT_ROOT"

# 激活 VU 环境（擦除训练需要 VU 的依赖）
echo "[ENV] 激活 VU 环境..."
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"

# 检查 VU 是否有 conda 环境
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

# 配置参数
MODEL_PATH="$FT_ROOT/models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
FORGET_MANIFEST="$VU_ROOT/data/splits/nudity_forget_composed_wan.jsonl"
RETAIN_MANIFEST="$VU_ROOT/data/splits/nudity_retain_composed_wan.jsonl"
LATENT_CACHE_DIR="$FT_ROOT/data/wan5b/unlearn_latents"
OUTPUT_DIR="$FT_ROOT/models/unlearn/exp015_wan5b_nudity_${METHOD}"

# 训练参数（统一配置）
UNLEARN_STEPS=600
SAVE_EVERY=100
LORA_RANK=8
LORA_ALPHA=16.0
LEARNING_RATE=1e-5
BATCH_SIZE=1

echo "[CONFIG] 训练配置:"
echo "  方法: $METHOD"
echo "  模型: $MODEL_PATH"
echo "  Forget manifest: $FORGET_MANIFEST ($(wc -l < "$FORGET_MANIFEST") 条)"
echo "  Retain manifest: $RETAIN_MANIFEST ($(wc -l < "$RETAIN_MANIFEST") 条)"
echo "  Latent 缓存: $LATENT_CACHE_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  步数: $UNLEARN_STEPS, 保存间隔: $SAVE_EVERY"
echo "  LoRA: rank=$LORA_RANK, alpha=$LORA_ALPHA"
echo "  学习率: $LEARNING_RATE, batch_size=$BATCH_SIZE"

# 检查 latent 缓存是否存在
if [ ! -d "$LATENT_CACHE_DIR" ]; then
    echo "[ERROR] Latent 缓存不存在: $LATENT_CACHE_DIR"
    echo "  请先运行缓存阶段（复用 Exp 012 或执行 --cache）"
    exit 1
fi

LATENT_COUNT=$(find "$LATENT_CACHE_DIR" -name "*.pt" | wc -l)
echo "  ✓ Latent 缓存已存在 ($LATENT_COUNT 个文件)"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm/logs

# 执行擦除训练
echo ""
echo "[TRAIN] 开始 $METHOD 擦除训练..."
echo "========================================"


python3 scripts/run_unlearn_wan5b.py \
    --train \
    --method "NPO" \
    --vu-root "$VU_ROOT" \
    --model-path "$MODEL_PATH" \
    --latent-manifest "$LATENT_CACHE_DIR/latent_manifest.jsonl" \
    --train-dir "$OUTPUT_DIR" \
    --unlearn-steps "$UNLEARN_STEPS" \
    --save-every "$SAVE_EVERY" \
    --learning-rate "$LEARNING_RATE" \
    --batch-size "$BATCH_SIZE"

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] $METHOD 擦除训练完成"
    echo "  输出目录: $OUTPUT_DIR"
    ls -lh "$OUTPUT_DIR"
else
    echo "[FAILED] $METHOD 擦除训练失败 (exit code: $EXIT_CODE)"
fi
echo "Time: $(date)"
echo "========================================"

exit $EXIT_CODE
