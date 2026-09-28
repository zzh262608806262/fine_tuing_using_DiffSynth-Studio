#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -C fat
#SBATCH -t 02:00:00
#SBATCH -J exp014-stage1-test
#SBATCH -o slurm/logs/exp014_stage1_test-%j.out
#SBATCH -e slurm/logs/exp014_stage1_test-%j.err

# ============================================================================
# Exp014 Stage 1 测试：擦除训练（使用 VU 标准配置）
#
# 测试目标：
#   - 验证环境配置正确
#   - 测试 600 步擦除训练（VU 标准）
#   - 确认脚本和参数都对齐
#
# 与 Exp012 的关键差异：
#   - UNLEARN_STEPS: 1200 → 600 (VU 标准配置)
#   - SAVE_EVERY: 200 → 100 (生成 6 个 checkpoint)
# ============================================================================

set -euo pipefail

# ---- 环境配置（完全照抄 exp012 成功的配置） ----
VU_REPO="/home/x_jiage/jiage/video-unlearning"
FT_ROOT="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"
VU_VENV_ACTIVATE="$VU_REPO/.venv/bin/activate"
VU_CONDA_ENV="diffsynth"

# ---- Exp014 专用配置 ----
METHOD="GradAscent"
UNLEARN_STEPS=600      # VU 标准（exp012 用的是 1200）
SAVE_EVERY=100         # 每 100 步保存（生成 6 个 checkpoint）
TRAIN_DIR="models/unlearn/exp014_wan5b_nudity_600steps"

# ---- 数据路径（复用 exp012 的缓存数据） ----
WAN_MODEL_PATH="$FT_ROOT/models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
CACHE_FORGET="$FT_ROOT/data/wan5b/unlearn_baseline_forget.jsonl"
CACHE_RETAIN="$FT_ROOT/data/wan5b/unlearn_baseline_retain.jsonl"
CACHE_H=480
CACHE_W=736

cd "$FT_ROOT"

# ---- 激活环境（完全照抄 exp012） ----
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
if [ -f "$VU_VENV_ACTIVATE" ]; then
  echo "✓ 使用 VU uv venv: $VU_VENV_ACTIVATE"
  source "$VU_VENV_ACTIVATE"
else
  echo "✓ 兜底 conda activate $VU_CONDA_ENV"
  conda activate "$VU_CONDA_ENV"
fi

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/proj/berzelius-aiics-real/users/x_jiage/huggingface_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1

mkdir -p slurm/logs

echo "=================================================="
echo "Exp014 Stage 1 测试 - 擦除训练"
echo "=================================================="
echo "节点: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo ""
echo "配置:"
echo "  方法: $METHOD"
echo "  步数: $UNLEARN_STEPS (VU 标准)"
echo "  保存间隔: $SAVE_EVERY"
echo "  输出目录: $TRAIN_DIR"
echo "=================================================="
echo ""

# ---- 环境检查 ----
echo "🔍 阶段 0: 环境检查"
echo "-----------------------------------"

echo "✓ Python 版本:"
python --version

echo ""
echo "✓ 关键脚本存在性:"
for script in run_unlearn_wan5b.py merge_unlearn_lora_wan5b.py; do
    if [ -f "scripts/$script" ]; then
        echo "  ✓ scripts/$script"
    else
        echo "  ✗ scripts/$script (缺失)"
        exit 1
    fi
done

echo ""
echo "✓ 模型路径:"
if [ -d "$WAN_MODEL_PATH" ]; then
    echo "  ✓ $WAN_MODEL_PATH"
else
    echo "  ✗ $WAN_MODEL_PATH (缺失)"
    exit 1
fi

echo ""
echo "✓ 缓存数据:"
for manifest in "$CACHE_FORGET" "$CACHE_RETAIN"; do
    if [ -f "$manifest" ]; then
        echo "  ✓ $manifest"
    else
        echo "  ✗ $manifest (缺失)"
        exit 1
    fi
done

echo ""
echo "=================================================="
echo "环境检查通过！开始训练..."
echo "=================================================="
echo ""

# ---- Stage 1: 缓存 latent（如果还没缓存的话） ----
LATENT_DIR="data/wan5b/unlearn_latents"
if [ -d "$LATENT_DIR" ] && [ "$(ls -A $LATENT_DIR/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "✓ 检测到已有 latent 缓存，跳过缓存步骤"
else
    echo "🔄 阶段 1a: 缓存 latent"
    echo "-----------------------------------"
    python3 -u scripts/run_unlearn_wan5b.py \
        --vu-root "$VU_REPO" \
        --cache \
        --forget-manifest "$CACHE_FORGET" \
        --retain-manifest "$CACHE_RETAIN" \
        --cache-height "$CACHE_H" \
        --cache-width "$CACHE_W" \
        --model-path "$WAN_MODEL_PATH"

    echo ""
    echo "✅ Latent 缓存完成"
    echo ""
fi

# ---- Stage 1b: 擦除训练（600步） ----
echo "🔥 阶段 1b: 擦除训练（$METHOD, $UNLEARN_STEPS 步）"
echo "-----------------------------------"

python3 -u scripts/run_unlearn_wan5b.py \
    --vu-root "$VU_REPO" \
    --train \
    --method "$METHOD" \
    --unlearn-steps "$UNLEARN_STEPS" \
    --save-every "$SAVE_EVERY" \
    --train-dir "$TRAIN_DIR" \
    --model-path "$WAN_MODEL_PATH"

echo ""
echo "=================================================="
echo "✅ Stage 1 测试完成！"
echo "=================================================="
echo ""

# ---- 产物检查 ----
echo "📦 产物检查:"
echo "-----------------------------------"
if [ -d "$TRAIN_DIR" ]; then
    echo "输出目录: $TRAIN_DIR"
    ls -lh "$TRAIN_DIR/"
    echo ""

    # 检查 checkpoint 数量
    checkpoint_count=$(ls -1 "$TRAIN_DIR"/adapter_step*.pt "$TRAIN_DIR"/adapter_final.pt 2>/dev/null | wc -l)
    echo "生成的 checkpoint 数量: $checkpoint_count"
    echo "预期: 7 个 (step_100, step_200, ..., step_600, final)"

    if [ "$checkpoint_count" -ge 7 ]; then
        echo "✅ Checkpoint 数量正确"
    else
        echo "⚠️  Checkpoint 数量不足"
    fi
else
    echo "❌ 输出目录不存在: $TRAIN_DIR"
    exit 1
fi

echo ""
echo "=================================================="
echo "下一步: 如果测试成功，可以运行 Stage 2 (合并)"
echo "命令: sbatch slurm/exp014_stage2_test.sbatch"
echo "=================================================="
