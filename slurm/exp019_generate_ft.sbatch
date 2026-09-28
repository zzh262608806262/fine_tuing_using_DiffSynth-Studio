#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 2:00:00
#SBATCH -J exp019_gen_ft
#SBATCH -o outputs/exp019_generate_ft_%j.log
#SBATCH -e outputs/exp019_generate_ft_%j.err

# Exp019 视频生成：GradDiff erased + finetune LoRA (step-50000)

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

echo "========================================"
echo "Exp019 视频生成: GradDiff Fine-tuned"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "========================================"

# 激活环境
source /home/x_jiage/jiage/video-unlearning/.venv/bin/activate
echo "✓ 已激活 VU .venv 环境"

# 设置环境变量
export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 路径配置
BASE_MODEL="models/wan5b/exp019_graddiff_erased"
LORA_PATH="models/finetune/exp019_graddiff_ft/step-50000.safetensors"
OUTPUT_DIR="outputs/exp019/graddiff_ft"

echo ""
echo "[配置]"
echo "  方法: GradDiff"
echo "  基座模型: ${BASE_MODEL}"
echo "  LoRA: ${LORA_PATH}"
echo "  输出: ${OUTPUT_DIR}"
echo "========================================"

# 前置检查
if [ ! -d "$BASE_MODEL" ]; then
    echo "❌ 基座模型不存在: $BASE_MODEL"
    exit 1
fi

if [ ! -f "$LORA_PATH" ]; then
    echo "❌ LoRA 不存在: $LORA_PATH"
    exit 1
fi

echo "✅ 前置检查通过"
echo ""

# 生成视频（复用exp018的生成脚本）
echo "[GENERATE] 开始生成 99 条评测视频..."
python -u scripts/exp018_generate_videos.py \
    --method graddiff \
    --base_model "$BASE_MODEL" \
    --lora_path "$LORA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --frames 17 \
    --height 480 \
    --width 720 \
    --steps 50 \
    --cfg 5.0 \
    --flow_shift 5.0 \
    --fps 15

if [ $? -ne 0 ]; then
    echo "[ERROR] 生成失败"
    exit 1
fi

echo ""
echo "[DONE] 视频生成完成"
echo "  输出目录: $OUTPUT_DIR"
video_count=$(find "$OUTPUT_DIR" -name "*.mp4" 2>/dev/null | wc -l)
echo "  总计: ${video_count} 个视频（预期 99）"

if [ "$video_count" -eq 99 ]; then
    echo "  ✅ 数量正确"
else
    echo "  ⚠️  数量不符"
fi

echo ""
echo "Time: $(date)"
echo "========================================"
