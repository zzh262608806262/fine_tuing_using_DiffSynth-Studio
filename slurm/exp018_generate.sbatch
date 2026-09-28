#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 2:00:00
#SBATCH -J exp018_gen
#SBATCH -o slurm/logs/exp018_generate-%x-%j.out
#SBATCH -e slurm/logs/exp018_generate-%x-%j.err

# Exp018 Stage 4: 视频生成（微调后的 epoch-19）
# 用法: sbatch --job-name=exp018_gen_esd slurm/exp018_generate.sbatch esd

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "用法: sbatch --job-name=exp018_gen_<method> $0 <method>"
    echo "可用方法: esd, npo, grad_ascent, anchor_distill"
    exit 1
fi

METHOD="$1"
EPOCH="19"  # 只生成最终 checkpoint

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

echo "========================================"
echo "Exp018 视频生成: ${METHOD} epoch-${EPOCH}"
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
BASE_MODEL="models/wan5b/exp015_${METHOD}_erased"
LORA_PATH="models/finetune/exp018_${METHOD}_ft/epoch-${EPOCH}.safetensors"
OUTPUT_DIR="outputs/exp018/${METHOD}_ft_e${EPOCH}"

echo ""
echo "[配置]"
echo "  方法: ${METHOD}"
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

# 生成视频
echo "[GENERATE] 开始生成 99 条评测视频..."
python -u scripts/exp018_generate_videos.py \
    --method "$METHOD" \
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
