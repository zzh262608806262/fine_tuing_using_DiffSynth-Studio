#!/bin/bash
#SBATCH -A berzelius-2024-502
#SBATCH --gpus=1
#SBATCH -t 00:30:00
#SBATCH -J validate_distill_wan5b
#SBATCH -o slurm/logs/validate_distill_wan5b-%j.out
#SBATCH -e slurm/logs/validate_distill_wan5b-%j.err

# Wan2.2-TI2V-5B 蒸馏模型验证作业
# 目的: 验证蒸馏后的模型能否用少量步数生成视频

set -e

echo "=== Wan2.2-TI2V-5B 蒸馏模型验证 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

# 进入项目目录
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 激活环境
echo "[1/3] 激活 diffsynth 环境..."
source ~/.bashrc
conda activate diffsynth
echo ""

# 配置参数
DISTILLED_MODEL="${DISTILLED_MODEL:-./models/train/Wan2.2-TI2V-5B_distill/epoch-1.safetensors}"
NUM_STEPS="${NUM_STEPS:-4}"
CFG_SCALE="${CFG_SCALE:-1.0}"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-outputs/distill_wan5b_validate_${NUM_STEPS}step.mp4}"
PROMPT="${PROMPT:-A playful dog running across a lush green meadow}"
INPUT_IMAGE="${INPUT_IMAGE:-}"

echo "[2/3] 验证配置:"
echo "  蒸馏模型: $DISTILLED_MODEL"
echo "  推理步数: $NUM_STEPS"
echo "  CFG Scale: $CFG_SCALE"
echo "  输出视频: $OUTPUT_VIDEO"
echo "  提示词: $PROMPT"
echo "  输入图片: ${INPUT_IMAGE:-无（纯文本生成）}"
echo ""

# 创建输出目录
mkdir -p "$(dirname "$OUTPUT_VIDEO")"

# 运行验证
echo "[3/3] 运行推理验证..."
if [ -n "$INPUT_IMAGE" ] && [ -f "$INPUT_IMAGE" ]; then
    python scripts/distill_wan5b.py \
        --mode validate \
        --distilled_model_path "$DISTILLED_MODEL" \
        --num_inference_steps "$NUM_STEPS" \
        --cfg_scale "$CFG_SCALE" \
        --output_video "$OUTPUT_VIDEO" \
        --prompt "$PROMPT" \
        --input_image "$INPUT_IMAGE" \
        --seed 0
else
    python scripts/distill_wan5b.py \
        --mode validate \
        --distilled_model_path "$DISTILLED_MODEL" \
        --num_inference_steps "$NUM_STEPS" \
        --cfg_scale "$CFG_SCALE" \
        --output_video "$OUTPUT_VIDEO" \
        --prompt "$PROMPT" \
        --seed 0
fi

EXIT_CODE=$?

echo ""
echo "=== 验证完成 ==="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 验证成功"
    echo "视频已保存: $OUTPUT_VIDEO"
    ls -lh "$OUTPUT_VIDEO"
else
    echo "✗ 验证失败"
fi

exit $EXIT_CODE
