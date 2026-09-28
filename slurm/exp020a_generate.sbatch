#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 6:00:00
#SBATCH -J exp020a-generate
#SBATCH -o slurm/logs/exp020a_generate_%j.out
#SBATCH -e slurm/logs/exp020a_generate_%j.err

# Exp020a: 生成NPOMasked擦除后的视频

set -e

echo "=================================================="
echo "Exp020a: 生成NPOMasked擦除后视频"
echo "=================================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "=================================================="
echo ""

# 激活环境
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

# 路径配置
MODEL_DIR="models/wan5b/exp020a_npo_masked_erased"
PROMPTS_FILE="data/exp015/nudity_eval_prompts.jsonl"
OUTPUT_DIR="outputs/exp020a/npo_masked_erased"

echo "生成配置:"
echo "  模型: $MODEL_DIR"
echo "  提示词: $PROMPTS_FILE (99条)"
echo "  输出: $OUTPUT_DIR"
echo "  参数: 17帧, 480×736, 50步, cfg=5.0, seed=0"
echo ""

# 检查输入
if [ ! -d "$MODEL_DIR" ]; then
    echo "[ERROR] 模型不存在: $MODEL_DIR"
    echo "  请先完成模型合并"
    exit 1
fi

if [ ! -f "$PROMPTS_FILE" ]; then
    echo "[ERROR] 提示词文件不存在: $PROMPTS_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# 运行生成（使用Exp020a专用脚本）
python scripts/exp020a_generate_videos.py \
    --model-dir "$MODEL_DIR" \
    --prompts-file "$PROMPTS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --num-frames 17 \
    --height 480 \
    --width 736 \
    --num-inference-steps 50 \
    --guidance-scale 5.0 \
    --seed 0 \
    --fps 8

if [ $? -ne 0 ]; then
    echo "❌ 生成失败！"
    exit 1
fi

# 统计生成结果
VIDEO_COUNT=$(find "$OUTPUT_DIR" -name "*.mp4" | wc -l)
echo ""
echo "=================================================="
echo "✅ 生成完成"
echo "  生成视频数: $VIDEO_COUNT / 99"
echo "完成时间: $(date)"
echo "=================================================="
