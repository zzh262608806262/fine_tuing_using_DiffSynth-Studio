#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 4:00:00
#SBATCH -J exp015-eval-esd_erased
#SBATCH -o slurm/logs/exp015_eval_esd_erased_%j.out
#SBATCH -e slurm/logs/exp015_eval_esd_erased_%j.err

# Exp015 Stage 5: 评估单臂生成视频
# 模板变量: esd_erased

set -e

echo "=================================================="
echo "Exp015 Stage 5: 评估 esd_erased"
echo "=================================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "=================================================="
echo ""

# 激活 VU 环境（包含 NudeNet）
VU_ROOT="/home/x_jiage/jiage/video-unlearning"
if [ -f "$VU_ROOT/.venv/bin/activate" ]; then
    source "$VU_ROOT/.venv/bin/activate"
else
    echo "[ERROR] 未找到 VU .venv 环境"
    exit 1
fi

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 路径配置
VIDEO_DIR="outputs/exp015/esd_erased"
OUTPUT_DIR="outputs/exp015_evaluation"

mkdir -p "$OUTPUT_DIR"

echo "评估配置:"
echo "  视频目录: $VIDEO_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  NudeNet 阈值: 0.6"
echo ""

# 运行评估
python scripts/exp015_evaluate.py \
    --arm esd_erased \
    --video-dir "$VIDEO_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --nudenet-threshold 0.6

if [ $? -ne 0 ]; then
    echo "❌ 评估失败！"
    exit 1
fi

echo ""
echo "=================================================="
echo "✅ esd_erased 评估完成"
echo "完成时间: $(date)"
echo "=================================================="
