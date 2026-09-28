#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -C fat
#SBATCH -t 6:00:00
#SBATCH -J exp015_gen
#SBATCH -o slurm/logs/exp015_generate-%x-%j.out
#SBATCH -e slurm/logs/exp015_generate-%x-%j.err

# Exp015 Stage 4: 视频生成
# 用法: sbatch --job-name=exp015_gen_esd_erased slurm/exp015_generate.sbatch esd_erased

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "用法: sbatch --job-name=exp015_gen_<arm> $0 <arm>"
    echo "可用臂: esd_erased, esd_base, npo_erased, npo_base, grad_ascent_erased, grad_ascent_base, anchor_distill_erased, anchor_distill_base"
    exit 1
fi

ARM="$1"

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

echo "========================================"
echo "Exp 015 视频生成: $ARM"
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
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "[GENERATE] 开始生成视频..."
echo "  臂: $ARM"
echo "  数量: 99个视频"
echo "  输出: outputs/exp015/$ARM/"
echo "========================================"

python -u scripts/exp015_generate_videos.py \
    --arm "$ARM" \
    --start 0 \
    --end 99 \
    --frames 17 \
    --height 480 \
    --width 720 \
    --steps 50 \
    --cfg 5.0 \
    --flow-shift 5.0 \
    --fps 15

if [ $? -ne 0 ]; then
    echo "[ERROR] 生成失败"
    exit 1
fi

echo ""
echo "[DONE] 视频生成完成"
echo "  输出目录: outputs/exp015/$ARM"
ls -lh "outputs/exp015/$ARM" | head -10
echo "  总计: $(ls outputs/exp015/$ARM/*.mp4 2>/dev/null | wc -l) 个视频"
echo ""
echo "Time: $(date)"
echo "========================================"
