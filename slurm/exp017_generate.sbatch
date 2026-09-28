#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 4:00:00
#SBATCH -J exp017_gen
#SBATCH -o slurm/logs/exp017_gen_%A_%a.out
#SBATCH -e slurm/logs/exp017_gen_%A_%a.err

# Exp017 蒸馏模型批量视频生成
# 每个作业生成一个模型的99条视频

set -e

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

echo "=== Exp017 批量视频生成 ==="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "模型臂: $ARM"
echo ""

# GPU信息
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# 激活环境
source ~/.bashrc
conda activate ft_diffsynth

# 设置环境变量
export DIFFSYNTH_SKIP_DOWNLOAD=true

# 生成视频
echo "开始生成 $ARM (99条视频)..."
python -u scripts/exp017_generate_videos.py \
    --arm $ARM \
    --start 0 \
    --end 99 \
    --steps 4 \
    --cfg 1.0

EXIT_CODE=$?

echo ""
echo "=== 生成完成 ==="
echo "结束时间: $(date)"
echo "退出码: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 成功生成所有视频"
    echo "输出目录: outputs/exp017/$ARM/"
    ls -lh outputs/exp017/$ARM/*.mp4 | wc -l
    echo "个视频文件"
else
    echo "❌ 生成失败"
fi

exit $EXIT_CODE
