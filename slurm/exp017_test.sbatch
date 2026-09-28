#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 0:30:00
#SBATCH -J exp017_test
#SBATCH -o slurm/logs/exp017_test_%j.out
#SBATCH -e slurm/logs/exp017_test_%j.err

# Exp017 蒸馏模型测试（在GPU节点上）

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

echo "=== Exp017 蒸馏模型测试 ==="
echo "节点: $(hostname)"
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# 激活环境
source ~/.bashrc
conda activate ft_diffsynth

# 设置环境变量
export DIFFSYNTH_SKIP_DOWNLOAD=true

# 测试base模型
echo "[1/2] 测试 base 模型..."
python scripts/exp017_test_distill_generation.py \
    --arm base \
    --test-prompt "A tiger walking in the forest" \
    --output-dir outputs/exp017_test

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ base 模型测试成功！"
    echo ""
    echo "[2/2] 测试所有5个模型..."
    python scripts/exp017_test_distill_generation.py \
        --arm all \
        --test-prompt "A tiger walking in the forest" \
        --output-dir outputs/exp017_test
else
    echo "❌ base 模型测试失败"
    exit 1
fi

echo ""
echo "=== 测试完成 ==="
echo "生成的视频:"
ls -lh outputs/exp017_test/*.mp4
