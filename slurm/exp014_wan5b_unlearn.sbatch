#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 06:00:00
#SBATCH -J exp014-unlearn
#SBATCH -o logs/exp014_unlearn_%j.out
#SBATCH -e logs/exp014_unlearn_%j.err

# Exp 014: VU标准配置擦除（rank 8, 600步）
# 对比Exp 012: rank相同，但步数从1200→600
# 目的：验证VU推荐的600步是否足够

set -e

echo "=================================================="
echo "Exp 014 - Wan5B Nudity Unlearning (VU Standard)"
echo "=================================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

# 安装缺失依赖
pip install omegaconf hydra-core -q

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 配置
UNLEARN_STEPS=600          # VU exp504标准（vs Exp012的1200）
SAVE_EVERY=100             # 每100步存点（6个checkpoint）
LEARNING_RATE=1e-5         # VU标准
TRAIN_DIR="models/unlearn/exp014_wan5b_nudity_600steps"

echo "📊 配置:"
echo "  - 训练步数: $UNLEARN_STEPS (VU标准)"
echo "  - 存点间隔: $SAVE_EVERY 步"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - 输出目录: $TRAIN_DIR"
echo ""

# [阶段1] 缓存latents
echo "🔧 [1/3] 缓存训练数据latents..."
if [ ! -f "data/wan5b/unlearn_latents/latent_manifest.jsonl" ]; then
    python scripts/run_unlearn_wan5b.py --cache
    echo "✅ Latents缓存完成"
else
    echo "✅ Latents已存在，跳过缓存"
fi
echo ""

# [阶段2] 擦除训练
echo "🚀 [2/3] 擦除训练 (GradAscent, 600步)..."
python scripts/run_unlearn_wan5b.py \
    --train \
    --model-path "models/Wan-AI/Wan2.2-TI2V-5B-Diffusers" \
    --unlearn-steps $UNLEARN_STEPS \
    --save-every $SAVE_EVERY \
    --learning-rate $LEARNING_RATE \
    --train-dir "$TRAIN_DIR" \
    --log-every 50

echo "✅ 擦除训练完成"
echo ""

# [阶段3] 探针生成（跳过，显存不足）
echo "⏭️  [3/3] 探针生成已跳过（显存限制，将在后续评估阶段验证）"
echo ""

echo "=================================================="
echo "✅ Exp 014擦除阶段完成！"
echo "完成时间: $(date)"
echo ""
echo "产物："
echo "  - Adapter: $TRAIN_DIR/"
echo "  - Checkpoints: step_{100,200,300,400,500,600}"
echo ""
echo "下一步："
echo "  运行合并脚本: sbatch slurm/exp014_wan5b_merge.sbatch"
echo "=================================================="
