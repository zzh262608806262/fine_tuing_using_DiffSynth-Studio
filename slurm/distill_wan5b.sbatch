#!/bin/bash
#SBATCH -A berzelius-2024-502
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -J distill_wan5b
#SBATCH -o slurm/logs/distill_wan5b-%j.out
#SBATCH -e slurm/logs/distill_wan5b-%j.err

# Wan2.2-TI2V-5B 蒸馏训练作业
# 目的: 将 5B 模型蒸馏到 4-8 步推理
# 预计时间: 8-10 小时（取决于数据集大小和重复次数）

set -e

echo "=== Wan2.2-TI2V-5B 蒸馏训练 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

# 进入项目目录
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 激活 conda 环境
echo "[1/4] 激活 diffsynth 环境..."
source ~/.bashrc
conda activate diffsynth
python --version
echo ""

# 检查 GPU
echo "[2/4] 检查 GPU 状态..."
nvidia-smi
echo ""

# 配置参数
DATASET_BASE="${DATASET_BASE:-data/diffsynth_example_dataset/wanvideo/Wan2.2-TI2V-5B}"
DATASET_META="${DATASET_META:-data/diffsynth_example_dataset/wanvideo/Wan2.2-TI2V-5B/metadata.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-./models/train/Wan2.2-TI2V-5B_distill}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
DATASET_REPEAT="${DATASET_REPEAT:-160}"
NUM_FRAMES="${NUM_FRAMES:-49}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
DOWNLOAD_DATASET="${DOWNLOAD_DATASET:-true}"

echo "[3/4] 训练配置:"
echo "  数据集路径: $DATASET_BASE"
echo "  Metadata: $DATASET_META"
echo "  输出路径: $OUTPUT_PATH"
echo "  学习率: $LEARNING_RATE"
echo "  训练轮数: $NUM_EPOCHS"
echo "  数据集重复: $DATASET_REPEAT"
echo "  视频尺寸: ${HEIGHT}x${WIDTH}x${NUM_FRAMES}"
echo "  下载数据集: $DOWNLOAD_DATASET"
echo ""

# 运行蒸馏训练
echo "[4/4] 启动蒸馏训练..."
if [ "$DOWNLOAD_DATASET" = "true" ]; then
    python scripts/distill_wan5b.py \
        --mode train \
        --dataset_base_path "$DATASET_BASE" \
        --dataset_metadata_path "$DATASET_META" \
        --output_path "$OUTPUT_PATH" \
        --learning_rate "$LEARNING_RATE" \
        --num_epochs "$NUM_EPOCHS" \
        --dataset_repeat "$DATASET_REPEAT" \
        --num_frames "$NUM_FRAMES" \
        --height "$HEIGHT" \
        --width "$WIDTH" \
        --use_gradient_checkpointing \
        --enable_tensorboard_log
else
    python scripts/distill_wan5b.py \
        --mode train \
        --no_download_dataset \
        --dataset_base_path "$DATASET_BASE" \
        --dataset_metadata_path "$DATASET_META" \
        --output_path "$OUTPUT_PATH" \
        --learning_rate "$LEARNING_RATE" \
        --num_epochs "$NUM_EPOCHS" \
        --dataset_repeat "$DATASET_REPEAT" \
        --num_frames "$NUM_FRAMES" \
        --height "$HEIGHT" \
        --width "$WIDTH" \
        --use_gradient_checkpointing \
        --enable_tensorboard_log
fi

EXIT_CODE=$?

echo ""
echo "=== 训练完成 ==="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 蒸馏训练成功完成"
    echo "模型保存在: $OUTPUT_PATH"
    ls -lh "$OUTPUT_PATH"/*.safetensors 2>/dev/null || echo "（检查输出目录获取详情）"
else
    echo "✗ 蒸馏训练失败"
fi

exit $EXIT_CODE
