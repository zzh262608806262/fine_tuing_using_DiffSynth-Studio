#!/bin/bash
#SBATCH --job-name=exp015_ft_grad_diff
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/exp015_ft_grad_diff-%j.out
#SBATCH --error=slurm/logs/exp015_ft_grad_diff-%j.err

# Exp 015 微调脚本模板
# 对擦除后的模型进行良性微调（与 Exp 012 配置一致）

set -euo pipefail

METHOD="${METHOD:-grad_diff}"

echo "========================================"
echo "Exp 015 微调: $grad_diff_erased"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "========================================"

FT_ROOT="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"
cd "$FT_ROOT"

# 激活环境
echo "[ENV] 激活 diffsynth 环境..."
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

export PATH="/home/x_jiage/.conda/envs/diffsynth/bin:$PATH"
export LD_LIBRARY_PATH="/home/x_jiage/.conda/envs/diffsynth/lib:$LD_LIBRARY_PATH"

# 验证环境
python -c "import torch; print(f'  torch: {torch.__version__}')"
python -c "import pandas; print(f'  pandas: {pandas.__version__}')"

# 路径配置
ERASED_MODEL="models/wan5b/exp015_$grad_diff_erased"
TRAIN_DATA="data/tiger_dataset/metadata_100.csv"
OUTPUT_DIR="models/finetune/exp015_$grad_diff_erased_lora10e2"

echo "[CONFIG] 微调配置:"
echo "  方法: $METHOD"
echo "  擦除后模型: $ERASED_MODEL"
echo "  训练数据: $TRAIN_DATA"
echo "  输出目录: $OUTPUT_DIR"

# 检查输入
if [ ! -d "$ERASED_MODEL" ]; then
    echo "[ERROR] 擦除后模型不存在: $ERASED_MODEL"
    echo "  请先完成 Stage 2（LoRA 合并）"
    exit 1
fi

if [ ! -f "$TRAIN_DATA" ]; then
    echo "[ERROR] 训练数据不存在: $TRAIN_DATA"
    exit 1
fi

echo "  ✓ 输入文件检查通过"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 执行微调（复用 Exp 012 的 finetune_wan5b.py，与 Exp 012 完全一致）
echo ""
echo "[FINETUNE] 开始微调..."
echo "========================================"

python3 scripts/finetune_wan5b.py \
    --model-dir "$ERASED_MODEL" \
    --data-csv "$TRAIN_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --dataset-repeat 10 \
    --num-epochs 2 \
    --learning-rate 1e-4 \
    --lora-rank 32 \
    --lora-alpha 32.0 \
    --num-frames 25 \
    --height 480 \
    --width 832 \
    --lora-target-modules q,k,v,o,ffn.0,ffn.2 \
    --use-gradient-checkpointing \
    --seed 0

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] $grad_diff 微调完成"
    echo "  输出目录: $OUTPUT_DIR"
    ls -lh "$OUTPUT_DIR"
else
    echo "[FAILED] $grad_diff 微调失败 (exit code: $EXIT_CODE)"
fi
echo "Time: $(date)"
echo "========================================"

exit $EXIT_CODE
