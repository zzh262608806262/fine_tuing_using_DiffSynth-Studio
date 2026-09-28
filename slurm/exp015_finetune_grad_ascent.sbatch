#!/bin/bash
#SBATCH --job-name=exp015_finetune_grad_ascent
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/exp015_finetune_grad_ascent-%j.out
#SBATCH --error=slurm/logs/exp015_finetune_grad_ascent-%j.err

# Exp 015 微调脚本 - ESD
# 对 ESD 擦除后的模型进行良性微调

set -euo pipefail

METHOD="grad_ascent"

echo "========================================"
echo "Exp 015 微调: $METHOD"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "========================================"

FT_ROOT="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"
VU_ROOT="/home/x_jiage/jiage/video-unlearning"
cd "$FT_ROOT"

# 激活环境（使用 VU .venv，与 Exp012 一致）
echo "[ENV] 激活环境..."
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"

# 使用 VU .venv（有完整的 transformers/accelerate）
if [ -f "$VU_ROOT/.venv/bin/activate" ]; then
    echo "  使用 VU .venv 环境"
    source "$VU_ROOT/.venv/bin/activate"
else
    echo "[ERROR] 未找到 VU .venv 环境"
    exit 1
fi

export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0

# 路径配置
ERASED_MODEL="models/wan5b/exp015_${METHOD}_erased"
OUTPUT_DIR="models/finetune/exp015_${METHOD}_erased_lora10e2"
DATASET_PATH="data/tiger_dataset/metadata_100.csv"

# 微调参数（与 Exp 012 对齐）
DATASET_REPEAT=10
NUM_EPOCHS=2
LEARNING_RATE=1e-4
LORA_RANK=32
LORA_TARGETS="q,k,v,o,ffn.0,ffn.2"
HEIGHT=480
WIDTH=832
NUM_FRAMES=25
BATCH_SIZE=1

echo "[CONFIG] 微调配置:"
echo "  方法: $METHOD"
echo "  擦除模型: $ERASED_MODEL"
echo "  数据集: $DATASET_PATH"
echo "  输出: $OUTPUT_DIR"
echo "  超参: repeat=$DATASET_REPEAT, epochs=$NUM_EPOCHS, lr=$LEARNING_RATE, rank=$LORA_RANK"
echo ""

# 检查输入
if [ ! -d "$ERASED_MODEL" ]; then
    echo "[ERROR] 擦除后基座不存在: $ERASED_MODEL"
    echo "  请先完成 Stage 2（合并）"
    exit 1
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "[ERROR] 数据集不存在: $DATASET_PATH"
    exit 1
fi

echo "  ✓ 输入文件检查通过"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 构建 model_id_with_origin_paths（DiffSynth 本地定位格式）
# 格式：model_id:pattern,model_id:pattern,... （逗号分隔，每个文件独立指定 model_id）
MODEL_ID="wan5b/exp015_${METHOD}_erased"
MODEL_PATHS="${MODEL_ID}:diffusion_pytorch_model*.safetensors,${MODEL_ID}:models_t5_umt5-xxl-enc-bf16.safetensors,${MODEL_ID}:Wan2.2_VAE.safetensors"

# 调用 accelerate 直接训练
echo ""
echo "[TRAIN] 开始微调..."
echo "========================================"

accelerate launch examples/wanvideo/model_training/train.py \
    --dataset_base_path "data/tiger_dataset" \
    --dataset_metadata_path "$DATASET_PATH" \
    --dataset_repeat "$DATASET_REPEAT" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num_frames "$NUM_FRAMES" \
    --model_id_with_origin_paths "${MODEL_PATHS}" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --gradient_accumulation_steps 1 \
    --weight_decay 0.01 \
    --output_path "$OUTPUT_DIR" \
    --remove_prefix_in_ckpt pipe.dit. \
    --lora_base_model dit \
    --lora_target_modules "$LORA_TARGETS" \
    --lora_rank "$LORA_RANK" \
    --use_gradient_checkpointing \
    --enable_tensorboard_log

if [ $? -ne 0 ]; then
    echo "[ERROR] 微调失败"
    exit 1
fi

echo ""
echo "========================================"
echo "[SUCCESS] $METHOD 微调完成"
echo "  输出目录: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
echo "Time: $(date)"
echo "========================================"
