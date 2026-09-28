#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=A100-SXM4-80GB:1
#SBATCH -t 48:00:00
#SBATCH -J exp017_distill
#SBATCH -o slurm/logs/exp017_distill_%A_%a.out
#SBATCH -e slurm/logs/exp017_distill_%A_%a.err

# Exp 017: 擦除后模型蒸馏训练
# 参数化：ARM（base / grad_ascent / esd / npo）

set -e

echo "=== Exp 017: 擦除后模型蒸馏训练 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

# 必需参数
ARM="${ARM:?必须指定 ARM 参数 (base/grad_ascent/esd/npo)}"

# 进入项目目录
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 设置 PYTHONPATH 确保使用当前项目的 DiffSynth-Studio
export PYTHONPATH="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio:$PYTHONPATH"

# 激活环境
echo "[1/3] 激活 diffsynth 环境..."
source ~/.bashrc
conda activate ft_diffsynth
python --version
echo ""

# 定义变量（在 case 之前）
OUTPUT_PATH="models/train/exp017_${ARM}_distill"
DATASET_BASE="${DATASET_BASE:-data/tiger_dataset}"
DATASET_META="${DATASET_META:-data/tiger_dataset/metadata_50_distill.csv}"

echo "[2/3] 配置信息:"
echo "  ARM: $ARM"
echo "  输出路径: $OUTPUT_PATH"
echo "  训练数据: $DATASET_BASE"
echo "  Metadata: $DATASET_META"
echo ""

# 配置路径和模型
WORK_DIR="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"

case "$ARM" in
    base)
        # base 使用 model_id（modelscope）
        USE_MODEL_ID=true
        MODEL_PATHS="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.safetensors,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.safetensors"
        ;;
    grad_ascent)
        # 擦除模型：3个模型，第1个是DIT（3个分片），第2个是T5，第3个是VAE
        USE_MODEL_ID=false
        DIFFUSION_FILES=(${WORK_DIR}/models/wan5b/exp015_grad_ascent_erased/diffusion_pytorch_model*.safetensors)
        # DIT模型的内层数组（分片列表）
        DIT_ARRAY="["
        for f in "${DIFFUSION_FILES[@]}"; do DIT_ARRAY+="\"$f\","; done
        DIT_ARRAY="${DIT_ARRAY%,}]"  # 移除最后一个逗号
        # 外层数组：3个模型
        MODEL_PATHS="[$DIT_ARRAY,"
        MODEL_PATHS+="\"${WORK_DIR}/models/wan5b/exp015_grad_ascent_erased/models_t5_umt5-xxl-enc-bf16.safetensors\","
        MODEL_PATHS+="\"${WORK_DIR}/models/wan5b/exp015_grad_ascent_erased/Wan2.2_VAE.safetensors\"]"
        ;;
    esd)
        USE_MODEL_ID=false
        DIFFUSION_FILES=(${WORK_DIR}/models/wan5b/exp015_esd_erased/diffusion_pytorch_model*.safetensors)
        DIT_ARRAY="["
        for f in "${DIFFUSION_FILES[@]}"; do DIT_ARRAY+="\"$f\","; done
        DIT_ARRAY="${DIT_ARRAY%,}]"
        MODEL_PATHS="[$DIT_ARRAY,"
        MODEL_PATHS+="\"${WORK_DIR}/models/wan5b/exp015_esd_erased/models_t5_umt5-xxl-enc-bf16.safetensors\","
        MODEL_PATHS+="\"${WORK_DIR}/models/wan5b/exp015_esd_erased/Wan2.2_VAE.safetensors\"]"
        ;;
    npo)
        USE_MODEL_ID=false
        DIFFUSION_FILES=(${WORK_DIR}/models/wan5b/exp015_npo_erased/diffusion_pytorch_model*.safetensors)
        DIT_ARRAY="["
        for f in "${DIFFUSION_FILES[@]}"; do DIT_ARRAY+="\"$f\","; done
        DIT_ARRAY="${DIT_ARRAY%,}]"
        MODEL_PATHS="[$DIT_ARRAY,"
        MODEL_PATHS+="\"${WORK_DIR}/models/wan5b/exp015_npo_erased/models_t5_umt5-xxl-enc-bf16.safetensors\","
        MODEL_PATHS+="\"${WORK_DIR}/models/wan5b/exp015_npo_erased/Wan2.2_VAE.safetensors\"]"
        ;;
    anchor_distill)
        USE_MODEL_ID=false
        DIFFUSION_FILES=(${WORK_DIR}/models/wan5b/exp015_anchor_distill_erased/diffusion_pytorch_model*.safetensors)
        DIT_ARRAY="["
        for f in "${DIFFUSION_FILES[@]}"; do DIT_ARRAY+="\"$f\","; done
        DIT_ARRAY="${DIT_ARRAY%,}]"
        MODEL_PATHS="[$DIT_ARRAY,"
        MODEL_PATHS+="\"${WORK_DIR}/models/wan5b/exp015_anchor_distill_erased/models_t5_umt5-xxl-enc-bf16.safetensors\","
        MODEL_PATHS+="\"${WORK_DIR}/models/wan5b/exp015_anchor_distill_erased/Wan2.2_VAE.safetensors\"]"
        ;;
    *)
        echo "错误: 未知的 ARM=$ARM"
        exit 1
        ;;
esac

echo "[3/3] 启动蒸馏训练（$ARM）..."

# 根据模型类型选择参数
if [ "$USE_MODEL_ID" = true ]; then
    MODEL_ARG="--model_id_with_origin_paths"
else
    MODEL_ARG="--model_paths"
fi

# 统一使用 distill_wan5b.py 包装脚本
python scripts/distill_wan5b.py \
    --mode train \
    --no_download_dataset \
    --dataset_base_path "$DATASET_BASE" \
    --dataset_metadata_path "$DATASET_META" \
    --output_path "$OUTPUT_PATH" \
    --dataset_repeat 5 \
    --learning_rate 1e-5 \
    --num_epochs 10 \
    --height 480 \
    --width 736 \
    --num_frames 17 \
    $MODEL_ARG "$MODEL_PATHS" \
    --extra_inputs seed,rand_device \
    --use_gradient_checkpointing \
    --accelerate_config examples/wanvideo/model_training/full/accelerate_config_single_gpu.yaml

EXIT_CODE=$?

echo ""
echo "=== 训练完成 ==="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 蒸馏训练成功"
    echo "模型保存在: $OUTPUT_PATH"
    ls -lh "$OUTPUT_PATH"/*.safetensors 2>/dev/null || echo "（检查输出目录）"
else
    echo "✗ 蒸馏训练失败"
fi

exit $EXIT_CODE
