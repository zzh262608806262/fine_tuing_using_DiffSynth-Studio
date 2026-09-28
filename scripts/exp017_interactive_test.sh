#!/bin/bash
# Exp 017 交互式测试脚本
# 在 GPU 节点上测试蒸馏训练命令

set -e

echo "=========================================="
echo "Exp 017 蒸馏训练交互式测试"
echo "=========================================="
echo ""

# 激活环境
echo "[1/5] 激活 diffsynth 环境..."
source ~/.bashrc
conda activate ft_diffsynth
python --version
echo ""

# 测试环境
echo "[2/5] 测试环境..."
python scripts/test_env.py
echo ""

# 进入项目目录
cd "$(dirname "$(dirname "$(readlink -f "$0")")")" # Auto: project root

# 测试参数
ARM="${ARM:-base}"
OUTPUT_PATH="models/train/exp017_${ARM}_distill_interactive_test"
DATASET_BASE="data/tiger_dataset"
DATASET_META="data/tiger_dataset/metadata_100.csv"

echo "[3/5] 测试配置:"
echo "  ARM: $ARM"
echo "  输出: $OUTPUT_PATH"
echo "  数据集: $DATASET_BASE"
echo ""

# 构建模型路径
case "$ARM" in
    base)
        MODEL_PATHS="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.safetensors,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.safetensors"
        ;;
    grad_ascent)
        MODEL_PATHS="models/wan5b/exp015_grad_ascent_erased:diffusion_pytorch_model*.safetensors,models/wan5b/exp015_grad_ascent_erased:models_t5_umt5-xxl-enc-bf16.safetensors,models/wan5b/exp015_grad_ascent_erased:Wan2.2_VAE.safetensors"
        ;;
    esd)
        MODEL_PATHS="models/wan5b/exp015_esd_erased:diffusion_pytorch_model*.safetensors,models/wan5b/exp015_esd_erased:models_t5_umt5-xxl-enc-bf16.safetensors,models/wan5b/exp015_esd_erased:Wan2.2_VAE.safetensors"
        ;;
    *)
        echo "错误: ARM 必须是 base/grad_ascent/esd"
        exit 1
        ;;
esac

echo "[4/5] 模型路径:"
echo "  $MODEL_PATHS"
echo ""

# 打印完整命令
echo "[5/5] 启动蒸馏训练（测试模式：仅1个epoch，小数据集）..."
echo "命令："
cat <<'CMD'
python scripts/distill_wan5b.py \
    --mode train \
    --no_download_dataset \
    --dataset_base_path "$DATASET_BASE" \
    --dataset_metadata_path "$DATASET_META" \
    --output_path "$OUTPUT_PATH" \
    --dataset_repeat 10 \
    --learning_rate 1e-5 \
    --num_epochs 1 \
    --height 480 \
    --width 736 \
    --num_frames 17 \
    --model_id_with_origin_paths "$MODEL_PATHS" \
    --use_gradient_checkpointing \
    --enable_tensorboard_log
CMD
echo ""

# 询问是否继续
read -p "按 Enter 继续执行，或 Ctrl+C 取消..."

# 执行训练（测试模式）
python scripts/distill_wan5b.py \
    --mode train \
    --no_download_dataset \
    --dataset_base_path "$DATASET_BASE" \
    --dataset_metadata_path "$DATASET_META" \
    --output_path "$OUTPUT_PATH" \
    --dataset_repeat 10 \
    --learning_rate 1e-5 \
    --num_epochs 1 \
    --height 480 \
    --width 736 \
    --num_frames 17 \
    --model_id_with_origin_paths "$MODEL_PATHS" \
    --use_gradient_checkpointing \
    --enable_tensorboard_log

echo ""
echo "=========================================="
echo "✓ 测试完成！"
echo "=========================================="
echo ""
echo "如果训练正常启动并运行了几个 step，说明配置正确。"
echo "现在可以："
echo "  1. Ctrl+C 停止测试"
echo "  2. 使用 sbatch 提交完整的训练作业"
