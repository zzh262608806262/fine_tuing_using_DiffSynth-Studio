#!/bin/bash
#SBATCH -A berzelius-2026-50
#SBATCH --gpus=1
#SBATCH -C "thin"
#SBATCH -t 2:00:00
#SBATCH -o /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio/slurm/logs/quantize_wan5b_%A_${ARM}.log

# Wan2.2-TI2V-5B 量化任务
# 用法: sbatch --export=ARM=base,MODE=infer slurm/quantize_wan5b.sbatch

cd /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

export DIFFSYNTH_SKIP_DOWNLOAD=true

# 默认参数
ARM=${ARM:-base}
MODE=${MODE:-infer}
METHOD=${METHOD:-bitsandbytes_nf4}

echo "=========================================="
echo "Wan2.2-TI2V-5B 量化任务"
echo "  ARM:    $ARM"
echo "  MODE:   $MODE"
echo "  METHOD: $METHOD"
echo "=========================================="

# 创建输出目录
mkdir -p outputs/quantized
mkdir -p models/quantized

case $MODE in
    infer)
        # 在线量化推理
        OUTPUT_VIDEO="outputs/quantized/${ARM}_${METHOD}_test.mp4"
        python -u scripts/quantize_wan5b.py \
            --mode infer \
            --arm "$ARM" \
            --method "$METHOD" \
            --output_video "$OUTPUT_VIDEO" \
            --num_inference_steps 30 \
            --num_frames 17 \
            --seed 42
        ;;

    save)
        # 保存量化 checkpoint
        OUTPUT_PATH="models/quantized/Wan2.2-TI2V-5B_${ARM}_${METHOD}"
        python -u scripts/quantize_wan5b.py \
            --mode save \
            --arm "$ARM" \
            --method "$METHOD" \
            --output_path "$OUTPUT_PATH"
        ;;

    load)
        # 加载量化 checkpoint 推理
        QUANTIZED_PATH="models/quantized/Wan2.2-TI2V-5B_${ARM}_${METHOD}.safetensors"
        OUTPUT_VIDEO="outputs/quantized/${ARM}_${METHOD}_loaded_test.mp4"
        python -u scripts/quantize_wan5b.py \
            --mode load \
            --arm "$ARM" \
            --quantized_path "$QUANTIZED_PATH" \
            --output_video "$OUTPUT_VIDEO" \
            --num_inference_steps 30 \
            --num_frames 17 \
            --seed 42
        ;;

    *)
        echo "错误: 未知的 MODE=$MODE (可选: infer, save, load)"
        exit 1
        ;;
esac

echo "任务完成: ARM=$ARM, MODE=$MODE"
