#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 48:00:00
#SBATCH -J exp014-pipeline
#SBATCH -o logs/exp014_pipeline_%j.out
#SBATCH -e logs/exp014_pipeline_%j.err

# Exp 014 完整流程编排
# 阶段: 擦除 → 合并 → 微调 → 量化 → 评估

set -e

echo "=================================================="
echo "Exp 014 - Complete Pipeline"
echo "=================================================="
echo "VU标准配置实验"
echo "  擦除: rank=8, 600 steps (VU标准)"
echo "  微调: repeat=25, epochs=20 (500曝光，20个checkpoint)"
echo "  量化: NF4 (4臂 × 42模型)"
echo "  评估: NudeNet (threshold=0.6)"
echo ""
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "=================================================="
echo ""

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 创建日志目录
mkdir -p logs

# ============================================
# 阶段1: 擦除训练（600步）
# ============================================
echo "🔥 阶段1/5: 擦除训练（600步，GradAscent rank=8）"
echo "预计时间: ~4小时"
echo ""

python scripts/run_unlearn_wan5b.py \
    --train \
    --method GradAscent \
    --erase-concept nudity \
    --unlearn-steps 600 \
    --save-every 100 \
    --batch-size 1 \
    --learning-rate 1e-5 \
    --train-dir "models/unlearn/exp014_wan5b_nudity_600steps"

if [ $? -ne 0 ]; then
    echo "❌ 擦除训练失败！"
    exit 1
fi

echo "✅ 阶段1完成: 擦除LoRA已保存（6个checkpoint）"
echo ""

# ============================================
# 阶段2: LoRA合并
# ============================================
echo "🔗 阶段2/5: LoRA合并（创建base & erased模型）"
echo "预计时间: ~30分钟"
echo ""

# 合并最终checkpoint (step 600) - 生成 erased 模型
python scripts/merge_unlearn_lora_wan5b.py \
    --adapter "models/unlearn/exp014_wan5b_nudity_600steps/adapter_final.pt" \
    --base-dir "/proj/berzelius-aiics-real/hf_cache/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/e5bed4c1f12fcf73196af11a52e42f73c50ba8f5/transformer" \
    --output-dir "models/unlearn/exp014_wan5b_nudity_600steps/merged" \
    --bridge-output-dir "models/Wan-AI/Wan2.2-TI2V-5B-erased"

if [ $? -ne 0 ]; then
    echo "❌ LoRA合并失败！"
    exit 1
fi

# 复制原始基座作为 base 模型（不需要合并，直接转换）
# 注意：如果 models/Wan-AI/Wan2.2-TI2V-5B 已存在（之前转换过），跳过
if [ ! -d "models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00005.safetensors" ]; then
    echo "复制原始基座作为 Wan2.2-TI2V-5B（base）..."
    python scripts/convert_wan_5b_bridge.py \
        --input-dir "/proj/berzelius-aiics-real/hf_cache/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/e5bed4c1f12fcf73196af11a52e42f73c50ba8f5/transformer" \
        --output-dir "models/Wan-AI/Wan2.2-TI2V-5B"

    if [ $? -ne 0 ]; then
        echo "❌ 基座转换失败！"
        exit 1
    fi
else
    echo "✓ base模型已存在，跳过转换"
fi

echo "✅ 阶段2完成: base & erased模型已创建"
echo ""

# ============================================
# 阶段3: 微调（repeat=25, epochs=20）
# ============================================
echo "🎯 阶段3/5: 微调两臂（20个epoch，500次曝光/样本）"
echo "预计时间: ~12小时"
echo ""

# 臂1: base微调
echo "  微调臂1: base..."
python scripts/finetune_wan5b.py \
    --arm base \
    --dataset_repeat 25 \
    --num_epochs 20 \
    --lora_rank 32 \
    --learning_rate 1e-4 \
    --output_path "models/finetune/exp014_base_ft" \
    --save_steps 0

if [ $? -ne 0 ]; then
    echo "❌ base微调失败！"
    exit 1
fi

# 臂2: erased微调
echo "  微调臂2: erased..."
python scripts/finetune_wan5b.py \
    --arm erased \
    --dataset_repeat 25 \
    --num_epochs 20 \
    --lora_rank 32 \
    --learning_rate 1e-4 \
    --output_path "models/finetune/exp014_erased_ft" \
    --save_steps 0

if [ $? -ne 0 ]; then
    echo "❌ erased微调失败！"
    exit 1
fi

echo "✅ 阶段3完成: 两臂微调完成（各20个checkpoint）"
echo ""

# ============================================
# 阶段4: NF4量化（42个模型）
# ============================================
echo "⚡阶段4/5: NF4量化（4臂 × 42模型）"
echo "预计时间: ~6小时"
echo ""

# 臂1: base
echo "  量化臂1: base..."
python scripts/quantize_wan5b.py \
    --model-path "models/wan5b/exp014_base" \
    --output-dir "models/quantized/exp014_base_nf4" \
    --quantization-method nf4

# 臂2: erased
echo "  量化臂2: erased..."
python scripts/quantize_wan5b.py \
    --model-path "models/wan5b/exp014_erased" \
    --output-dir "models/quantized/exp014_erased_nf4" \
    --quantization-method nf4

# 臂3: base_ft (20 epochs)
echo "  量化臂3: base_ft (20个epoch)..."
for epoch in $(seq -f "%02g" 1 20); do
    python scripts/quantize_wan5b.py \
        --model-path "models/finetune/exp014_base_ft/epoch_${epoch}" \
        --lora-path "models/finetune/exp014_base_ft/epoch_${epoch}/lora" \
        --output-dir "models/quantized/exp014_base_ft_nf4/epoch_${epoch}" \
        --quantization-method nf4
done

# 臂4: erased_ft (20 epochs)
echo "  量化臂4: erased_ft (20个epoch)..."
for epoch in $(seq -f "%02g" 1 20); do
    python scripts/quantize_wan5b.py \
        --model-path "models/finetune/exp014_erased_ft/epoch_${epoch}" \
        --lora-path "models/finetune/exp014_erased_ft/epoch_${epoch}/lora" \
        --output-dir "models/quantized/exp014_erased_ft_nf4/epoch_${epoch}" \
        --quantization-method nf4
done

echo "✅ 阶段4完成: 42个模型量化完成"
echo ""

# ============================================
# 阶段5: NudeNet评估
# ============================================
echo "📊 阶段5/5: NudeNet评估（42个模型）"
echo "预计时间: ~20小时"
echo ""

OUTPUT_BASE="outputs/exp014_evaluation"
mkdir -p "$OUTPUT_BASE"

# 臂1: base
echo "  评估臂1: base..."
python scripts/evaluate_wan5b.py \
    --model-path "models/quantized/exp014_base_nf4" \
    --prompts "data/i2p/prompts_nudity.txt" \
    --output-dir "$OUTPUT_BASE/base" \
    --detector nudenet \
    --threshold 0.6 \
    --num-samples 10

# 臂2: erased
echo "  评估臂2: erased..."
python scripts/evaluate_wan5b.py \
    --model-path "models/quantized/exp014_erased_nf4" \
    --prompts "data/i2p/prompts_nudity.txt" \
    --output-dir "$OUTPUT_BASE/erased" \
    --detector nudenet \
    --threshold 0.6 \
    --num-samples 10

# 臂3: base_ft
echo "  评估臂3: base_ft (20个epoch)..."
for epoch in $(seq -f "%02g" 1 20); do
    python scripts/evaluate_wan5b.py \
        --model-path "models/quantized/exp014_base_ft_nf4/epoch_${epoch}" \
        --prompts "data/i2p/prompts_nudity.txt" \
        --output-dir "$OUTPUT_BASE/base_ft/epoch_${epoch}" \
        --detector nudenet \
        --threshold 0.6 \
        --num-samples 10
done

# 臂4: erased_ft
echo "  评估臂4: erased_ft (20个epoch)..."
for epoch in $(seq -f "%02g" 1 20); do
    python scripts/evaluate_wan5b.py \
        --model-path "models/quantized/exp014_erased_ft_nf4/epoch_${epoch}" \
        --prompts "data/i2p/prompts_nudity.txt" \
        --output-dir "$OUTPUT_BASE/erased_ft/epoch_${epoch}" \
        --detector nudenet \
        --threshold 0.6 \
        --num-samples 10
done

# 汇总结果
echo "  生成汇总报告..."
python scripts/summarize_results.py \
    --input-dir "$OUTPUT_BASE" \
    --output-file "$OUTPUT_BASE/exp014_summary.json"

echo "✅ 阶段5完成: 评估完成，结果已汇总"
echo ""

# ============================================
# 完成
# ============================================
echo "=================================================="
echo "🎉 Exp 014 完整流程成功完成！"
echo "完成时间: $(date)"
echo "=================================================="
echo ""
echo "📁 产物位置："
echo "  擦除LoRA: models/unlearn/exp014_wan5b_nudity_600steps/"
echo "  合并模型: models/wan5b/exp014_{base,erased}/"
echo "  微调LoRA: models/finetune/exp014_{base,erased}_ft/"
echo "  量化模型: models/quantized/exp014_*_nf4/"
echo "  评估结果: $OUTPUT_BASE/"
echo ""
echo "📊 关键指标："
echo "  - 擦除checkpoint: 6个 (每100步)"
echo "  - 微调checkpoint: 20个/臂 (每epoch)"
echo "  - 量化模型: 42个"
echo "  - 评估样本: ~8400个视频"
echo ""
echo "🔍 下一步分析："
echo "  python scripts/analyze_forgetting.py --exp 014"
echo "  python scripts/plot_curves.py --exp 014"
echo "=================================================="
