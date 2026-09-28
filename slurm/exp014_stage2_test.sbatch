#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -C fat
#SBATCH -t 01:00:00
#SBATCH -J exp014-stage2-test
#SBATCH -o slurm/logs/exp014_stage2_test-%j.out
#SBATCH -e slurm/logs/exp014_stage2_test-%j.err

# ============================================================================
# Exp014 Stage 2 测试：LoRA 合并 + 桥转换
#
# 依赖：Stage 1 必须成功完成（生成 adapter_final.pt）
#
# 输出：
#   - models/unlearn/exp014_wan5b_nudity_600steps/merged/ (diffusers 格式)
#   - models/Wan-AI/Wan2.2-TI2V-5B-exp014-erased/ (DiffSynth 格式)
# ============================================================================

set -euo pipefail

FT_ROOT="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"
cd "$FT_ROOT"

# ---- 激活环境 ----
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface

echo "=================================================="
echo "Exp014 Stage 2 测试 - LoRA 合并"
echo "=================================================="
echo "节点: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "=================================================="
echo ""

# ---- 配置 ----
ADAPTER="models/unlearn/exp014_wan5b_nudity_600steps/adapter_final.pt"
BASE_TRANSFORMER="models/Wan-AI/Wan2.2-TI2V-5B-Diffusers/transformer"
MERGED_DIR="models/unlearn/exp014_wan5b_nudity_600steps/merged"
ERASED_DIR="models/Wan-AI/Wan2.2-TI2V-5B-exp014-erased"

# ---- 前置检查 ----
echo "🔍 前置检查:"
echo "-----------------------------------"

if [ ! -f "$ADAPTER" ]; then
    echo "❌ 错误: 擦除 LoRA 不存在: $ADAPTER"
    echo "   请先运行 Stage 1: sbatch slurm/exp014_stage1_test.sbatch"
    exit 1
fi
echo "✓ 擦除 LoRA: $ADAPTER"

if [ ! -d "$BASE_TRANSFORMER" ]; then
    echo "❌ 错误: 基座 transformer 不存在: $BASE_TRANSFORMER"
    exit 1
fi
echo "✓ 基座 transformer: $BASE_TRANSFORMER"

echo ""
echo "=================================================="
echo "开始合并..."
echo "=================================================="
echo ""

# ---- 合并 + 桥转换 ----
echo "🔗 阶段 2: LoRA 合并 + 桥转换"
echo "-----------------------------------"

python3 -u scripts/merge_unlearn_lora_wan5b.py \
    --adapter "$ADAPTER" \
    --base-dir "$BASE_TRANSFORMER" \
    --output-dir "$MERGED_DIR" \
    --bridge-output-dir "$ERASED_DIR" \
    --bridge-dtype bf16 \
    --max-shard-size-gb 4.0 \
    --strict \
    --check-hash \
    --force

if [ $? -ne 0 ]; then
    echo "❌ 合并失败！"
    exit 1
fi

echo ""
echo "✅ 合并完成"
echo ""

# ---- 补齐 T5/VAE 软链 ----
echo "🔗 补齐 T5/VAE 软链"
echo "-----------------------------------"

SHARED="models/DiffSynth-Studio/Wan-Series-Converted-Safetensors"
mkdir -p "$ERASED_DIR"

for f in models_t5_umt5-xxl-enc-bf16.safetensors Wan2.2_VAE.safetensors; do
    if [ ! -e "$ERASED_DIR/$f" ]; then
        ln -sf "$PWD/$SHARED/$f" "$ERASED_DIR/$f"
        echo "  ✓ 创建软链: $f"
    else
        echo "  ✓ 已存在: $f"
    fi
done

echo ""
echo "=================================================="
echo "✅ Stage 2 测试完成！"
echo "=================================================="
echo ""

# ---- 产物检查 ----
echo "📦 产物检查:"
echo "-----------------------------------"

echo "1. Merged diffusers transformer:"
if [ -d "$MERGED_DIR" ]; then
    ls -lh "$MERGED_DIR"/ | head -10
    echo "   文件数: $(ls -1 "$MERGED_DIR"/*.safetensors 2>/dev/null | wc -l)"
else
    echo "   ❌ 目录不存在"
fi

echo ""
echo "2. DiffSynth 擦除后基座:"
if [ -d "$ERASED_DIR" ]; then
    ls -lh "$ERASED_DIR"/ | head -15

    # 检查关键文件
    echo ""
    echo "   关键文件检查:"
    for f in models_t5_umt5-xxl-enc-bf16.safetensors Wan2.2_VAE.safetensors; do
        if [ -e "$ERASED_DIR/$f" ]; then
            echo "   ✓ $f"
        else
            echo "   ✗ $f (缺失)"
        fi
    done
else
    echo "   ❌ 目录不存在"
fi

echo ""
echo "3. 合并报告:"
if [ -f "$MERGED_DIR/merge_report.json" ]; then
    python3 - <<'PY'
import json
with open("models/unlearn/exp014_wan5b_nudity_600steps/merged/merge_report.json") as f:
    r = json.load(f)
print("   Summary:", json.dumps(r["summary"], ensure_ascii=False))
print("   Scaling:", r["scaling"])
print("   Shard rewrites:", len(r["shard_rewrites"]), "files")
if r.get("unmatched"):
    print("   ⚠️  Unmatched keys:", len(r["unmatched"]))
PY
else
    echo "   ⚠️  merge_report.json 不存在"
fi

echo ""
echo "=================================================="
echo "下一步: 如果测试成功，可以运行完整的 pipeline"
echo "=================================================="
