#!/bin/bash
# Exp 015 前置条件检查脚本

set -euo pipefail

FT_ROOT="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"
VU_ROOT="/home/x_jiage/jiage/video-unlearning"

cd "$FT_ROOT"

echo "========================================"
echo "Exp 015 前置条件检查"
echo "Time: $(date)"
echo "========================================"

# 错误计数
ERRORS=0

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_file() {
    local file=$1
    local desc=$2
    if [ -e "$file" ]; then
        echo -e "${GREEN}✓${NC} $desc: $file"
    else
        echo -e "${RED}✗${NC} $desc 缺失: $file"
        ((ERRORS++))
    fi
}

check_dir() {
    local dir=$1
    local desc=$2
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓${NC} $desc: $dir"
    else
        echo -e "${RED}✗${NC} $desc 缺失: $dir"
        ((ERRORS++))
    fi
}

check_count() {
    local pattern=$1
    local desc=$2
    local min_count=$3
    local count=$(find $pattern 2>/dev/null | wc -l)
    if [ "$count" -ge "$min_count" ]; then
        echo -e "${GREEN}✓${NC} $desc: $count 个文件 (需要 ≥$min_count)"
    else
        echo -e "${RED}✗${NC} $desc: $count 个文件 (需要 ≥$min_count)"
        ((ERRORS++))
    fi
}

echo ""
echo "[1/6] 检查 VU 仓库..."
check_dir "$VU_ROOT" "VU 仓库根目录"
check_dir "$VU_ROOT/src" "VU src 包"
check_dir "$VU_ROOT/configs/methods/training" "VU 方法配置"
check_file "$VU_ROOT/data/splits/nudity_forget_composed_wan.jsonl" "Forget manifest"
check_file "$VU_ROOT/data/splits/nudity_retain_composed_wan.jsonl" "Retain manifest"

echo ""
echo "[2/6] 检查 Exp 012 产物（复用资源）..."
check_dir "models/Wan-AI/Wan2.2-TI2V-5B" "base 桥产物（DiffSynth 格式）"
check_dir "models/Wan-AI/Wan2.2-TI2V-5B-Diffusers" "5B diffusers 基座"
check_dir "data/wan5b/unlearn_baseline" "基线视频缓存"
check_count "data/wan5b/unlearn_latents/latents/*.pt" "Latent 缓存文件" 25

echo ""
echo "[3/6] 检查训练数据..."
check_file "data/tiger_dataset/metadata_100.csv" "Tiger 训练数据"

echo ""
echo "[4/6] 检查共享资源..."
check_file "models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors" "T5 编码器"
check_file "models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors" "VAE"

echo ""
echo "[5/6] 检查脚本和作业文件..."
check_file "scripts/run_unlearn_wan5b_multi_method.py" "多方法擦除驱动脚本"
check_file "scripts/merge_unlearn_lora_wan5b.py" "LoRA 合并脚本（复用 Exp 012）"
check_file "scripts/finetune_wan5b.py" "微调脚本（复用 Exp 012）"

for method in grad_diff esd npo anchor_distill; do
    check_file "slurm/exp015_unlearn_${method}.sbatch" "${method} 擦除作业"
    check_file "slurm/exp015_merge_${method}.sbatch" "${method} 合并作业"
    check_file "slurm/exp015_finetune_${method}.sbatch" "${method} 微调作业"
done

check_file "slurm/exp015_submit.sh" "全流程提交脚本"

echo ""
echo "[6/6] 检查 VU 方法配置文件..."
for method in grad_diff esd npo anchor_distill; do
    check_file "$VU_ROOT/configs/methods/training/${method}_wan.yaml" "${method} 配置"
done

echo ""
echo "========================================"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ 所有前置条件满足！${NC}"
    echo ""
    echo "可以开始实验:"
    echo "  bash slurm/exp015_submit.sh"
else
    echo -e "${RED}✗ 发现 $ERRORS 个问题${NC}"
    echo ""
    echo "请先解决上述问题，然后重新运行此脚本"
fi
echo "========================================"

exit $ERRORS
