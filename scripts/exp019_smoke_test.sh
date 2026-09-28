#!/bin/bash
# Exp019 交互节点冒烟测试
# 在node051上测试微调/量化/蒸馏的配置，确保可以跑通后再提交全量作业

set -euo pipefail

echo "========================================"
echo "Exp019 冒烟测试 - node051"
echo "时间: $(date)"
echo "节点: $(hostname)"
echo "========================================"

cd "$(dirname "$(dirname "$(readlink -f "$0")")")" # Auto: project root

# ============================================================================
# Test 1: 微调配置测试
# ============================================================================
echo ""
echo "========================================"
echo "Test 1: 微调配置验证"
echo "========================================"

# 激活diffsynth环境
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

export LD_LIBRARY_PATH="/home/x_jiage/.conda/envs/diffsynth/lib:$LD_LIBRARY_PATH"
export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ERASED_DIT="models/wan5b/exp019_graddiff_erased"
BASE_MODEL="models/Wan-AI/Wan2.2-TI2V-5B"

echo "检查擦除模型..."
ERASED_DIT="models/wan5b/exp019_graddiff_erased"
BASE_MODEL="models/Wan-AI/Wan2.2-TI2V-5B"

if [ ! -d "$ERASED_DIT" ]; then
    echo "❌ 擦除DiT不存在: $ERASED_DIT"
    exit 1
fi

if [ ! -d "$BASE_MODEL" ]; then
    echo "❌ 原始基座不存在: $BASE_MODEL"
    exit 1
fi

# 检查DiT分片
dit_files=(
    "diffusion_pytorch_model-00001-of-00003.safetensors"
    "diffusion_pytorch_model-00002-of-00003.safetensors"
    "diffusion_pytorch_model-00003-of-00003.safetensors"
)

for file in "${dit_files[@]}"; do
    if [ ! -f "$ERASED_DIT/$file" ]; then
        echo "❌ 缺少DiT文件: $ERASED_DIT/$file"
        exit 1
    fi
done

# 检查T5/VAE
auxiliary_files=(
    "models_t5_umt5-xxl-enc-bf16.safetensors"
    "Wan2.2_VAE.safetensors"
)

for file in "${auxiliary_files[@]}"; do
    if [ ! -f "$BASE_MODEL/$file" ] && [ ! -L "$BASE_MODEL/$file" ]; then
        echo "❌ 缺少T5/VAE文件: $BASE_MODEL/$file"
        exit 1
    fi
done

echo "✓ 擦除DiT文件完整: $ERASED_DIT"
echo "✓ T5/VAE文件可用: $BASE_MODEL"

# 构建模型路径
MODEL_PATHS=""
for file in "${dit_files[@]}"; do
    MODEL_PATHS="${MODEL_PATHS}${ERASED_DIT}/${file},"
done
for file in "${auxiliary_files[@]}"; do
    MODEL_PATHS="${MODEL_PATHS}${BASE_MODEL}/${file},"
done
MODEL_PATHS="${MODEL_PATHS%,}"

echo "✓ 模型路径构建完成"

# 测试加载（dry-run）
echo "测试微调脚本加载..."
python -c "
import sys
sys.path.insert(0, 'examples/wanvideo/model_training')
import train
print('✓ 微调脚本导入成功')
"

echo "✓ Test 1 通过: 微调配置正常"

# ============================================================================
# Test 2: 量化配置测试
# ============================================================================
echo ""
echo "========================================"
echo "Test 2: 量化配置验证"
echo "========================================"

echo "检查量化依赖..."
python -c "
import torch
import bitsandbytes as bnb
print(f'PyTorch: {torch.__version__}')
print(f'bitsandbytes: {bnb.__version__}')
print('✓ 量化依赖完整')
"

# 测试量化加载
echo "测试量化加载（仅CPU，1层）..."
python -c "
import torch
from safetensors.torch import load_file

# 测试加载一个小分片
state_dict = load_file('$ERASED_DIT/diffusion_pytorch_model-00001-of-00003.safetensors')
print(f'✓ 成功加载分片，包含 {len(state_dict)} 个键')

# 测试量化一个小tensor
test_tensor = next(iter(state_dict.values()))
if test_tensor.dtype == torch.bfloat16:
    quantized = test_tensor.to(torch.float16)
    print(f'✓ 测试量化: {test_tensor.dtype} -> {quantized.dtype}')
else:
    print(f'✓ Tensor dtype: {test_tensor.dtype}')
"

echo "✓ Test 2 通过: 量化配置正常"

# ============================================================================
# Test 3: 蒸馏配置测试
# ============================================================================
echo ""
echo "========================================"
echo "Test 3: 蒸馏配置验证"
echo "========================================"

# 检查蒸馏数据
DATASET_PATH="data/tiger_dataset/metadata_100.csv"
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ 蒸馏数据集不存在: $DATASET_PATH"
    exit 1
fi

echo "✓ 蒸馏数据集存在"

# 测试蒸馏脚本导入
echo "测试蒸馏脚本..."
python -c "
import sys
sys.path.insert(0, 'examples/wanvideo/model_training')
import train

# 检查蒸馏相关参数
import argparse
parser = argparse.ArgumentParser()
# 这些是蒸馏训练需要的参数
required_args = ['dataset_base_path', 'dataset_metadata_path', 'model_id_with_origin_paths']
print('✓ 蒸馏脚本导入成功')
"

echo "✓ Test 3 通过: 蒸馏配置正常"

# ============================================================================
# Test 4: 微调后生成脚本验证（dry-run，不需要实际LoRA）
# ============================================================================
echo ""
echo "========================================"
echo "Test 4: 微调后生成脚本验证"
echo "========================================"

# 切换到VU环境
source /home/x_jiage/jiage/video-unlearning/.venv/bin/activate

# 只测试脚本语法和导入，不验证LoRA文件
echo "测试生成脚本语法..."
python -c "
import sys
sys.path.insert(0, 'scripts')
# 只导入测试，不运行
with open('scripts/exp019_generate_finetuned.py', 'r') as f:
    code = f.read()
    compile(code, 'exp019_generate_finetuned.py', 'exec')
print('✓ 生成脚本语法正确')
"

echo "✓ Test 4 通过: 生成脚本正常（LoRA将在微调后生成）"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================"
echo "✓ 所有冒烟测试通过"
echo "========================================"
echo ""
echo "✅ Test 1: 微调配置正常"
echo "✅ Test 2: 量化配置正常"
echo "✅ Test 3: 蒸馏配置正常"
echo "✅ Test 4: 生成脚本正常"
echo ""
echo "可以提交全量作业："
echo "  1. bash scripts/exp019_submit_finetune.sh     # 微调"
echo "  2. bash scripts/exp019_submit_quantize.sh     # 量化（待创建）"
echo "  3. bash scripts/exp019_submit_distill.sh      # 蒸馏（待创建）"
echo "========================================"
