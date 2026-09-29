#!/bin/bash
# 完整环境安装脚本
# 包含 DiffSynth 和 Video Unlearning 所需的所有依赖

set -e  # 遇到错误立即退出

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "DiffSynth + Video Unlearning 环境安装"
echo "=========================================="
echo ""
echo "项目路径: $PROJECT_ROOT"
echo ""

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda 未安装！请先安装 Miniforge3 或 Miniconda"
    exit 1
fi
echo "✅ Conda 已安装: $(conda --version)"

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU 检测到:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
else
    echo "⚠️  未检测到 NVIDIA GPU"
fi
echo ""

# ============================================
# 1. 创建 Conda 环境
# ============================================
echo "1. 创建 Conda 环境..."
echo "-------------------------------------------"

ENV_NAME="diffsynth"

if conda env list | grep -q "^${ENV_NAME} "; then
    read -p "环境 '${ENV_NAME}' 已存在，是否删除并重新创建？ (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除旧环境..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "跳过环境创建，使用现有环境"
        conda activate "$ENV_NAME"
        SKIP_CONDA=1
    fi
fi

if [ -z "$SKIP_CONDA" ]; then
    if [ -f "environment.yml" ]; then
        echo "使用 environment.yml 创建环境..."
        conda env create -f environment.yml
    else
        echo "使用默认配置创建环境..."
        conda create -n "$ENV_NAME" python=3.10 -y
        conda activate "$ENV_NAME"

        # 安装 PyTorch with CUDA
        echo "安装 PyTorch with CUDA 11.8..."
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    fi
fi

echo "✅ Conda 环境创建完成"
echo ""

# 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
echo "✅ 已激活环境: $ENV_NAME"
echo ""

# ============================================
# 2. 安装核心依赖
# ============================================
echo "2. 安装核心依赖..."
echo "-------------------------------------------"

if [ -f "requirements.txt" ]; then
    echo "从 requirements.txt 安装..."
    pip install -r requirements.txt
else
    echo "手动安装核心包..."

    # Hugging Face
    pip install transformers>=4.30.0 diffusers>=0.20.0 accelerate>=0.20.0 peft>=0.4.0 safetensors>=0.3.0

    # 量化
    pip install bitsandbytes>=0.41.0

    # 视频处理
    pip install opencv-python pillow imageio imageio-ffmpeg av decord

    # 文本处理
    pip install sentencepiece protobuf tokenizers

    # 安全检测
    pip install nudenet

    # API
    pip install openai anthropic

    # 工具
    pip install pandas pyyaml tqdm einops python-dotenv

    # 监控
    pip install tensorboard wandb
fi

echo "✅ 核心依赖安装完成"
echo ""

# ============================================
# 3. 安装 DiffSynth（本地）
# ============================================
echo "3. 安装 DiffSynth (本地)..."
echo "-------------------------------------------"

if [ -d "diffsynth" ]; then
    echo "检测到本地 DiffSynth 目录"
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
    echo "已添加到 PYTHONPATH"
else
    echo "⚠️  未找到 diffsynth/ 目录"
fi
echo ""

# ============================================
# 4. 配置环境变量
# ============================================
echo "4. 配置环境变量..."
echo "-------------------------------------------"

if [ ! -f ".env" ]; then
    if [ -f "env.template" ]; then
        echo "从模板创建 .env..."
        cp env.template .env
        echo "⚠️  请编辑 .env 文件配置路径!"
        echo "   vim .env"
    else
        echo "创建基础 .env..."
        cat > .env << ENVEOF
export PROJECT_ROOT="${PROJECT_ROOT}"
export VU_ROOT="${PROJECT_ROOT}/../video-unlearning"
export WAN_MODEL_PATH="${PROJECT_ROOT}/models/Wan-AI/Wan2.2-TI2V-5B"
export CONDA_ROOT="$(conda info --base)"
ENVEOF
    fi
fi

source .env
echo "✅ 环境变量已加载"
echo ""

# ============================================
# 5. 验证安装
# ============================================
echo "5. 验证安装..."
echo "-------------------------------------------"

python3 << 'PYEOF'
import sys

packages = {
    "torch": "PyTorch",
    "transformers": "Transformers",
    "diffusers": "Diffusers",
    "accelerate": "Accelerate",
    "peft": "PEFT",
    "cv2": "OpenCV",
    "PIL": "Pillow",
    "nudenet": "NudeNet",
    "openai": "OpenAI",
}

failed = []
for module, name in packages.items():
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "installed")
        print(f"✅ {name}: {version}")
    except ImportError:
        print(f"❌ {name}: NOT INSTALLED")
        failed.append(name)

# 检查 CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f"\n✅ CUDA available: {torch.version.cuda}")
        print(f"✅ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("\n⚠️  CUDA not available")
except Exception as e:
    print(f"\n❌ CUDA check failed: {e}")

# 检查本地 DiffSynth
try:
    import diffsynth
    print("\n✅ DiffSynth: installed (local)")
except ImportError:
    print("\n⚠️  DiffSynth: not in PYTHONPATH")

if failed:
    print(f"\n❌ 以下包安装失败: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\n🎉 所有依赖安装成功!")
PYEOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 验证失败，请检查错误信息"
    exit 1
fi

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 激活环境: conda activate $ENV_NAME"
echo "2. 配置路径: vim .env && source .env"
echo "3. 下载数据: python download_hf_data.py"
echo "4. 验证: python -c 'from scripts.run_unlearn_wan5b import main'"
echo ""
echo "环境详情:"
echo "  Conda 环境: $ENV_NAME"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "N/A")')"
echo ""
