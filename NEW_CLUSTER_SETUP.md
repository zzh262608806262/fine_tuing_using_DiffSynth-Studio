# 新集群部署完整指南

**目标**: 在新集群上从零开始部署完整的实验环境

**最后更新**: 2026-09-22

---

## 📋 目录

1. [前置要求](#1-前置要求)
2. [克隆代码和数据](#2-克隆代码和数据)
3. [环境配置](#3-环境配置)
4. [验证安装](#4-验证安装)
5. [Claude Code 安装（可选）](#5-claude-code-安装可选)
6. [常见问题](#6-常见问题)

---

## 1. 前置要求

### 硬件需求
- **GPU**: 至少 1 × A100 (80GB) 或同等算力
- **存储**: ~150GB 可用空间
  - 代码: ~500MB
  - 基座模型: ~10GB (Wan2.2-TI2V-5B)
  - HuggingFace 数据: ~4GB
  - 工作空间: ~100GB

### 软件依赖
- **操作系统**: Linux (测试于 Ubuntu 20.04+)
- **Python**: 3.10+
- **CUDA**: 11.8+ (与 PyTorch 兼容)
- **Git**: 2.0+
- **Conda**: Miniforge3 或 Miniconda

---

## 2. 克隆代码和数据

### 2.1 克隆 GitHub 代码

```bash
# 创建工作目录
mkdir -p ~/projects
cd ~/projects

# 克隆代码仓库
git clone https://github.com/zzh262608806262/fine_tuing_using_DiffSynth-Studio.git
cd fine_tuing_using_DiffSynth-Studio

# 检查文件完整性
bash check_github_integrity.sh
```

### 2.2 下载 HuggingFace 数据

```bash
# 安装 HuggingFace CLI
pip install huggingface_hub

# 下载数据集 (~4GB)
python3 << 'EOF'
from huggingface_hub import snapshot_download

print("开始下载 HuggingFace 数据集...")
snapshot_download(
    repo_id="littlepig404/wan5b-unlearning-artifacts",
    repo_type="dataset",
    local_dir="./hf_data",
    max_workers=4  # 并行下载
)
print("✅ 下载完成！")
EOF
```

**预计下载时间**: 10-30 分钟（取决于网络速度）

### 2.3 下载基座模型

```bash
# 创建模型目录
mkdir -p models/Wan-AI

# 方法 1: 从 HuggingFace 下载（如果有）
# hf download Wan-AI/Wan2.2-TI2V-5B --local-dir models/Wan-AI/Wan2.2-TI2V-5B

# 方法 2: 从原集群复制（推荐，更快）
# rsync -avP old_cluster:/path/to/Wan2.2-TI2V-5B models/Wan-AI/

# 方法 3: 联系原作者获取
```

**注意**: Wan2.2-TI2V-5B (~10GB) 可能需要授权访问。

---

## 3. 环境配置

### 3.1 配置环境变量

```bash
# 复制模板
cp env.template .env

# 编辑配置
vim .env
```

**关键配置项**:

```bash
# 项目路径
export PROJECT_ROOT="$(pwd)"

# Video Unlearning 项目（如果有外部依赖）
export VU_ROOT="${PROJECT_ROOT}/../video-unlearning"  # 或实际路径

# Wan 模型路径
export WAN_MODEL_PATH="${PROJECT_ROOT}/models/Wan-AI/Wan2.2-TI2V-5B"
export WAN_MODEL_DIFFUSERS="${WAN_MODEL_PATH}-Diffusers"

# Conda 路径
export CONDA_ROOT="${HOME}/miniforge3"  # 或你的 conda 路径

# 其他路径
export TIGER_DATASET="${PROJECT_ROOT}/data/tiger_dataset"
export OUTPUTS_ROOT="${PROJECT_ROOT}/outputs"
export MODELS_ROOT="${PROJECT_ROOT}/models"
```

**加载环境变量**:

```bash
# 临时加载（当前 shell）
source .env

# 永久加载（添加到 ~/.bashrc）
echo "source $(pwd)/.env" >> ~/.bashrc
```

### 3.2 创建 Conda 环境

```bash
# 创建基础环境
conda create -n diffsynth python=3.10 -y
conda activate diffsynth

# 安装 PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装项目依赖
pip install -r requirements.txt  # 如果有

# 或手动安装核心依赖
pip install transformers accelerate diffusers peft
pip install opencv-python pillow numpy pandas
pip install safetensors sentencepiece protobuf
pip install bitsandbytes  # 量化支持
```

### 3.3 链接 HuggingFace 数据

```bash
# 链接模型
ln -s hf_data/models/unlearned models/unlearn
ln -s hf_data/models/finetuned models/finetune

# 链接数据
ln -s hf_data/datasets/tiger_dataset data/tiger_dataset

# 链接评估结果（可选）
ln -s hf_data/evaluations outputs/evaluations_backup
```

### 3.4 安装 Video Unlearning 依赖（如果需要）

```bash
# 克隆 VU 项目
cd ~/projects
git clone https://github.com/your-org/video-unlearning.git  # 替换为实际 URL
cd video-unlearning

# 安装 NudeNet 等评估工具
pip install nudenet
pip install openai anthropic  # 如果需要 GPT-4/Claude 评估

# 返回项目目录
cd ~/projects/fine_tuing_using_DiffSynth-Studio
```

---

## 4. 验证安装

### 4.1 检查环境变量

```bash
source .env

echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "WAN_MODEL_PATH: $WAN_MODEL_PATH"
echo "VU_ROOT: $VU_ROOT"

# 检查路径是否存在
ls -lh "$WAN_MODEL_PATH"
ls -lh models/unlearn/exp015_esd/
ls -lh models/finetune/exp018_esd_ft/
```

### 4.2 测试 Python 导入

```bash
python3 << 'EOF'
# 测试基本导入
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
print(f"✅ GPU count: {torch.cuda.device_count()}")

# 测试项目导入
import sys
sys.path.insert(0, ".")
from scripts.run_unlearn_wan5b import main
print("✅ Unlearn script import OK")

from scripts.evaluate_videos_nudenet import main
print("✅ Evaluation script import OK")

# 测试 VU 依赖（如果需要）
# sys.path.insert(0, os.getenv("VU_ROOT"))
# from src.eval.detectors.nudenet import NudeNetDetector
# print("✅ VU import OK")

print("\n🎉 All imports successful!")
EOF
```

### 4.3 运行 Smoke Test

```bash
# 测试模型加载
python3 << 'EOF'
from diffsynth import ModelManager
import os

model_path = os.getenv("WAN_MODEL_PATH")
print(f"Loading model from: {model_path}")

try:
    manager = ModelManager()
    manager.load_models([model_path])
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
EOF
```

### 4.4 测试 LoRA 加载

```bash
python3 << 'EOF'
from peft import PeftModel
import torch

# 测试加载擦除 LoRA
lora_path = "models/unlearn/exp015_esd"
print(f"Testing LoRA: {lora_path}")

# 这里需要实际的模型加载逻辑
# 只是检查文件是否存在
import os
if os.path.exists(f"{lora_path}/adapter_final.pt"):
    print("✅ LoRA file found!")
else:
    print("❌ LoRA file missing!")
EOF
```

---

## 5. Claude Code 安装（可选）

Claude Code 是 Anthropic 的 AI 编程助手，可以帮助调试和开发。

### 5.1 安装 Claude Code CLI

```bash
# 方法 1: 使用 npm (推荐)
npm install -g @anthropic-ai/claude-code

# 方法 2: 使用 pip
pip install claude-code

# 验证安装
claude-code --version
```

### 5.2 配置 Claude Code

```bash
# 登录 Claude
claude-code auth login

# 进入项目目录
cd ~/projects/fine_tuing_using_DiffSynth-Studio

# 初始化 Claude 配置（如果没有 .claude/ 目录）
mkdir -p .claude
```

### 5.3 使用 Claude Code

```bash
# 启动 Claude Code
claude-code

# 或直接运行命令
claude-code "帮我检查实验 Exp015 的结果"

# 在代码中使用
# Claude Code 会自动读取 .claude/CLAUDE.md 和 memory/ 中的文档
```

### 5.4 Claude Code 项目配置

本项目已配置 Claude 协作环境：

- **`.claude/CLAUDE.md`**: AI 协作指南
- **`memory/`**: 实验记录和项目文档
- **自动上下文**: Claude 会自动加载相关文档

**使用示例**:

```bash
# 询问实验结果
> 总结 Exp018 的微调回潮曲线

# 调试错误
> 为什么 exp019 的蒸馏模型加载失败？

# 生成代码
> 帮我写一个脚本评估 exp020a 的结果
```

---

## 6. 常见问题

### 6.1 CUDA 不可用

**症状**: `torch.cuda.is_available()` 返回 `False`

**解决**:
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 重新安装 PyTorch with CUDA
pip uninstall torch torchvision torchaudio
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 6.2 模块导入失败

**症状**: `ModuleNotFoundError: No module named 'XXX'`

**解决**:
```bash
# 检查是否在正确的 conda 环境
conda env list
conda activate diffsynth

# 重新安装依赖
pip install -r requirements.txt

# 或手动安装缺失的包
pip install <missing-package>
```

### 6.3 路径错误

**症状**: `FileNotFoundError: [Errno 2] No such file or directory`

**解决**:
```bash
# 检查环境变量
source .env
echo $PROJECT_ROOT
echo $WAN_MODEL_PATH

# 检查路径是否存在
ls -lh "$WAN_MODEL_PATH"

# 如果路径不对，重新编辑 .env
vim .env
source .env
```

### 6.4 内存不足

**症状**: `RuntimeError: CUDA out of memory`

**解决**:
```bash
# 1. 使用量化加载
# 在脚本中添加 quantization_config

# 2. 减少 batch size
# 修改脚本中的 batch_size=1

# 3. 使用梯度累积
# 修改 gradient_accumulation_steps

# 4. 使用更小的模型或更少的 LoRA rank
```

### 6.5 HuggingFace 下载慢

**症状**: 下载速度很慢或超时

**解决**:
```bash
# 使用镜像（中国大陆）
export HF_ENDPOINT=https://hf-mirror.com

# 或使用代理
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port

# 重新下载
python download_hf_data.py
```

### 6.6 SLURM 作业失败

**症状**: SLURM 作业立即失败或找不到命令

**解决**:
```bash
# 1. 检查 SLURM 配置
sinfo  # 查看可用分区
squeue -u $USER  # 查看作业队列

# 2. 修改 .sbatch 脚本中的路径
# 将所有绝对路径替换为 ${PROJECT_ROOT}

# 3. 确保在脚本中加载环境
echo "source ${PROJECT_ROOT}/.env" >> your_script.sbatch

# 4. 测试脚本
bash your_script.sbatch  # 本地测试
```

---

## 7. 快速开始示例

### 7.1 运行擦除实验

```bash
source .env
conda activate diffsynth

# 运行 ESD 擦除
python scripts/run_unlearn_wan5b.py \
    --method esd \
    --model-path $WAN_MODEL_PATH \
    --output-dir models/unlearn/my_esd_test \
    --steps 600
```

### 7.2 运行微调实验

```bash
# 使用擦除后的模型微调
python scripts/finetune_wan5b.py \
    --base-model models/unlearn/exp015_esd/adapter_final.pt \
    --data-csv data/tiger_dataset/metadata_100.csv \
    --output-dir models/finetune/my_esd_ft \
    --epochs 20
```

### 7.3 评估模型

```bash
# 生成测试视频
python scripts/exp015_generate_videos.py \
    --model-dir models/unlearn/exp015_esd \
    --output-dir outputs/my_test \
    --num-videos 99

# 评估违规率
python scripts/evaluate_videos_nudenet.py \
    --video-dir outputs/my_test \
    --output-json outputs/my_test_results.json
```

---

## 8. 项目结构说明

```
fine_tuing_using_DiffSynth-Studio/
├── .env                          # 环境变量配置（需创建）
├── .gitignore                    # Git 忽略规则
├── README.md                     # 项目说明
├── MIGRATION.md                  # 迁移指南
├── env.template                  # 环境变量模板
├── requirements.txt              # Python 依赖（如有）
│
├── .claude/                      # Claude Code 配置
│   └── CLAUDE.md                 # AI 协作指南
│
├── memory/                       # 项目文档和实验记录
│   ├── experiments.md            # 完整实验记录（Exp001-Exp020）
│   ├── MASTER_SUMMARY.md         # 项目总览
│   ├── QUICK_DATA_REFERENCE.md   # 数据快速查询
│   ├── group_meeting_presentation.md  # 组会汇报材料
│   └── ...                       # 其他文档
│
├── scripts/                      # Python 脚本
│   ├── run_unlearn_wan5b.py      # 擦除训练
│   ├── finetune_wan5b.py         # 微调训练
│   ├── quantize_wan5b.py         # 量化
│   ├── distill_wan5b.py          # 蒸馏
│   ├── evaluate_videos_nudenet.py # 评估
│   └── ...                       # 其他脚本
│
├── slurm/                        # SLURM 作业脚本
│   ├── exp015_submit.sh          # Exp015 提交脚本
│   ├── exp018_submit_all.sh      # Exp018 批量提交
│   └── ...                       # 其他 SLURM 脚本
│
├── models/                       # 模型目录（大文件，不在 Git）
│   ├── Wan-AI/                   # 基座模型
│   ├── unlearn/                  # 擦除 LoRA（HF 链接）
│   └── finetune/                 # 微调 checkpoints（HF 链接）
│
├── data/                         # 数据目录（大文件，不在 Git）
│   └── tiger_dataset/            # 训练数据（HF 链接）
│
├── outputs/                      # 输出目录（大文件，不在 Git）
│   └── evaluations_backup/       # 评估结果（HF 链接）
│
└── hf_data/                      # HuggingFace 下载数据（链接源）
    ├── models/
    ├── datasets/
    └── evaluations/
```

---

## 9. 资源链接

- **GitHub 代码**: https://github.com/zzh262608806262/fine_tuing_using_DiffSynth-Studio
- **HuggingFace 数据**: https://huggingface.co/datasets/littlepig404/wan5b-unlearning-artifacts
- **实验记录**: `memory/experiments.md`
- **组会汇报**: `memory/group_meeting_presentation.md`

---

## 10. 获取帮助

### 10.1 使用 Claude Code

如果安装了 Claude Code：
```bash
claude-code "我在新集群部署遇到问题：[描述问题]"
```

### 10.2 查看文档

```bash
# 查看实验记录
less memory/experiments.md

# 查看快速参考
less memory/QUICK_DATA_REFERENCE.md

# 查看迁移指南
less MIGRATION.md
```

### 10.3 GitHub Issues

https://github.com/zzh262608806262/fine_tuing_using_DiffSynth-Studio/issues

---

**部署完成后，你应该能够**:
- ✅ 运行所有实验脚本
- ✅ 加载擦除 LoRA 和微调 checkpoints
- ✅ 生成和评估视频
- ✅ 使用 Claude Code 辅助开发

**预计部署时间**: 1-2 小时（取决于下载速度和环境配置）
