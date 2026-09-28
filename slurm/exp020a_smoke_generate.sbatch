#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 0:30:00
#SBATCH -J exp020a_smoke_generate
#SBATCH -o slurm/logs/exp020a_smoke_generate_%j.out
#SBATCH -e slurm/logs/exp020a_smoke_generate_%j.err

# Exp020a: 生成冒烟测试 - 只生成1个视频验证流程

set -e

echo "=================================================="
echo "Exp020a: 生成冒烟测试 (1个视频)"
echo "=================================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "=================================================="
echo ""

# 激活环境
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

# 跳过网络下载，使用本地模型
export DIFFSYNTH_SKIP_DOWNLOAD=True

# 路径配置
MODEL_DIR="models/wan5b/exp020a_npo_masked_erased"
PROMPTS_FILE="data/exp015/nudity_eval_prompts.jsonl"
OUTPUT_DIR="outputs/exp020a/smoke_test_generate"

echo "生成配置:"
echo "  模型: $MODEL_DIR"
echo "  提示词: 只测试第1条"
echo "  输出: $OUTPUT_DIR"
echo "  参数: 17帧, 480×736, 50步, cfg=5.0, seed=0"
echo ""

# 检查输入
if [ ! -d "$MODEL_DIR" ]; then
    echo "[ERROR] 模型不存在: $MODEL_DIR"
    exit 1
fi

if [ ! -f "$PROMPTS_FILE" ]; then
    echo "[ERROR] 提示词文件不存在: $PROMPTS_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# 运行生成
python3 << 'PYEOF'
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

print("=" * 60)
print("加载模型...")

# 使用model_id方式加载（类似exp015）
MODEL_FILES = [
    "diffusion_pytorch_model*.safetensors",  # DiT分片文件
    "models_t5_umt5-xxl-enc-bf16.safetensors",  # Text encoder
    "Wan2.2_VAE.safetensors",  # VAE
]

model_configs = [
    ModelConfig(model_id="wan5b/exp020a_npo_masked_erased", origin_file_pattern=pattern)
    for pattern in MODEL_FILES
]

tokenizer_config = ModelConfig(
    model_id="Wan-AI/Wan2.1-T2V-1.3B",
    origin_file_pattern="google/umt5-xxl/"
)

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.float16,
    device="cuda",
    model_configs=model_configs,
    tokenizer_config=tokenizer_config,
)
print("✅ 模型加载完成")

# 读取第一条prompt
import json
with open("data/exp015/nudity_eval_prompts.jsonl", 'r') as f:
    prompt = json.loads(f.readline())['prompt']

print(f"\n生成视频: {prompt[:50]}...")

video = pipe(
    prompt=prompt,
    num_frames=17,
    height=480,
    width=736,
    num_inference_steps=50,
    guidance_scale=5.0,
    seed=0,
)

output_path = "outputs/exp020a/smoke_test_generate/video_000.mp4"
pipe.save_video(video, output_path, fps=8)

import os
file_size = os.path.getsize(output_path)
print(f"✅ 视频已保存: {output_path}")
print(f"   文件大小: {file_size} bytes")
print("=" * 60)
PYEOF

if [ $? -ne 0 ]; then
    echo "❌ 生成失败！"
    exit 1
fi

# 检查输出
if [ -f "$OUTPUT_DIR/video_000.mp4" ]; then
    FILE_SIZE=$(stat -c%s "$OUTPUT_DIR/video_000.mp4" 2>/dev/null)
    echo ""
    echo "=================================================="
    echo "✅ 冒烟测试成功！"
    echo "  视频文件: $OUTPUT_DIR/video_000.mp4"
    echo "  文件大小: $FILE_SIZE bytes"
    echo "完成时间: $(date)"
    echo "=================================================="
else
    echo "❌ 视频文件未生成"
    exit 1
fi
