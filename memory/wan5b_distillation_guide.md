# Wan2.2-TI2V-5B 蒸馏指南

## 概述

本指南介绍如何对 Wan2.2-TI2V-5B 模型进行知识蒸馏，将推理步数从 30-50 步压缩到 4-8 步，同时保持生成质量。

## 背景

### 什么是知识蒸馏？

知识蒸馏（Knowledge Distillation）是一种模型压缩技术，通过训练使模型在更少的推理步数下达到接近原始模型的效果。对于视频生成模型：
- **原始模型**：需要 30-50 步 diffusion 采样才能生成高质量视频
- **蒸馏后模型**：只需 4-8 步即可生成相似质量的视频，**速度提升 4-10 倍**

### Direct Distill 方法

本项目使用 DiffSynth-Studio 的 `direct_distill` 任务：
- 模型从纯噪声出发，在 N 步（如 4 步）去噪后直接与干净视频计算 MSE 损失
- 训练后模型学会用极少步数直接预测清晰结果

## 文件说明

### 核心脚本

- **`scripts/distill_wan5b.py`**：主蒸馏脚本
  - 基于 `scripts/distill.py`（Wan2.1-T2V-1.3B 蒸馏）改编
  - 适配 5B 模型特点：
    - 使用 Wan2.2_VAE.pth（3D VAE）
    - 支持 49 帧标准（5B TI2V）
    - 额外输入 `input_image`（图生视频）

### SLURM 作业脚本

- **`slurm/distill_wan5b.sbatch`**：蒸馏训练作业
  - 预计时间：8-10 小时
  - GPU 要求：1 块 A100（40GB 或 80GB）
  
- **`slurm/validate_distill_wan5b.sbatch`**：验证作业
  - 预计时间：5-10 分钟
  - 测试蒸馏后模型的快速推理能力

## 使用方法

### 1. 数据准备

#### 选项 A：使用官方数据集（推荐）

```bash
# 数据集会自动下载（通过 modelscope）
# 包含在训练脚本中，无需手动操作
```

#### 选项 B：使用自定义数据集

如果官方没有提供 Wan2.2-TI2V-5B 蒸馏数据集，你需要准备：

```
data/custom_distill_dataset/
├── metadata.csv          # 必需：包含 video, prompt 列
├── video_001.mp4
├── video_002.mp4
└── ...
```

**metadata.csv 格式**：
```csv
video,prompt
video_001.mp4,"A dog running in the park"
video_002.mp4,"Ocean waves at sunset"
```

### 2. 训练蒸馏模型

#### 快速启动（默认配置）

```bash
sbatch slurm/distill_wan5b.sbatch
```

#### 自定义配置

```bash
# 使用自定义数据集
sbatch --export=DATASET_BASE=data/custom_distill_dataset,\
DATASET_META=data/custom_distill_dataset/metadata.csv,\
DOWNLOAD_DATASET=false \
slurm/distill_wan5b.sbatch

# 调整超参数
sbatch --export=LEARNING_RATE=5e-6,\
NUM_EPOCHS=3,\
DATASET_REPEAT=200 \
slurm/distill_wan5b.sbatch
```

#### 关键超参数说明

| 参数 | 默认值 | 说明 | 建议范围 |
|------|--------|------|----------|
| `LEARNING_RATE` | 1e-5 | 学习率 | 5e-6 ~ 2e-5 |
| `NUM_EPOCHS` | 2 | 训练轮数 | 2 ~ 5 |
| `DATASET_REPEAT` | 160 | 数据集重复次数 | 100 ~ 200 |
| `NUM_FRAMES` | 49 | 视频帧数 | 25 / 49 / 81 |
| `HEIGHT` | 480 | 视频高度 | 480 / 640 |
| `WIDTH` | 832 | 视频宽度 | 832 / 1280 |

### 3. 验证蒸馏效果

训练完成后，测试蒸馏模型的推理速度和质量：

```bash
# 默认：4 步推理
sbatch slurm/validate_distill_wan5b.sbatch

# 测试 8 步推理
sbatch --export=NUM_STEPS=8,\
OUTPUT_VIDEO=outputs/distill_wan5b_8step.mp4 \
slurm/validate_distill_wan5b.sbatch

# 使用输入图片（TI2V 图生视频）
sbatch --export=INPUT_IMAGE=data/test_image.jpg,\
PROMPT="A person walking in the forest" \
slurm/validate_distill_wan5b.sbatch
```

### 4. 本地测试（可选）

如果想在本地快速测试：

```bash
# 训练模式
python scripts/distill_wan5b.py \
    --mode train \
    --no_download_dataset \
    --dataset_base_path data/my_dataset \
    --dataset_metadata_path data/my_dataset/metadata.csv \
    --num_epochs 1 \
    --dataset_repeat 10

# 验证模式
python scripts/distill_wan5b.py \
    --mode validate \
    --distilled_model_path ./models/train/Wan2.2-TI2V-5B_distill/epoch-1.safetensors \
    --num_inference_steps 4 \
    --output_video test_output.mp4
```

## 与 Wan2.1-1.3B 蒸馏的对比

| 特性 | Wan2.1-T2V-1.3B | Wan2.2-TI2V-5B |
|------|-----------------|----------------|
| 参数量 | 1.3B | 5B |
| VAE | Wan2.1_VAE.pth | Wan2.2_VAE.pth (3D) |
| 标准帧数 | 81 帧 | 49 帧 |
| 输入模式 | T2V（纯文本） | TI2V（文本+图片） |
| 训练时间 | ~4-6 小时 | ~8-10 小时 |
| 蒸馏目标 | 4 步推理 | 4-8 步推理 |

## 预期结果

### 速度提升

- **原始模型（30 步）**：~140 秒/视频
- **蒸馏模型（4 步）**：~20-30 秒/视频
- **加速比**：**约 5-7 倍**

### 质量保持

根据 Exp 003（1.3B 蒸馏）的经验：
- 视觉质量与原模型接近
- 细节略有损失（acceptable trade-off）
- CFG scale 建议降到 1.0（蒸馏模型特性）

## 常见问题

### Q1: 官方数据集下载失败怎么办？

如果 modelscope 下载失败，可以：
1. 使用 `--no_download_dataset` 跳过自动下载
2. 准备自己的数据集（参见"数据准备"）
3. 或使用 Wan2.2-TI2V-5B 的示例数据集（非蒸馏专用）

### Q2: 训练时 OOM（显存不足）怎么办？

尝试以下方法：
1. 减少 `NUM_FRAMES`（49 → 25）
2. 减少分辨率（832 → 640）
3. 启用 `use_gradient_checkpointing_offload`
4. 使用 80GB A100 节点

### Q3: 蒸馏后质量不理想怎么办？

可能的原因和解决方案：
1. **训练不足**：增加 `NUM_EPOCHS` 或 `DATASET_REPEAT`
2. **学习率过大**：降低到 5e-6
3. **目标步数太少**：尝试 6-8 步而非 4 步
4. **数据集不够多样**：准备更丰富的训练数据

### Q4: 如何选择蒸馏目标步数？

| 目标步数 | 速度提升 | 质量保持 | 推荐场景 |
|----------|----------|----------|----------|
| 4 步 | 最快（7-10x） | 良好 | 原型演示、快速迭代 |
| 6 步 | 快（5-7x） | 很好 | 平衡选择 |
| 8 步 | 较快（4-5x） | 优秀 | 追求质量 |

## 后续工作

### 实验记录

将蒸馏训练记录为新实验（如 Exp 016）：

```markdown
## Exp 016 — Wan2.2-TI2V-5B 4-step 蒸馏

- **Date**: 2026-08-31
- **Script**: `scripts/distill_wan5b.py`
- **Model**: 基座 `Wan-AI/Wan2.2-TI2V-5B`
- **Config**: 全参数蒸馏，lr=1e-5；2 epochs；数据=...；480×832×49 帧；蒸馏目标 `num_inference_steps=4`
- **Results**: （待训练完成后填写）
- **Artifacts**: `models/train/Wan2.2-TI2V-5B_distill/epoch-{0,1}.safetensors`
```

### 评估蒸馏效果

可以参考 Exp 007 的方法，对比：
- 原始模型 30 步 vs 蒸馏模型 4 步
- 生成质量（NudeNet、分类器评分）
- 推理速度（秒/视频）

### 量化蒸馏模型

可以进一步对蒸馏后的模型进行 NF4 量化（参考 Exp 004）：
- 蒸馏：30 步 → 4 步（7x 加速）
- 量化：FP16 → NF4（~4x 显存节省）
- **总效果**：更快 + 更省显存

## 参考

- DiffSynth-Studio 官方文档：[https://github.com/modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- Wan2.1-1.3B 蒸馏示例：`examples/wanvideo/model_training/special/direct_distill/`
- 本项目 Exp 003：Wan2.1-T2V-1.3B 4-step 蒸馏记录
