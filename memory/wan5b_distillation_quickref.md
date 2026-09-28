# Wan5B 蒸馏快速参考

## 一键启动

```bash
# 1. 训练蒸馏模型（8-10 小时）
sbatch slurm/distill_wan5b.sbatch

# 2. 验证效果（5 分钟）
sbatch slurm/validate_distill_wan5b.sbatch
```

## 核心文件

- `scripts/distill_wan5b.py` - 主脚本（train/validate 两模式）
- `slurm/distill_wan5b.sbatch` - 训练作业
- `slurm/validate_distill_wan5b.sbatch` - 验证作业

## 关键配置

| 参数 | 默认值 | 用途 |
|------|--------|------|
| LEARNING_RATE | 1e-5 | 学习率（建议 5e-6 ~ 2e-5）|
| NUM_EPOCHS | 2 | 训练轮数 |
| DATASET_REPEAT | 160 | 数据重复次数 |
| NUM_FRAMES | 49 | 视频帧数（5B 标准）|

## 自定义训练

```bash
# 使用自己的数据集
sbatch --export=DATASET_BASE=data/my_dataset,\
DATASET_META=data/my_dataset/metadata.csv,\
DOWNLOAD_DATASET=false \
slurm/distill_wan5b.sbatch

# 调整超参数
sbatch --export=LEARNING_RATE=5e-6,NUM_EPOCHS=3 \
slurm/distill_wan5b.sbatch
```

## 验证选项

```bash
# 测试不同步数
sbatch --export=NUM_STEPS=4 slurm/validate_distill_wan5b.sbatch   # 最快
sbatch --export=NUM_STEPS=8 slurm/validate_distill_wan5b.sbatch   # 高质量

# 图生视频（TI2V）
sbatch --export=INPUT_IMAGE=test.jpg slurm/validate_distill_wan5b.sbatch
```

## 预期效果

- **速度**：30 步 → 4 步 = **7x 加速**（140s → 20s/视频）
- **质量**：接近原模型（细节略有损失）
- **显存**：与原模型相同

## 与 1.3B 蒸馏的差异

| 特性 | 1.3B | 5B |
|------|------|-----|
| 参数 | 1.3B | 5B |
| VAE | Wan2.1 | Wan2.2 (3D) |
| 帧数 | 81 | 49 |
| 模式 | T2V | TI2V (支持图片输入) |
| 时长 | 4-6h | 8-10h |

## 常见问题

**Q: 数据集下载失败？**
```bash
# 准备 metadata.csv 后跳过下载
sbatch --export=DOWNLOAD_DATASET=false slurm/distill_wan5b.sbatch
```

**Q: 显存不足 (OOM)？**
- 减少帧数：`NUM_FRAMES=25`
- 降低分辨率：`WIDTH=640`
- 使用 80GB 节点

**Q: 质量不够好？**
- 增加训练：`NUM_EPOCHS=3, DATASET_REPEAT=200`
- 降低学习率：`LEARNING_RATE=5e-6`
- 提高目标步数：`NUM_STEPS=6` 或 `8`

## 完整文档

详见：`docs/wan5b_distillation_guide.md`
