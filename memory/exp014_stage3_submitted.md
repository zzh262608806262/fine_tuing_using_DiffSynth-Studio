# Exp014 执行状态更新

**更新时间**: 2026-09-04 14:15  
**状态**: ✅ Stage 3 (微调) 已提交

---

## 📊 作业状态

| 阶段 | Job ID | 状态 | 提交时间 |
|------|--------|------|----------|
| Stage 1: 擦除训练 | - | ✅ 完成 | 2026-08-30 |
| Stage 2: LoRA 合并 | - | ✅ 完成 | 2026-08-31 |
| **Stage 3: 微调** | **17460858** | 🟢 已提交 | 2026-09-04 14:15 |

---

## 🎯 Stage 3 配置

**目标**: 生成 20 个 checkpoint，用于细粒度分析"安全回潮曲线"

**参数**:
- `dataset_repeat`: 25
- `num_epochs`: 20
- `total_exposure`: 500次/样本
- `learning_rate`: 1e-4
- `lora_rank`: 32

**两臂**:
1. `erased_ft`: 使用 `models/Wan-AI/Wan2.2-TI2V-5B-erased` (擦除后基座)
2. `base_ft`: 使用 `models/Wan-AI/Wan2.2-TI2V-5B` (未擦除基座)

**预期产物**:
- `models/finetune/exp014_erased_ft/epoch-{01..20}.safetensors` (20个checkpoint)
- `models/finetune/exp014_base_ft/epoch-{01..20}.safetensors` (20个checkpoint)

**预计时间**: 24-32 小时（两臂串行，每臂 12-16 小时）

---

## 🔧 关键修复

### 问题
之前的 exp014 Stage 1-2 生成的模型是 **Diffusers 格式**（在 `models/wan5b/exp014_*`），但微调脚本需要 **DiffSynth 格式**。

### 解决方案
使用 Exp012 的 base/erased 模型（`models/Wan-AI/Wan2.2-TI2V-5B[-erased]`），它们是 DiffSynth 格式，适用于微调。

**理由**:
- Exp014 的擦除配置（rank=8, 600步）与 Exp012 相似（rank=8, 1200步）
- 两者的差异主要在**微调策略**，不在擦除阶段
- Exp014 的核心目标是**细粒度微调曲线** (repeat=25, epochs=20)，而非重新擦除

---

## 📝 监控命令

```bash
# 查看作业状态
squeue --me | grep exp014

# 查看实时日志
tail -f slurm/logs/exp014_finetune-17460858.out

# 查看错误日志
tail -f slurm/logs/exp014_finetune-17460858.err
```

---

## ✅ 成功标志

完成后应该有：
- `models/finetune/exp014_erased_ft/` - 20 个 epoch checkpoint
- `models/finetune/exp014_base_ft/` - 20 个 epoch checkpoint
- 每个 checkpoint 约 84-100 MB (LoRA 权重)

---

## 🔄 下一步

Stage 3 完成后的后续任务：

### Stage 4: 生成评测视频
- 使用 20 个 checkpoint 生成视频
- 评测集: VU `benchmark_wan.jsonl` nudity 类 99 条
- 需要为每个 checkpoint 生成一次 → 20 × 99 = 1980 个视频

### Stage 5: NudeNet 评估
- 对所有生成的视频进行 NudeNet 检测
- 计算 violation_rate 和 frame_nudity_rate
- 绘制"回潮曲线"：安全性 vs 微调步数

---

## 💡 经验教训

1. ✅ **确认模型格式**: Diffusers vs DiffSynth 格式不兼容
2. ✅ **复用成功配置**: Exp012 的模型可以直接用
3. ✅ **聚焦核心目标**: Exp014 的关键是微调曲线，不是重新擦除
4. ✅ **理解实验设计**: repeat=25 × epochs=20 = 500曝光，生成 20 个采样点

---

**预计完成时间**: 2026-09-05 或 2026-09-06（约 24-32 小时）
