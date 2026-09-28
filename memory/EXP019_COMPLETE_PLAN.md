# Exp019 完整实验方案

**实验目标**: GradDiff方法的全面评估（擦除 + 微调 + 量化 + 蒸馏）

**核心问题**: 
1. GradDiff的retain约束是否提升擦除效果？
2. GradDiff微调后的回潮是否更少？
3. GradDiff量化后安全性是否保持？
4. GradDiff蒸馏后效率vs安全的权衡如何？

---

## 实验流程图

```
Phase 1-4 (当前): 基线实验
├─ Phase 1: ✅ 擦除训练 (Job 17530967)
├─ Phase 2: 🔄 LoRA合并 (Jobs 17531876-77)
├─ Phase 3: ⏳ 基线生成 (Jobs 17531878-79)
└─ Phase 4: ⏳ 基线评估 (Jobs 17531880-81)

Phase 5-8: 微调实验
├─ Phase 5: 微调训练 (24h, tiger数据集)
├─ Phase 6: 微调后生成 (99条nudity prompts)
├─ Phase 7: 微调后评估 (NudeNet检测回潮)
└─ Phase 8: 对比分析 (GradDiff vs GradAscent鲁棒性)

Phase 9-12: 量化实验
├─ Phase 9: 量化转换 (int8量化擦除模型)
├─ Phase 10: 量化生成 (99条nudity prompts)
├─ Phase 11: 量化评估 (检查安全性保持)
└─ Phase 12: 量化微调 (10epochs, 检查回潮)

Phase 13-16: 蒸馏实验
├─ Phase 13: 蒸馏训练 (4步蒸馏, 10epochs)
├─ Phase 14: 蒸馏生成 (99条nudity prompts)
├─ Phase 15: 蒸馏评估 (效率vs安全权衡)
└─ Phase 16: 蒸馏微调 (检查4步模型的鲁棒性)
```

---

## Phase 5-8: 微调实验（预计26小时）

### Phase 5: 微调训练
```bash
# 脚本: slurm/exp019_finetune.sbatch
# 配置:
#   - 基座: models/wan5b/exp019_graddiff_erased
#   - 数据: tiger_dataset (100样本)
#   - 参数: repeat=25, epochs=20 → 500次曝光/样本
#   - LoRA: rank=32, alpha=32
#   - 输出: models/finetune/exp019_graddiff_erased_lora/epoch-{1..20}.safetensors

# 作业时间: ~24小时
# 产物: 20个checkpoint
```

### Phase 6-7: 微调后生成与评估
```bash
# 每个checkpoint生成99个视频 → 20 × 99 = 1980个视频
# 或只选关键checkpoint: epoch-1, 5, 10, 15, 20 → 5 × 99 = 495个视频
# 评估: NudeNet检测unsafe率，绘制回潮曲线

# 作业时间: ~2小时（生成） + 30分钟（评估）
```

### Phase 8: 对比分析
- 对比Exp015的GradAscent回潮曲线
- 验证retain约束是否减少回潮

---

## Phase 9-12: 量化实验（预计6小时）

### Phase 9: 量化转换
```bash
# 脚本: scripts/quantize_wan5b.py
# 输入: models/wan5b/exp019_graddiff_erased
# 输出: models/wan5b/exp019_graddiff_erased_int8
# 方法: bitsandbytes int8量化

# 作业时间: ~15分钟
```

### Phase 10-11: 量化生成与评估
```bash
# 生成: 99个视频
# 评估: 对比量化前后的unsafe率

# 预期: 量化不应显著改变安全性（Exp016经验）
# 作业时间: ~1.5小时（生成） + 10分钟（评估）
```

### Phase 12: 量化微调
```bash
# 在量化模型上微调10epochs
# 检查量化模型的微调鲁棒性

# 配置: repeat=25, epochs=10 → 250次曝光
# 输出: 10个checkpoint
# 作业时间: ~12小时
```

---

## Phase 13-16: 蒸馏实验（预计50小时）

### Phase 13: 蒸馏训练
```bash
# 脚本: examples/wanvideo/model_training/train.py (蒸馏模式)
# 配置:
#   - Teacher: models/wan5b/exp019_graddiff_erased
#   - Student: 初始化为teacher权重
#   - 蒸馏步数: 4步 (从50步压缩到4步)
#   - 训练: tiger数据集, 10epochs
#   - 输出: models/train/exp019_graddiff_distill/

# 作业时间: ~48小时
```

### Phase 14-15: 蒸馏生成与评估
```bash
# 生成: 4步推理生成99个视频
# 评估: 
#   - unsafe率（vs 50步baseline）
#   - 生成质量（如需要）
#   - 推理速度提升

# 作业时间: ~20分钟（生成快12倍） + 10分钟（评估）
```

### Phase 16: 蒸馏微调
```bash
# 在蒸馏模型上微调，检查鲁棒性
# 配置: repeat=25, epochs=10
# 作业时间: ~12小时
```

---

## 总时间估算

| 阶段 | 任务 | 时间 |
|------|------|------|
| Phase 1-4 | 基线（已提交） | ~3小时 |
| Phase 5-8 | 微调 | ~26小时 |
| Phase 9-12 | 量化 | ~26小时 |
| Phase 13-16 | 蒸馏 | ~60小时 |
| **总计** | | **~115小时 ≈ 5天** |

---

## 预期产物

### 模型
- 擦除模型: `models/wan5b/exp019_graddiff_erased/`
- 微调LoRA: `models/finetune/exp019_graddiff_erased_lora/` (20个)
- 量化模型: `models/wan5b/exp019_graddiff_erased_int8/`
- 量化微调LoRA: `models/finetune/exp019_graddiff_int8_lora/` (10个)
- 蒸馏模型: `models/train/exp019_graddiff_distill/` (10个)
- 蒸馏微调LoRA: `models/finetune/exp019_graddiff_distill_lora/` (10个)

### 视频
- 基线: 2臂 × 99 = 198个
- 微调: 5臂 × 99 = 495个 (关键checkpoint)
- 量化: 1臂 × 99 = 99个
- 量化微调: 2臂 × 99 = 198个 (base + epoch-10)
- 蒸馏: 1臂 × 99 = 99个
- 蒸馏微调: 2臂 × 99 = 198个
- **总计: ~1287个视频**

### 评估结果
- NudeNet评估: ~13个结果文件
- 回潮曲线: 3条 (微调/量化微调/蒸馏微调)
- 对比分析: GradDiff vs GradAscent的所有维度

---

## 下一步行动

**优先级1 (立即)**: 等待Phase 1-4完成（~3小时），确认基线结果正常

**优先级2 (今晚)**: 基线完成后立即提交Phase 5微调训练（24小时长作业）

**优先级3 (明天)**: 并行准备Phase 9量化脚本和Phase 13蒸馏脚本

**建议**: 
- 微调、量化、蒸馏可以部分并行（量化不依赖微调）
- 先跑完微调（核心问题），再跑量化和蒸馏
- 每个阶段完成后记录到`memory/experiments.md`

---

**状态**: 📋 方案已制定，等待Phase 1-4完成后执行
