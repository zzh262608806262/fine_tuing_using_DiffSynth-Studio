# Exp019 当前状态

**更新时间**: 2026-09-16 05:50

---

## ✅ 已完成

### Phase 1-4: 基线实验
- ✅ Phase 1: 擦除训练 (Job 17530967, 12分钟)
- ✅ Phase 2: LoRA合并 (Jobs 17531876-77, 1分钟)
- ✅ Phase 3: 视频生成 (Jobs 17531878-79, 22分钟)
  - Base: 99个视频
  - Erased: 99个视频
- 🔄 Phase 4: NudeNet评估 (Jobs 17533670-71, 进行中)
  - 修复了VU评估脚本参数问题（--method + --output-dir）
  - 预计5分钟完成

**产物**:
- 擦除模型: `models/wan5b/exp019_graddiff_erased/`
- 基线视频: `outputs/exp019/graddiff_{base,erased}/` (198个)
- 评估结果: 等待完成

---

## 📋 已准备（待提交）

### Phase 5-7: 微调实验
**脚本已创建**:
- `slurm/exp019_finetune.sbatch` - 微调训练（24h）
- `slurm/exp019_generate_finetuned.sbatch` - 微调后生成
- `scripts/exp019_generate_finetuned.py` - 生成脚本
- `slurm/exp019_eval_finetuned.sbatch` - 微调后评估
- `scripts/exp019_submit_finetune.sh` - 一键提交脚本

**配置**:
- repeat=25, epochs=20 → 500次曝光/样本
- LoRA rank=32
- 生成5个关键checkpoint: epoch-1, 5, 10, 15, 20
- 每个epoch生成99个视频 → 5 × 99 = 495个

**提交时机**: Phase 4评估完成后，确认基线结果正常

**提交命令**:
```bash
bash scripts/exp019_submit_finetune.sh
```

---

## 🔧 待开发

### Phase 9-12: 量化实验
**需要创建**:
1. `scripts/exp019_quantize.py` - 量化转换脚本
2. `slurm/exp019_generate_quantized.sbatch` - 量化生成
3. `slurm/exp019_finetune_quantized.sbatch` - 量化微调

**参考**: Exp016量化流程（如果存在）

### Phase 13-16: 蒸馏实验
**需要创建**:
1. `slurm/exp019_distill_train.sbatch` - 蒸馏训练
2. `slurm/exp019_generate_distilled.sbatch` - 蒸馏生成
3. `slurm/exp019_finetune_distilled.sbatch` - 蒸馏微调

**参考**: Exp017蒸馏流程

---

## 📊 预期时间线

| 时间点 | 任务 | 状态 |
|--------|------|------|
| 今天 17:00 | Phase 4评估完成 | 🔄 |
| 今天 17:10 | 提交Phase 5微调 | ⏳ |
| 明天 17:00 | Phase 5微调完成 | ⏳ |
| 明天 20:00 | Phase 6-7完成（5个checkpoint） | ⏳ |
| 后天 | 开发量化脚本 | ⏳ |
| 第3天 | 开发蒸馏脚本 | ⏳ |
| 第5天 | 所有实验完成 | ⏳ |

---

## 🎯 下一步行动

**立即** (Phase 4完成后):
1. 检查基线评估结果
2. 确认erased的unsafe率是否低于base
3. 如果结果正常，提交Phase 5微调

**明天**:
1. 监控微调训练进度
2. 准备量化脚本
3. 准备蒸馏脚本

**持续**:
- 记录所有结果到`memory/experiments.md`
- 更新本状态文件

---

**核心问题**: GradDiff的retain约束是否比GradAscent更鲁棒？
