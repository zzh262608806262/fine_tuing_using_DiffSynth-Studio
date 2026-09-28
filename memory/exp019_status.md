# Exp019 执行状态报告

**生成时间**: 2026-09-16  
**实验目标**: 补充 GradDiff 方法，验证 retain 约束对微调鲁棒性的影响

---

## 📋 Phase 完成状态

| Phase | 任务 | 状态 | Job ID | 备注 |
|-------|------|------|--------|------|
| **Phase 1** | 准备与验证 | ✅ 完成 | - | 所有依赖已就绪 |
| **Phase 2** | 擦除训练 | 🔄 运行中 | 17530936 | v1失败(参数错误)，v2已提交 |
| **Phase 3** | LoRA 合并 | ⏳ 待执行 | - | 脚本已准备 |
| **Phase 4** | 基线生成 | ⏳ 待执行 | - | 脚本已准备 |
| **Phase 5** | 基线评估 | ⏳ 待执行 | - | 脚本已准备 |
| **Phase 6** | 微调训练 | ⏳ 待执行 | - | 脚本已准备 |
| **Phase 7** | 微调后生成 | ⏳ 待执行 | - | 脚本已准备 |
| **Phase 8** | 微调后评估 | ⏳ 待执行 | - | 待创建脚本 |
| **Phase 9** | 结果整理 | ⏳ 待执行 | - | - |
| **Phase 10** | 验证检查 | ⏳ 待执行 | - | - |

---

## 🔧 已创建的脚本文件

1. ✅ `slurm/exp019_unlearn_graddiff.sbatch` - 擦除训练（已修复）
2. ✅ `slurm/exp019_merge_base.sbatch` - 合并 base 模型
3. ✅ `slurm/exp019_merge_erased.sbatch` - 合并 erased 模型
4. ✅ `slurm/exp019_generate_baseline.sbatch` - 基线视频生成
5. ✅ `slurm/exp019_eval_baseline.sbatch` - 基线评估
6. ✅ `slurm/exp019_finetune.sbatch` - 微调训练
7. ✅ `slurm/exp019_generate_ft.sbatch` - 微调后生成

---

## 🐛 已修复的问题

### 问题 1: 参数格式错误
- **错误**: 使用了 `--model_id` 而非 `--model-path`
- **修复**: 更新为正确的参数名（使用连字符）

### 问题 2: 环境配置错误
- **错误**: `module load Mambaforge/24.1.2-0` 和 `conda activate vu`
- **修复**: 
  - 模块名: `Mambaforge/23.3.1-1-hpc1-bdist`
  - 环境名: `video-unlearning`

### 问题 3: 缺少 --train 标志
- **错误**: 未指定 `--train` 标志
- **修复**: 添加 `--train` 参数

---

## 📊 预期时间线

- **Day 1 (今天)**: Phase 1-5 (准备 → 基线评估) - 约 2 小时
- **Day 2-3**: Phase 6 (微调训练) - 约 24 小时
- **Day 4**: Phase 7-10 (微调评估 → 整理) - 约 2 小时

**总计**: 约 26-28 小时

---

## 🎯 下一步行动

1. ⏳ **等待 Job 17530936 完成** (~10 分钟)
   - 监控命令: `squeue -j 17530936`
   - 日志位置: `outputs/exp019_graddiff_unlearn_17530936.log`

2. **验证擦除输出**
   - 检查 7 个 checkpoints: `models/unlearn/exp019_wan5b_nudity_graddiff/step-{100..600}`
   - 每个约 153M

3. **提交 LoRA 合并作业**
   - Base: `sbatch slurm/exp019_merge_base.sbatch`
   - Erased: `sbatch slurm/exp019_merge_erased.sbatch`

4. **继续后续 phases**

---

## 🔗 参考文档

- 完整规划: `memory/EXP019_TODOLIST.md`
- 实验记录: `memory/experiments.md` (Exp 019)
- Exp015 参考: `memory/experiments.md` Line 432

---

**备注**: 
- 本实验基于 Exp015/018 的成功经验
- GradDiff 是唯一带 retain 约束的方法
- 与 GradAscent 对比将揭示 retain 约束的实际效果
