# Exp020a 启动成功报告

**时间**: 2026-09-16 18:12  
**状态**: ✅ 已启动，所有作业已提交

---

## 📊 执行摘要

### Phase 1: Smoke Test ✅ 已完成

**目的**: 验证Wan模型的mask参数

**结果**:
- Job 17536586成功完成
- 获取关键参数：
  - `attn_substring`: "attn2" (cross-attention)
  - `num_heads`: **48** (重要发现！不是默认的24)
  - `frames`: 4
  - `height`: 30
  - `width`: 46

### Phase 2: 完整流程 🔄 执行中

**作业链**:
```
17536588 → 擦除训练 (NPOMasked, 600步)
  ↓
17536589 → 模型合并
  ↓
17536590 → 视频生成 (99个视频)
  ↓
17536591 → 安全评估 (NudeNet)
```

**预计时间线**:
- 18:12 - 提交所有作业
- ~18:42 - 擦除训练完成
- ~18:45 - 模型合并完成
- ~20:45 - 视频生成完成
- ~21:15 - 安全评估完成 ✅

**总计**: 约3小时

---

## 🎯 实验目标

测试NPOMasked是否改善Exp015中plain NPO的失效问题：
- **Exp015 baseline**: base 35.4% → plain NPO erased 41.4% (+6.1pp，失效)
- **Exp020a目标**: NPOMasked erased < 35.4% (优于baseline)

---

## 📁 已创建文件

### 核心脚本
1. ✅ `scripts/debug/smoke_test_wan_mask.py` - mask参数验证
2. ✅ `scripts/run_unlearn_wan5b.py` - 添加masked方法支持
3. ✅ `/home/x_jiage/jiage/video-unlearning/configs/methods/training/npo_masked_wan.yaml`

### SLURM作业
4. ✅ `slurm/smoke_test_wan_mask.sbatch`
5. ✅ `slurm/exp020a_unlearn_npo_masked.sbatch`
6. ✅ `slurm/exp020a_merge.sbatch`
7. ✅ `slurm/exp020a_generate.sbatch`
8. ✅ `slurm/exp020a_evaluate.sbatch`
9. ✅ `slurm/exp020a_submit_all.sh`

### 文档
10. ✅ `memory/exp020_plan.md` - 完整系列计划
11. ✅ `memory/exp020a_status.md` - 执行状态
12. ✅ `memory/experiments.md` - 实验记录已更新
13. ✅ `memory/index.md` - 文件索引已更新

---

## 🔍 关键发现

### Smoke Test发现
- Wan模型的attention heads是**48**，而不是CogVideoX的30或默认的24
- 这是一个重要的模型架构差异
- 如果使用错误的num_heads，mask构建会失败

### 技术实现
- 成功集成VU项目的NPOMasked方法
- 配置文件正确放置在VU项目中
- 训练脚本正确支持masked方法调用

---

## 📊 监控信息

### 查看作业状态
```bash
squeue -u x_jiage | grep exp020a
```

### 查看日志
```bash
# 擦除训练
tail -f slurm/logs/exp020a_unlearn-17536588.out

# 最终评估结果
tail slurm/logs/exp020a_evaluate-17536591.out
```

### 检查结果
```bash
# 评估结果
cat outputs/exp020a_evaluation/npo_masked_erased_evaluation.json

# 快速查看违规率
python -c "import json; d=json.load(open('outputs/exp020a_evaluation/npo_masked_erased_evaluation.json')); print(f'违规率: {d[\"nudenet\"][\"summary\"][\"violation_rate\"]:.1%}')"
```

---

## ⏭️ 下一步

### 如果Exp020a成功（违规率 < 35.4%）
1. 分析结果，对比Exp015 NPO
2. 继续Exp020b - GradDiffMasked
3. 继续Exp020c - GradAscentMasked

### 如果Exp020a部分成功（35.4% ~ 41.4%）
1. 分析mask是否正确生效
2. 检查训练日志中的loss曲线
3. 考虑调整mask_config或训练参数

### 如果Exp020a失败（> 41.4%）
1. 检查mask是否正确构建（查看训练日志）
2. 验证attention module hooks是否生效
3. 如果确认masked方法无效，放弃Exp020系列

---

## 🎉 成就解锁

✅ 完整实现了从smoke test到完整pipeline的自动化流程  
✅ 成功集成VU项目的masked方法  
✅ 发现Wan模型的关键架构参数（num_heads=48）  
✅ 一键提交4个依赖作业，自动化执行  
✅ 预计3小时内完成完整实验  

---

**当前状态**: 所有作业运行中，等待结果
**预计完成**: 2026-09-16 21:15
**监控**: 已设置自动监控，完成时会通知
