# Exp 015 项目状态报告

**生成时间**: 2026-08-31  
**状态**: ✅ 已就绪，可以启动

---

## 执行摘要

Exp 015 多方法擦除对比实验的所有脚本和前置条件已准备完毕。该实验将系统对比 VU 项目的 4 种擦除方法（GradDiff / ESD / NPO / AnchorDistill）在擦除后微调场景下的安全能力保持性，与 Exp 012 的 GradAscent 基线进行对比。

## ✅ 已完成的工作

### 1. 实验设计与文档
- ✅ 在 `memory/experiments.md` 创建了完整的 Exp 015 记录
- ✅ 创建了详细的使用文档 `docs/exp015_README.md`
- ✅ 更新了文件索引 `memory/index.md`

### 2. 核心脚本开发
- ✅ `scripts/run_unlearn_wan5b_multi_method.py` - 多方法擦除驱动脚本
  - 支持 5 种方法: GradAscent / GradDiff / ESD / NPO / AnchorDistill
  - 统一配置: rank=8, alpha=16, steps=600
  - 方法参数化，便于横向对比

### 3. SLURM 作业脚本（12个）
**擦除训练** (4个):
- ✅ `slurm/exp015_unlearn_grad_diff.sbatch`
- ✅ `slurm/exp015_unlearn_esd.sbatch`
- ✅ `slurm/exp015_unlearn_npo.sbatch`
- ✅ `slurm/exp015_unlearn_anchor_distill.sbatch`

**LoRA 合并** (4个):
- ✅ `slurm/exp015_merge_grad_diff.sbatch`
- ✅ `slurm/exp015_merge_esd.sbatch`
- ✅ `slurm/exp015_merge_npo.sbatch`
- ✅ `slurm/exp015_merge_anchor_distill.sbatch`

**微调** (4个):
- ✅ `slurm/exp015_finetune_grad_diff.sbatch`
- ✅ `slurm/exp015_finetune_esd.sbatch`
- ✅ `slurm/exp015_finetune_npo.sbatch`
- ✅ `slurm/exp015_finetune_anchor_distill.sbatch`

### 4. 自动化工具
- ✅ `slurm/exp015_submit.sh` - 全流程提交脚本
- ✅ `scripts/exp015_check_prerequisites.sh` - 前置条件检查脚本

### 5. 前置条件验证
所有 36 项前置条件检查通过：
- ✅ VU 仓库和方法配置 (5/5)
- ✅ Exp 012 产物（复用资源）(4/4)
- ✅ 训练数据 (1/1)
- ✅ 共享资源 (2/2)
- ✅ 脚本和作业文件 (20/20)
- ✅ VU 方法配置文件 (4/4)

---

## 📋 当前可执行的阶段

### Stage 1: 擦除训练 ✅ 就绪
```bash
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
bash slurm/exp015_submit.sh
```

**预期产物**:
- `models/unlearn/exp015_wan5b_nudity_grad_diff/final/`
- `models/unlearn/exp015_wan5b_nudity_esd/final/`
- `models/unlearn/exp015_wan5b_nudity_npo/final/`
- `models/unlearn/exp015_wan5b_nudity_anchor_distill/final/`

**预计时间**: ~2-4 小时/方法（600 steps）

### Stage 2: LoRA 合并 ✅ 就绪
等待 Stage 1 完成后手动提交，或添加依赖链自动提交。

### Stage 3: 微调 ✅ 就绪
等待 Stage 2 完成后手动提交。

---

## ⏳ 待开发的阶段

### Stage 4: 生成评测视频
**复用策略**: 
- 可以复用 Exp 012 的 `scripts/generate_wan5b_eval.py`
- 需要为每个方法的每个臂生成 99 条视频

**预期工作量**: 半天（参数化现有脚本）

### Stage 5: 评估
**复用策略**:
- 复用 Exp 012 的 `scripts/eval_porn_wan5b.py`
- 批量评估所有方法所有臂

**预期工作量**: 1-2 小时（修改批量调用逻辑）

### Stage 6: 对比分析
**新增脚本**: `scripts/analyze_multi_method_comparison.py`
- 横向对比: 5种方法的擦除效果
- 纵向对比: 5种方法的微调敏感度
- 安全保持率计算
- 生成可视化图表

**预期工作量**: 半天

---

## 🔄 与 Exp 012 的关系

### 复用资源
1. **base 桥产物**: `models/Wan-AI/Wan2.2-TI2V-5B/`
2. **Latent 缓存**: `data/wan5b/unlearn_latents/latents/` (27个文件)
3. **基线视频**: `data/wan5b/unlearn_baseline/`
4. **训练数据**: `data/tiger_dataset/metadata_100.csv`
5. **评测结果**: base/base_ft 的生成视频和评估可直接复用

### 复用脚本
1. `scripts/merge_unlearn_lora_wan5b.py` - LoRA 合并
2. `scripts/finetune_wan5b.py` - 微调
3. `scripts/generate_wan5b_eval.py` - 生成（待参数化）
4. `scripts/eval_porn_wan5b.py` - 评估（待参数化）

---

## 📊 预期结果

### 擦除效果排序（预测）
1. **ESD** - 最深擦除（负引导最激进）
2. **GradAscent** - 中等擦除（baseline）
3. **NPO** - 平衡擦除（负偏好优化）
4. **GradDiff** - 温和擦除（有 retain 约束）
5. **AnchorDistill** - 最温和（蒸馏到替代词）

### 微调鲁棒性排序（预测）
1. **GradDiff** - 最鲁棒（retain set 提供锚定）
2. **NPO** - 较鲁棒（偏好优化平衡）
3. **AnchorDistill** - 中等（概念替换）
4. **GradAscent** - 较易回潮
5. **ESD** - 最易回潮（擦除过深）

---

## 🚀 启动指令

### 1. 验证前置条件
```bash
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
bash scripts/exp015_check_prerequisites.sh
```

### 2. 启动 Stage 1（擦除训练）
```bash
bash slurm/exp015_submit.sh
```

### 3. 监控作业
```bash
# 查看作业状态
squeue -u $USER

# 查看作业记录
cat slurm/logs/exp015_pipeline_jobs.tsv

# 查看日志
tail -f slurm/logs/exp015_unlearn_grad_diff-<job_id>.out
```

---

## 📝 注意事项

1. **环境隔离**: 
   - 擦除训练使用 VU 环境（.venv 或 video-unlearning conda env）
   - 微调/生成使用 DiffSynth 环境（diffsynth conda env）

2. **计算资源**:
   - 每个擦除训练作业需要 1 GPU，约 2-4 小时
   - 4 个方法可并行执行
   - 建议在集群空闲时段提交

3. **存储需求**:
   - 每个方法的擦除 LoRA: ~100 MB
   - 每个擦除后基座: ~20 GB
   - 每个微调产物: ~200 MB
   - 总计约: 4 × 20 GB = 80 GB（主要是基座）

4. **后续工作**:
   - Stage 4-6 需要在 Stage 1-3 完成后再开发
   - 可以在 Stage 1 运行期间准备 Stage 4-6 的脚本

---

## 📚 参考文档

- **实验记录**: `memory/experiments.md` → Exp 015
- **使用指南**: `docs/exp015_README.md`
- **文件索引**: `memory/index.md`
- **VU 项目**: `/home/x_jiage/jiage/video-unlearning/`
- **Exp 012 基线**: `memory/experiments.md` → Exp 012

---

## ✅ 总结

Exp 015 的 Stage 1-3（擦除、合并、微调）已完全就绪，所有脚本和前置条件均已准备完毕。可以立即启动实验。Stage 4-6（生成、评估、分析）的脚本可以在 Stage 1 运行期间开发，预计总开发工作量约 1-1.5 天。

**建议行动**:
1. ✅ 立即启动 Stage 1: `bash slurm/exp015_submit.sh`
2. ⏳ 等待 Stage 1 完成后手动提交 Stage 2
3. ⏳ 在 Stage 1 运行期间开发 Stage 4-6 的脚本
