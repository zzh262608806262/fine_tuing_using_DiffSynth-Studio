# Exp015 执行计划

## 目标
对比 4 种 VU 擦除方法（GradAscent/ESD/NPO/AnchorDistill）在擦除后微调场景下的安全回潮情况。

## 实验设计
- **方法**: 4 种（GradAscent, ESD, NPO, AnchorDistill）
- **臂**: 每种方法 2 臂（erased, erased_ft）+ baseline (base, base_ft from Exp012)
- **评测**: 与 Exp012 相同（NudeNet + 分类器，VU benchmark nudity 99 条）

## 执行阶段

### Stage 0: 准备 ✅
- [x] 复用 Exp012 latent cache
- [x] 复用 Exp012 基座模型
- [x] 创建脚本框架

### Stage 1: 擦除训练（600 steps）
| 方法 | Job ID | 状态 | 产物 | 备注 |
|------|--------|------|------|------|
| ESD | 17428668 | ✅ 完成 (11min14s) | `models/unlearn/exp015_wan5b_nudity_esd/` | 7 checkpoints |
| NPO | 17428669 | ✅ 完成 (9min12s) | `models/unlearn/exp015_wan5b_nudity_npo/` | 7 checkpoints |
| GradAscent | 17428685 | ✅ 完成 (7min) | `models/unlearn/exp015_wan5b_nudity_grad_ascent/` | 7 checkpoints |
| AnchorDistill | 17428692 | ✅ 完成 (8min) | `models/unlearn/exp015_wan5b_nudity_anchor_distill/` | 7 checkpoints, 使用空 anchor |

**失败尝试**:
- Job 17428667 (GradDiff) - 方法不支持
- Job 17428670 (AnchorDistill v1, `anchor="a person"`) - 概念词检查失败
- Job 17428681 (GradAscent v1) - 环境配置错误

### Stage 2: LoRA 合并
| 方法 | Job ID | 状态 | 产物 |
|------|--------|------|------|
| ESD | 17428695 | ✅ 完成 | `models/wan5b/exp015_esd_erased/` (5 safetensors) |
| NPO | 17428696 | ✅ 完成 | `models/wan5b/exp015_npo_erased/` (5 safetensors) |
| GradAscent | 17428697 | ✅ 完成 | `models/wan5b/exp015_grad_ascent_erased/` (5 safetensors) |
| AnchorDistill | 17428713 | ✅ 完成 | `models/wan5b/exp015_anchor_distill_erased/` (5 safetensors) |

**失败尝试**:
- Job 17428673/17428674 (ESD/NPO v1) - 环境错误（diffsynth 缺少 diffusers）
- Job 17428686/17428687 (ESD/NPO v2) - 参数错误（--lora-dir 应为 --adapter）
- Job 17428709 (AnchorDistill v1) - 脚本头部未修改，执行了 ESD

### Stage 3: 微调（repeat=10, epochs=2）
| 方法 | Job ID | 状态 | 产物 |
|------|--------|------|------|
| ESD | 17429770 | ✅ 完成 (12:46) | `models/finetune/exp015_esd_erased_lora10e2/` |
| NPO | 17429771 | ✅ 完成 (12:47) | `models/finetune/exp015_npo_erased_lora10e2/` |
| GradAscent | 17429772 | ✅ 完成 (12:47) | `models/finetune/exp015_grad_ascent_erased_lora10e2/` |
| AnchorDistill | 17429773 | ✅ 完成 (12:59) | `models/finetune/exp015_anchor_distill_erased_lora10e2/` |

**产物验证**:
- ✅ 每个方法都有 `epoch-0.safetensors` 和 `epoch-1.safetensors` (153M each)
- ✅ 每个方法都有 `training_args.json`
- ✅ 每个方法都有 `tensorboard_log/` 目录

**失败记录**:
- v1-v4: 数据集路径、环境、参数名、accelerate 参数等错误
- v5 (Jobs 17429720-17429723): ❌ model_id_with_origin_paths 分隔符错误（用 `;` 而非 `,`）
- v6 (Jobs 17429770-17429773): ✅ 成功完成（共修复11个错误）

配置：
- Dataset: tiger_dataset/metadata_100.csv（良性，100 样本）
- LoRA: rank=32, lr=1e-4, targets=q,k,v,o,ffn.0,ffn.2
- 训练: repeat=10 × epochs=2 = 20 曝光/样本
- 预计时间: ~30 分钟/方法

### Stage 4: 生成评测视频（99 条）

**状态**: 🔄 运行中（v2，已修复导入错误）

**作业 ID**（使用 Berzelius-2026-243 账户）:
- esd_erased: 17449669 ✓ RUNNING
- esd_base: 17449670 ✓ RUNNING
- npo_erased: 17449671 ✓ RUNNING
- npo_base: 17449672 ✓ RUNNING
- grad_ascent_erased: 17449673 ✓ RUNNING
- grad_ascent_base: 17449674 ✓ RUNNING
- anchor_distill_erased: 17449675 ✓ RUNNING
- anchor_distill_base: 17449676 ✓ RUNNING

**总计**: 8臂 × 99条 = 792个视频

**配置**: 17f, 480×720, steps=50, cfg=5.0

**产物**: `outputs/exp015/<arm>/<idx:03d>.mp4`

**失败记录**:
- v1 (Jobs 17449135-17449142): ❌ 导入错误（使用了错误的 DiffSynth API）
- v2 (Jobs 17449669-17449676): ✓ 运行中（已修复，模型加载中）

**预计完成时间**: ~1.5-2小时

### Stage 5: 评估（NudeNet + 分类器）
⏳ 待 Stage 4 完成

指标:
- violation_rate（视频级）
- frame_nudity_rate（帧级）
- porn 检出率（分类器 thr=0.2/0.3/0.5）

### Stage 6: 对比分析
⏳ 待 Stage 5 完成

分析维度:
- 横向: 4 种方法的擦除效果（erased vs base）
- 纵向: 4 种方法的微调敏感度（erased_ft - erased）
- 量化: 安全保持率 = (erased_ft - erased) / (base - erased)

## 脚本清单

### 擦除训练
- `slurm/exp015_unlearn_esd.sbatch` ✅
- `slurm/exp015_unlearn_npo.sbatch` ✅
- `slurm/exp015_unlearn_grad_ascent.sbatch` ✅
- `slurm/exp015_unlearn_anchor_distill.sbatch` ✅

### LoRA 合并
- `slurm/exp015_merge_esd.sbatch` ✅
- `slurm/exp015_merge_npo.sbatch` ✅
- `slurm/exp015_merge_grad_ascent.sbatch` ✅
- `slurm/exp015_merge_anchor_distill.sbatch` ⏳ 待创建

### 微调
- `slurm/exp015_finetune_*.sbatch` ⏳ 待创建（4 个）

### 生成与评估
- `slurm/exp015_generate_*.sbatch` ⏳ 待创建（8 个：4 方法 × 2 臂）
- `scripts/exp015_evaluate_all.py` ⏳ 待创建
- `scripts/exp015_analyze_comparison.py` ⏳ 待创建

## 时间估算

- Stage 1: ~10 min/方法 → 40 min total（并行）
- Stage 2: ~5 min/方法 → 20 min total（并行）
- Stage 3: ~30 min/方法 → 2 hours total（并行）
- Stage 4: ~1 hour/臂 → 8 hours total（并行）
- Stage 5: ~30 min total
- Stage 6: ~1 hour（分析）

**总计**: ~12 hours（假设充分并行）

## 下一步行动

1. ✅ Stage 1 完成（4 方法擦除训练）
2. ✅ Stage 2 完成（4 方法 LoRA 合并）
3. ✅ Stage 3 完成（4 方法微调，历经11个错误修复）
4. 🔄 Stage 4 运行中（8臂视频生成，Jobs 17434688-17434695）
   - 已提交：8个作业，等待调度
   - 预计完成：14:30（约1.5小时）
   - 产物：792个视频（8臂 × 99条）
5. ⏳ Stage 5-6: 待 Stage 4 完成后启动
2. ⏳ 等待 AnchorDistill (Job 17428692) 完成（预计 10 分钟）
3. ✅ 验证 Stage 1 产物（checkpoints 完整性）
4. 🔄 提交 GradAscent/AnchorDistill 的 Stage 2 合并
5. ⏳ 创建 Stage 3 微调脚本
6. ⏳ 提交 Stage 3 作业

## 验证检查点

每个 Stage 完成后验证：
- [ ] Stage 1: 7 个 .pt 文件（step100-600 + training_trace.jsonl）
- [ ] Stage 2: DiffSynth 格式模型（.safetensors + T5/VAE 软链接）
- [ ] Stage 3: LoRA checkpoints（epoch-0, epoch-1）
- [ ] Stage 4: 99 个视频文件
- [ ] Stage 5: JSON 结果文件
- [ ] Stage 6: 对比报告 markdown
