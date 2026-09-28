# Exp015 进度表格 - VU 擦除方法对齐能力检验

**实验目标**: 对比 4 种 VU 擦除方法（ESD, NPO, GradAscent, AnchorDistill）在擦除后微调场景下的安全回潮情况

**实验设计**: 每种方法 2 臂（erased, base）× 微调前后 = 共 8 个臂进行对比

---

## 总体进度

| 阶段 | ESD | NPO | GradAscent | AnchorDistill | 状态 |
|------|:---:|:---:|:----------:|:-------------:|:----:|
| **Stage 1**: 擦除训练 | ✅ | ✅ | ✅ | ✅ | 完成 |
| **Stage 2**: LoRA 合并 | ✅ | ✅ | ✅ | ✅ | 完成 |
| **Stage 3**: 良性微调 | ✅ | ✅ | ✅ | ✅ | 完成 |
| **Stage 4**: 生成视频 | 🔄 | 🔄 | 🔄 | 🔄 | 运行中 |
| **Stage 5**: 评估 | ⏳ | ⏳ | ⏳ | ⏳ | 待执行 |
| **Stage 6**: 分析对比 | ⏳ | ⏳ | ⏳ | ⏳ | 待执行 |

**进度**: 6 阶段中已完成 3 阶段，当前第 4 阶段运行中

---

## Stage 1: 擦除训练（Unlearning）✅

| 方法 | Job ID | 耗时 | 产物路径 | Checkpoints |
|------|--------|------|----------|-------------|
| **ESD** | 17428668 | 11min | `models/wan5b_unlearn/exp015_esd/` | 7 个 ckpt |
| **NPO** | 17428669 | 9min | `models/wan5b_unlearn/exp015_npo/` | 7 个 ckpt |
| **GradAscent** | 17428685 | 7min | `models/wan5b_unlearn/exp015_grad_ascent/` | 7 个 ckpt |
| **AnchorDistill** | 17428686 | 8min | `models/wan5b_unlearn/exp015_anchor_distill/` | 7 个 ckpt |

**配置**:
- 基座: Wan2.2-TI2V-5B
- 数据: nudity 类 100 条
- 参数: rank=32, lr=1e-4, steps=600
- 输出: 擦除 LoRA (每 100 步存档)

**失败尝试**: 3 次（GradDiff 方法不支持、AnchorDistill 概念词错误、环境配置错误）

---

## Stage 2: LoRA 合并 ✅

| 方法 | Job ID | 完成时间 | 产物路径 | 文件数 |
|------|--------|----------|----------|--------|
| **ESD** | 17429172 | 09:11 | `models/wan5b/exp015_esd_erased/` | 3 件套 |
| **NPO** | 17429173 | 09:11 | `models/wan5b/exp015_npo_erased/` | 3 件套 |
| **GradAscent** | 17429174 | 09:11 | `models/wan5b/exp015_grad_ascent_erased/` | 3 件套 |
| **AnchorDistill** | 17429175 | 09:12 | `models/wan5b/exp015_anchor_distill_erased/` | 3 件套 |

**产物**: 每个方法的擦除后基座模型
- DiT: `diffusion_pytorch_model-*.safetensors` (3 个分片)
- T5: `models_t5_umt5-xxl-enc-bf16.safetensors` (软链)
- VAE: `Wan2.2_VAE.safetensors` (软链)

**失败尝试**: 3 次（环境错误、参数错误、脚本头部未修改）

---

## Stage 3: 良性微调 ✅

| 方法 | Job ID | 完成时间 | 产物路径 | 大小 |
|------|--------|----------|----------|------|
| **ESD** | 17429770 | 12:46 | `models/finetune/exp015_esd_erased_lora10e2/` | 153M |
| **NPO** | 17429771 | 12:47 | `models/finetune/exp015_npo_erased_lora10e2/` | 153M |
| **GradAscent** | 17429772 | 12:47 | `models/finetune/exp015_grad_ascent_erased_lora10e2/` | 153M |
| **AnchorDistill** | 17429773 | 12:59 | `models/finetune/exp015_anchor_distill_erased_lora10e2/` | 153M |

**配置**:
- 数据: tiger_dataset 100 样本（良性）
- 参数: repeat=10, epochs=2, rank=32, lr=1e-4
- 总曝光: 20 次/样本
- 耗时: ~30-50 分钟/方法

**产物**: 每个方法的 `epoch-0.safetensors` 和 `epoch-1.safetensors`

**历经错误**: 11 个错误，6 次提交（v1-v6），最终在 v6 成功

---

## Stage 4: 生成评测视频 🔄 (当前运行中)

| 臂 | Job ID | 节点 | 状态 | 进度 | 产物路径 |
|----|--------|------|------|------|----------|
| **esd_erased** | 17449669 | node066 | 🔄 RUNNING | 0/99 | `outputs/exp015/esd_erased/` |
| **esd_base** | 17449670 | node066 | 🔄 RUNNING | 0/99 | `outputs/exp015/esd_base/` |
| **npo_erased** | 17449671 | node066 | 🔄 RUNNING | 0/99 | `outputs/exp015/npo_erased/` |
| **npo_base** | 17449672 | node068 | 🔄 RUNNING | 0/99 | `outputs/exp015/npo_base/` |
| **grad_ascent_erased** | 17449673 | node069 | 🔄 RUNNING | 0/99 | `outputs/exp015/grad_ascent_erased/` |
| **grad_ascent_base** | 17449674 | node070 | 🔄 RUNNING | 0/99 | `outputs/exp015/grad_ascent_base/` |
| **anchor_distill_erased** | 17449675 | node070 | 🔄 RUNNING | 0/99 | `outputs/exp015/anchor_distill_erased/` |
| **anchor_distill_base** | 17449676 | node073 | 🔄 RUNNING | 0/99 | `outputs/exp015/anchor_distill_base/` |

**配置**:
- 评测集: VU benchmark_wan.jsonl nudity domain 99 条
- 视频参数: 17 帧, 480×720, steps=50, cfg=5.0
- 总视频数: 8 臂 × 99 条 = **792 个视频**
- 预计完成: ~1.5-2 小时

**失败尝试**: v1 (Jobs 17449135-17449142) DiffSynth API 导入错误，已在 v2 修复

**账户更新**: 已从 Berzelius-2026-50 更新为 Berzelius-2026-243

---

## Stage 5: 评估 ⏳ (待 Stage 4 完成)

**评估指标**:
- **NudeNet**: violation_rate（视频级）、frame_nudity_rate（帧级）
- **分类器**: porn 检出率（threshold=0.2/0.3/0.5）
- **对比维度**: erased vs base, 微调前 vs 微调后

**预期分析**:
- 安全回潮情况：erased_ft 相比 erased 的安全退化程度
- 方法对比：4 种擦除方法的抗回潮能力排序
- 基线对比：与 base/base_ft 的对比

---

## Stage 6: 交叉对比分析 ⏳ (待 Stage 5 完成)

**分析内容**:
- 4 种方法的擦除效果对比
- 良性微调后的安全回潮程度对比
- 最优方法推荐

---

## 错误统计与修复

**总计**: 12 个错误，分布于 3 个阶段

### Stage 1 错误（3 个）
1. GradDiff 方法不支持
2. AnchorDistill 概念词验证失败
3. 环境配置错误（torch/transformers 版本冲突）

### Stage 2 错误（无新增，复用 Stage 1 的环境修复）

### Stage 3 错误（11 个，v1-v6 迭代）
4. 数据集路径缺少 `data/` 前缀
5. 环境选择错误（VU .venv vs diffsynth conda）
6. 模板变量未替换
7. 数据集路径错误（重复）
8. train.py 参数名错误（`--dataset_path` vs `--dataset_base_path`）
9. accelerate 参数传递错误
10. model_id_with_origin_paths 格式错误（分隔符 `;` vs `,`）
11. ... (共 11 个)

### Stage 4 错误（1 个）
12. DiffSynth API 导入错误（ModelManager vs WanVideoPipeline.from_pretrained）

---

## 时间线

| 时间 | 事件 | 耗时 |
|------|------|------|
| 08:41 | Stage 1 开始（擦除训练） | - |
| 08:47 | Stage 1 完成 | ~6 分钟 |
| 09:00 | Stage 2 完成（LoRA 合并） | ~13 分钟 |
| 09:00-12:59 | Stage 3（微调，历经 6 次提交） | ~4 小时 |
| 12:59 | Stage 3 完成 | - |
| ~14:00 | Stage 4 开始（视频生成 v2） | - |
| **预计 ~16:00** | **Stage 4 完成** | **~2 小时** |

**当前总耗时**: ~5-6 小时（含错误修复）
**预计总耗时**: ~7-8 小时（至 Stage 4 完成）

---

## 输出文件位置

**项目目录**: `/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/`

### 模型产物
- 擦除 LoRA: `models/wan5b_unlearn/exp015_{method}/`
- 擦除后基座: `models/wan5b/exp015_{method}_erased/`
- 微调 LoRA: `models/finetune/exp015_{method}_erased_lora10e2/`

### 生成视频
- 输出目录: `outputs/exp015/{arm}/`
- 文件命名: `{idx:03d}.mp4` (000-098)
- 总计: 8 臂 × 99 条 = 792 个视频

### 日志文件
- 作业日志: `slurm/logs/exp015_*.{out,err}`
- 错误记录: `memory/exp015_errors.md`
- 执行计划: `memory/exp015_execution_plan.md`

### 脚本文件
- 生成脚本: `scripts/exp015_generate_videos.py`
- SLURM 作业: `slurm/exp015_generate.sbatch`
- 批量提交: `slurm/exp015_submit_generate_all.sh`

---

## 下一步行动

1. ⏳ 等待 Stage 4 完成（预计 ~16:00）
2. ⏳ 验收 792 个视频产物
3. ⏳ 启动 Stage 5 评估
4. ⏳ 完成 Stage 6 对比分析
5. ⏳ 撰写实验报告

---

**备注**: 
- Exp015 不包含量化和蒸馏，专注于擦除方法的对齐能力检验
- 量化和蒸馏是其他实验（如 Exp003-004）的内容
- 本实验对比 4 种 VU 擦除方法在良性微调后的安全回潮情况
