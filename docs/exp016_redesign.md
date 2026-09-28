# Exp016 重新设计

## 问题分析

**原设计的问题**：
- ❌ 量化了Exp015的8个"擦除+微调LoRA"模型
- ❌ 混淆了两个研究问题：
  - Exp015研究：微调对安全对齐的影响
  - Exp016应研究：量化对安全对齐的影响

## 正确的实验设计

### 研究目标
评估**量化（NF4）对安全对齐（擦除效果）的影响**

### 研究问题（RQ）
1. 量化是否会削弱擦除效果？
2. 不同擦除方法在量化后的鲁棒性如何？
3. 量化对原始模型的生成质量影响如何？

### 实验设计

**10个臂**（5个模型 × 2个精度）：

| 模型 | FP16/BF16 | NF4 | 说明 |
|------|-----------|-----|------|
| baseline | baseline_fp16 | baseline_nf4 | 原始Wan2.2-TI2V-5B |
| esd_erased | esd_erased_fp16 | esd_erased_nf4 | ESD纯擦除 |
| npo_erased | npo_erased_fp16 | npo_erased_nf4 | NPO纯擦除 |
| grad_ascent_erased | grad_ascent_erased_fp16 | grad_ascent_erased_nf4 | GradAscent纯擦除 |
| anchor_distill_erased | anchor_distill_erased_fp16 | anchor_distill_erased_nf4 | AnchorDistill纯擦除 |

**注意**：所有模型都是**纯擦除**，不加任何LoRA

### 对比维度

#### 1. 量化对原始模型的影响
```
baseline_fp16 vs baseline_nf4
→ 量化是否降低生成质量？
→ 量化对原始模型的不安全内容生成的影响
```

#### 2. 量化对擦除效果的影响
```
对每个擦除方法：
  (baseline_fp16 - erased_fp16) vs (baseline_nf4 - erased_nf4)
  
擦除保持率 = (baseline_nf4 - erased_nf4) / (baseline_fp16 - erased_fp16)
→ 擦除保持率 = 1.0：量化完全保持擦除效果
→ 擦除保持率 < 1.0：量化削弱了擦除效果
→ 擦除保持率 > 1.0：量化增强了擦除效果（罕见）
```

#### 3. 不同方法在量化后的对比
```
横向对比（NF4空间）：
  esd_erased_nf4 vs npo_erased_nf4 vs grad_ascent_erased_nf4 vs anchor_distill_erased_nf4
→ 哪种擦除方法在量化后效果最好？
```

### 实验流程

#### Stage 0: 生成FP16基线（如果还没有）
```bash
# 检查是否已有FP16评估
ls outputs/exp016_baseline/baseline/*.mp4
ls outputs/exp016_baseline/esd_erased/*.mp4
...

# 如果没有，生成5个模型的视频（不加LoRA）
sbatch slurm/exp016_gen_fp16_baseline.sbatch
sbatch slurm/exp016_gen_fp16_esd_erased.sbatch
sbatch slurm/exp016_gen_fp16_npo_erased.sbatch
sbatch slurm/exp016_gen_fp16_grad_ascent_erased.sbatch
sbatch slurm/exp016_gen_fp16_anchor_distill_erased.sbatch
```

产物：`outputs/exp016_baseline/<model>/*.mp4`（5×99=495个视频）

#### Stage 1: 量化5个模型
```bash
# 量化原始+4个擦除模型
python scripts/exp016_quantize.py --model baseline
python scripts/exp016_quantize.py --model esd_erased
python scripts/exp016_quantize.py --model npo_erased
python scripts/exp016_quantize.py --model grad_ascent_erased
python scripts/exp016_quantize.py --model anchor_distill_erased
```

产物：`models/quantized/exp016_<model>_nf4/`（5个量化模型）

#### Stage 2: 生成NF4视频
```bash
# 为5个量化模型生成视频
sbatch slurm/exp016_gen_nf4_baseline.sbatch
sbatch slurm/exp016_gen_nf4_esd_erased.sbatch
sbatch slurm/exp016_gen_nf4_npo_erased.sbatch
sbatch slurm/exp016_gen_nf4_grad_ascent_erased.sbatch
sbatch slurm/exp016_gen_nf4_anchor_distill_erased.sbatch
```

产物：`outputs/exp016/<model>_nf4/*.mp4`（5×99=495个视频）

#### Stage 3: 评估FP16+NF4
```bash
# 评估10个臂（5 FP16 + 5 NF4）
python scripts/exp016_evaluate_all.py
```

产物：`outputs/exp016_evaluation/<model>_{fp16|nf4}_evaluation.json`（10个文件）

#### Stage 4: 对比分析
```bash
python scripts/exp016_compare_quantization.py
```

产物：
- `outputs/exp016_evaluation/quantization_impact_report.md`
- `outputs/exp016_evaluation/quantization_comparison_table.csv`

### 脚本清单

#### 需要修改的脚本
- [ ] `scripts/exp016_quantize.py`
  - 移除LoRA逻辑
  - 只量化5个基座模型
  
- [ ] `scripts/exp016_generate_videos.py`
  - 移除LoRA逻辑
  - 支持FP16和NF4两种模式
  
- [ ] `scripts/exp016_evaluate.py`
  - 支持FP16和NF4对比

#### 需要新建的脚本
- [ ] `scripts/exp016_generate_fp16_baseline.py`（生成FP16基线视频）
- [ ] `scripts/exp016_compare_quantization.py`（量化前后对比分析）

#### Slurm脚本
- [ ] `slurm/exp016_gen_fp16_template.sbatch`（FP16生成模板）
- [ ] `slurm/exp016_gen_nf4_template.sbatch`（NF4生成模板）
- [ ] `slurm/exp016_quantize_template.sbatch`（量化模板）

### 预期结果

#### 表格1: 量化对各模型的影响

| 模型 | FP16 violation | NF4 violation | 差异 | 擦除保持率 |
|------|----------------|---------------|------|-----------|
| baseline | 80% | 78% | -2% | - |
| esd_erased | 30% | 35% | +5% | 90% |
| npo_erased | 25% | 32% | +7% | 87% |
| grad_ascent_erased | 40% | 45% | +5% | 90% |
| anchor_distill_erased | 35% | 40% | +5% | 90% |

擦除保持率计算：
```
baseline差异 = 80% - 78% = 2%（下降）
esd擦除效果 FP16 = 80% - 30% = 50%
esd擦除效果 NF4 = 78% - 35% = 43%
esd擦除保持率 = 43% / 50% = 86%
```

#### 表格2: 量化后的方法对比

| 方法 | NF4 violation | 排名 |
|------|---------------|------|
| npo_erased_nf4 | 32% | 1 |
| anchor_distill_erased_nf4 | 40% | 2 |
| grad_ascent_erased_nf4 | 45% | 3 |
| esd_erased_nf4 | 35% | 2 |

### 时间估算

- Stage 0: 如果需要生成FP16基线 → 2小时（5臂并行）
- Stage 1: 量化5个模型 → 1小时（并行）
- Stage 2: 生成NF4视频 → 2小时（5臂并行）
- Stage 3: 评估10个臂 → 30分钟
- Stage 4: 对比分析 → 10分钟

**总计**: ~5.5小时（假设需要从头开始）

### 执行计划

1. 检查是否已有FP16基线视频
2. 如果没有，生成FP16基线（Stage 0）
3. 量化5个模型（Stage 1）
4. 生成NF4视频（Stage 2）
5. 评估10个臂（Stage 3）
6. 生成对比报告（Stage 4）

### 与Exp015的关系

- **Exp015**: 微调对安全对齐的影响（8臂，都包含LoRA微调）
- **Exp016**: 量化对安全对齐的影响（10臂，都不包含LoRA，纯擦除模型）
- **两个实验正交独立**，研究不同维度的问题

### 未来可能的Exp017

如果要研究"量化+微调"的联合影响，可以设计Exp017：
- 量化Exp015的8个臂（擦除+LoRA微调）
- 评估量化对"擦除后再微调"场景的影响
