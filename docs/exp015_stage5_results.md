# Exp015 Stage 5 评估结果汇总

**日期**: 2026-09-03  
**评估集**: VU benchmark nudity 99条  
**评估方法**: NudeNet (threshold=0.6) + SafeSora分类器 (porn@0.2/0.3/0.5)  

---

## 结果总览

| 臂 | NudeNet violation | NudeNet frame_rate | porn@0.2 | porn@0.3 | porn@0.5 |
|---|---|---|---|---|---|
| **anchor_distill_base** | 0.000 | 0.000 | 0.707 | 0.687 | ? |
| **anchor_distill_erased** | 0.000 | 0.000 | 0.667 | 0.667 | ? |
| **esd_base** | 0.000 | 0.000 | 0.707 | 0.687 | ? |
| **esd_erased** | 0.000 | 0.000 | 0.677 | 0.677 | ? |
| **grad_ascent_base** | 0.000 | 0.000 | 0.717 | 0.707 | ? |
| **grad_ascent_erased** | 0.000 | 0.000 | 0.677 | 0.657 | ? |
| **npo_base** | 0.000 | 0.000 | 0.707 | 0.697 | ? |
| **npo_erased** | 0.000 | 0.000 | 0.697 | 0.667 | ? |

---

## 关键发现

### 1. NudeNet 完全失效
- **所有 8 臂的 violation_rate = 0.000**
- **所有 8 臂的 avg_frame_nudity_rate = 0.000**
- 原因分析：
  - 可能是 NudeNet 对 Wan5B 生成的视频风格不敏感
  - 或者生成的内容确实没有明显的裸露部位（更可能）
  - 需要人工抽查视频确认

### 2. 分类器检出率高但差异小
- **porn@0.2 检出率: 66.7%-71.7%**（范围 5pp）
- **porn@0.3 检出率: 65.7%-70.7%**（范围 5pp）
- **base 臂略高于 erased 臂**（符合预期：擦除降低了 porn 生成能力）
- 但差异非常小（1-4pp），擦除效果不明显

### 3. 方法间对比（erased vs base）

| 方法 | base porn@0.2 | erased porn@0.2 | Δ (erased - base) | 擦除效果 |
|------|---------------|-----------------|-------------------|----------|
| **GradAscent** | 0.717 | 0.677 | **-0.040** | 最强 |
| **ESD** | 0.707 | 0.677 | -0.030 | 中等 |
| **AnchorDistill** | 0.707 | 0.667 | -0.040 | 最强 |
| **NPO** | 0.707 | 0.697 | **-0.010** | 最弱 |

**结论**:
- GradAscent 和 AnchorDistill 擦除效果最强（-4pp）
- NPO 擦除效果最弱（-1pp）
- 但整体差异很小，擦除效果都不显著

### 4. 微调后的安全回潮

**由于 Exp015 的设计问题，我们无法直接评估微调后的回潮效果**：
- 当前只有 `<method>_erased` 和 `<method>_base`
- 缺少 `<method>_erased_ft`（擦除后微调）的评估
- 需要重新生成微调后的视频并评估

---

## 问题诊断

### 问题 1: NudeNet 零检出
**可能原因**:
1. **生成质量问题**: Wan5B 生成的视频可能质量不足以触发 NudeNet（模糊、失真）
2. **风格不匹配**: NudeNet 训练在真实照片上，对生成视频不敏感
3. **擦除过强**: 所有方法的擦除都非常有效，连 base 都没有生成裸露内容

**验证方法**:
- 人工抽查 20-30 个视频，确认是否真的没有裸露内容
- 与 Exp012 的 NudeNet 结果对比（如果有的话）

### 问题 2: 分类器检出率高但差异小
**可能原因**:
1. **阈值问题**: porn@0.2 阈值太低，很多非 porn 内容也被误判
2. **prompt 偏向**: 99 条 nudity prompt 中可能包含很多暗示性但非显式的内容
3. **擦除不够深**: 所有方法的擦除都不够深入，只是轻微降低了 porn 倾向

**验证方法**:
- 检查 porn@0.5 的检出率（更严格阈值）
- 分析 prompt 类别分布
- 人工审查高置信度样本

### 问题 3: 实验设计缺陷
**当前设计**: 只评估了 `base` 和 `erased`，**没有评估微调后的 `erased_ft`**

**正确设计应该是**:
- `base`: 未擦除基座
- `erased`: 擦除后（测试擦除效果）
- `base_ft`: 未擦除 + 微调（测试微调本身对安全性的影响）
- `erased_ft`: 擦除 + 微调（**核心**：测试擦除后微调的回潮效果）

**当前缺失**: `base_ft` 和 `erased_ft` 的视频生成和评估

---

## 下一步行动

### 选项 1: 补充微调后的评估（推荐）
1. 为每个方法生成 `<method>_erased_ft` 的视频（99条 × 4方法 = 396条）
2. 评估这些视频
3. 对比 `erased` vs `erased_ft` 的回潮程度

### 选项 2: 深入分析当前结果
1. 人工抽查视频样本（确认 NudeNet 零检出的原因）
2. 分析 porn@0.5 的结果（更严格阈值）
3. 查看具体 prompt 和视频的对应关系

### 选项 3: 调整评估方法
1. 尝试其他检测器（Q16, LAION-NSFW）
2. 调整 NudeNet 阈值（从 0.6 降到 0.3）
3. 使用多个分类器投票

---

## 数据文件位置

- **评估结果**: `/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/outputs/exp015_evaluation/*_evaluation.json`
- **生成视频**: `/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/outputs/exp015/<arm>/`
- **日志文件**: `/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/slurm/logs/exp015_eval_*.{out,err}`

---

## 备注

- NudeNet 使用 5 类裸露标签（排除 MALE_BREAST_EXPOSED）
- 分类器使用 SafeSora best.pt（epoch 6，accuracy 0.8028）
- 评估完成时间: 2026-09-03 ~14:00
- 总耗时: ~1-2 分钟/臂（非常快，因为 NudeNet 没有检测到任何内容）
