# Exp020 - Masked 方法实验设计

**创建日期**: 2026-09-16  
**实验目标**: 验证空间约束（mask）能否解决擦除方法的"齐步降解"问题

---

## 🎯 核心研究问题

### 背景问题（Exp015 发现）
- GradAscent/GradDiff/NPO 擦除后违规率反而上升（+5~+6pp）
- 学长在 VU 项目文档中提到：**无约束的全局 LoRA 最便宜的降低 forget loss 方式 = 让整个画面退化（dither）**
- 空间约束的目标：把 forget loss 限制在 concept 出现的区域，避免全局退化

### 本实验要回答的问题
1. **Attention Mask 是否有效？** Masked 版本 vs Non-masked 版本擦除效果对比
2. **哪个方法受益最大？** GradAscent/GradDiff/NPO 的 Masked 版本效果排序
3. **Mask 是否影响微调鲁棒性？** Masked 版本的回潮幅度是否更小
4. **SAM Mask 是否更好？** （可选）SAM 分割 vs Attention mask 质量对比

---

## 📐 实验设计

### 阶段 1: Attention Mask 验证（优先级最高）

**实验臂**（3 个方法 × 2 种训练配置）:

| 实验臂 | 方法 | Mask 类型 | 对比基线 |
|--------|------|-----------|----------|
| **graddiff_masked** | GradDiff | Attention | Exp019 graddiff (无mask) |
| **gradascent_masked** | GradAscent | Attention | Exp015 grad_ascent (无mask) |
| **npo_masked** | NPO | Attention | Exp015 npo (无mask) |

**为什么选这 3 个**：
- ✅ VU 项目已实现（`grad_diff_masked.py`, `grad_ascent_masked.py`, `npo_masked.py`）
- ✅ 直接可用，零开发成本
- ✅ 覆盖有/无 retain 约束两种情况

**不包括 ESD**：
- ❌ VU 项目没有 ESD 的 Masked 版本
- ❌ ESD 在 Exp015 中已经有擦除效果（-5.1pp），不是齐步降解的受害者

---

### 阶段 2: SAM Mask 对比（可选，低优先级）

**前置条件**: 阶段 1 证明 Attention Mask 有效

**实验设计**:
- 选择阶段 1 中效果最好的 1 个方法
- 对比 Attention Mask vs SAM Mask（中心两点提示）
- 评估掩码质量差异对擦除效果的影响

---

## 🔧 技术配置

### Mask 生成配置

**Attention Mask 参数**（参考 VU 配置文件）:
```yaml
# configs/methods/training/grad_diff_masked_cogvideox.yaml
mask_config:
  type: attention          # 注意力掩码
  threshold: 0.5           # 二值化阈值
  dilation: 1              # 膨胀核大小（1=不膨胀）
  concept: "nudity"        # 要擦除的概念
```

**SAM Mask 参数**（如果做阶段 2）:
```yaml
mask_config:
  type: sam                # SAM 分割掩码
  model_id: facebook/sam-vit-base
  prompt_strategy: center_two_points  # 中心两点提示
  # 点位置：[w//2, h//2] 和 [w//2, h*0.62]
```

### 擦除训练配置（与 Exp015/019 一致）

```yaml
# 保持与基线实验一致，只改 mask
model: Wan-AI/Wan2.2-TI2V-5B
lora_rank: 8
learning_rate: 1e-5
steps: 600
batch_size: 1

# GradDiff 特有
retain_weight: 1.0

# NPO 特有
beta: 0.1
```

---

## 📊 评估指标

### 主要指标
1. **擦除效果**: Base → Erased 违规率变化
2. **视觉质量**: 是否出现全局退化（需人工检查样本）
3. **微调鲁棒性**: Erased → Fine-tuned 回潮幅度

### 对比维度

**维度 1: Masked vs Non-masked（核心）**
```
GradDiff:      Exp019 (无mask) vs Exp020 graddiff_masked
GradAscent:    Exp015 (无mask) vs Exp020 gradascent_masked  
NPO:           Exp015 (无mask) vs Exp020 npo_masked
```

**维度 2: 方法横向对比**
```
在 Masked 约束下，哪个方法擦除效果最好？
GradDiff-masked vs GradAscent-masked vs NPO-masked
```

**维度 3: Mask 质量（可选）**
```
Attention Mask vs SAM Mask（如果做阶段 2）
```

---

## 📦 数据资产规划

### 生成数据量

**阶段 1: Attention Mask**（推荐）
```
3 方法 × (99 base + 99 erased + 99 fine-tuned) = 891 视频
```

**阶段 2: SAM Mask**（可选）
```
1 方法 × (99 base + 99 erased + 99 fine-tuned) = 297 视频
```

### 存储位置
```
models/unlearn/exp020_masked_{graddiff,gradascent,npo}/
models/wan5b/exp020_masked_{graddiff,gradascent,npo}_{base,erased}/
models/finetune/exp020_masked_{graddiff,gradascent,npo}_ft/
outputs/exp020/{masked_graddiff,masked_gradascent,masked_npo}_{base,erased,ft_e19}/
outputs/exp020_evaluation/*.json
```

---

## ⏱️ 时间估算

### 阶段 1: Attention Mask（推荐优先做）

| 步骤 | 时间 | 说明 |
|------|------|------|
| **准备验证** | 30min | 检查 VU masked 方法支持，验证配置文件 |
| **擦除训练** | 30min | 3 方法并行（各 ~10min） |
| **LoRA 合并** | 30min | 6 个模型（3方法 × 2臂） |
| **基线生成** | 3h | 6 臂 × 99 视频（可并行） |
| **基线评估** | 1.5h | NudeNet 评估 |
| **微调训练** | 72h | 3 方法并行（各 ~24h） |
| **微调生成** | 1.5h | 3 臂 × 99 视频 |
| **微调评估** | 45min | NudeNet 评估 |
| **分析报告** | 2h | 对比分析、可视化 |
| **总计** | **~80h** | 主要是 3 个微调作业（可并行） |

**Wall-clock 时间**: ~3.5 天（微调并行）

### 阶段 2: SAM Mask（可选）

| 步骤 | 时间 | 说明 |
|------|------|------|
| **SAM 掩码生成** | 3h | 18 videos × 17 frames SAM 推理 |
| **训练+评估** | 26h | 与单个方法的阶段 1 相同 |
| **总计** | **~29h** | |

---

## 🚀 执行计划

### 推荐策略：分两期执行

#### **第一期：最小验证（Exp020a）**
- **只做 GradDiff Masked**（1 个方法）
- **目标**: 快速验证 mask 是否有效
- **成本**: 297 视频，~26h wall-clock
- **决策点**: 如果有效 → 继续其他方法；如果无效 → 放弃 Masked 路线

#### **第二期：完整对比（Exp020b）**
- **前置条件**: Exp020a 证明 mask 有效
- **内容**: 补充 GradAscent-masked 和 NPO-masked
- **成本**: 594 视频，~48h wall-clock

#### **第三期：SAM 对比（Exp020c）**
- **前置条件**: Exp020b 完成，且想进一步提升
- **内容**: 选最优方法做 SAM Mask 版本
- **成本**: 297 视频，~29h wall-clock

---

## 📋 检查清单

### Exp020a 启动前验证
- [ ] VU 项目 `grad_diff_masked.py` 存在且可用
- [ ] Wan5B 模型支持 masked 训练（检查是否需要适配）
- [ ] Retain manifest 可用（GradDiff 需要）
- [ ] `run_unlearn_wan5b.py` 支持 `--method GradDiffMasked`

### Exp020a 执行流程
1. [ ] 擦除训练（masked）
2. [ ] LoRA 合并（base + erased）
3. [ ] 基线生成（2 × 99 视频）
4. [ ] 基线评估
5. [ ] **决策点**: 对比 Exp019（无mask）vs Exp020a（masked）
6. [ ] 如果擦除效果改善 → 继续微调；否则 → 停止
7. [ ] 微调训练
8. [ ] 微调后生成+评估
9. [ ] 完整分析报告

---

## 🎓 预期发现

### 假设 1: Mask 有效（期望结果）
- **擦除效果**: Masked 版本违规率显著下降（vs 无 mask 版本）
- **视觉质量**: 不再出现全局退化/抖动
- **解释**: 空间约束防止 LoRA 通过退化整个画面来降低 loss

### 假设 2: Mask 无效（备选结果）
- **擦除效果**: Masked vs Non-masked 无显著差异
- **可能原因**:
  - Attention Mask 质量太差（需尝试 SAM）
  - Wan5B 模型的擦除机制与 CogVideoX 不同
  - 需要调整 mask threshold/dilation 参数

### 假设 3: 方法排序
- **期望**: GradDiff-masked > NPO-masked > GradAscent-masked
- **原因**: Retain 约束 + 空间约束的双重保护

---

## 🔗 相关文档

- VU Masked 实现: `/home/x_jiage/jiage/video-unlearning/src/unlearning/training/grad_diff_masked.py`
- VU Mask 生成: `/home/x_jiage/jiage/video-unlearning/src/unlearning/training/concept_mask.py`
- VU 设计文档: `/home/x_jiage/jiage/video-unlearning/docs/attention_mask_notes.md`
- 基线实验: `memory/experiments.md` (Exp015, Exp019)
- VU 方法分析: `memory/VU_METHODS_COMPLETE_ANALYSIS.md`

---

## ⚠️ 风险与应对

### 风险 1: Wan5B 不支持 masked 训练
- **概率**: 中（VU 项目主要用 CogVideoX 开发）
- **应对**: 需要适配 `concept_mask.py` 中的注意力捕获逻辑
- **工作量**: 1-2 天

### 风险 2: Attention Mask 质量差
- **概率**: 中（学长文档中提到过这个担忧）
- **应对**: 
  - 先可视化几个样本的 mask
  - 调整 threshold/dilation 参数
  - 如果实在不行，切换到 SAM Mask

### 风险 3: Masked 方法仍然无效
- **概率**: 低-中
- **应对**: 
  - 分析失败原因（mask 质量？模型特性？）
  - 回到 ESD 方向（唯一有效的基础方法）
  - 考虑 ESDGen（不依赖 mask）

---

## 💡 后续扩展方向

如果 Exp020 证明 Masked 方法有效：

1. **ESD + Mask**: 尝试给 ESD 加空间约束（需自己实现）
2. **Mask 参数优化**: Grid search threshold/dilation
3. **多尺度 Mask**: 不同层用不同粒度的 mask
4. **Adaptive Mask**: 根据训练过程动态调整 mask 区域

---

**总结**: 建议先做 **Exp020a (GradDiff-masked)**，快速验证 mask 是否有效，再决定是否投入更多资源做完整对比。
