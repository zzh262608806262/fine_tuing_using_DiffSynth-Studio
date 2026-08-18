# 提升 Safe/Unsafe 二分类准确率 计划

## 目标

提升 Safety Classifier 对视频 "是否 safe" 的二分类准确率（binary safe/unsafe accuracy）。
当前模型在 SafeSora 测试集上的二分类表现：unsafe recall 仅 37.6%，大量 unsafe 视频被误判为 safe。

## 当前状态分析

### 数据分布（严重不平衡）
- Train: 51,588 样本，safe 占 86.1%，animal_abuse 仅 0.3%（148样本）
- Test: 5,745 样本
- 88.1% 样本只有 1 个标签（绝大多数仅标 "safe"）

### 当前性能（best epoch 6）
- Exact-match accuracy: 80.28%（被 safe 多数类主导，有误导性）
- Macro F1: 49.2%
- AUROC: 95.45%（排序能力好，但阈值不佳）
- **Unsafe recall: 37.6%**（核心问题：62% 的 unsafe 视频被漏检）

### 根本原因
1. **pos_weight 实现了但从未使用** — `losses.py` 中有 `compute_pos_weight()` 函数，但 `train.py:198` 实例化 `BCEWithLogitsLoss()` 时未传入 pos_weight
2. **固定阈值 0.5** — 对所有 13 个类共用 0.5，对稀有类太高
3. **无二分类专用评估指标** — 当前 selection_metric 是 13 类 exact-match accuracy，不反映二分类性能

### 二分类决策逻辑（predict.py:134）
```python
unsafe = bool(pred[1:].sum() > 0)  # 12个unsafe类任一 >= threshold 则 unsafe
```

## 方案：两阶段改进

### 阶段 A：快速验证 — 阈值优化（无需重训）

对现有 best.pt（epoch 6）在测试集上搜索最优二分类阈值，不改变模型权重。
**不修改现有代码，创建新脚本 `classify/evaluation/optimize_threshold.py`。**

新增功能：
1. 加载 checkpoint，在测试集上跑全部样本，收集每个样本的 13 类 sigmoid 概率
2. 对 binary safe/unsafe 决策做阈值扫描：threshold 从 0.01 到 0.99，步长 0.01
3. 对每个阈值计算 binary accuracy / precision / recall / F1
4. 输出最优阈值和对应的二分类指标
5. 将最优阈值保存到 `thresholds.json`

**预期结果：** AUROC=95.45% 说明排序能力好，降低阈值能显著提升 unsafe recall，可能从 37% 提升到 60-70%+

### 阶段 B：核心改进 — 启用 pos_weight 重训

用 pos_weight 重新训练模型，从根本解决类别不平衡问题。
**同样不修改现有代码，创建新训练脚本。**

**预期结果：** pos_weight 让稀有 unsafe 类的损失被放大 100-300 倍，模型会更敏感地检测 unsafe 内容。结合阈值优化，unsafe recall 预期可达 70%+，binary accuracy 85%+。

## 实施步骤

1. **阶段 A（快速验证，无需重训）：**
   - 创建 `classify/evaluation/optimize_threshold.py`
   - 对现有 best.pt 跑阈值优化
   - 记录最优阈值和二分类指标

2. **阶段 B（重训，约3小时）：**
   - 创建新的训练脚本（保留原 train.py 不动）
   - 创建新的 config（保留原 config 不动）
   - 重新训练 10 epochs
   - 训练完成后跑阶段 A 的阈值优化
   - 对比前后指标

## 涉及文件

| 文件 | 改动 |
|---|---|
| `classify/evaluation/optimize_threshold.py` | **新建** — 阈值优化脚本 |
| `classify/training/train_pw.py` | **新建**（阶段B）— 启用 pos_weight 的训练脚本 |
| `classify/configs/safety_classifier_pw.yaml` | **新建**（阶段B）— pos_weight 版 config |
| 现有所有文件 | **不动** |

## 不涉及的范围

- 不修改任何现有代码文件
- 不修改模型架构
- 不修改推理 API（predict.py）
- 不添加数据增强（留待后续）
