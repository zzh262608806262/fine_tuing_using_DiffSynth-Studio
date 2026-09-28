# VU 方法可行性详细分析

**创建日期**: 2026-09-16  
**目的**: 详细解释为什么某些方法不适合你的研究

---

## ❓ 澄清：ESDGen vs ESD

### 你的疑问是对的！

**ESD** 和 **ESDGen** 是**两个不同的方法**：

| 方法 | 原理 | 数据需求 | Exp015状态 |
|------|------|----------|-----------|
| **ESD** | 负引导，基于缓存的 forget 视频 | 需要 forget 视频 | ✅ 已测试 |
| **ESDGen** | 负引导，模型自己生成 x_t | 只需 prompt，不需视频 | ❌ 未测试 |

### 关键区别

**ESD（Exp015 已做）**:
```python
# 使用缓存的真实视频
x_t = noise(real_video_latent)  # 从真实视频加噪
pred = model(x_t, concept_prompt)
target = null_pred - η * (concept_pred - null_pred)
loss = ||pred - target||²
```

**ESDGen（未做）**:
```python
# 模型自己生成 x_t，不需要真实视频
x_t = DDIM_sample(noise, concept_prompt, current_model)  # 部分去噪
pred = model(x_t, concept_prompt)
target = null_pred - η * (concept_pred - null_pred)
loss = ||pred - target||²
```

### 为什么提出 ESDGen？

1. **ESD 需要真实 unsafe 视频** - 数据获取困难
2. **ESDGen 只需要 prompt** - 模型自己生成 unsafe 内容来擦除
3. **理论上更强** - 直接在模型生成轨迹上擦除

### 你的实验中

- ✅ **Exp015 用的是 ESD** (η=1.0, 需要 forget 视频)
- ❌ **ESDGen 未测试** (η=3.0 通常, 不需要 forget 视频)

### 是否需要做 ESDGen？

**我的建议**: **不需要**

**理由**:
1. **数据需求不同** - 你已有 forget 视频，ESD 更适合
2. **ESDGen 主要优势是"不需要真实视频"** - 但你已经有了
3. **核心研究问题是"微调鲁棒性"** - 用 ESD 还是 ESDGen 影响不大
4. **参数 η 的影响** - 可以在 ESD 上测试不同的 η（1.0 vs 3.0）

**如果想测试参数敏感性**:
- 用 **ESD + η=3.0**（而非 ESDGen）
- 更简单，直接对比 ESD(η=1.0) vs ESD(η=3.0)

---

## ❌ 为什么 Masked 方法不可行？

### 1. Masked 方法的原理

**普通方法** (GradAscent/ESD/NPO):
```python
loss = mean(error)  # 整个视频的平均误差
```

**Masked 方法** (GradAscentMasked/NPOMasked):
```python
# 只在概念出现的区域计算 loss
mask = detect_concept_region(video)  # 需要空间 mask！
loss = masked_mean(error, mask)
```

### 2. 需要什么样的 Mask？

**Mask 是视频的空间标注**，标记"哪些像素包含要擦除的概念"：

```
视频帧 1:
┌─────────────────┐
│    背景         │
│  ┌────────┐    │  <- Mask = 1（裸体区域）
│  │ 裸体   │    │
│  └────────┘    │
│    背景         │  <- Mask = 0（背景区域）
└─────────────────┘

需要对每一帧的每个像素标注！
```

### 3. 如何生成 Mask？（⚠️ 旧理解，已过时）

**~~方法 A: 人工标注~~** (错误理解)
- ~~需要标注 18 个 forget 视频 × 17 帧 × 每帧标注~~
- ~~工作量：至少 1-2 周~~
- ~~成本：极高~~

**方法 B: VU 项目实际使用的方式**
```python
# 方式 1: Attention Mask（完全自动）
# VU 已实现，零人工标注
from src.unlearning.training.concept_mask import attention_concept_mask
mask = attention_concept_mask(query, key, ...)  # 从注意力权重计算

# 方式 2: SAM Mask（自动分割）
# 用 SAM 模型自动分割，需要提示点
from src.unlearning.training.sam_mask import SamMasker
masker = SamMasker()
mask = masker.mask(frame)  # 中心两点提示
```

### 4. ⚠️ 更新：之前的理解有误（2026-09-16）

**之前认为 Masked 方法不可行，是因为**：
- ❌ 误以为需要人工逐帧标注
- ❌ 没有仔细阅读 VU 的 mask 实现代码

**实际情况**：
- ✅ VU 提供完全自动的 Attention Mask
- ✅ 也支持 SAM 自动分割（需要提示策略）
- ✅ **零人工标注工作量**

**关于 VU 作者的"expected to fail"评论**：
- 这是针对早期版本的评论
- 后续 VU 项目仍然实现并保留了 Masked 方法
- 值得实验验证是否能解决 Exp015 的"齐步降解"问题

### 5. 新的结论：Masked 方法值得尝试

**为什么现在推荐**：
- ✅ 可能解决 Exp015 的"齐步降解"问题
- ✅ Attention Mask 零额外成本
- ✅ 验证空间约束对微调鲁棒性的影响

**详见**: `memory/exp020_masked_methods_design.md`

---

## ❌ 为什么 Encoder 方法不可行？

### 1. Encoder vs DiT 的区别

**你的实验流程**:
```
Text Prompt
    ↓
[Text Encoder] ← 如果在这里擦除
    ↓
Text Embeddings
    ↓
[DiT Transformer] ← 你的微调在这里！
    ↓
Video Latents
```

**问题**: **擦除和微调不在同一层级**

### 2. 为什么不匹配？

**场景 A: Encoder 擦除 + DiT 微调**
```
Encoder: 已擦除 "nudity" 概念
        ↓
DiT:    接收到"被修改的"文本嵌入
        然后被微调

问题: DiT 微调可能恢复的不是"被擦除的概念"，
      而是"学习新的映射关系"
```

这不是"回潮"，而是"DiT 学会了新的方式绕过 Encoder 的擦除"

**场景 B: DiT 擦除 + DiT 微调**（你现在的方案）
```
DiT: 直接擦除生成 unsafe 内容的能力
     然后被微调

问题: 微调可能直接恢复 DiT 生成 unsafe 的能力

这才是真正的"回潮"！
```

### 3. 研究问题不匹配

**你的核心问题**: 微调是否让**已擦除的能力**恢复？

**Encoder 方法测试的是**: 微调 DiT 是否能绕过 Encoder 的限制？

这是**两个不同的研究问题**！

### 4. 对照实验失效

如果做 Encoder 擦除：

```
Base:          Encoder未擦 + DiT未微调
Erased:        Encoder擦除 + DiT未微调  ← 不同层级
Finetuned:     Encoder擦除 + DiT微调    ← 混合效应

无法分离：
- DiT微调导致的回潮
- DiT学习绕过Encoder的新路径
```

**结论**: **不做 Encoder 方法**

---

## ❌ 为什么 FTTP 不推荐？

### 1. FTTP 的原理

**FTTP = Fine-Tuning Then Pruning**

```python
# 不是 LoRA，是全参数训练
loss = exp(-λ * forget_loss) + retain_weight * retain_loss

# 特点
- forget_loss: 只在 mask 标记的概念区域
- retain_loss: 在同一视频的背景区域
- 需要 mask！
```

### 2. 为什么不推荐？

#### 问题 1: 需要 Mask
FTTP 也是 Masked 方法，需要空间 mask（同上所有问题）

#### 问题 2: 全参数训练
```python
# 你现在的方法（LoRA）
训练参数: ~84MB (rank=32 LoRA)
训练时间: ~24小时

# FTTP（全参数）
训练参数: ~10GB (整个 DiT)
训练时间: 可能 10倍（需要更小的学习率）
内存需求: 更高
```

#### 问题 3: 微调流程不同

**你的微调**:
```python
# 在 DiT 上加 LoRA
base_model (冻结) + LoRA_微调 (训练)
```

**FTTP 后微调**:
```python
# FTTP 是全参数训练，没有 LoRA 分离
FTTP_model (全参数已修改)

如何微调？
选项 A: 继续全参数微调 → 成本极高
选项 B: 在 FTTP 模型上加 LoRA → 但 base 已经不是原始的了
```

#### 问题 4: 对照性差

```
GradAscent (LoRA擦除) + LoRA微调
vs
FTTP (全参数擦除) + ???微调

不是同一个实验范式！
```

### 3. 如果一定要对比全参数？

**更好的方案**: 不做 LoRA 擦除，做全参数擦除

```
方案 A (你现在): LoRA擦除 + LoRA微调
方案 B (更好):   全参数擦除 + LoRA微调

但这需要重新训练所有方法（几周的计算）
```

**结论**: **不推荐 FTTP**，除非：
- 你想专门研究"全参数 vs LoRA"
- 有大量额外计算资源
- 能接受 1-2 周的额外工作

---

## ❌ 为什么 T2VUnlearning 不适用？

### 1. T2VUnlearning 是什么？

查看代码：
```python
# t2v_unlearning.py
"""
Adam, matching the configuration this method was validated under.
"""
```

这是**特定论文的方法**，可能：
- 针对特定模型设计
- 有特定的超参数依赖
- 可能不通用于 Wan5B

### 2. 为什么不适用？

**问题 1**: VU 没有提供 Wan5B 配置
```bash
$ ls video-unlearning/configs/methods/training/*t2v*
# 空！没有 t2v_unlearning_wan.yaml
```

**问题 2**: 需要研究实现
- 需要阅读 T2VUnlearning 的论文
- 需要理解其特定假设
- 需要验证是否适用于 Wan5B

**问题 3**: 不确定性高
- 可能与 VU 其他方法原理重复
- 可能需要特定数据格式
- 调试成本未知

**结论**: **暂不考虑**，除非：
- VU 明确说它是通用方法
- 有 Wan5B 配置文件
- 原理与已测试方法显著不同

---

## ✅ 总结：可行性矩阵

| 方法 | 可行性 | 原因 | 建议 |
|------|--------|------|------|
| **GradDiff** | ✅ 高 | 配置齐全，流程已验证 | **必做** |
| ESD (η=3.0) | ⚪ 中 | 需修改配置，但简单 | 可选 |
| ESDGen | ⚪ 低 | 原理不同，你的场景不需要 | 不做 |
| FTTP | ❌ 低 | 需 mask + 全参数，成本高 | 不做 |
| Masked变体 | ❌ 极低 | 需 mask 标注，理论上会失败 | 不做 |
| Encoder变体 | ❌ 极低 | 层级不匹配，研究问题不同 | 不做 |
| T2VUnlearning | ❌ 未知 | 无配置，原理不明 | 暂不考虑 |

---

## 📋 修正后的实验计划

### 必做
1. **Exp019 - GradDiff** (~26小时)
   - 补全 5 个基础方法
   - 验证 retain 约束作用

2. **Exp021 - 81帧验证** (~8小时)
   - 提升可信度
   - 验证 17 帧结论

### 可选
3. **Exp020 - ESD 参数研究** (~26小时)
   - 测试 ESD (η=1.0 vs η=3.0)
   - 注意：不是 ESDGen，是 ESD 的不同参数

### 不做
- ❌ ESDGen - 原理不同，你的场景不需要
- ❌ FTTP - 需 mask + 全参数
- ❌ Masked 变体 - 需标注，理论上失败
- ❌ Encoder 变体 - 层级不匹配
- ❌ T2VUnlearning - 不明确

---

## 💡 关键洞察

### 你的研究核心
**"DiT 微调对 DiT 擦除效果的影响"**

### 适合的方法
- ✅ 在 **DiT** 上做擦除
- ✅ 在 **DiT** 上做微调
- ✅ 使用 **LoRA** 统一范式
- ✅ 不需要空间 mask

### 不适合的方法
- ❌ 在 **Encoder** 上擦除（层级不匹配）
- ❌ 需要 **mask** 的方法（额外标注成本）
- ❌ **全参数**方法（范式不同）
- ❌ 原理不明的**特定方法**

---

## 🎯 最终建议

**核心实验**: Exp019 (GradDiff) + Exp021 (81帧验证)

**完成后你有**:
- 5 种方法系统对比
- 17帧 + 81帧双重验证
- Retain 约束作用分析
- 完整的擦除-微调数据

**足够写一篇高质量论文！**

其他方法要么不适用，要么成本收益比太低。
