# VU 擦除方法对齐能力检验项目总体进度（Exp012-015）

**项目目标**: 系统性检验 VU（video-unlearning）擦除方法在良性微调场景下的对齐能力稳定性

**核心问题**: 擦除后的模型经过良性微调，被擦除的不安全概念是否会回潮？不同擦除方法的鲁棒性如何？

---

## 项目架构

```
Exp012: 基线实验（GradAscent, 1200步）
   ↓
Exp013: 量化评估方法验证
   ↓
Exp014: VU标准配置实验（GradAscent, 600步，细粒度checkpoint）
   ↓
Exp015: 多方法对比（ESD/NPO/GradAscent/AnchorDistill）
```

---

## 总体进度一览

| 实验 | 主要内容 | 擦除方法 | 状态 | 完成度 |
|------|---------|---------|------|--------|
| **Exp012** | 基线实验（1200步擦除+良性微调） | GradAscent | 🔄 运行中 | 90% |
| **Exp013** | 量化模型评估方法验证 | - | ⏳ Pending | 0% |
| **Exp014** | VU标准配置（600步，细粒度checkpoint） | GradAscent | ⏳ Pending | 0% |
| **Exp015** | 4种方法对比 | ESD/NPO/GradAscent/AnchorDistill | 🔄 运行中 | 60% |

**整体进度**: 4 个实验中，2 个运行中，2 个待启动

---

## Exp012: 基线实验（GradAscent 1200步）🔄

**目标**: 建立擦除后微调的基线，验证"安全回潮"现象是否存在

### 实验设计

**4 臂对比**:
- `base`: 未擦除基座（基线安全率）
- `erased`: 擦除后基座（擦除效果）
- `base_ft`: 未擦除 + 微调（解耦"微调本身降安全"）
- `erased_ft`: 擦除 + 微调（**主臂**，回潮检验）

### 配置

| 项目 | 配置 |
|------|------|
| 擦除方法 | GradAscent (`loss=-l_f`) |
| 擦除参数 | rank=8, alpha=16, lr=1e-5, **steps=1200** |
| 擦除数据 | nudity 18条 forget + 9条 retain |
| 微调数据 | tiger_dataset 100样本（良性） |
| 微调参数 | repeat=10, epochs=2 (20次曝光/样本) |
| 评测集 | VU benchmark nudity 99条 |

### 执行状态

| 阶段 | Job ID | 状态 | 说明 |
|------|--------|------|------|
| J0: 下载基座 | 17370719 | ✅ | Wan2.2-TI2V-5B-Diffusers |
| J1: 桥转换+验证 | 17377126 | ✅ | 逐键比对全 PASS |
| J1c: 基线重生成 | 17377127 | ✅ | 27条视频 |
| J2: 擦除训练 | 17377128 | 🔄 | GradAscent 1200步 |
| J3: LoRA合并 | 17377129 | ⏳ | 依赖 J2 |
| J4: 双臂微调 | 17377130 | ⏳ | base_ft + erased_ft |
| J5-8: 生成视频 | 17377131-17377134 | ⏳ | 4臂 × 99条 |
| J9: 评估 | 17377135 | ⏳ | NudeNet + 分类器 |

**历经错误**: 14次提交（J0-J1 前13次），主要错误：
- 分辨率问题（720→736，满足32倍数约束）
- 视频保存编码器问题（pyav codec）
- pandas 依赖和 C++ 库版本冲突

**当前进度**: J1c 完成，J2 运行中，预计完成时间待定

### 预期产物

- 擦除 LoRA: `models/unlearn/wan5b_nudity_grad_ascent/`
- 擦除后基座: `models/Wan-AI/Wan2.2-TI2V-5B-erased/`
- 微调 LoRA: `models/train/Wan2.2-TI2V-5B_{base,erased}_lora10e2/`
- 生成视频: `outputs/wan5b_eval/{base,erased,base_ft,erased_ft}/`
- 评估结果: `outputs/wan5b_eval/classify_results/`

---

## Exp013: 量化评估方法验证 ⏳

**目标**: 验证量化模型（NF4）的评估一致性

### 实验设计

- 使用 Exp012 的基座模型进行 NF4 量化
- 对比量化前后的 NudeNet 评估结果
- 验证量化模型评估的可靠性

### 配置

| 项目 | 配置 |
|------|------|
| 量化方法 | NF4 (4-bit) |
| 测试模型 | Exp012 的 4 个臂 |
| 评估方法 | NudeNet |

**状态**: Pending（等待 Exp012 完成）

---

## Exp014: VU标准配置实验 ⏳

**目标**: 使用 VU 作者的标准配置（600步），生成细粒度 checkpoint 分析回潮曲线

### 实验设计

**4 臂对比**（同 Exp012）:
- `base` / `erased` / `base_ft` / `erased_ft`

### 配置

| 项目 | 配置 | vs Exp012 |
|------|------|-----------|
| 擦除方法 | GradAscent | 相同 |
| 擦除步数 | **600步** | ⬇️ 1200→600 |
| 擦除 checkpoint | 6个（每100步） | 相同 |
| 微调参数 | **repeat=25, epochs=20** | ⬆️ 10×2→25×20 |
| 微调 checkpoint | **20个**（每epoch） | ⬆️ 2→20 |
| 总曝光 | 500次/样本 | ⬆️ 20→500 |

### 关键差异

1. **擦除步数更少**（600 vs 1200）—— 测试 VU 标准配置的擦除效果
2. **微调 checkpoint 更密集**（20个 vs 2个）—— 绘制细粒度回潮曲线
3. **包含量化流程**（42个模型：base×1 + erased×1 + base_ft×20 + erased_ft×20）

### 预期产物

- 擦除 LoRA: `models/unlearn/exp014_wan5b_nudity_600steps/step_{100..600}/`
- 微调 LoRA: `models/finetune/exp014_{base,erased}_ft/epoch_{01..20}/`
- 量化模型: `models/quantized/exp014_*_nf4/`（42个）
- 回潮曲线: 安全率随微调步数的变化

**状态**: Pending（规划完成，待 Exp012 完成后启动）

---

## Exp015: 4种擦除方法对比实验 🔄

**目标**: 对比 4 种 VU 擦除方法的抗回潮能力

### 实验设计

**8 臂对比**（4方法 × 2版本）:
- ESD: `esd_erased`, `esd_base`
- NPO: `npo_erased`, `npo_base`
- GradAscent: `grad_ascent_erased`, `grad_ascent_base`
- AnchorDistill: `anchor_distill_erased`, `anchor_distill_base`

**设计理念**: 复用 Exp012 的 base/base_ft 作为基线，只需生成 4 种擦除方法的 erased 和 erased_ft

### 配置

| 项目 | 配置 |
|------|------|
| 擦除方法 | **ESD / NPO / GradAscent / AnchorDistill** |
| 擦除参数 | rank=32, lr=1e-4, **steps=600** |
| 擦除数据 | nudity 100条 |
| 微调数据 | tiger_dataset 100样本 |
| 微调参数 | repeat=10, epochs=2 (20次曝光) |
| 评测集 | VU benchmark nudity 99条 |

### 执行进度

| 阶段 | ESD | NPO | GradAscent | AnchorDistill | 状态 |
|------|:---:|:---:|:----------:|:-------------:|:----:|
| **Stage 1**: 擦除训练 | ✅ | ✅ | ✅ | ✅ | 完成 |
| **Stage 2**: LoRA合并 | ✅ | ✅ | ✅ | ✅ | 完成 |
| **Stage 3**: 良性微调 | ✅ | ✅ | ✅ | ✅ | 完成 |
| **Stage 4**: 生成视频 | 🔄 | 🔄 | 🔄 | 🔄 | 运行中 |
| **Stage 5**: 评估 | ⏳ | ⏳ | ⏳ | ⏳ | 待执行 |
| **Stage 6**: 对比分析 | ⏳ | ⏳ | ⏳ | ⏳ | 待执行 |

### Stage 1-3 详情（已完成）✅

| Stage | 耗时 | 产物 | Job IDs |
|-------|------|------|---------|
| Stage 1: 擦除训练 | ~6-11分钟/方法 | 4个擦除 LoRA | 17428668,17428669,17428685,17428686 |
| Stage 2: LoRA合并 | ~13分钟 | 4个擦除后基座 | 17429172-17429175 |
| Stage 3: 良性微调 | ~30-50分钟/方法 | 4个微调 LoRA (153M) | 17429770-17429773 |

**历经错误**: 12个错误，主要包括：
- 数据集路径错误
- 环境配置错误（VU .venv vs diffsynth conda）
- AnchorDistill 概念词验证失败
- train.py 参数名错误
- accelerate 参数传递错误
- model_id_with_origin_paths 格式错误
- DiffSynth API 导入错误

### Stage 4: 生成视频（运行中）🔄

**配置**: 8臂 × 99条 = **792个视频**

| 臂 | Job ID | 节点 | 状态 | 产物路径 |
|----|--------|------|------|----------|
| esd_erased | 17449669 | node066 | 🔄 | `outputs/exp015/esd_erased/` |
| esd_base | 17449670 | node066 | 🔄 | `outputs/exp015/esd_base/` |
| npo_erased | 17449671 | node066 | 🔄 | `outputs/exp015/npo_erased/` |
| npo_base | 17449672 | node068 | 🔄 | `outputs/exp015/npo_base/` |
| grad_ascent_erased | 17449673 | node069 | 🔄 | `outputs/exp015/grad_ascent_erased/` |
| grad_ascent_base | 17449674 | node070 | 🔄 | `outputs/exp015/grad_ascent_base/` |
| anchor_distill_erased | 17449675 | node070 | 🔄 | `outputs/exp015/anchor_distill_erased/` |
| anchor_distill_base | 17449676 | node073 | 🔄 | `outputs/exp015/anchor_distill_base/` |

**视频参数**: 17帧, 480×720, steps=50, cfg=5.0

**预计完成**: ~1.5-2小时（约16:00）

**账户更新**: 已从 Berzelius-2026-50 更新为 Berzelius-2026-243

### Stage 5-6（待执行）⏳

**Stage 5: 评估**
- NudeNet: violation_rate, frame_nudity_rate
- 分类器: porn 检出率（threshold=0.2/0.3/0.5）

**Stage 6: 对比分析**
- 4种方法的擦除效果对比
- 良性微调后的安全回潮程度对比
- 最优方法推荐

### 预期发现

| 方法 | 预期特点 |
|------|---------|
| **GradAscent** | 基线方法（loss=-l_f），无 retain 机制 |
| **ESD** | 擦除更深，但微调后可能回潮更明显 |
| **NPO** | 擦除-保留平衡最优 |
| **AnchorDistill** | 擦除最温和，微调鲁棒性最强 |

---

## 项目时间线

| 日期 | 事件 | 实验 |
|------|------|------|
| 2026-08-25 | 规划与脚本开发 | Exp012 |
| 2026-08-25 13:00 | 首次提交（J0-J9） | Exp012 |
| 2026-08-26 | 修复14个错误，重复提交 | Exp012 |
| 2026-08-26 13:35 | 第14次提交，J2运行中 | Exp012 |
| 2026-08-30 | 规划 | Exp014 |
| 2026-08-31 08:41 | 开始执行 | Exp015 |
| 2026-08-31 12:59 | Stage 1-3 完成 | Exp015 |
| 2026-08-31 ~14:00 | Stage 4 开始 | Exp015 |
| **2026-08-31 ~16:00** | **Stage 4 预计完成** | **Exp015** |

---

## 错误统计与教训

### Exp012 错误（14次提交）

**主要问题**:
1. 分辨率约束（720→736，32倍数）
2. 视频编码器（pyav codec）
3. Python 依赖（pandas, libstdc++）
4. 环境变量冲突（PYTHONNOUSERSITE）

**教训**: 首次使用新环境时，先进行小规模烟雾测试

### Exp015 错误（12个）

**阶段分布**:
- Stage 1: 3个（路径、环境、概念词）
- Stage 3: 11个（数据集、参数、API）
- Stage 4: 1个（API 导入）

**教训**: 参考成功案例的完整命令，严格验证参数格式

---

## 输出文件位置

### Exp012

**项目目录**: `/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/`

- 基座模型: `models/Wan-AI/Wan2.2-TI2V-5B{,-erased}/`
- 擦除 LoRA: `models/unlearn/wan5b_nudity_grad_ascent/`
- 微调 LoRA: `models/train/Wan2.2-TI2V-5B_{base,erased}_lora10e2/`
- 生成视频: `outputs/wan5b_eval/{base,erased,base_ft,erased_ft}/`
- 评估结果: `outputs/wan5b_eval/classify_results/`

### Exp015

**项目目录**: `/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/`

- 擦除 LoRA: `models/wan5b_unlearn/exp015_{method}/`
- 擦除后基座: `models/wan5b/exp015_{method}_erased/`
- 微调 LoRA: `models/finetune/exp015_{method}_erased_lora10e2/`
- 生成视频: `outputs/exp015/{arm}/`
- 日志文件: `slurm/logs/exp015_*.{out,err}`

---

## 下一步行动

### Exp012（当前焦点）
1. ⏳ 等待 J2 擦除训练完成
2. ⏳ 验证 J3-J9 依赖链正常执行
3. ⏳ 分析首个基线实验结果

### Exp015（并行进行）
1. 🔄 监控 Stage 4 视频生成（预计 16:00 完成）
2. ⏳ 验收 792 个视频
3. ⏳ 启动 Stage 5 评估
4. ⏳ 完成 Stage 6 对比分析

### Exp013-014
1. ⏳ 等待 Exp012 完成后启动
2. ⏳ Exp013: 量化评估验证
3. ⏳ Exp014: 细粒度回潮曲线分析

---

## 项目意义

**学术价值**:
- 系统性评估 VU 擦除方法在实际应用场景（良性微调）下的鲁棒性
- 量化"安全回潮"现象，为擦除方法改进提供实证依据
- 对比多种擦除方法，找出最优的对齐保持策略

**实用价值**:
- 指导安全模型的微调策略设计
- 为生产环境中的模型更新提供风险评估
- 推动更鲁棒的擦除方法研究

---

**备注**: 
- 本项目专注于擦除方法的对齐能力检验，不包含量化和蒸馏的主体研究
- 量化仅作为评估方法验证（Exp013）或产物压缩（Exp014）
- 所有实验使用相同的评测协议，确保结果可比性
