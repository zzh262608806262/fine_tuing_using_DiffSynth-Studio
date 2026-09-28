# Exp019 - VU 方法补充计划

**创建日期**: 2026-09-16  
**目标**: 补充 VU 其他擦除方法，完善"微调对擦除效果影响"的研究

---

## 📊 当前状态

### 已完成的方法 (Exp015-018)
- ✅ **GradAscent** - 擦除失效（+5.1pp），微调后更差（+6.1pp）
- ✅ **ESD** - 擦除最好（-5.1pp），但微调回潮严重（+10.1pp）
- ✅ **NPO** - 擦除失效（+6.1pp），微调后更差（+7.1pp）
- ✅ **AnchorDistill** - 擦除轻微有效（-2.0pp），微调回潮中等（+8.1pp）
- ❌ **GradDiff** - Exp015 尝试但失败，未重跑

### 研究空白
1. **缺少 retain 约束方法的对比** - GradDiff 是唯一有 retain 的基础方法
2. **缺少全参数方法** - 现有全部是 LoRA，FTTP 是全参数
3. **ESD 参数敏感性未测试** - 只测了 η=1.0，未测 η=3.0

---

## 🎯 推荐补充方案

### 📌 方案 A: 最小补充（1个方法，性价比最高）

**补充方法**: GradDiff

**理由**:
1. **唯一缺失的基础方法** - Exp015 尝试但失败
2. **直接对比价值高** - GradDiff = GradAscent + retain，可对比 retain 的作用
3. **计算成本低** - 与其他方法相同的流程，已有 retain manifest

**实验设计**:
```
Exp019A - GradDiff 完整流程

Stage 1: 擦除训练
  - Method: GradDiff
  - 参数: retain_weight=1.0, rank=8, alpha=16.0, steps=600
  - 数据: nudity_forget + nudity_retain

Stage 2: 基线评估
  - base vs erased
  - 对比 GradAscent (无 retain) vs GradDiff (有 retain)

Stage 3: 微调训练
  - erased 模型微调 (repeat=25, epochs=20)
  
Stage 4: 微调后评估
  - 回潮幅度对比
  
Stage 5: 对比分析
  - GradAscent vs GradDiff 的擦除效果差异
  - GradAscent vs GradDiff 的微调鲁棒性差异
```

**预期发现**:
- GradDiff 应该比 GradAscent 擦除效果更好（有 retain 约束）
- GradDiff 应该比 GradAscent 微调鲁棒性更强

**计算成本**: 
- 1 方法 × 2 臂 (base/erased) × 99 视频 = 198 个视频
- 1 方法 × 1 臂 (微调) × 99 视频 = 99 个视频
- **总计**: 297 个视频 + 评估

**预计时间**: 
- 擦除: 10 分钟
- 微调: 24 小时
- 生成: 1 小时
- 评估: 1 小时
- **总计**: ~26 小时

---

### 📌 方案 B: 标准补充（2个方法，平衡性价比）

**补充方法**: GradDiff + ESDGen

**新增**: ESDGen（ESD 的增强版）

**理由**:
- ESD 是 Exp015 唯一有效的方法（-5.1pp）
- ESDGen 使用更强负引导（η=3.0 vs 1.0）
- 测试"更深擦除"是否意味着"更强回潮"

**实验设计**:
```
Exp019B - GradDiff + ESDGen

方法 1: GradDiff (同方案A)

方法 2: ESDGen
  - 参数: negative_guidance=3.0 (vs ESD的1.0)
  - 其他参数同 Exp015
  - 对比维度: ESD (η=1.0) vs ESDGen (η=3.0)
    - 擦除深度
    - 微调鲁棒性
```

**预期发现**:
- ESDGen 擦除应该更深（比 ESD 的 -5.1pp 更多）
- 但微调回潮可能更严重（验证"深度-鲁棒性"权衡）

**计算成本**:
- 2 方法 × 2 臂 × 99 = 396 个基线视频
- 2 方法 × 1 臂 × 99 = 198 个微调视频
- **总计**: 594 个视频

**预计时间**: ~52 小时（2 × 方案A）

---

### 📌 方案 C: 完整补充（3个方法，覆盖所有空白）

**补充方法**: GradDiff + ESDGen + FTTP

**新增**: FTTP（全参数方法）

**理由**:
- FTTP 是全参数训练，不是 LoRA
- 两阶段策略：先微调再剪枝
- 可能对微调更鲁棒（因为训练方式不同）

**挑战**:
- FTTP 计算成本可能更高（全参数）
- 可能需要修改脚本（不确定 VU 的 FTTP 是否支持 Wan5B）

**计算成本**:
- 3 方法 × 2 臂 × 99 = 594 个基线视频
- 3 方法 × 1 臂 × 99 = 297 个微调视频
- **总计**: 891 个视频

**预计时间**: ~78 小时

**风险**: FTTP 可能需要额外调试

---

## 💡 我的建议

### 推荐：方案 A（GradDiff）

**原因**:
1. **性价比最高** - 只需 ~26 小时
2. **研究价值大** - 直接对比 retain 约束的作用
3. **风险低** - 流程已验证，只需重跑 Exp015 失败的部分
4. **论文贡献清晰** - "retain 约束能否提升微调鲁棒性"

### 如果时间充裕，再考虑方案 B

**条件**:
- 论文需要更多方法对比
- 想探索 ESD 的参数敏感性
- 有额外 2-3 天计算时间

### 不推荐方案 C（除非必要）

**原因**:
- FTTP 全参数训练，与现有 LoRA 方法不在同一范式
- 可能需要额外调试
- 对核心研究问题的贡献不大

---

## 🔧 具体实施计划（方案 A）

### Step 1: 修复 GradDiff 支持
```bash
# 检查 Exp015 失败原因
grep "GradDiff" memory/experiments.md

# 确认 run_unlearn_wan5b.py 已支持 GradDiff
# （已有代码，应该可以直接用）
```

### Step 2: 准备数据
```bash
# 确认 retain manifest 存在
ls /home/x_jiage/jiage/video-unlearning/data/manifests/nudity_retain_composed_wan.jsonl

# 确认基线视频缓存可复用（来自 Exp015）
ls data/wan5b/unlearn_baseline/
```

### Step 3: 提交擦除训练
```bash
# 创建 slurm/exp019_unlearn_graddiff.sbatch
# 参数: method=GradDiff, retain_weight=1.0, steps=600
sbatch slurm/exp019_unlearn_graddiff.sbatch
```

### Step 4: 合并 LoRA + 生成 base/erased 视频
```bash
# 复用 Exp015 流程
# Stage 2: 合并
# Stage 4: 生成 2×99 视频
```

### Step 5: 基线评估
```bash
# NudeNet 评估
# 对比 GradAscent vs GradDiff
```

### Step 6: 微调训练
```bash
# 使用 GradDiff erased 模型
# repeat=25, epochs=20
# 生成 20 个 checkpoints
```

### Step 7: 微调后评估
```bash
# 评估 epoch 19
# 计算回潮幅度
# 对比 GradAscent vs GradDiff 的微调鲁棒性
```

---

## 📈 预期论文贡献

补充 GradDiff 后，你可以回答：

1. **Retain 约束是否有效？**
   - GradDiff (有 retain) vs GradAscent (无 retain)
   - 擦除效果对比
   - 微调鲁棒性对比

2. **为什么现有方法失效？**
   - 4/5 方法擦除失效或微调回潮严重
   - 可能是缺少 retain 约束
   - GradDiff 数据验证这一假设

3. **方法选择建议**
   - 如果 GradDiff 表现好 → "需要 retain 约束"
   - 如果 GradDiff 也失效 → "当前擦除方法都不够鲁棒"

---

## ⚠️ 注意事项

### 1. GradDiff 之前为什么失败？
查看 Exp015 记录：
```
Job 17428667 FAILED - 方法不支持
```

可能原因：
- `run_unlearn_wan5b.py` 当时没有 GradDiff 分支
- 需要确认现在是否已修复

**行动**: 先测试 GradDiff 是否可用

### 2. 视频长度问题
当前所有视频 17 帧，可能不够：
- 考虑用 81 帧重新生成 GradDiff（如果 17 帧结论不够可靠）
- 或者同时生成 17 帧和 81 帧对比

### 3. 与现有数据对比
确保 GradDiff 使用相同配置：
- 相同评测集（99 条 nudity prompts）
- 相同生成参数（17帧/480×720/50步）
- 相同评估方法（NudeNet threshold=0.6）

---

## 📊 补充后的完整方法列表

| 方法 | Exp015基线 | Exp016量化 | Exp018微调 | Exp019补充 |
|------|-----------|-----------|-----------|-----------|
| GradAscent | ✅ | ✅ | ✅ | - |
| ESD | ✅ | ✅ | ✅ | - |
| NPO | ✅ | ✅ | ✅ | - |
| AnchorDistill | ✅ | ✅ | ✅ | - |
| **GradDiff** | ❌ | - | - | **🎯 补充** |
| ESDGen | - | - | - | 可选 |
| FTTP | - | - | - | 可选 |

补充 GradDiff 后，基础方法就完整了。

---

## 🚀 下一步行动

### 立即执行（方案 A）
1. [ ] 确认 GradDiff 是否可用（测试运行）
2. [ ] 准备 Exp019 脚本和配置
3. [ ] 提交擦除训练作业
4. [ ] 等待结果，分析对比

### 如果时间允许（方案 B）
5. [ ] 添加 ESDGen（测试 η=3.0）
6. [ ] 对比 ESD vs ESDGen

### 论文写作准备
7. [ ] 整理所有方法对比表
8. [ ] 分析 retain 约束的作用
9. [ ] 讨论方法失效的原因

---

**总结**: 建议优先补充 GradDiff（方案 A），性价比最高，研究价值大。
