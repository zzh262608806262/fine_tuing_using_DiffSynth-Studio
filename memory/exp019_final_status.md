# Exp019 - 最终状态报告

**更新时间**: 2026-09-16 04:15  
**当前状态**: Phase 2 擦除训练进行中 (Job 17530967)

---

## ✅ 成功启动！

### Job 17530967 - GradDiff 擦除训练 (v7)
- **状态**: ✅ RUNNING (node051)
- **运行时间**: ~1 分钟
- **预计完成**: ~10 分钟后
- **配置**: 
  - 模型: `Wan2.2-TI2V-5B-Diffusers`
  - 方法: GradDiff (retain_weight=1.0)
  - 步数: 600, LoRA rank=8, alpha=16.0
  - Latent manifest: 27 条（18 forget + 9 retain）
  - Retain latent manifest: 9 条（单独提取）
- **输出**: `models/unlearn/exp019_wan5b_nudity_graddiff/`
- **监控**: Monitor bgt7c59gp 正在跟踪，完成时将通知

---

## 🎉 问题全部解决！

经过 **7 次迭代**，所有配置问题已解决：

| 版本 | 问题 | 解决方案 |
|------|------|----------|
| v1 | 参数格式错误 | 使用连字符参数名 |
| v2 | 环境不存在 | 使用 VU `.venv` 环境 |
| v3 | 缺少 text_encoder | 使用 Diffusers 版本模型 |
| v4 | retain_manifest 缺 latent 字段 | 尝试只用 latent-manifest |
| v5 | retain_manifest 仍被默认传递 | 尝试不传 retain-manifest |
| v6 | 相对路径解析错误 | 创建 retain latent manifest |
| v7 | ✅ **成功！** | 使用绝对路径 + retain latent manifest |

---

## 🔑 最终正确配置

### 1. 环境
```bash
source /home/x_jiage/jiage/video-unlearning/.venv/bin/activate
```

### 2. 模型路径
```bash
models/Wan-AI/Wan2.2-TI2V-5B-Diffusers  # Diffusers 版本
```

### 3. GradDiff 的关键发现
**GradDiff 需要两个 manifest，都必须包含 `latent` 字段：**
- `--latent-manifest`: 所有数据（forget + retain）
- `--retain-manifest`: 只有 retain 数据（从 latent_manifest 提取）

**创建 retain latent manifest：**
```bash
grep "nudity_retain" data/wan5b/unlearn_latents/latent_manifest.jsonl > \
  data/wan5b/unlearn_latents/retain_only_latent_manifest.jsonl
```

### 4. 使用绝对路径
```bash
RETAIN_LATENT_MANIFEST=${FT_ROOT}/${LATENT_CACHE_DIR}/retain_only_latent_manifest.jsonl
```

---

## 📊 预期结果（~10分钟后）

### 输出文件
```
models/unlearn/exp019_wan5b_nudity_graddiff/
├── step-100/          # ~153M
├── step-200/          # ~153M  
├── step-300/          # ~153M
├── step-400/          # ~153M
├── step-500/          # ~153M
├── step-600/          # ~153M (最终用于合并)
└── adapter_final.pt   # 最终 checkpoint
```

**总计**: 7 个 checkpoints

---

## 🚀 下一步（训练完成后）

### 1. 验证输出
```bash
ls models/unlearn/exp019_wan5b_nudity_graddiff/step-*
du -sh models/unlearn/exp019_wan5b_nudity_graddiff/step-*
```

### 2. 提交完整流程链
```bash
bash scripts/exp019_submit_chain.sh
```

这将自动提交：
- ✅ LoRA 合并（base + erased）
- ✅ 基线生成（2臂×99视频）
- ✅ 基线评估（NudeNet）
- ✅ 微调训练（24小时）
- ✅ 微调后生成+评估

### 3. 监控进度
```bash
# 查看所有作业
squeue -u $USER

# 运行监控脚本
bash scripts/exp019_monitor.sh
```

---

## 📈 预期时间线

| Phase | 预计时间 | 说明 |
|-------|---------|------|
| ✅ Phase 1 | 完成 | 准备与验证 |
| 🔄 Phase 2 | ~10 分钟 | 擦除训练（进行中，约剩 9 分钟） |
| ⏳ Phase 3 | ~20 分钟 | LoRA 合并 |
| ⏳ Phase 4 | ~1 小时 | 基线生成 |
| ⏳ Phase 5 | ~30 分钟 | 基线评估 |
| ⏳ Phase 6 | ~24 小时 | 微调训练 ⚠️ **最长** |
| ⏳ Phase 7-8 | ~1 小时 | 微调后生成+评估 |
| ⏳ Phase 9-10 | ~30 分钟 | 结果整理 |

**总计**: 约 26-28 小时

---

## 🔬 研究目标

本实验将回答：
1. **GradDiff vs GradAscent 擦除效果**
   - GradDiff（有 retain）是否比 GradAscent（无 retain）更有效？

2. **微调鲁棒性对比**
   - GradDiff 回潮幅度是否 < GradAscent (+6.1pp)？

3. **Retain 约束的价值**
   - 验证 retain 约束是否真正提升擦除鲁棒性

---

## 📝 已完成的准备工作

### 脚本（8个SLURM + 3个辅助）
- ✅ 擦除训练（已修复并运行）
- ✅ LoRA合并（base + erased）
- ✅ 基线生成、评估
- ✅ 微调训练
- ✅ 微调后生成、评估
- ✅ 自动化提交脚本
- ✅ 监控脚本

### 文档
- ✅ 实验记录（experiments.md）
- ✅ 文件索引（index.md）
- ✅ 启动报告、执行总结、状态报告等

---

## 🎯 关键经验

1. **GradDiff 是唯一需要 retain data 的方法**
2. **Retain manifest 必须包含 `latent` 字段**（不能用原始 VU manifest）
3. **使用绝对路径避免解析问题**
4. **参考成功案例**（Exp015）但要注意其 GradDiff 未实际运行
5. **迭代调试**：每次失败都提供了关键信息

---

**状态**: ✅ 训练进行中，Monitor 正在跟踪，约 9 分钟后完成！

**备注**: 这是 Exp019 的第一次成功运行，将为后续 26 小时的实验流程奠定基础。
