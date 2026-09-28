# Exp019 当前状态报告

**更新时间**: 2026-09-16 03:45  
**当前状态**: Phase 2 擦除训练进行中

---

## 🔄 正在运行

### Job 17530960 - GradDiff 擦除训练
- **状态**: RUNNING (node038)
- **开始时间**: 约 03:44
- **预计时长**: ~10 分钟
- **配置**: 
  - 模型: `Wan2.2-TI2V-5B-Diffusers`
  - 方法: GradDiff (retain_weight=1.0)
  - 步数: 600, 保存间隔: 100
  - LoRA: rank=8, alpha=16.0
- **输出**: `models/unlearn/exp019_wan5b_nudity_graddiff/`
- **监控**: Monitor 任务已启动，将在完成时通知

---

## 🐛 已修复的所有问题

| 版本 | Job ID | 问题 | 修复 |
|------|--------|------|------|
| v1 | 17530930 | 参数格式错误（下划线 vs 连字符） | 使用 `--model-path` 等连字符参数 |
| v2 | 17530936 | 环境配置错误 | 使用 VU `.venv` 环境 |
| v3 | 17530937 | 模型路径错误（找不到 text_encoder） | 使用 `Wan2.2-TI2V-5B-Diffusers` |
| v4 | 17530960 | ✅ **所有问题已修复** | 使用正确的 Diffusers 路径 + latent-manifest |

---

## 📊 关键修复点

### 1. 环境配置
```bash
# ✅ 正确
source /home/x_jiage/jiage/video-unlearning/.venv/bin/activate

# ❌ 错误
conda activate video-unlearning  # 环境不存在
```

### 2. 模型路径
```bash
# ✅ 正确 - Diffusers 版本（有 text_encoder 子目录）
models/Wan-AI/Wan2.2-TI2V-5B-Diffusers

# ❌ 错误 - DiffSynth 版本（无 text_encoder）
models/Wan-AI/Wan2.2-TI2V-5B
```

### 3. Latent 缓存
```bash
# ✅ 正确 - 使用预先缓存的 latent manifest
--latent-manifest data/wan5b/unlearn_latents/latent_manifest.jsonl

# ❌ 错误 - cache-dir 会重新缓存
--cache-dir data/wan5b/unlearn_baseline
```

---

## 🎯 下一步

### 当训练完成后（~10分钟）：

1. **验证输出**
   ```bash
   ls models/unlearn/exp019_wan5b_nudity_graddiff/step-*
   # 应该有 7 个 checkpoints
   ```

2. **提交后续作业**
   ```bash
   bash scripts/exp019_submit_chain.sh
   ```

3. **监控进度**
   ```bash
   bash scripts/exp019_monitor.sh
   ```

---

## 📈 预期时间线

- ✅ Phase 1: 准备与验证 - 已完成
- 🔄 Phase 2: 擦除训练 - 进行中 (~10 分钟)
- ⏳ Phase 3: LoRA 合并 - 待开始 (~20 分钟)
- ⏳ Phase 4: 基线生成 - 待开始 (~1 小时)
- ⏳ Phase 5: 基线评估 - 待开始 (~30 分钟)
- ⏳ Phase 6: 微调训练 - 待开始 (~24 小时) ⚠️ **最长**
- ⏳ Phase 7-8: 微调后生成评估 - 待开始 (~1 小时)
- ⏳ Phase 9-10: 结果整理 - 待开始 (~30 分钟)

**总计**: ~26-28 小时

---

**监控任务**: 已启动 Monitor，训练完成时将收到通知
