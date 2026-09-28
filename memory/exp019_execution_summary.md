# Exp019 执行总结

**更新时间**: 2026-09-16 04:00  
**当前状态**: Phase 2 擦除训练进行中（Job 17530963）

---

## ✅ 已完成的工作

### Phase 1: 准备与验证
- ✅ 验证 GradDiff 支持
- ✅ 确认数据准备
- ✅ 创建实验记录

### 脚本准备（全部完成）
1. ✅ 擦除训练、LoRA合并、视频生成、评估等 8 个 SLURM 脚本
2. ✅ 自动化提交脚本（依赖链编排）
3. ✅ 进度监控脚本
4. ✅ 冒烟测试脚本

---

## 🔄 当前进行中

### Job 17530963 - GradDiff 擦除训练 (v5)
- **状态**: RUNNING (node051, 已运行 ~1 分钟)
- **配置**: 
  - 模型: Wan2.2-TI2V-5B-Diffusers
  - 方法: GradDiff (retain_weight=1.0)
  - 步数: 600, LoRA rank=8
  - Latent manifest: 27 条（包含 forget + retain）
- **预计时长**: ~10 分钟
- **输出**: `models/unlearn/exp019_wan5b_nudity_graddiff/`
- **监控**: Monitor 任务 bnvwl0hsk 已启动

---

## 🐛 问题修复历史（5 次迭代）

| 版本 | Job ID | 问题 | 根因 | 修复 |
|------|--------|------|------|------|
| v1 | 17530930 | 参数格式错误 | 使用了下划线（`--model_id`） | 改用连字符（`--model-path`） |
| v2 | 17530936 | 环境不存在 | `conda activate video-unlearning` 失败 | 使用 VU `.venv` 环境 |
| v3 | 17530937 | 找不到 text_encoder | 使用 DiffSynth 版本路径 | 使用 Diffusers 版本 |
| v4 | 17530960 | KeyError: 'latent' | `retain_manifest` 指向原始 jsonl（无 latent 字段） | 只使用 `--latent-manifest` |
| v5 | 17530963 | ✅ **运行中** | - | GradDiff 从 latent_manifest 自动分离 forget/retain |

---

## 🔑 关键经验教训

### 1. 环境配置
```bash
# ✅ 正确 - 使用 VU 项目的 .venv
source /home/x_jiage/jiage/video-unlearning/.venv/bin/activate
```

### 2. 模型路径
```bash
# ✅ 正确 - Diffusers 版本（有完整子目录结构）
models/Wan-AI/Wan2.2-TI2V-5B-Diffusers
```

### 3. GradDiff 的特殊性
- GradDiff 需要 retain 数据，但**不需要单独的** `--retain-manifest`
- 只需要 `--latent-manifest`，它会从中自动分离 forget 和 retain
- `latent_manifest.jsonl` 中的数据已经包含 `latent` 字段（预先缓存）

### 4. 参考 Exp015 成功经验
- Exp015 的 GradDiff 脚本也**只使用** `--latent-manifest`
- 避免传递原始的 VU manifest（缺少 latent 字段）

---

## 🎯 下一步（训练完成后）

### 1. 验证输出（预计 ~10 分钟后）
```bash
# 检查 checkpoints（应该有 7 个）
ls models/unlearn/exp019_wan5b_nudity_graddiff/step-*

# 每个约 153M
du -sh models/unlearn/exp019_wan5b_nudity_graddiff/step-*
```

### 2. 提交后续作业链
```bash
# 自动化提交完整流程
bash scripts/exp019_submit_chain.sh
```

### 3. 监控进度
```bash
# 查看所有作业
squeue -u $USER

# 运行监控脚本
bash scripts/exp019_monitor.sh
```

---

## 📊 预期时间线

| Phase | 预计时间 | 说明 |
|-------|---------|------|
| ✅ Phase 1 | 完成 | 准备与验证 |
| 🔄 Phase 2 | ~10 分钟 | 擦除训练（进行中） |
| ⏳ Phase 3 | ~20 分钟 | LoRA 合并 |
| ⏳ Phase 4 | ~1 小时 | 基线生成（2臂×99视频） |
| ⏳ Phase 5 | ~30 分钟 | 基线评估 |
| ⏳ Phase 6 | ~24 小时 | 微调训练 ⚠️ **最长** |
| ⏳ Phase 7-8 | ~1 小时 | 微调后生成+评估 |
| ⏳ Phase 9-10 | ~30 分钟 | 结果整理与验证 |

**总计**: 约 26-28 小时

---

## 🔬 研究目标（提醒）

1. **GradDiff vs GradAscent 擦除效果**
   - GradDiff（有 retain）是否比 GradAscent（无 retain）更有效？

2. **微调鲁棒性对比**
   - GradDiff 回潮是否 < GradAscent (+6.1pp)？

3. **Retain 约束的价值**
   - 验证 retain 约束是否真正提升擦除鲁棒性

---

## 📝 文档更新

- ✅ `memory/experiments.md` - 添加 Exp019 记录
- ✅ `memory/index.md` - 更新文件索引
- ✅ `memory/exp019_status.md` - 状态报告
- ✅ `memory/exp019_启动报告.md` - 启动报告
- ✅ `memory/exp019_current_status.md` - 当前状态
- ✅ 本文件 - 执行总结

---

**备注**: 
- Monitor 任务正在跟踪训练进度，完成时会自动通知
- 所有后续脚本已准备就绪，可无缝衔接
- 经过 5 次迭代，所有配置问题已解决
