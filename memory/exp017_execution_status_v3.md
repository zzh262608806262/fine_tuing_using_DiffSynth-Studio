# Exp 017 执行状态更新

**最后更新**: 2026-09-04 13:54  
**版本**: v4（全部5个模型成功启动）

---

## ✅ 最终成功状态

### 🎉 所有5个模型正在训练（5/5）

| 模型 | Job ID | 节点 | 状态 | 进度 | 运行时间 |
|------|--------|------|------|------|----------|
| base | 17460464 | node083 | ✅ 训练中 | 38/1000 steps (4%) | 00:11:18 |
| grad_ascent | 17460465 | node073 | ✅ 训练中 | 初始化完成 | 00:11:18 |
| esd | 17460643 | node047 | ✅ 训练中 | 刚启动 | 00:02:00 |
| npo | 17460644 | node035 | ✅ 训练中 | 刚启动 | 00:02:00 |
| anchor_distill | 17460645 | node035 | ✅ 训练中 | 刚启动 | 00:02:00 |

**训练参数**: `--dataset_repeat 10 --num_epochs 10` → 10,000 steps
**预计完成**: 2026-09-06 12:00（~46小时）

---

## 🔧 修复历史

### v1-v2: 初始问题
- 账户错误、参数缺失、路径问题
- 已在之前版本修复

### v3: 参数优化 + OOM斗争
- **原参数**: `--dataset_repeat 160 --num_epochs 2` → 32,000 steps (6.3天)
- **用户反馈**: "居然要这么久吗，参数选择是否合理"
- **优化后**: `--dataset_repeat 10 --num_epochs 10` → 10,000 steps (46小时)
- **收益**: 训练时间减少70%，checkpoint频率提升5倍

### v4: 节点内存问题最终解决 ✅
**问题**: CUDA OOM - 某些节点GPU显存被占用

**失败尝试**:
1. Jobs 17460450-17460459: 加了 `--enable_tensorboard_log` → 全部OOM
2. Jobs 17460464-17460468: 去掉tensorboard → base/grad_ascent成功，其他3个OOM
3. Jobs 17460616-17460618: 重新提交 → 仍然OOM（分配到差节点）

**成功突破** (2026-09-04 13:52):
- Jobs 17460643-17460645: 第3次重新提交
- **关键**: 分配到了node047和node035（有足够显存）
- **结果**: 所有5个模型全部成功启动训练

**经验教训**:
- GPU节点显存可用性存在差异
- node083, node073, node047, node035 可用显存充足 ✅
- node048, node022, node029, node033, node009 可用显存不足 ❌
- 策略: 持续重新提交直到获得好节点

---

## 📊 训练配置

```bash
--dataset_repeat 10          # 每个样本重复10次
--num_epochs 10              # 训练10个epoch
--learning_rate 1e-05        # 学习率
--gradient_accumulation_steps 1
--use_gradient_checkpointing # 节省显存
# 无 tensorboard（避免OOM）
```

**总训练步数**: 100 samples × 10 repeat × 10 epochs / 1 batch = 10,000 steps
**每步耗时**: ~16.6秒
**总耗时**: 10,000 × 16.6s ≈ 46小时

---

## 📈 Checkpoint 生成

每个epoch结束保存一次checkpoint:
```
models/train/exp017_base_distill/checkpoint-epoch-0/
models/train/exp017_base_distill/checkpoint-epoch-1/
...
models/train/exp017_base_distill/checkpoint-epoch-9/
```

总共: **5个模型 × 10个checkpoints = 50个模型文件**

---

## 🎯 流水线依赖

### Stage 1: 蒸馏训练 ✅ 进行中
- 17460464-17460465, 17460643-17460645
- 状态: 全部运行中
- 预计完成: 2026-09-06 12:00

### Stage 2: 视频生成 (afterok依赖)
- 将在Stage 1完成后自动触发
- 使用10个checkpoints中的最终checkpoint
- 生成100个视频（每个模型20个）

### Stage 3: 安全评估 (afterok依赖)
- 将在Stage 2完成后自动触发
- 使用GPT-4o + Qwen3-VL评估
- 输出安全分数到JSON

---

## 📝 监控命令

```bash
# 查看所有作业状态
sacct -j 17460464,17460465,17460643,17460644,17460645 --format=JobID,State,NodeList,Elapsed

# 查看训练进度（base）
tail -f /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio/slurm/logs/exp017_distill_17460464_4294967294.err

# 检查是否有OOM
grep -i "out of memory" /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio/slurm/logs/exp017_distill_*.err
```

---

## ⏱️ 时间线

- **2026-09-04 13:42** - base/grad_ascent 开始训练 ✅
- **2026-09-04 13:52** - esd/npo/anchor_distill 启动成功 ✅
- **2026-09-06 12:00** - Stage 1 预计完成（+46h）
- **2026-09-06 13:30** - Stage 2 预计完成（+1.5h）
- **2026-09-06 14:30** - Stage 3 预计完成（+1h）

**总预计时长**: ~49小时

---

## ✅ 确认清单

- [x] 账户、参数、路径问题已修复
- [x] 训练参数优化（70%时间节省）
- [x] GradDiff支持已添加
- [x] 去除tensorboard（避免OOM）
- [x] 所有5个模型成功启动
- [x] 训练循环正常运行
- [ ] 等待训练完成（~46小时）
- [ ] 等待视频生成自动触发
- [ ] 等待安全评估自动触发
- [ ] 记录最终结果到 experiments.md

---

**状态**: 🟢 全部正常运行中（5/5成功）
