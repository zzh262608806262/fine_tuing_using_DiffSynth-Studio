# Exp014 Stage 3 监控记录

**Job ID**: 17460858  
**启动时间**: 2026-09-04 14:15  
**监控状态**: 🟢 活跃

---

## 监控配置

- ✅ 持续监控已启动（Monitor task: bfel4ddqr）
- ✅ 自动检测错误关键词
- ✅ 每10分钟报告进度
- ✅ 检测成功/失败状态

---

## 常见问题与修复方案

### 问题 1: 模型路径错误
**症状**: `FileNotFoundError: models/Wan-AI/...`
**修复**: 检查模型路径，确认 T5/VAE 软链存在

### 问题 2: CUDA OOM
**症状**: `CUDA out of memory`
**修复**: 
- 降低 batch size
- 使用 gradient checkpointing（已启用）
- 申请 fat 节点（已申请）

### 问题 3: 数据集加载失败
**症状**: `FileNotFoundError: data/tiger_dataset/...`
**修复**: 检查 metadata_100.csv 路径

### 问题 4: LoRA 保存失败
**症状**: `Permission denied` 或 `Disk quota exceeded`
**修复**: 检查磁盘空间和权限

### 问题 5: Accelerate 配置错误
**症状**: `accelerate launch` 相关错误
**修复**: 检查 GPU 数量和配置

---

## 预期时间线

| 时间点 | 预期事件 |
|--------|----------|
| 14:15 | ✅ 作业提交 |
| 14:15-15:00 | 等待调度（PD 状态） |
| 15:00-15:10 | 环境初始化 + 前置检查 |
| 15:10-03:00 (次日) | Erased 臂微调（约12小时） |
| 03:00-15:00 (次日) | Base 臂微调（约12小时） |
| 15:00 | ✅ 完成 |

**总计**: 24-32 小时

---

## 监控输出解读

- `[PENDING]`: 作业在队列中等待
- `[RUNNING]`: 作业正在运行
- `[PROGRESS]`: 定期进度报告
- `[ERROR]`: 检测到错误，需要修复
- `[SUCCESS]`: 作业成功完成
- `[FAILED]`: 作业失败或被取消

---

## 手动检查命令

```bash
# 查看作业状态
squeue -j 17460858

# 查看完整输出日志
tail -100 slurm/logs/exp014_finetune-17460858.out

# 查看错误日志
tail -100 slurm/logs/exp014_finetune-17460858.err

# 查看实时日志
tail -f slurm/logs/exp014_finetune-17460858.out
```

---

**最后更新**: 2026-09-04 14:20  
**下次检查**: 自动（每60秒）
