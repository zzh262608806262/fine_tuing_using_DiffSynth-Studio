# Exp012 运行时状态

## 当前运行：快速验证版（推荐）

- **提交时间**: 2026-08-26 18:04:41
- **版本**: 快速验证版（几小时完成）
- **监控启动**: 2026-08-26 18:05 (task bvc2fs07h)

### 优化点
1. **擦除训练**: 1200步 → 200步
2. **微调**: repeat=10×epochs=2 → repeat=1×epochs=1 (20次曝光 → 1次曝光)
3. **生成**: 只生成 fast 集 29条（跳过 benchmark 99条）
4. **预计耗时**: **几小时**（而非原版的3-5天）

## 历史：完整版（已取消）

- **提交时间**: 2026-08-26 17:41:13
- **状态**: 已取消（用户认为3-5天太长，改用快速验证版）
- **原因**: 首次验证流程应该快速迭代，不应该等3-5天

## 作业链

```
J0  17379128  wan5b-download    (6h)   → 下载 5B 模型 34.2GB
  ↓
J1  17379129  wan5b-bridge      (8h)   → 桥转换 + 原生 DiT 逐键比对
  ↓
J1c 17379130  wan5b-baseline    (4h)   → 重生成 27 条基线视频
  ↓
J2  17379131  wan5b-unlearn     (8h)   → nudity 擦除训练 (GradAscent 1200步)
  ↓
J3  17379132  wan5b-merge       (12h)  → 擦除 LoRA 合并
  ↓
J4  17379133  wan5b-finetune    (3d)   → 两臂微调 (base+FT, erased+FT)
  ↓
GEN 17379134  base              (3d)   → 四臂评测视频生成
    17379135  erased            (3d)
    17379136  base_ft           (3d)
    17379137  erased_ft         (3d)
  ↓
J9  17379138  wan5b-eval-judge  (24h)  → porn 专项判别
```

**预计总耗时**: 3-5天（取决于队列等待和生成速度）

## 已修复的技术问题

1. ✅ GLIBCXX_3.4.29 缺失 → 修复 LD_LIBRARY_PATH 设置
2. ✅ pandas 依赖缺失 → conda install pandas + libstdcxx-ng
3. ✅ 分辨率问题 → 720→736 (VAE 要求 32 倍数)
4. ✅ /scratch 路径消失 → 重新下载到本地
5. ✅ 软链接断裂 → 改用绝对路径
6. ✅ 各种路径和环境问题

## 监控设置

- **后台监控**: Monitor 工具 (task b5f1ho22g)
- **检查间隔**: 5分钟
- **监控模式**: persistent (会话级持久)
- **事件通知**: 状态变化时自动通知

## 注意事项

⚠️ **GPU idle reaper**: 如果作业申请 GPU 但不使用会在整 1 小时被 CANCELLED
  - 本实验所有作业都实际使用 GPU，应该不会触发

⚠️ **会话断连**: Monitor 工具绑定到当前 Claude 会话
  - 用户断连后监控会停止
  - 但 slurm 作业本身在集群上继续运行
  - 重新连接后可用 `squeue --me` 或 `sacct` 查看状态

## 手动检查命令

```bash
# 查看当前作业
squeue --me

# 查看详细状态
sacct -j 17379128,17379129,17379130,17379131,17379132,17379133,17379134,17379135,17379136,17379137,17379138 \
  --format=JobID,JobName,State,ExitCode,Elapsed,NodeList

# 查看特定作业日志
tail -f slurm/logs/wan5b_*-<JOBID>.out
tail -f slurm/logs/wan5b_*-<JOBID>.err

# 使用现有监控脚本
bash scripts/monitor_exp012.sh check    # 一次性检查
bash scripts/monitor_exp012.sh report   # 生成报告
```

## 多GPU优化（待实施）

长时间作业（>8h）可考虑多GPU并行：
- wan5b_finetune.sbatch (3天) - 可能支持多GPU
- wan5b_eval_gen.sbatch (3天) - 已经是4个作业并行
- wan5b_eval_judge.sbatch (24h) - 可考虑数据并行

需要检查训练脚本是否支持多GPU（DDP/FSDP）。

## 下一步

- ✅ 作业已提交
- ✅ 监控已启动
- ⏳ 等待 J0 下载完成
- ⏳ 等待整个流水线完成
- 待定: 根据运行结果更新 experiments.md
