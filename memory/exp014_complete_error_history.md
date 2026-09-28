# Exp014 Stage 3 错误修复完整记录

**最后更新**: 2026-09-06 (用户两天后返回检查)  
**当前版本**: v4 (Job 17491077)

---

## 完整错误历史

### ❌ v1 (Job 17460858) - 2026-09-04 14:15
- **错误**: `ValueError: numpy.dtype size changed`
- **原因**: pandas 2.0.3 与 numpy 二进制不兼容
- **运行时间**: ~1 分钟
- **修复**: 升级 pandas → 2.3.3

### ❌ v2 (Job 17461003) - 2026-09-04 14:45
- **错误**: `ImportError: GLIBCXX_3.4.29 not found`
- **原因**: 升级 numpy 2.2.6 需要更新的 C++ 库
- **运行时间**: ~1 分钟
- **修复**: 安装 conda libstdcxx-ng + 设置 LD_LIBRARY_PATH

### ❌ v3 (Job 17461048) - 2026-09-04 14:56
- **错误**: `ModuleNotFoundError: No module named 'tensorboard'`
- **原因**: `--enable_tensorboard_log` 需要 tensorboard 模块
- **运行时间**: ~2 分钟
- **关键**: **训练实际已开始** (`0/2500 [00:00<?, ?it/s]`)
- **修复**: 移除 `--enable_tensorboard_log` 参数

### ✅ v4 (Job 17491077) - 2026-09-06
- **修复**: 移除 tensorboard 依赖
- **状态**: 已提交
- **预期**: 成功运行 24-32 小时

---

## 根本问题链

```
1. DiffSynth 需要 pandas (数据集加载)
   ↓
2. pandas 2.0.3 与 numpy 不兼容
   ↓
3. 升级 pandas 2.3.3 需要 numpy 2.2.6
   ↓
4. numpy 2.2.6 需要 GLIBCXX_3.4.29
   ↓
5. 系统 C++ 库版本太旧
   ↓
6. 需要 conda 的 libstdcxx-ng
   ↓
7. tensorboard 是可选依赖，但被启用了
   ↓
8. 移除 tensorboard → ✅ 应该成功
```

---

## 关键发现

**v3 的日志显示训练已经开始**：
```
0/2500 [00:00<?, ?it/s]
```

这意味着：
- ✅ 环境配置正确
- ✅ 模型加载成功
- ✅ 数据集准备完成
- ✅ 训练循环已启动
- ❌ 仅在第一次日志写入时因 tensorboard 失败

**v4 只需移除 tensorboard，训练应该能正常运行**

---

## v4 修改点

```diff
- python -u scripts/finetune_wan5b.py ... --enable_tensorboard_log
+ python -u scripts/finetune_wan5b.py ... # 移除 tensorboard
```

保留所有其他配置：
- ✅ LD_LIBRARY_PATH (C++ 库)
- ✅ repeat=25, epochs=20
- ✅ gradient checkpointing
- ✅ 所有其他参数

---

## 预期结果

如果 v4 成功：
- **Erased 臂**: 12-16 小时，生成 20 个 checkpoint
- **Base 臂**: 12-16 小时，生成 20 个 checkpoint
- **总计**: 24-32 小时，40 个 checkpoint

产物：
```
models/finetune/exp014_erased_ft/
├── epoch-01.safetensors
├── epoch-02.safetensors
├── ...
└── epoch-20.safetensors

models/finetune/exp014_base_ft/
├── epoch-01.safetensors
├── epoch-02.safetensors
├── ...
└── epoch-20.safetensors
```

---

## 监控计划

因为用户两天后才返回，说明：
1. 自动监控没有正常工作（可能会话已结束）
2. 需要手动检查作业状态

**检查命令**：
```bash
# 查看作业状态
sacct -j 17491077 --format=JobID,State,ExitCode,Elapsed

# 查看日志
tail -100 slurm/logs/exp014_finetune_v4-17491077.out

# 检查产物
ls -lh models/finetune/exp014_*/epoch-*.safetensors | wc -l
```

---

**当前状态**: 等待用户确认 v4 是否成功运行
