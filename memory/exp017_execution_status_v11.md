# Exp 017 执行状态 - v11 最终版

**最后更新**: 2026-09-03 19:33  
**版本**: v11 - JSON model_paths 参数  
**状态**: 🟢 作业运行中

---

## 📊 当前作业状态 (v11)

### Stage 1: 蒸馏训练

| ARM | Job ID | 状态 | 节点 | 开始时间 |
|-----|--------|------|------|----------|
| base | 17458319 | ⏳ **RUNNING** | node042 | 19:32 |
| grad_ascent | 17458320 | ⏳ **RUNNING** | node011 | 19:32 |
| esd | 17458321 | ⏳ **RUNNING** | node019 | 19:32 |

**环境**: `ft_diffsynth` (独立环境)  
**监控**: bafe20m2l (每30秒检查)

---

## 🎯 最终解决方案（11次迭代）

### 问题总结

1. **环境问题** (v1-v8)
   - transformers/torch 版本不兼容
   - pandas/numpy 版本不兼容
   - DeepSpeed 缺失

2. **路径问题** (v9-v11)
   - `--model_id_with_origin_paths` 总是尝试从 modelscope 下载
   - 相对路径、绝对路径都被误认为仓库ID

### 最终方案

**使用 `--model_paths` JSON格式传递本地路径**

```bash
# base: 使用 modelscope ID
--model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:..."

# 擦除模型: 使用 JSON 本地路径
--model_paths '{"dit":"/absolute/path/to/model.safetensors","text_encoder":"...","vae":"..."}'
```

---

## 🔧 完整迭代历史（11次）

| 版本 | 核心问题 | 解决方案 | 结果 |
|------|----------|----------|------|
| v1 | 账户错误 | 修改账户 | ❌ |
| v2 | 缺少参数 | 添加参数 | ❌ |
| v3 | base 路径 | 改脚本 | ❌ |
| v4 | 参数为空 | 提前定义 | ❌ |
| v5 | diffusers格式 + ImportError | 统一脚本 | ❌ |
| v6 | DeepSpeed 缺失 | 单GPU配置 | ❌ |
| v7 | pandas GLIBCXX | 降级 pandas | ❌ |
| v8 | numpy/pandas 二进制不兼容 | 尝试重装 | ❌ |
| v9 | 环境共享冲突 | **独立环境 ft_diffsynth** | ❌ (路径问题) |
| v10 | 相对路径被误认为仓库ID | 绝对路径 | ❌ (仍被误认) |
| **v11** | **model_id_with_origin_paths 总是下载** | **--model_paths JSON格式** | ⏳ **运行中** |

---

## 💡 核心发现

### 1. DiffSynth-Studio 路径处理机制

**`--model_id_with_origin_paths`**:
- 格式: `repo_id:file_pattern`
- 总是尝试从 modelscope 下载
- 即使使用绝对路径也会被当作仓库ID

**`--model_paths`** (正确方式):
- 格式: JSON `{"dit":"path","text_encoder":"path","vae":"path"}`
- 直接使用本地文件
- 优先于 `model_id_with_origin_paths`

### 2. 环境隔离的重要性

创建 `ft_diffsynth` 独立环境:
```bash
conda create --name ft_diffsynth --clone diffsynth
conda activate ft_diffsynth
pip install numpy==1.26.4 --force-reinstall --no-deps
pip install pandas==2.0.3 --force-reinstall --no-cache-dir --no-deps
```

### 3. 调试策略

**有效**:
- 在登录节点测试环境
- 仔细阅读工具文档/代码
- 理解参数的真实含义

**无效** (我们走的弯路):
- 反复猜测和尝试
- 11次迭代才找到正确参数

---

## 📁 关键文件

### 环境
- `ft_diffsynth` conda 环境
  - numpy 1.26.4
  - pandas 2.0.3
  - transformers 4.44.0
  - tokenizers 0.19.1

### 配置
- `slurm/exp017_distill.sbatch` - 使用 `--model_paths` JSON格式
- `accelerate_config_single_gpu.yaml` - 单GPU配置

### 文档
- `memory/exp017_execution_status_v11.md` - 本文件
- `memory/ft_environment_separation_plan.md` - 环境分离方案
- `memory/exp017_environment_issue.md` - 环境问题详细分析

---

## ⏱️ 预计时间线

- **19:32** ✅ v11 作业开始运行
- **19:36** - 预计训练开始（模型加载完成）
- **次日 04:00-06:00** - Stage 1 完成（8-10小时）
- **次日 06:00-07:30** - Stage 2 完成（生成视频）
- **次日 07:30-08:30** - Stage 3 完成（安全评估）

**总预计时长**: ~11-13 小时

---

## 📖 监控

### 自动监控
```bash
# Monitor task: bafe20m2l
# 每30秒检查一次，最多20次
```

### 手动检查
```bash
# 状态
squeue -u $USER | grep exp017

# 日志
tail -f slurm/logs/exp017_distill_17458319_*.out  # base
tail -f slurm/logs/exp017_distill_17458320_*.out  # grad_ascent
tail -f slurm/logs/exp017_distill_17458321_*.out  # esd

# 错误
tail -f slurm/logs/exp017_distill_17458319_*.err
```

---

## ✅ 成功关键因素

1. ✅ **独立环境**: `ft_diffsynth` 避免版本冲突
2. ✅ **正确参数**: `--model_paths` JSON格式用于本地路径
3. ✅ **环境验证**: `python scripts/test_env.py` 通过
4. ⏳ **等待确认**: 监控器将报告训练是否成功启动

---

## 🔮 下一步

1. **等待监控器确认** - 训练成功开始 (2-3分钟)
2. **如果成功** - 等待 8-10 小时完成
3. **如果失败** - 需要更深入调试或联系 DiffSynth-Studio 作者
4. **完成后** - 分析结果并更新实验记录

---

**这是第11次尝试。希望这次能成功！** 🤞
