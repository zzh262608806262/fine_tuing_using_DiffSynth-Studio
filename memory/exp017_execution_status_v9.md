# Exp 017 执行状态 - v9 最终版

**最后更新**: 2026-09-03 19:23  
**版本**: v9 - 独立环境 ft_diffsynth  
**状态**: 🟢 作业运行中，使用独立环境

---

## 📊 当前作业状态 (v9)

### Stage 1: 蒸馏训练

| ARM | Job ID | 状态 | 节点 | 开始时间 |
|-----|--------|------|------|----------|
| base | 17458249 | ⏳ **RUNNING** | node048 | 19:22 |
| grad_ascent | 17458250 | ⏳ **RUNNING** | node017 | 19:22 |
| esd | 17458251 | ⏳ **RUNNING** | node082 | 19:22 |

**环境**: `ft_diffsynth` (独立环境)

### Stage 2-3: 等待依赖

- Stage 2: Jobs 17458252-17458254 (PENDING)
- Stage 3: Job 17458255 (PENDING)

---

## 🎯 最终解决方案：环境隔离

### 问题根源

经过 **8 次迭代**发现：
- `diffsynth` 环境被多个项目共享
- numpy 2.2.6 与 pandas 2.0.3 二进制不兼容
- 环境在2026-08-31至09-03间被自动升级

### 最终方案

**创建独立的 `ft_diffsynth` 环境**

```bash
# 克隆环境
conda create --name ft_diffsynth --clone diffsynth

# 修复版本
conda activate ft_diffsynth
pip install numpy==1.26.4 --force-reinstall --no-deps
pip install pandas==2.0.3 --force-reinstall --no-cache-dir --no-deps

# 验证
python scripts/test_env.py
```

### 环境配置

```
Python:        3.10.20
torch:         2.13.0+cu130
transformers:  4.44.0
tokenizers:    0.19.1
numpy:         1.26.4  ← 降级（兼容 pandas）
pandas:        2.0.3   ← 重新编译（匹配 numpy）
accelerate:    ✓ 可导入
```

---

## 🔧 修改的文件

### 1. 环境配置
- 新建: `ft_diffsynth` conda 环境

### 2. SLURM 脚本
- `slurm/exp017_distill.sbatch`: `diffsynth` → `ft_diffsynth`

### 3. Python 脚本
- `scripts/distill_wan5b.py`: 更新环境引用
- `scripts/exp017_interactive_test.sh`: 更新环境引用

### 4. 文档
- `memory/ft_environment_separation_plan.md`: 环境分离方案
- `memory/exp017_execution_status_v9.md`: 本文件

---

## 📈 完整迭代历史（8次）

| 版本 | 核心问题 | 解决方案 | 结果 |
|------|----------|----------|------|
| v1 | 账户错误 | 修改账户名 | ❌ |
| v2 | 缺少参数 | 添加参数 | ❌ |
| v3 | base 路径错误 | 改用 distill_wan5b.py | ❌ |
| v4 | 参数为空 | 提前定义变量 | ❌ |
| v5 | diffusers 格式 + ImportError | 统一脚本 | ❌ |
| v6 | DeepSpeed 缺失 | 单GPU配置 | ❌ |
| v7 | pandas GLIBCXX 错误 | 降级 pandas | ❌ |
| v8 | numpy/pandas 二进制不兼容 | 尝试重装 | ❌ |
| **v9** | **环境共享导致版本冲突** | **独立环境** | ⏳ **运行中** |

---

## 💡 核心经验

### 1. 环境隔离的重要性

**错误做法**: 多个项目共享一个 conda 环境
- FT 项目和 VU 项目都用 `diffsynth`
- 一个项目的升级影响另一个项目

**正确做法**: 每个项目独立环境
- FT 项目: `ft_diffsynth`
- VU 项目: `diffsynth` 或独立环境
- 跨项目调用时显式切换

### 2. 版本固定

**创建 requirements.txt**:
```bash
conda activate ft_diffsynth
pip freeze > requirements_ft_diffsynth.txt
```

未来重建:
```bash
conda create -n ft_diffsynth_new python=3.10
pip install -r requirements_ft_diffsynth.txt
```

### 3. 调试策略

**有效路径**:
1. 在登录节点测试环境（accelerate 不需要 GPU）
2. 发现根本问题（环境共享）
3. 彻底解决（环境隔离）

**无效路径** (我们走过的弯路):
- 反复修改配置 → 提交 sbatch → 失败 → 再修改
- 8次迭代才找到根因

---

## ⏱️ 预计时间线

- **19:22** ✅ v9 作业开始运行
- **19:26** - 预计训练开始
- **次日 04:00-06:00** - Stage 1 完成
- **次日 06:00-07:30** - Stage 2 完成
- **次日 07:30-08:30** - Stage 3 完成

**总预计时长**: ~11-13 小时

---

## 📖 监控

### 自动监控
- Monitor task: btn26ws8e
- 检查频率: 每30秒
- 最多15次检查

### 手动检查
```bash
# 状态
squeue -u $USER | grep exp017

# 日志
tail -f slurm/logs/exp017_distill_17458249_*.out  # base

# 验证环境
conda activate ft_diffsynth
python -c "import pandas; import numpy; print(f'numpy {numpy.__version__}, pandas {pandas.__version__}')"
```

---

## ✅ 下一步

1. **等待确认** - 监控器报告训练开始
2. **如果成功** - 等待 8-10 小时完成
3. **完成后** - 分析结果，更新 experiments.md
4. **长期** - 为其他实验也创建独立环境

---

**关键成功因素**:
- ✅ 独立环境隔离
- ✅ numpy 1.26.4 + pandas 2.0.3 兼容
- ✅ 验证测试通过
- ⏳ 等待训练启动确认
