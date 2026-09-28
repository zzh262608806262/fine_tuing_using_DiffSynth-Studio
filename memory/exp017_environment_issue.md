# Exp 017 环境问题报告

**日期**: 2026-09-03  
**严重程度**: 🔴 阻塞性问题  
**状态**: 需要人工修复

---

## 🚨 问题描述

所有 Exp 017 蒸馏作业失败，原因是 **diffsynth conda 环境损坏**。

### 错误信息

```
ImportError: cannot import name 'NP_SUPPORTED_MODULES' from 'torch._dynamo.utils'
```

### 根本原因

PyTorch 和 transformers 版本不兼容：

```bash
$ conda activate diffsynth && pip list | grep -E "torch|transformers"
torch                  2.13.0
torchvision            0.28.0
transformers           5.14.1
```

- **torch 2.13.0**: 非常新的开发版本，内部 API 已变更
- **transformers 5.14.1**: 期望旧版 torch API，导致 `NP_SUPPORTED_MODULES` 找不到

### 影响范围

- ❌ **所有使用 accelerate 的训练任务**（蒸馏、微调等）
- ✅ **纯推理任务**可能不受影响（不依赖 transformers）

### 时间线

- **8月31日**: Exp 015 成功运行（环境正常）
- **9月3日**: Exp 017 全部失败（环境已损坏）
- **可能原因**: 期间有包自动升级或人工更新

---

## 🔧 解决方案

### 方案1: 降级 transformers（推荐，快速）

```bash
conda activate diffsynth
pip install transformers==4.44.0  # 兼容 torch 2.13 的版本
```

### 方案2: 降级 torch（稳定，但耗时）

```bash
conda activate diffsynth
pip install torch==2.4.0 torchvision==0.19.0
# 或者使用 conda:
conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 方案3: 重建环境（最彻底）

```bash
# 备份当前环境列表
conda activate diffsynth
pip list > ~/diffsynth_env_backup_20260903.txt

# 删除并重建
conda deactivate
conda env remove -n diffsynth
conda create -n diffsynth python=3.10
conda activate diffsynth

# 重新安装核心包（参考 DiffSynth-Studio 文档）
pip install torch==2.4.0 torchvision==0.19.0
pip install transformers==4.44.0 accelerate diffusers
# ... 其他依赖
```

---

## 🧪 验证修复

运行以下命令验证环境是否修复：

```bash
conda activate diffsynth
python -c "
from accelerate.commands.accelerate_cli import main
print('✓ accelerate 可以导入')
"
```

如果不报错，说明环境修复成功。

---

## 📋 失败的作业列表

### v4 (17453288-17453294)
- 17453288 (base): ImportError
- 17453289 (grad_ascent): 擦除模型路径问题
- 17453290 (esd): 擦除模型路径问题

### v5 (17453328-17453334)
- 17453328 (base): ImportError
- 17453329 (grad_ascent): ImportError
- 17453330 (esd): ImportError

**所有作业在同一个环境错误处失败**

---

## 🔄 修复后的重新提交步骤

1. 修复环境（选择上述方案之一）
2. 验证环境正常
3. 重新提交：
   ```bash
   cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
   bash slurm/exp017_submit_all.sh
   ```

---

## 💡 预防措施

为了避免未来环境再次损坏：

1. **固定版本**: 在 `environment.yml` 或 `requirements.txt` 中明确指定版本
2. **环境快照**: 定期导出可用的环境配置
   ```bash
   conda env export > environment_working_20260903.yml
   ```
3. **禁用自动更新**: 不要使用 `pip install --upgrade` 除非明确需要

---

**下一步行动**: 请用户选择一个修复方案并执行，然后我会重新提交 Exp 017。
