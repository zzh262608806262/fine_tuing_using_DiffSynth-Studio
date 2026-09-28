# Exp014 错误修复记录 v2

**更新时间**: 2026-09-04 14:55

---

## ❌ 错误 #1: Pandas/Numpy 二进制不兼容
- **Job**: 17460858
- **错误**: `ValueError: numpy.dtype size changed`
- **修复**: 升级 pandas/numpy
- **结果**: ❌ 引发错误 #2

---

## ❌ 错误 #2: GLIBCXX 版本不足
- **Job**: 17461003
- **错误**: `ImportError: /lib64/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found`
- **根因**: 升级 numpy 2.2.6 后需要更新的 C++ 库（GLIBCXX_3.4.29），但系统库版本太旧
- **修复**: 
  1. 安装 conda 的 `libstdcxx-ng` 包
  2. 设置 `LD_LIBRARY_PATH` 使用 conda 环境的 C++ 库
- **Job**: 17461048 (v3)

---

## ✅ v3 修复方案

```bash
# 1. 安装新版 C++ 库
conda install -y libstdcxx-ng -c conda-forge

# 2. 强制使用 conda 的库
export LD_LIBRARY_PATH="/home/x_jiage/.conda/envs/diffsynth/lib:$LD_LIBRARY_PATH"

# 3. 验证库版本包含 GLIBCXX_3.4.29
strings ~/.conda/envs/diffsynth/lib/libstdc++.so.6 | grep GLIBCXX
```

---

## 📊 当前状态

| 版本 | Job ID | 状态 | 错误 |
|------|--------|------|------|
| v1 | 17460858 | FAILED | pandas/numpy 不兼容 |
| v2 | 17461003 | FAILED | GLIBCXX版本不足 |
| **v3** | **17461048** | **🟡 等待调度** | - |

---

## 🔍 根本问题分析

这是一个**依赖链问题**：

1. DiffSynth 需要 pandas（用于 UnifiedDataset）
2. 原 pandas 2.0.3 与 numpy 不兼容
3. 升级到 pandas 2.3.3 需要 numpy 2.2.6
4. numpy 2.2.6 需要 GLIBCXX_3.4.29
5. 系统 `/lib64/libstdc++.so.6` 版本太旧
6. 需要使用 conda 环境的新版 C++ 库

---

## 💡 经验教训

1. ❌ 不要孤立地升级单个包（pandas）
2. ✅ 升级后要检查整个依赖链
3. ✅ C++ 库版本是关键依赖
4. ✅ conda 环境需要正确配置 LD_LIBRARY_PATH

---

**下一步**: 等待 v3 启动，验证 C++ 库修复是否生效
