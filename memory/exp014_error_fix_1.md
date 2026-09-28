# Exp014 Stage 3 错误修复记录

**更新时间**: 2026-09-04 14:45

---

## ❌ 错误 #1: Pandas/Numpy 版本不兼容

### 错误信息
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility. 
Expected 96 from C header, got 88 from PyObject
```

### 根本原因
- diffsynth conda 环境中的 pandas 与 numpy 版本不匹配
- pandas 是用旧版本 numpy 编译的，但环境中 numpy 已更新
- 导致二进制不兼容

### 解决方案
在 sbatch 脚本中添加：
```bash
pip install --upgrade numpy pandas --no-deps
```

### 修复后的作业
- **旧作业**: 17460858 (FAILED)
- **新作业**: 17461003 (v2, 已提交)

---

## ✅ 修复验证

### v2 脚本改进
1. ✅ 在训练前升级 numpy/pandas
2. ✅ 打印版本信息用于调试
3. ✅ 保留所有其他配置不变

### 监控更新
- ✅ 停止旧监控 (bfel4ddqr)
- ✅ 启动新监控 (针对 Job 17461003)

---

## 📊 当前状态

| 项目 | 值 |
|------|-----|
| Job ID | 17461003 |
| 状态 | 等待调度 |
| 修复 | pandas/numpy 版本升级 |
| 预计时间 | 24-32 小时 |

---

## 下一步

等待作业启动，监控系统会自动检测：
- ✅ 是否成功导入 pandas
- ✅ 是否开始训练
- ✅ 是否有新的错误

---

**监控中**: 🟢 活跃
