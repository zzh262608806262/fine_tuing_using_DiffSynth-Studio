# Exp 017 执行状态 v7

**最后更新**: 2026-09-03 18:54  
**版本**: v7 - 单GPU配置（无需 DeepSpeed）  
**状态**: 🟡 作业运行中，等待确认训练开始

---

## 📊 当前作业状态 (v7)

### Stage 1: 蒸馏训练

| ARM | Job ID | 状态 | 节点 | 开始时间 |
|-----|--------|------|------|----------|
| base | 17453822 | ⏳ **RUNNING** | node024 | 18:52 |
| grad_ascent | 17453823 | ⏳ **RUNNING** | node033 | 18:52 |
| esd | 17453824 | ⏳ **RUNNING** | node033 | 18:52 |

**状态**: 等待确认训练开始（监控器运行中）

### Stage 2-3: 等待依赖

- Stage 2: Jobs 17453825-17453827 (PENDING)
- Stage 3: Job 17453828 (PENDING)

---

## 🔧 完整修复历史

| 版本 | Job IDs | 问题 | 修复 | 状态 |
|------|---------|------|------|------|
| v1 | 17451254-260 | 账户错误 | 改为 Berzelius-2026-243 | ❌ |
| v2 | 17451262-268 | 缺少参数 | 添加 --use_gradient_checkpointing | ❌ |
| v3 | 17452966-972 | base 路径错误 | base 用 distill_wan5b.py | ❌ |
| v4 | 17453288-294 | 参数为空 | 在 case 前定义变量 | ❌ |
| v5 | 17453328-334 | diffusers格式 + ImportError | 统一用 distill_wan5b.py | ❌ |
| v6 | 17453814-820 | DeepSpeed 缺失 | （未完成修复就取消） | ❌ |
| **v7** | **17453822-828** | **需要单GPU配置** | **创建 accelerate_config_single_gpu.yaml** | ⏳ **运行中** |

---

## 🛠️ 环境修复详情

### 问题1: transformers/torch 版本不兼容

**症状**: `ImportError: cannot import name 'NP_SUPPORTED_MODULES'`

**原因**: 
- torch 2.13.0（新开发版）
- transformers 5.14.1（期望旧版 torch API）
- tokenizers 0.22.2（与 transformers 4.x 不兼容）

**修复**:
```bash
conda activate diffsynth
pip install transformers==4.44.0 --no-deps
pip install 'tokenizers>=0.19,<0.20'
```

**验证**: ✅ `python scripts/test_env.py` 通过

### 问题2: DeepSpeed 缺失

**症状**: `ImportError: DeepSpeed is not installed`

**原因**: 默认 accelerate 配置使用 DeepSpeed Zero2

**修复**: 创建单GPU配置
```yaml
# examples/wanvideo/model_training/full/accelerate_config_single_gpu.yaml
distributed_type: NO
num_processes: 1
mixed_precision: bf16
```

---

## 📁 新增文件

1. **scripts/test_env.py** - 环境测试脚本
2. **scripts/exp017_interactive_test.sh** - 交互式测试脚本（备用）
3. **accelerate_config_single_gpu.yaml** - 单GPU配置
4. **memory/exp017_environment_issue.md** - 环境问题详细文档

---

## ⏱️ 预计时间线

- **18:52** ✅ v7 作业开始运行
- **18:55** - 预计训练开始（模型加载）
- **次日 03:00-05:00** - Stage 1 完成（8-10h）
- **次日 05:00-06:30** - Stage 2 完成（+1.5h）
- **次日 06:30-07:30** - Stage 3 完成（+1h）

**总预计时长**: ~11-13 小时

---

## 📖 监控

### 自动监控
- Monitor task b5wjfxpao: 每30秒检查训练进度

### 手动检查
```bash
# 查看作业
squeue -u $USER | grep exp017

# 查看日志
tail -f slurm/logs/exp017_distill_17453822_*.out  # base
tail -f slurm/logs/exp017_distill_17453823_*.out  # grad_ascent
tail -f slurm/logs/exp017_distill_17453824_*.out  # esd
```

---

## ✅ 经验总结

### 环境管理教训
1. **版本固定**: torch/transformers 应该在 requirements.txt 中固定版本
2. **定期备份**: `conda env export > environment_working.yml`
3. **依赖检查**: accelerate 配置的依赖（DeepSpeed）应提前验证

### 调试流程优化
1. **环境测试优先**: 在登录节点先修复环境问题
2. **配置简化**: 单GPU训练不需要 DeepSpeed
3. **快速迭代**: sbatch 提交 → 检查日志 → 修复 比交互式节点更高效

---

**下一步**: 等待监控器确认训练开始，然后等待 8-10 小时完成蒸馏。
