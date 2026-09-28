# Exp019 蒸馏修复完整记录

**日期**: 2026-09-17  
**状态**: 第3次提交，等待验证

---

## 🔄 失败与修复历程

### 第1次失败 (Job 17536559)
- **错误**: `KeyError: 'seed'`
- **原因**: 数据集缺少蒸馏任务需要的字段
- **修复**: 创建 `metadata_100_distill.csv`，添加 `seed, rand_device, num_inference_steps, cfg_scale`

### 第2次失败 (Job 17539922)
- **错误**: `CUDA out of memory` (40GB GPU不够)
- **原因**: 分配到A100-40GB，但需要>40GB显存
- **修复**: 添加 `gradient_accumulation_steps=4` 减少显存占用

### 第3次提交 (Job 17539949) ⏳
- **配置**: metadata_100_distill.csv + gradient_accumulation_steps=4
- **状态**: 等待调度和验证

---

## ✅ 冒烟测试经验

**成功配置** (node072, A100-80GB):
- 数据集: metadata_100_distill.csv (含seed等字段)
- 配置: mixed_precision='no' (避免BFloat16+fp16冲突)
- 结果: 3条数据训练成功，生成epoch-0.safetensors (9.4GB)

**正式作业差异**:
- 数据量: 3条 → 100条 × repeat 10 = 1000条
- GPU: 80GB → **40GB** (分配到不同节点)
- 显存: 冒烟测试约30GB → 正式训练>40GB

---

## 🔧 当前配置 (Job 17539949)

```bash
--gradient_accumulation_steps 4  # 新增，减少显存
--dataset_repeat 10
--num_epochs 10
--use_gradient_checkpointing
--model_paths (擦除后DiT + T5 + VAE)
--extra_inputs seed,rand_device
```

**预期效果**: 
- 显存占用: ~40GB → ~30GB (减少25%)
- 训练速度: 不变 (梯度累积4步后更新一次)

---

## 📊 三次修复的根本问题

### 问题1: 数据格式不一致
- 假设: Exp017和Exp019用同样的数据格式
- 现实: Exp017的数据集有额外字段
- 教训: 必须检查参考案例的实际数据格式

### 问题2: 冒烟测试环境不代表
- 假设: 冒烟测试成功 = 正式作业成功
- 现实: 冒烟测试在80GB GPU，正式作业可能分配40GB
- 教训: 冒烟测试应该模拟最坏情况（40GB GPU）

### 问题3: 显存需求未评估
- 假设: 有GPU就能运行
- 现实: 5B模型 + 480×736×17视频 需要大显存
- 教训: 提交前应该估算显存需求

---

## 💡 改进建议

### 1. 冒烟测试应该在40GB GPU上运行
```bash
# 申请40GB GPU进行冒烟测试
srun --gpus=1 --constraint="GPUMEM40GB" bash test_exp019_distill_smoketest.sh
```

### 2. 添加显存监控
```python
# 在训练脚本中添加
import torch
print(f"Peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
```

### 3. 自动调整gradient_accumulation_steps
```python
# 根据GPU显存自动设置
gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
if gpu_memory_gb < 50:
    gradient_accumulation_steps = 4
```

---

## 🎯 下一步

1. **监控Job 17539949**: 等待启动并检查是否成功
2. **如果再次OOM**: 
   - 选项A: 增加gradient_accumulation_steps到8
   - 选项B: 减小batch size
   - 选项C: 申请80GB GPU (但可能等很久)
3. **如果成功**: 更新冒烟测试脚本，加入显存优化配置

---

**当前状态**: Job 17539949等待调度中
