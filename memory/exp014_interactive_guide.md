# Exp014 交互式调试指南

**创建时间**: 2026-09-03  
**状态**: 准备就绪

---

## 问题背景

Exp014 的 pipeline 脚本在 sbatch 提交时反复失败，主要问题：
1. 脚本名称不匹配（旧版本的脚本名）
2. 参数名称可能有变化
3. 路径配置可能不一致

经过3次失败尝试后，决定采用**交互式节点调试**策略。

---

## 交互式调试步骤

### Step 1: 申请 GPU 节点

```bash
# 申请一个 80GB GPU 节点（Exp014 需要大显存）
init1gf

# 查看申请的节点
squeue --me
```

记录节点名称和 Job ID（例如：node084, Job ID 17452500）

### Step 2: 进入 GPU 节点

```bash
# 假设申请到的节点是 node084
jobsh node084
```

或使用：
```bash
# 假设 Job ID 是 17452500
srun --overlap --jobid=17452500 --pty bash
```

### Step 3: 运行测试脚本

在 GPU 节点上运行：

```bash
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
bash scripts/exp014_interactive_test.sh
```

这个脚本会：
- ✓ 检查环境（Python、脚本、模型、缓存）
- ✓ 测试擦除训练（仅10步，约5-10分钟）
- ✓ 显示合并脚本的帮助信息（用于确认参数）

### Step 4: 根据测试结果调整

#### 如果阶段1（擦除训练）失败：

检查错误信息，常见问题：
- 模型路径不对
- 缓存视频缺失
- 环境依赖缺失

#### 如果阶段1成功：

会生成测试产物在 `models/unlearn/exp014_test_10steps/`

然后手动测试阶段2（合并），根据帮助信息调整参数：

```bash
# 查看合并脚本的正确参数
python scripts/merge_unlearn_lora_wan5b.py --help

# 根据帮助信息调整命令
python scripts/merge_unlearn_lora_wan5b.py \
    --adapter "models/unlearn/exp014_test_10steps/adapter_final.pt" \
    <正确的参数...>
```

### Step 5: 验证完整后更新 pipeline

当所有阶段在交互式节点上都能成功运行后：
1. 记录正确的命令和参数
2. 更新 `slurm/exp014_wan5b_pipeline.sbatch`
3. 重新提交 sbatch

---

## 测试脚本说明

`scripts/exp014_interactive_test.sh` 做了以下简化：
- **擦除步数**: 600 → 10（快速验证，避免等4小时）
- **保存间隔**: 100 → 5
- **输出目录**: 使用 `exp014_test_` 前缀（不污染正式实验）

如果10步测试通过，说明配置正确，只需改回600步即可。

---

## 常见问题排查

### Q1: `run_unlearn_wan5b.py` 的参数有哪些？

在节点上运行：
```bash
python scripts/run_unlearn_wan5b.py --help
```

### Q2: 缓存视频路径对吗？

检查：
```bash
ls -R data/wan5b/unlearn_baseline/
```

应该看到 `nudity_forget/` 和 `nudity_retain/` 两个目录。

### Q3: 基座模型在哪里？

检查：
```bash
ls -lh models/Wan-AI/Wan2.2-TI2V-5B-Diffusers/transformer/
```

应该看到 `diffusion_pytorch_model-*.safetensors` 文件。

---

## 完成标准

交互式调试成功的标志：
1. ✅ 擦除训练10步能跑通（生成 adapter_*.pt）
2. ✅ LoRA 合并能跑通（生成 merged 模型）
3. ✅ 记录了正确的命令和参数

达到以上标准后，可以：
- 更新 pipeline 脚本
- 提交正式的 600步擦除训练
- 继续后续的微调、量化、评估阶段

---

## 资源释放

测试完成后，记得释放 GPU 节点：

```bash
# 退出节点
exit

# 释放节点（使用申请时的 Job ID）
scancel <Job_ID>
```

---

## 下一步

1. **立即行动**: 申请节点并运行测试脚本
2. **记录结果**: 将测试输出保存到 `logs/exp014_interactive_test.log`
3. **反馈问题**: 如遇到新错误，记录完整的错误信息
4. **更新 pipeline**: 成功后更新正式的 sbatch 脚本

---

**重要提醒**:
- 公用集群，注意其他用户的作业
- 只进入自己申请的节点
- 测试完成后及时释放资源
