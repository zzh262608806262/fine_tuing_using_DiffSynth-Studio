# Exp014 修复报告 - 基于 Exp012 成功经验

**日期**: 2026-09-04  
**状态**: 已创建测试脚本，等待提交

---

## 🔍 问题诊断

### 原因分析

通过对比 **Exp012（成功）** vs **Exp014（失败）**，找到了根本原因：

| 问题 | Exp014（错误） | Exp012（正确） |
|------|----------------|----------------|
| 脚本名称 | `unlearn_wan5b.py` | `run_unlearn_wan5b.py` |
| 脚本名称 | `merge_lora_wan5b.py` | `merge_unlearn_lora_wan5b.py` |
| 参数方式 | 命令行直接传参 | 环境变量 + 统一入口 |
| 环境管理 | 混乱 | VU venv 优先，conda 兜底 |

**核心问题**：Exp014 的 pipeline 脚本是从旧版本复制的，包含过时的脚本名和参数。

---

## ✅ 解决方案

### 采用 Exp012 的成功模式

1. **环境配置**（完全照抄）：
   ```bash
   # 优先使用 VU 的 uv venv
   source /home/x_jiage/jiage/video-unlearning/.venv/bin/activate
   # 找不到时兜底 conda
   conda activate diffsynth
   ```

2. **脚本调用**（使用统一入口）：
   ```bash
   # 通过 run_unlearn_wan5b.py 统一入口调用
   python3 scripts/run_unlearn_wan5b.py \
       --vu-root "$VU_REPO" \
       --train \
       --method GradAscent \
       --unlearn-steps 600 \
       --model-path "$WAN_MODEL_PATH"
   ```

3. **参数传递**（环境变量）：
   ```bash
   METHOD="GradAscent"
   UNLEARN_STEPS=600
   SAVE_EVERY=100
   ```

---

## 📋 创建的测试脚本

### 1. Stage 1: 擦除训练
- **脚本**: `slurm/exp014_stage1_test.sbatch`
- **功能**: 
  - 环境检查（Python、脚本、模型、缓存）
  - 600步擦除训练（VU标准配置）
  - 生成 6 个 checkpoint（每100步）
- **时间**: 约 2 小时

### 2. Stage 2: LoRA 合并
- **脚本**: `slurm/exp014_stage2_test.sbatch`
- **功能**:
  - 合并擦除 LoRA
  - 桥转换到 DiffSynth 格式
  - 补齐 T5/VAE 软链
- **时间**: 约 30 分钟
- **依赖**: Stage 1 必须成功

### 3. 提交脚本
- **脚本**: `slurm/exp014_submit_test.sh`
- **功能**: 一键提交两个阶段，自动设置依赖关系

---

## 🎯 关键改进

### 与 Exp012 对齐的配置

| 配置项 | Exp014 | Exp012 | 说明 |
|--------|--------|--------|------|
| 擦除步数 | 600 | 1200 | VU 标准 vs 2倍量 |
| 保存间隔 | 100 | 200 | 生成更多 checkpoint |
| 环境管理 | ✅ 统一 | ✅ 统一 | VU venv 优先 |
| 脚本入口 | ✅ 修复 | ✅ 正确 | `run_unlearn_wan5b.py` |
| 参数传递 | ✅ 修复 | ✅ 正确 | 环境变量 |

### 测试特点

1. **分阶段测试**: 先测试前两个阶段，验证配置正确
2. **自动检查**: 每个阶段都有前置检查和产物验证
3. **详细日志**: 输出清晰的进度信息和错误提示
4. **依赖管理**: 使用 `--dependency=afterok` 确保顺序

---

## 🚀 提交命令

```bash
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 提交测试流程（Stage 1 + Stage 2）
bash slurm/exp014_submit_test.sh
```

提交后会输出：
- Job ID（用于监控）
- 日志文件路径
- 监控命令

---

## 📊 预期结果

### Stage 1 产物
```
models/unlearn/exp014_wan5b_nudity_600steps/
├── adapter_step_100.pt
├── adapter_step_200.pt
├── adapter_step_300.pt
├── adapter_step_400.pt
├── adapter_step_500.pt
├── adapter_step_600.pt
└── adapter_final.pt
```

### Stage 2 产物
```
models/unlearn/exp014_wan5b_nudity_600steps/merged/
├── diffusion_pytorch_model-00001-of-00005.safetensors
├── ...
└── merge_report.json

models/Wan-AI/Wan2.2-TI2V-5B-exp014-erased/
├── wan_video_dit_*.safetensors
├── models_t5_umt5-xxl-enc-bf16.safetensors -> (软链)
└── Wan2.2_VAE.safetensors -> (软链)
```

---

## 📝 监控和调试

### 监控命令
```bash
# 查看作业状态
squeue --me

# 实时监控
watch -n 60 'squeue --me'

# 查看日志（替换 <JOB_ID> 为实际的 Job ID）
tail -f slurm/logs/exp014_stage1_test-<JOB_ID>.out
tail -f slurm/logs/exp014_stage2_test-<JOB_ID>.out
```

### 如果失败
1. 查看错误日志：`slurm/logs/exp014_stage*_test-*.err`
2. 检查前置条件：模型、缓存、环境
3. 对比 exp012 的成功日志

---

## 🔄 后续步骤

如果 Stage 1 + Stage 2 测试成功：

1. **验证产物**：
   - 检查 checkpoint 数量和大小
   - 验证合并报告
   - 确认软链正确

2. **继续完整 pipeline**：
   - Stage 3: 微调（repeat=25, epochs=20）
   - Stage 4: 量化（NF4）
   - Stage 5: 评估（NudeNet）

3. **更新正式脚本**：
   - 将测试脚本的配置应用到正式 pipeline
   - 提交完整的 exp014 流程

---

## 💡 经验总结

### 为什么 Exp012 能跑通？

1. **使用统一入口**：`run_unlearn_wan5b.py` 封装了所有 VU 调用
2. **环境隔离清晰**：VU venv 优先，避免依赖冲突
3. **脚本命名准确**：没有使用过时的脚本名
4. **参数传递规范**：通过环境变量，不是硬编码

### 教训

1. ❌ **不要复制旧版本脚本**：可能包含过时的配置
2. ✅ **参考最近成功的实验**：Exp012 是可靠的模板
3. ✅ **分阶段测试**：不要一次提交完整 pipeline
4. ✅ **环境管理规范**：VU 脚本用 VU 环境，FT 脚本用 FT 环境
