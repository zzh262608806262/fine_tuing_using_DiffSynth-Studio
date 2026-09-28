# Exp014 执行状态 - 基于 Exp012 成功配置

**更新时间**: 2026-09-04 13:46  
**状态**: ✅ Stage 1 + Stage 2 测试已提交

---

## 📊 作业状态

| Stage | Job ID | 状态 | 预计时间 | 说明 |
|-------|--------|------|----------|------|
| Stage 1: 擦除训练 | 17460630 | PD (排队中) | ~2 小时 | 600步，VU标准配置 |
| Stage 2: LoRA合并 | 17460631 | PD (依赖中) | ~30分钟 | 等待 Stage 1 完成 |

**当前状态**: 
- ✅ Stage 1 已进入队列 (状态: PD - Priority)
- ✅ Stage 2 已设置依赖 (状态: PD - Dependency)
- ⏳ 等待调度器分配 GPU 节点

---

## 🔧 修复内容

### 关键改进（基于 Exp012 成功经验）

1. **环境配置**：
   - ✅ 使用 VU uv venv 优先，conda 兜底
   - ✅ 完全照抄 exp012 的环境激活方式
   
2. **脚本调用**：
   - ✅ 使用正确的脚本名：`run_unlearn_wan5b.py`（不是 `unlearn_wan5b.py`）
   - ✅ 使用正确的脚本名：`merge_unlearn_lora_wan5b.py`（不是 `merge_lora_wan5b.py`）
   
3. **参数传递**：
   - ✅ 使用统一入口 `--vu-root`, `--train`, `--method`
   - ✅ 不是直接调用 VU 内部脚本

4. **配置对齐**：
   - UNLEARN_STEPS: 600 (VU标准，exp012是1200)
   - SAVE_EVERY: 100 (生成6个checkpoint)
   - METHOD: GradAscent
   - 模型/数据路径：完全复用 exp012

---

## 📝 监控命令

```bash
# 查看作业状态
squeue --me | grep exp014

# 实时监控（推荐）
watch -n 60 'squeue --me | grep exp014'

# 查看 Stage 1 日志（作业开始运行后）
tail -f slurm/logs/exp014_stage1_test-17460630.out

# 查看错误日志（如果失败）
tail -f slurm/logs/exp014_stage1_test-17460630.err
```

---

## 🎯 预期产物

### Stage 1 完成后

**目录**: `models/unlearn/exp014_wan5b_nudity_600steps/`

**文件**:
- `adapter_step_100.pt` (LoRA checkpoint @ 100步)
- `adapter_step_200.pt` (LoRA checkpoint @ 200步)
- `adapter_step_300.pt` (LoRA checkpoint @ 300步)
- `adapter_step_400.pt` (LoRA checkpoint @ 400步)
- `adapter_step_500.pt` (LoRA checkpoint @ 500步)
- `adapter_step_600.pt` (LoRA checkpoint @ 600步)
- `adapter_final.pt` (最终 LoRA，与 step_600 相同)
- `training_protocol.yaml` (训练配置记录)
- `training_trace.jsonl` (训练日志)

**大小**: 每个 LoRA checkpoint 约 10-20 MB

### Stage 2 完成后

**目录 1**: `models/unlearn/exp014_wan5b_nudity_600steps/merged/`
- Diffusers 格式的合并后 transformer
- `diffusion_pytorch_model-*.safetensors` (5个分片)
- `merge_report.json` (合并报告)

**目录 2**: `models/Wan-AI/Wan2.2-TI2V-5B-exp014-erased/`
- DiffSynth 格式的擦除后基座
- `wan_video_dit_*.safetensors` (多个文件)
- 软链: `models_t5_umt5-xxl-enc-bf16.safetensors`
- 软链: `Wan2.2_VAE.safetensors`

---

## ✅ 成功标志

### Stage 1 成功
- ✓ 生成 7 个文件（6个step + 1个final）
- ✓ 日志显示 "all stages done"
- ✓ 无错误信息

### Stage 2 成功
- ✓ merged 目录包含 5 个 safetensors 分片
- ✓ erased 目录包含 DiffSynth 格式的 DiT
- ✓ merge_report.json 显示 unmatched=0
- ✓ T5/VAE 软链正确

---

## 🔄 下一步

### 如果测试成功

1. **验证产物**：
   ```bash
   # 检查文件数量
   ls -lh models/unlearn/exp014_wan5b_nudity_600steps/
   ls -lh models/unlearn/exp014_wan5b_nudity_600steps/merged/
   ls -lh models/Wan-AI/Wan2.2-TI2V-5B-exp014-erased/
   ```

2. **继续完整流程**：
   - Stage 3: 微调（两臂：base_ft + erased_ft）
   - Stage 4: 量化（NF4，42个模型）
   - Stage 5: 生成评测视频
   - Stage 6: NudeNet 评估

3. **创建完整 pipeline**：
   - 基于测试脚本创建正式的 exp014 pipeline
   - 包含所有 6 个阶段

### 如果测试失败

1. **检查日志**：
   ```bash
   # 查看错误
   tail -100 slurm/logs/exp014_stage1_test-17460630.err
   
   # 查看完整输出
   cat slurm/logs/exp014_stage1_test-17460630.out
   ```

2. **对比 Exp012**：
   - 检查环境变量差异
   - 对比脚本调用方式
   - 验证路径配置

3. **逐步调试**：
   - 先确保环境检查通过
   - 单独测试 latent 缓存
   - 单独测试擦除训练

---

## 📚 相关文档

- 详细修复报告: `memory/exp014_fix_final.md`
- Exp012 成功记录: `memory/experiments.md` (第372-376行)
- 交互式调试指南: `memory/exp014_interactive_guide.md` (已过时，现用 sbatch)

---

## 💡 关键经验

1. **参考成功实验**：Exp012 是可靠的模板，直接复用其配置
2. **分阶段测试**：不要一次提交完整 pipeline，逐步验证
3. **环境管理规范**：VU 脚本用 VU 环境，避免依赖冲突
4. **使用统一入口**：`run_unlearn_wan5b.py` 封装了所有 VU 调用

---

**预计完成时间**: 2026-09-04 16:15 (约2.5小时后)
