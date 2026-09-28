# Exp016 量化评估 - 工作总结

**会话时间**: 2026-09-03 后台任务  
**状态**: Stage 1 完成 ✅ | Stage 2 运行中 ⏳ | Stage 3 就绪 📋

---

## 任务概述

对 Exp015 的 8 个模型臂（4种擦除方法 × 2版本）进行 NF4 量化，评估量化对安全性能的影响。

---

## 完成工作

### ✅ Stage 1: 模型量化（已完成）

**成果**:
- 8个量化模型全部成功保存
- 每个模型 ~2.5GB（原始 ~10GB，节省 75%）
- 位置: `models/quantized/exp016_*_nf4/`

**调试过程**:
- 经历 7 次失败（v1-v7）
- 核心问题：量化 API 理解、分片模型加载、LoRA 处理
- 最终方案：
  1. 使用 `model_id` + `DIFFSYNTH_SKIP_DOWNLOAD=true` 加载分片模型
  2. 只 `load_lora()`，不 merge
  3. 正确的量化 API：`quant_config.quantize_model(pipe.dit, ...)`
  4. 使用 `sed` 替换 sbatch 模板变量

**详细记录**: `docs/exp016_quantization_debugging.md`

**Jobs**: 17451853-17451860（全部 COMPLETED）

---

### ⏳ Stage 2: 视频生成（运行中）

**当前状态**:
- **Jobs**: 17452120-17452127（v4 尝试）
- **进度**: 1/8 运行中，7/8 排队
- **启动时间**: 2026-09-03 16:24
- **预计完成**: 1-2 小时后

**配置**:
- 每臂生成 99 条 nudity domain 视频
- 评测集: VU benchmark_wan.jsonl
- 输出: `outputs/exp016/<arm>/`

**调试过程**:
- v1: 模块加载冲突（Miniforge3 vs Mambaforge）
- v2: conda 环境路径错误
- v3: 量化模型加载方式错误（无法识别模型类型）
- v4: **正确方案** - 先加载原始结构，再应用量化并加载量化权重

**关键修复** (v4):
```python
# 1. 加载原始模型结构
pipe = WanVideoPipeline.from_pretrained(model_configs=原始配置)

# 2. 应用量化（创建量化结构）
quant_config.quantize_model(pipe.dit, ...)

# 3. 加载量化权重
quantized_state = load_file(量化safetensors)
pipe.dit.load_state_dict(quantized_state, strict=False)
```

**监控**: 持久监控已启动（任务 bdo1cxbqk，2小时超时）

---

### 📋 Stage 3: 评估（已就绪）

**准备完成**:
- 评估脚本: `scripts/exp016_evaluate.py`
- 提交脚本: `scripts/submit_exp016_eval_all.sh`
- SLURM 模板: `slurm/exp016_eval_template.sbatch`

**评估指标**:
1. NudeNet: violation_rate, frame_nudity_rate (threshold=0.6)
2. 安全分类器: porn 检出率 (threshold=0.2/0.3/0.5)
3. 对比报告: 量化前后性能差异

**输出**:
- `outputs/exp016_evaluation/<arm>_evaluation.json`（8个）
- `outputs/exp016_evaluation/quantization_comparison.md`（对比报告）

**下一步**: 等待生成完成后运行 `bash scripts/submit_exp016_eval_all.sh`

---

## 文件清单

### 脚本
- ✅ `scripts/exp016_quantize.py` - 量化脚本
- ✅ `scripts/exp016_generate_videos.py` - 生成脚本
- ✅ `scripts/exp016_evaluate.py` - 评估脚本
- ✅ `scripts/submit_exp016_all.sh` - 批量提交量化
- ✅ `scripts/submit_exp016_gen_all.sh` - 批量提交生成
- ✅ `scripts/submit_exp016_eval_all.sh` - 批量提交评估

### SLURM 模板
- ✅ `slurm/exp016_quant_template.sbatch`
- ✅ `slurm/exp016_gen_template.sbatch`
- ✅ `slurm/exp016_eval_template.sbatch`

### 文档
- ✅ `docs/exp016_quantization_debugging.md` - 量化调试记录（v1-v8）
- ✅ `docs/exp016_progress_report.md` - 进度报告
- ✅ `memory/experiments.md` - 实验记录更新

---

## 关键经验

### 量化模型处理
1. **保存**: 只保存量化后的 DiT state_dict
2. **加载**: 不能直接用 ModelConfig(path=...)，需要：
   - 先加载原始模型结构
   - 应用量化配置（创建量化层结构）
   - 加载量化权重
3. **推理**: 量化模型可以正常推理，无需额外处理

### 环境配置
- 使用 `conda activate main`（不是 mamba）
- 不需要 `module load`（conda 已包含）

### 调试策略
- 检查成功的参考脚本（如量化脚本）
- 逐步验证每个组件
- 保存详细的错误记录

---

## 监控命令

```bash
# 查看生成任务状态
squeue -j 17452120,17452121,17452122,17452123,17452124,17452125,17452126,17452127

# 检查最新日志
tail -f slurm/logs/exp016_gen_esd_erased_nf4_17452120.out

# 查看生成进度
ls outputs/exp016/*/*.mp4 | wc -l

# 等生成完成后提交评估
bash scripts/submit_exp016_eval_all.sh
```

---

## 待办事项

- [ ] 等待 Stage 2 生成完成（监控中，预计1-2小时）
- [ ] 提交 Stage 3 评估任务
- [ ] 分析量化前后性能差异
- [ ] 撰写 Exp016 完整报告

---

## 产物位置

```
models/quantized/
├── exp016_esd_erased_nf4/           (2.5GB) ✅
├── exp016_esd_base_nf4/             (2.5GB) ✅
├── exp016_npo_erased_nf4/           (2.5GB) ✅
├── exp016_npo_base_nf4/             (2.5GB) ✅
├── exp016_grad_ascent_erased_nf4/   (2.5GB) ✅
├── exp016_grad_ascent_base_nf4/     (2.5GB) ✅
├── exp016_anchor_distill_erased_nf4/(2.5GB) ✅
└── exp016_anchor_distill_base_nf4/  (2.5GB) ✅

outputs/exp016/
├── esd_erased_nf4/                  (生成中...)
├── esd_base_nf4/                    (排队)
├── npo_erased_nf4/                  (排队)
├── npo_base_nf4/                    (排队)
├── grad_ascent_erased_nf4/          (排队)
├── grad_ascent_base_nf4/            (排队)
├── anchor_distill_erased_nf4/       (排队)
└── anchor_distill_base_nf4/         (排队)

outputs/exp016_evaluation/           (待生成)
```
