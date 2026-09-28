# Exp016 执行进度报告

**更新时间**: 2026-09-03 16:18

---

## 总览

Exp016 旨在对 Exp015 的 8 个模型臂进行 NF4 量化评估。

**当前状态**: Stage 1 完成 ✅ | Stage 2 运行中 ⏳ | Stage 3 待启动

---

## Stage 1: 模型量化 ✅ COMPLETED

### 成功完成 (v8)

经过 7 次失败调试，最终在 v8 成功完成全部 8 个臂的量化：

| Job ID | 臂 | 状态 | 耗时 | 模型大小 |
|--------|---|------|------|----------|
| 17451853 | esd_erased | ✅ COMPLETED | 1m18s | 2.5GB |
| 17451854 | esd_base | ✅ COMPLETED | 2m44s | 2.5GB |
| 17451855 | npo_erased | ✅ COMPLETED | 2m03s | 2.5GB |
| 17451856 | npo_base | ✅ COMPLETED | 2m43s | 2.5GB |
| 17451857 | grad_ascent_erased | ✅ COMPLETED | 1m18s | 2.5GB |
| 17451858 | grad_ascent_base | ✅ COMPLETED | 1m18s | 2.5GB |
| 17451859 | anchor_distill_erased | ✅ COMPLETED | 1m18s | 2.5GB |
| 17451860 | anchor_distill_base | ✅ COMPLETED | 1m40s | 2.5GB |

**关键成功要素**:
1. 使用 `model_id` + `DIFFSYNTH_SKIP_DOWNLOAD=true` 处理分片模型
2. 只 `load_lora()`，不 merge
3. 使用 `quant_config.quantize_model(pipe.dit, ...)` API
4. 使用 `sed` 替换模板变量

**产物**: `models/quantized/exp016_*_nf4/`（每个 ~2.5GB，相比原始 ~10GB 节省 75%）

**详细调试记录**: 见 `docs/exp016_quantization_debugging.md`

---

## Stage 2: 视频生成 ⏳ RUNNING (v3)

### 当前状态 (Jobs 17451905-17451912)

| Job ID | 臂 | 状态 | 运行时间 |
|--------|---|------|----------|
| 17451905 | esd_erased_nf4 | ✅ RUNNING | 0:48 |
| 17451906 | esd_base_nf4 | ✅ RUNNING | 0:16 |
| 17451907 | npo_erased_nf4 | ✅ RUNNING | 0:16 |
| 17451908 | npo_base_nf4 | ✅ RUNNING | 0:16 |
| 17451909 | grad_ascent_erased_nf4 | ✅ RUNNING | 0:16 |
| 17451910 | grad_ascent_base_nf4 | ⏳ PENDING | - |
| 17451911 | anchor_distill_erased_nf4 | ⏳ PENDING | - |
| 17451912 | anchor_distill_base_nf4 | ⏳ PENDING | - |

**配置**:
- 每臂生成 99 条 nudity domain 视频
- 评测集: VU benchmark_wan.jsonl
- 输出: `outputs/exp016/<arm>/`
- 预计耗时: 1-2 小时/臂

**历史失败**:
- v1 (Jobs 17451872-17451881): 模块加载冲突
- v2 (Jobs 17451882-17451890): conda 环境路径错误

**修复**: 使用 `conda activate main`（与量化任务一致）

---

## Stage 3: 评估 📋 READY

### 准备就绪

**脚本**: `scripts/exp016_evaluate.py`
- NudeNet: violation_rate, frame_nudity_rate (threshold=0.6)
- 安全分类器: porn 检出率 (threshold=0.2/0.3/0.5)
- 对比报告: 量化前后性能差异

**提交脚本**: `scripts/submit_exp016_eval_all.sh`

**输出**:
- 评估结果: `outputs/exp016_evaluation/<arm>_evaluation.json`
- 对比报告: `outputs/exp016_evaluation/quantization_comparison.md`

**预计耗时**: ~30分钟/臂

---

## 下一步

1. ⏳ 等待 Stage 2 生成完成（预计 1-2 小时）
2. 📋 提交 Stage 3 评估任务: `bash scripts/submit_exp016_eval_all.sh`
3. 📊 分析量化前后的性能差异

---

## 文件清单

### 脚本
- `scripts/exp016_quantize.py` - 量化脚本
- `scripts/exp016_generate_videos.py` - 生成脚本
- `scripts/exp016_evaluate.py` - 评估脚本
- `scripts/submit_exp016_all.sh` - 批量提交量化
- `scripts/submit_exp016_gen_all.sh` - 批量提交生成
- `scripts/submit_exp016_eval_all.sh` - 批量提交评估

### SLURM 模板
- `slurm/exp016_quant_template.sbatch` - 量化作业模板
- `slurm/exp016_gen_template.sbatch` - 生成作业模板
- `slurm/exp016_eval_template.sbatch` - 评估作业模板

### 文档
- `docs/exp016_quantization_debugging.md` - 量化调试完整记录
- `memory/experiments.md` - 实验记录（包含 Exp016）

---

## 监控命令

```bash
# 查看生成任务状态
squeue -u x_jiage | grep exp016-gen

# 查看最新日志
tail -f slurm/logs/exp016_gen_esd_erased_nf4_17451905.out

# 检查生成进度
ls -lh outputs/exp016/esd_erased_nf4/*.mp4 | wc -l
```
