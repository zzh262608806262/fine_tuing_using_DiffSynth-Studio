# Exp 015 快速参考卡片

## 🎯 一句话概括
对比 VU 的 4 种擦除方法（GradDiff/ESD/NPO/AnchorDistill）在擦除后微调场景下的安全鲁棒性

## 📍 当前状态
✅ **Stage 1-3 已就绪**（擦除、合并、微调）  
⏳ Stage 4-6 待开发（生成、评估、分析）

## ⚡ 快速启动

```bash
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 检查前置条件
bash scripts/exp015_check_prerequisites.sh

# 启动 Stage 1（4 个擦除训练作业并行）
bash slurm/exp015_submit.sh
```

## 📊 监控命令

```bash
# 查看作业状态
squeue -u $USER

# 查看作业记录
cat slurm/logs/exp015_pipeline_jobs.tsv

# 实时查看日志
tail -f slurm/logs/exp015_unlearn_grad_diff-<JOB_ID>.out
```

## 🔗 关键路径

| 类型 | 路径 |
|------|------|
| 实验记录 | `memory/experiments.md` → Exp 015 |
| 使用指南 | `docs/exp015_README.md` |
| 状态报告 | `docs/exp015_status_report.md` |
| 驱动脚本 | `scripts/run_unlearn_wan5b_multi_method.py` |
| 提交脚本 | `slurm/exp015_submit.sh` |

## 🎨 方法对比

| 方法 | 核心思想 | 特殊参数 | 需要 retain |
|------|----------|----------|-------------|
| GradAscent | loss = -l_f | - | ❌ |
| GradDiff | loss = l_f - λ·l_r | retain_weight=1.0 | ✅ |
| ESD | 负引导擦除 | negative_guidance=1.0 | ❌ |
| NPO | 负偏好优化 | beta=0.1 | ❌ |
| AnchorDistill | 蒸馏到替代词 | anchor="a person" | ❌ |

## 📦 预期产物

```
models/unlearn/exp015_wan5b_nudity_<method>/final/        # 擦除 LoRA
models/wan5b/exp015_<method>_erased/                       # 擦除后基座
models/finetune/exp015_<method>_erased_lora10e2/epoch-{0,1}  # 微调 LoRA
```

## ⏱️ 预估时间

- Stage 1: 2-4h/方法（4 个并行）
- Stage 2: ~1h/方法（4 个并行）
- Stage 3: ~2h/方法（4 个并行）

## 🔧 故障排查

### 擦除训练失败
```bash
# 检查 latent 缓存
ls data/wan5b/unlearn_latents/latents/*.pt | wc -l  # 应该 ≥25

# 查看错误日志
cat slurm/logs/exp015_unlearn_<method>-<JOB_ID>.err
```

### LoRA 合并失败
```bash
# 检查擦除 LoRA 是否生成
ls models/unlearn/exp015_wan5b_nudity_<method>/final/
```

### 微调失败
```bash
# 检查擦除后基座
ls models/wan5b/exp015_<method>_erased/
```

## 📝 手动提交（如果不用自动脚本）

```bash
# Stage 1: 擦除训练（4 个方法并行）
sbatch slurm/exp015_unlearn_grad_diff.sbatch
sbatch slurm/exp015_unlearn_esd.sbatch
sbatch slurm/exp015_unlearn_npo.sbatch
sbatch slurm/exp015_unlearn_anchor_distill.sbatch

# Stage 2: LoRA 合并（等 Stage 1 完成）
for method in grad_diff esd npo anchor_distill; do
    sbatch slurm/exp015_merge_${method}.sbatch
done

# Stage 3: 微调（等 Stage 2 完成）
for method in grad_diff esd npo anchor_distill; do
    sbatch slurm/exp015_finetune_${method}.sbatch
done
```

## 💡 关键设计

1. **复用 Exp 012**: base/latent/训练数据全部复用
2. **统一配置**: rank=8, alpha=16, steps=600
3. **方法参数化**: 便于扩展和对比
4. **阶段解耦**: 每个 stage 独立，便于调试

## 🎯 预期发现

| 维度 | 最强 → 最弱 |
|------|-------------|
| 擦除深度 | ESD > GradAscent > NPO > GradDiff > AnchorDistill |
| 微调鲁棒性 | GradDiff > NPO > AnchorDistill > GradAscent > ESD |

## 📞 下一步

1. ✅ 启动 Stage 1
2. ⏳ 等待完成后启动 Stage 2
3. ⏳ 同时开发 Stage 4-6 脚本

---

**最后更新**: 2026-08-31  
**状态**: ✅ 所有前置条件满足，可以启动
