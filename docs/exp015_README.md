# Exp 015: 多方法擦除对比实验

## 实验目标

系统对比 VU 项目的 5 种擦除方法在"擦除后微调"场景下的安全能力保持性：
1. **GradAscent** (baseline, Exp 012): loss = -l_f
2. **GradDiff**: loss = l_f - λ·l_r (需要 retain set)
3. **ESD**: 负引导擦除, negative_guidance=1.0
4. **NPO**: 负偏好优化, beta=0.1
5. **AnchorDistill**: 蒸馏到替代词, anchor="a person"

## 研究问题

1. 哪种擦除方法最能抵抗微调后的安全回潮？
2. 不同方法的擦除深度与微调敏感度关系如何？
3. 量化后安全性能是否保持一致？

## 实验设计

**5种方法 × 4臂设计** = 20条流水线（复用 Exp 012 的 base/base_ft）

每种方法的 4 臂：
- `base`: 未擦除基座（复用 Exp 012）
- `<method>_erased`: 该方法擦除后
- `base_ft`: 未擦除 + 微调（复用 Exp 012）
- `<method>_erased_ft`: 该方法擦除 + 微调

## 前置条件

1. **Exp 012 已完成**（复用其资源）:
   - base 桥产物: `models/Wan-AI/Wan2.2-TI2V-5B/`
   - latent 缓存: `data/wan5b/unlearn_latents/`
   - base/base_ft 的生成视频和评估结果

2. **环境准备**:
   - VU 环境（擦除训练）: `.venv` 或 `conda activate video-unlearning`
   - DiffSynth 环境（微调、生成）: `conda activate diffsynth`

3. **数据资产**:
   - 5B diffusers 基座: `models/Wan-AI/Wan2.2-TI2V-5B-Diffusers/`
   - 基线视频: `data/wan5b/unlearn_baseline/`
   - 训练数据: `data/tiger_dataset/metadata_100.csv`

## 流程说明

### Stage 1: 擦除训练（4个新方法并行）

```bash
# 提交 4 个擦除训练作业（并行执行）
bash slurm/exp015_submit.sh
```

或手动提交单个方法：
```bash
sbatch slurm/exp015_unlearn_grad_diff.sbatch
sbatch slurm/exp015_unlearn_esd.sbatch
sbatch slurm/exp015_unlearn_npo.sbatch
sbatch slurm/exp015_unlearn_anchor_distill.sbatch
```

**产物**:
- `models/unlearn/exp015_wan5b_nudity_<method>/final/` (4个)

**监控**:
```bash
# 查看作业状态
squeue -u $USER

# 查看日志
tail -f slurm/logs/exp015_unlearn_grad_diff-<job_id>.out
```

### Stage 2: LoRA 合并（4个方法并行，依赖 Stage 1）

```bash
# 等待 Stage 1 完成后，提交合并作业
for method in grad_diff esd npo anchor_distill; do
    sbatch slurm/exp015_merge_${method}.sbatch
done
```

**产物**:
- `models/wan5b/exp015_<method>_erased/` (4个 DiffSynth 格式基座)

### Stage 3: 微调（4个方法并行，依赖 Stage 2）

```bash
# 等待 Stage 2 完成后，提交微调作业
for method in grad_diff esd npo anchor_distill; do
    sbatch slurm/exp015_finetune_${method}.sbatch
done
```

**产物**:
- `models/finetune/exp015_<method>_erased_lora10e2/epoch-{0,1}` (4个)

### Stage 4: 生成评测视频（待实现）

为每个方法的每个臂生成 99 条评测视频（VU benchmark_wan.jsonl nudity 类）

### Stage 5: 评估（待实现）

使用 NudeNet + 安全分类器对所有方法所有臂进行评估

### Stage 6: 对比分析（待实现）

生成多方法对比报告

## 统一配置（确保公平对比）

### 擦除参数（所有方法）
- LoRA: rank=8, alpha=16.0
- 训练: steps=600, batch_size=1, lr=1e-5
- 层范围: layer 0-29（transformer 全层）
- 采样: guidance_scale=1.0

### 微调参数（所有方法，与 Exp 012 一致）
- dataset_repeat=10, num_epochs=2
- learning_rate=1e-4, lora_rank=32
- 数据: tiger_dataset/metadata_100.csv（良性）

### 评测协议（所有方法）
- 判别器: NudeNet (threshold=0.6) + 安全分类器 porn 类
- 评测集: VU benchmark_wan.jsonl nudity 类 99 条
- 指标: violation_rate, frame_nudity_rate, porn 检出率

## 方法特定配置

1. **GradAscent** (baseline, Exp 012 已完成):
   - 无额外参数

2. **GradDiff**:
   - retain_weight=1.0
   - retain_manifest=nudity_retain_composed_wan.jsonl

3. **ESD**:
   - negative_guidance=1.0

4. **NPO**:
   - beta=0.1

5. **AnchorDistill**:
   - anchor="a person"

## 文件结构

```
fine_tuing_using_DiffSynth-Studio/
├── scripts/
│   └── run_unlearn_wan5b_multi_method.py    # 多方法擦除驱动脚本
├── slurm/
│   ├── exp015_submit.sh                      # 全流程提交脚本
│   ├── exp015_unlearn_*.sbatch              # 擦除训练作业 (4个)
│   ├── exp015_merge_*.sbatch                # LoRA 合并作业 (4个)
│   ├── exp015_finetune_*.sbatch             # 微调作业 (4个)
│   └── logs/exp015_pipeline_jobs.tsv        # 作业记录
├── models/
│   ├── unlearn/exp015_wan5b_nudity_*/       # 擦除 LoRA (4个)
│   ├── wan5b/exp015_*_erased/               # 擦除后基座 (4个)
│   └── finetune/exp015_*_erased_lora10e2/   # 微调 LoRA (4个)
└── outputs/
    └── exp015_evaluation/                    # 评估结果
```

## 预期结果

### 擦除效果（<method>_erased vs base）
- **最深擦除**: ESD（负引导最激进）
- **平衡擦除**: GradDiff（有 retain 约束）
- **温和擦除**: AnchorDistill（蒸馏到替代词）

### 微调鲁棒性（<method>_erased_ft 的回潮程度）
- **最鲁棒**: GradDiff（retain set 提供锚定）
- **适中**: NPO（负偏好优化平衡）
- **易回潮**: ESD（擦除过深，容易恢复）

### 安全保持率
安全保持率 = (<method>_erased_ft - <method>_erased) / (base - <method>_erased)
- 值越小 = 微调后回潮越少 = 方法越鲁棒

## 监控和调试

### 查看作业状态
```bash
# 查看运行中的作业
squeue -u $USER

# 查看历史作业
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList

# 查看作业记录
cat slurm/logs/exp015_pipeline_jobs.tsv
```

### 查看日志
```bash
# 实时监控
tail -f slurm/logs/exp015_unlearn_grad_diff-<job_id>.out

# 查看错误
cat slurm/logs/exp015_unlearn_grad_diff-<job_id>.err
```

### 检查产物
```bash
# 检查擦除 LoRA
ls -lh models/unlearn/exp015_wan5b_nudity_*/final/

# 检查擦除后基座
ls -lh models/wan5b/exp015_*_erased/

# 检查微调产物
ls -lh models/finetune/exp015_*_erased_lora10e2/
```

## 故障排查

### 擦除训练失败
1. 检查 latent 缓存是否存在: `data/wan5b/unlearn_latents/`
2. 检查 VU 环境是否正确激活
3. 查看详细错误日志: `slurm/logs/exp015_unlearn_*-<job_id>.err`

### LoRA 合并失败
1. 确认擦除 LoRA 已生成: `models/unlearn/exp015_wan5b_nudity_*/final/`
2. 检查基座路径: `models/Wan-AI/Wan2.2-TI2V-5B-Diffusers/`
3. 确认 diffsynth 环境已激活

### 微调失败
1. 确认擦除后基座已生成: `models/wan5b/exp015_*_erased/`
2. 检查训练数据: `data/tiger_dataset/metadata_100.csv`
3. 验证 pandas 和 libstdcxx-ng 已安装

## 参考

- **Exp 012**: GradAscent 基线（1200 steps）
- **Exp 014**: VU 标准配置（600 steps, 细粒度回潮曲线）
- **VU 项目**: /home/x_jiage/jiage/video-unlearning/
- **方法配置**: video-unlearning/configs/methods/training/*_wan.yaml

## 下一步

1. ✅ Stage 1: 擦除训练（脚本已就绪）
2. ✅ Stage 2: LoRA 合并（脚本已就绪）
3. ✅ Stage 3: 微调（脚本已就绪）
4. ⏳ Stage 4: 生成评测视频（待开发）
5. ⏳ Stage 5: 评估（待开发）
6. ⏳ Stage 6: 对比分析（待开发）

当前可以启动 Stage 1-3，后续阶段复用 Exp 012 的生成和评估脚本。
