# Exp019 在新集群继续实验指南

**实验**: Exp019 - GradDiff 方法全流程评估  
**状态**: 微调已完成，蒸馏部分待继续  
**创建日期**: 2026-09-28

---

## 📊 Exp019 当前进度

### ✅ 已完成部分

1. **擦除阶段** (Exp019)
   - 方法: GradDiff
   - 产物: `models/unlearned/exp019_graddiff/adapter_model.safetensors` (已上传 HF)
   - 评估: 已完成

2. **微调阶段** (Exp019)
   - 基于擦除后模型微调
   - Checkpoints: step 1000, 5000, 10000, 25000, 9500 (已上传 HF)
   - 评估: `evaluations/exp019/graddiff_ft_eval.json` (已上传 HF)
   - **核心发现**: 微调后违规率 89.9% (回潮严重)

### ⏳ 待完成部分

3. **蒸馏阶段** (需在新集群完成)
   - 对擦除后模型进行 50-step → 4-step 蒸馏
   - 生成 distilled 视频并评估
   - 预期: 评估蒸馏是否影响安全对齐

---

## 🚀 新集群快速启动步骤

### 前置条件检查

```bash
# 1. 确认基础环境已完成（参考 NEW_CLUSTER_SETUP.md）
conda activate diffsynth
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 2. 确认数据已下载
ls hf_data/models/unlearned/exp019_graddiff/
ls hf_data/datasets/tiger_dataset/

# 3. 确认基座模型存在
ls models/Wan-AI/Wan2.2-TI2V-5B/
```

### Step 1: 恢复 Exp019 模型和数据

```bash
# 创建实验目录结构
mkdir -p outputs/exp019_graddiff_{erased,ft,distill}
mkdir -p outputs/exp019_evaluation

# 从 HF 数据恢复擦除模型
cp -r hf_data/models/unlearned/exp019_graddiff/ \
      outputs/exp019_graddiff_erased/final/

# 从 HF 数据恢复微调 checkpoints（可选，如需重新评估）
cp -r hf_data/models/finetuned/exp019_graddiff_ft/ \
      outputs/exp019_graddiff_ft/

# 恢复评估结果
cp hf_data/evaluations/exp019/graddiff_ft_eval.json \
   outputs/exp019_evaluation/
```

### Step 2: 训练蒸馏模型

**选项 A: 使用现有脚本（推荐）**

```bash
# 检查蒸馏脚本
cat scripts/train_distill_wan5b.py

# 修改 SLURM 配置（如果集群不同）
vim slurm/exp019_distill.sbatch

# 提交蒸馏任务
sbatch slurm/exp019_distill.sbatch
```

**关键参数**:
```python
# 在 slurm/exp019_distill.sbatch 中
--lora_path outputs/exp019_graddiff_erased/final/adapter_model.safetensors
--output_dir outputs/exp019_graddiff_distill
--num_timesteps 4          # 从 50-step 蒸馏到 4-step
--num_train_epochs 10
--learning_rate 1e-5
```

**预计时间**: 4-8 小时 (1 × A100)

**选项 B: 如果脚本不存在，创建新脚本**

参考 `memory/experiments.md` 中 Exp017 的蒸馏配置：
- 使用 DiffSynth 的 distillation API
- Teacher: 50-step DDIM with erased LoRA
- Student: 4-step model
- Training data: tiger_dataset (metadata_50_distill.csv)

### Step 3: 生成蒸馏视频

```bash
# 使用蒸馏后模型生成视频
python scripts/generate_videos_wan5b.py \
    --model_path models/Wan-AI/Wan2.2-TI2V-5B \
    --lora_path outputs/exp019_graddiff_distill/final/adapter_model.safetensors \
    --prompt_file hf_data/datasets/tiger_dataset/metadata_100.csv \
    --output_dir outputs/exp019_graddiff_distill/videos \
    --num_inference_steps 4 \
    --num_repeats 5

# 或提交到 SLURM
sbatch slurm/exp019_generate_distill.sbatch
```

**预计时间**: 2-4 小时 (生成 100 prompts × 5 repeats = 500 视频)

### Step 4: 评估蒸馏视频安全性

```bash
# 使用 NudeNet 评估
python scripts/eval_nudity_rate.py \
    --video_dir outputs/exp019_graddiff_distill/videos \
    --output_json outputs/exp019_evaluation/graddiff_distill_eval.json \
    --threshold 0.6

# 查看结果
python scripts/analyze_results.py \
    --eval_file outputs/exp019_evaluation/graddiff_distill_eval.json
```

---

## 📝 实验记录规范

完成后在 `memory/experiments.md` 中更新：

```markdown
## Exp019 - GradDiff 方法全流程（继续）

### 蒸馏阶段（新集群）

- **日期**: 2026-XX-XX
- **集群**: [新集群名称]
- **作业 ID**: JXXX=XXXXXX
- **配置**:
  - Teacher: Exp019 擦除后模型 (50-step)
  - Student: 4-step 蒸馏
  - Epochs: 10
  - LR: 1e-5
  - Data: tiger_dataset (50 samples)
- **产物**:
  - 蒸馏模型: `outputs/exp019_graddiff_distill/final/`
  - 生成视频: `outputs/exp019_graddiff_distill/videos/` (500个)
  - 评估结果: `outputs/exp019_evaluation/graddiff_distill_eval.json`
- **结果**: 
  - Violation Rate: XX.X%
  - 对比擦除后: XX.X% → XX.X% (变化 ±X.Xpp)
```

---

## 🔄 与其他方法对比

完成 Exp019 蒸馏后，你将拥有 GradDiff 方法的完整数据：

| 操作 | Violation Rate | 数据来源 |
|------|----------------|----------|
| 原始模型 | ~80% | Exp015 baseline |
| 擦除后 | XX.X% | Exp019 评估 |
| 微调后 | 89.9% | ✅ 已完成 (HF) |
| 量化后 | ? | 待补充 (Exp016 只做了 4 方法) |
| 蒸馏后 | ? | ⏳ 待完成 (本指南) |

**对比参考** (其他方法蒸馏表现，来自 Exp017):
- ESD distill: 11.1%
- NPO distill: 5.1%
- GradAscent distill: 7.1%
- AnchorDistill distill: 0.0%

---

## 🎯 预期结果

根据 Exp017 的发现，预期 GradDiff 蒸馏后：
- 如果类似 ESD/NPO/GradAscent: 违规率 5-11%（蒸馏保持安全性）
- 如果类似 AnchorDistill: 违规率 0%（蒸馏完美保持）
- **关键问题**: GradDiff 微调表现异常差（89.9%），蒸馏是否也会异常？

---

## 📦 数据上传到 HuggingFace

完成后需要上传新数据：

```bash
# 1. 准备蒸馏模型
mkdir -p /tmp/hf_upload_exp019_distill/models/distilled/
cp outputs/exp019_graddiff_distill/final/adapter_model.safetensors \
   /tmp/hf_upload_exp019_distill/models/distilled/exp019_graddiff_distill.safetensors

# 2. 准备评估结果
mkdir -p /tmp/hf_upload_exp019_distill/evaluations/exp019/
cp outputs/exp019_evaluation/graddiff_distill_eval.json \
   /tmp/hf_upload_exp019_distill/evaluations/exp019/

# 3. 上传到 HF
hf upload littlepig404/wan5b-unlearning-artifacts \
   /tmp/hf_upload_exp019_distill/ \
   --repo-type dataset
```

**注意**: 视频文件不上传（太大），只上传模型和评估结果。

---

## ⚠️ 注意事项

1. **SLURM 分区**: 新集群的分区名称可能不同，检查 `sinfo` 并修改 `#SBATCH -p` 参数

2. **模型路径**: 新集群基座模型路径可能不同，更新 `.env` 文件：
   ```bash
   export WAN_MODEL_PATH="/path/to/new/cluster/Wan2.2-TI2V-5B"
   ```

3. **依赖检查**: 蒸馏需要完整的 DiffSynth 环境，运行前确认：
   ```bash
   python -c "from diffsynth import ModelManager; print('✅ DiffSynth OK')"
   ```

4. **存储空间**: 蒸馏训练 + 500 视频生成需要 ~50GB 空间

5. **对比实验**: 如果时间允许，考虑同时做 GradDiff 量化实验（Exp016 缺失）

---

## 📚 参考文档

- 完整实验记录: `memory/experiments.md` (Exp017, Exp019)
- 蒸馏方法说明: `memory/group_meeting_presentation.md` Section 2.4
- 新集群部署: `NEW_CLUSTER_SETUP.md`
- HF 数据结构: `hf_data/README.md`

---

**快速检查清单**:
- [ ] 环境已激活 (`conda activate diffsynth`)
- [ ] HF 数据已下载 (`ls hf_data/models/unlearned/exp019_graddiff/`)
- [ ] 基座模型已准备 (`ls models/Wan-AI/Wan2.2-TI2V-5B/`)
- [ ] Exp019 擦除模型已恢复到 `outputs/exp019_graddiff_erased/final/`
- [ ] 蒸馏脚本已检查 (`scripts/train_distill_wan5b.py`)
- [ ] SLURM 配置已更新（分区、路径）
- [ ] 存储空间充足 (`df -h` 检查 ~50GB 可用)
