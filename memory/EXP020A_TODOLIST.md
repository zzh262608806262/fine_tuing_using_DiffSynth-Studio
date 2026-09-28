# Exp020a - GradDiff Masked 验证实验 TodoList

**实验目标**: 验证 Attention Mask 能否解决 Exp015 的"齐步降解"问题  
**预计时间**: ~26 小时  
**核心问题**: 空间约束是否提升擦除效果和微调鲁棒性？

---

## 📋 Phase 0: 前置检查（1小时）

### ✅ Task 0.1: 验证 VU Masked 方法支持
```bash
# 检查 GradDiff Masked 实现
ls -lh /home/x_jiage/jiage/video-unlearning/src/unlearning/training/grad_diff_masked.py

# 检查 concept mask 模块
ls -lh /home/x_jiage/jiage/video-unlearning/src/unlearning/training/concept_mask.py

# 检查配置文件（参考 CogVideoX 的配置）
ls -lh /home/x_jiage/jiage/video-unlearning/configs/methods/training/grad_diff_masked_cogvideox.yaml
```

**预期结果**: 3 个文件都存在

---

### ✅ Task 0.2: 研究 Masked 方法接口
```bash
# 阅读 GradDiff Masked 实现
cd /home/x_jiage/jiage/video-unlearning
grep -A 30 "class.*GradDiff.*Masked" src/unlearning/training/grad_diff_masked.py | head -50

# 阅读配置文件
cat configs/methods/training/grad_diff_masked_cogvideox.yaml
```

**需要确认的参数**:
- `mask_config.type`: 应该是 "attention"
- `mask_config.threshold`: 默认 0.5
- `mask_config.dilation`: 默认 1（不膨胀）
- `mask_config.concept`: "nudity" 或从 prompt 自动提取

---

### ✅ Task 0.3: 检查 Wan5B 模型兼容性
```bash
# 关键问题：VU 的 Masked 方法主要针对 CogVideoX 开发
# 需要确认 Wan5B 的注意力结构是否兼容

# 检查 VU 是否已支持 Wan 模型的 masked 训练
grep -r "Wan" /home/x_jiage/jiage/video-unlearning/configs/methods/training/ | grep -i mask
```

**如果不支持**: 需要创建 `grad_diff_masked_wan.yaml` 配置文件

---

### ✅ Task 0.4: 验证 run_unlearn_wan5b.py 是否支持 Masked
```bash
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 检查脚本是否已支持 GradDiffMasked
grep -n "GradDiffMasked\|grad_diff_masked" scripts/run_unlearn_wan5b.py
```

**预期结果**:
- ✅ 如果已支持：可以直接用
- ❌ 如果不支持：需要添加 method 分支

**修复方案**（如果需要）:
```python
# 在 scripts/run_unlearn_wan5b.py 添加
METHOD_MAP = {
    'GradAscent': 'grad_ascent_wan.yaml',
    'GradDiff': 'grad_diff_wan.yaml',
    'GradDiffMasked': 'grad_diff_masked_wan.yaml',  # 新增
    # ...
}
```

---

### ✅ Task 0.5: 创建 Wan5B 的 Masked 配置文件（如果需要）

**如果 VU 没有 Wan 的 masked 配置**，需要创建：

```bash
cd /home/x_jiage/jiage/video-unlearning

# 基于 CogVideoX 的配置创建 Wan 版本
cp configs/methods/training/grad_diff_masked_cogvideox.yaml \
   configs/methods/training/grad_diff_masked_wan.yaml
```

**编辑配置文件**:
```yaml
# grad_diff_masked_wan.yaml
method: grad_diff_masked

training:
  steps: 600
  batch_size: 1
  learning_rate: 1e-5
  lora_rank: 8
  retain_weight: 1.0  # GradDiff 特有

mask_config:
  type: attention          # 使用 Attention Mask
  threshold: 0.5           # 二值化阈值
  dilation: 1              # 不膨胀
  concept: "nudity"        # 或从 prompt 自动提取
  
# Wan5B 模型特定配置（如果需要）
model_specific:
  attention_type: "cross"  # Wan 用 attn2 (cross-attention)
  # CogVideoX 用 attn1 (joint attention)
```

---

## 📋 Phase 1: 擦除训练（准备 + 30分钟）

### ✅ Task 1.1: 准备擦除训练脚本
```bash
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 创建 SLURM 脚本
vi slurm/exp020a_unlearn_graddiff_masked.sbatch
```

**脚本内容**:
```bash
#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -J exp020a_graddiff_masked_unlearn
#SBATCH -o outputs/exp020a_graddiff_masked_unlearn_%j.log
#SBATCH -e outputs/exp020a_graddiff_masked_unlearn_%j.err

# 加载环境
module load Mambaforge/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate vu

# 训练参数
METHOD=GradDiffMasked  # 或 GradDiff --use-mask（取决于脚本实现）
STEPS=600
RETAIN_WEIGHT=1.0
BATCH_SIZE=1
LR=1e-5

# Mask 配置
MASK_TYPE=attention
MASK_THRESHOLD=0.5
MASK_DILATION=1
CONCEPT="nudity"

# 路径
VU_ROOT=/home/x_jiage/jiage/video-unlearning
MODEL_ROOT=models/Wan-AI/Wan2.2-TI2V-5B
CACHE_ROOT=data/wan5b/unlearn_baseline
FORGET_MANIFEST=${VU_ROOT}/data/manifests/nudity_forget_composed_wan.jsonl
RETAIN_MANIFEST=${VU_ROOT}/data/manifests/nudity_retain_composed_wan.jsonl
OUTPUT_DIR=models/unlearn/exp020a_wan5b_nudity_graddiff_masked

# 运行训练
# 注意：参数名可能需要根据实际脚本调整
python scripts/run_unlearn_wan5b.py \
    --method ${METHOD} \
    --model_id ${MODEL_ROOT} \
    --concept_manifest ${FORGET_MANIFEST} \
    --retain_manifest ${RETAIN_MANIFEST} \
    --retain_weight ${RETAIN_WEIGHT} \
    --cache ${CACHE_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --unlearn_steps ${STEPS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --save_every 100 \
    --mask_type ${MASK_TYPE} \
    --mask_threshold ${MASK_THRESHOLD} \
    --mask_dilation ${MASK_DILATION} \
    --mask_concept ${CONCEPT}

echo "Exp020a GradDiff Masked 擦除训练完成"
```

---

### ✅ Task 1.2: 提交擦除训练
```bash
# 提交作业
sbatch slurm/exp020a_unlearn_graddiff_masked.sbatch

# 记录 Job ID
JOB_ID=$(squeue -u $USER -n exp020a_graddiff_masked_unlearn -h -o "%i")
echo "Job ID: ${JOB_ID}"

# 更新 memory/experiments.md
```

**监控命令**:
```bash
# 查看作业状态
squeue -j ${JOB_ID}

# 查看日志（特别关注 mask 生成信息）
tail -f outputs/exp020a_graddiff_masked_unlearn_${JOB_ID}.log

# 应该看到类似的日志：
# "Generating attention masks for concept 'nudity'..."
# "Mask coverage: 23.4%, attention peak: 0.82"
```

**预期时间**: ~10-30 分钟（取决于 mask 计算开销）

---

### ✅ Task 1.3: 验证擦除输出
```bash
# 检查 checkpoint 数量
ls models/unlearn/exp020a_wan5b_nudity_graddiff_masked/

# 应该有 7 个：step-100 到 step-600
du -sh models/unlearn/exp020a_wan5b_nudity_graddiff_masked/step-*

# 检查是否有 mask 相关的诊断文件（如果 VU 保存的话）
ls models/unlearn/exp020a_wan5b_nudity_graddiff_masked/*.json
ls models/unlearn/exp020a_wan5b_nudity_graddiff_masked/masks/
```

---

### ✅ Task 1.4: 创建实验记录
```bash
# 在 memory/experiments.md 添加 Exp020a 记录
```

**记录内容**:
```markdown
## Exp 020a — GradDiff Masked 验证实验

- **Date**: 2026-09-XX（启动日期）
- **目的**: 验证 Attention Mask 能否解决"齐步降解"问题
- **方法**: GradDiff + Attention Mask
- **对比基线**: Exp019 GradDiff (无 mask)
- **研究问题**: 
  1. Masked 版本擦除效果是否优于无 mask 版本？
  2. Masked 版本视觉质量是否更好（无全局退化）？
  3. Masked 版本的微调鲁棒性如何？
- **实验设计**: 
  - 擦除训练: GradDiffMasked, mask_type=attention, threshold=0.5
  - Mask 配置: concept="nudity", dilation=1
  - 基线评估: masked_base vs masked_erased
  - 微调训练: repeat=25, epochs=20
  - 回潮评估: 对比 Exp019 (无 mask)
- **Job 记录**: 
  - 擦除训练: J0=XXXXX
  - 微调训练: J1=XXXXX
- **Results**: Pending
- **Artifacts**: 
  - 擦除 LoRA: `models/unlearn/exp020a_wan5b_nudity_graddiff_masked/`
  - 合并模型: `models/wan5b/exp020a_masked_graddiff_{base,erased}/`
  - 微调 LoRA: `models/finetune/exp020a_masked_graddiff_ft/`
  - 视频: `outputs/exp020a/{masked_graddiff_base,masked_graddiff_erased,masked_graddiff_ft_e19}/`
  - 评估: `outputs/exp020a_evaluation/`
```

---

## 📋 Phase 2: LoRA 合并（10分钟）

### ✅ Task 2.1: 合并 base 模型
```bash
vi slurm/exp020a_merge_base.sbatch
```

```bash
#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 0:15:00
#SBATCH -J exp020a_merge_base

python scripts/merge_lora_wan5b.py \
    --base_model models/Wan-AI/Wan2.2-TI2V-5B \
    --output_dir models/wan5b/exp020a_masked_graddiff_base \
    --no-lora
```

---

### ✅ Task 2.2: 合并 erased 模型
```bash
vi slurm/exp020a_merge_erased.sbatch
```

```bash
#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 0:15:00
#SBATCH -J exp020a_merge_erased

python scripts/merge_lora_wan5b.py \
    --base_model models/Wan-AI/Wan2.2-TI2V-5B \
    --lora_path models/unlearn/exp020a_wan5b_nudity_graddiff_masked/step-600 \
    --output_dir models/wan5b/exp020a_masked_graddiff_erased
```

---

### ✅ Task 2.3: 提交合并作业
```bash
sbatch slurm/exp020a_merge_base.sbatch
sbatch slurm/exp020a_merge_erased.sbatch

# 验证输出
du -sh models/wan5b/exp020a_masked_graddiff_{base,erased}
```

---

## 📋 Phase 3: 基线视频生成（1小时）

### ✅ Task 3.1: 准备生成脚本
```bash
vi slurm/exp020a_generate_baseline.sbatch
```

```bash
#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -J exp020a_gen_baseline
#SBATCH --array=0-1  # 2 个臂：base, erased

ARMS=(masked_graddiff_base masked_graddiff_erased)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

python scripts/exp015_generate_videos.py \
    --method masked_graddiff \
    --arm ${ARM} \
    --model_path models/wan5b/exp020a_${ARM} \
    --output_dir outputs/exp020a/${ARM} \
    --num_videos 99 \
    --frames 17 \
    --height 480 \
    --width 720 \
    --steps 50 \
    --guidance_scale 5.0
```

---

### ✅ Task 3.2: 提交生成作业
```bash
sbatch slurm/exp020a_generate_baseline.sbatch

# 验证结果
ls outputs/exp020a/masked_graddiff_base/*.mp4 | wc -l  # 应该是 99
ls outputs/exp020a/masked_graddiff_erased/*.mp4 | wc -l  # 应该是 99
```

**预期时间**: ~30 分钟/臂

---

## 📋 Phase 4: 基线评估 + 关键决策点（30分钟）

### ✅ Task 4.1: NudeNet 评估
```bash
vi slurm/exp020a_eval_baseline.sbatch
```

```bash
#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -J exp020a_eval_baseline
#SBATCH --array=0-1

ARMS=(masked_graddiff_base masked_graddiff_erased)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

python scripts/evaluate_videos_nudenet.py \
    --video_dir outputs/exp020a/${ARM} \
    --output_file outputs/exp020a_evaluation/${ARM}_evaluation.json \
    --threshold 0.6
```

```bash
sbatch slurm/exp020a_eval_baseline.sbatch
```

---

### ✅ Task 4.2: 🔥 关键决策点 - 对比 Masked vs Non-masked

```bash
python3 << 'EOF'
import json

# Exp019 (GradDiff 无 mask)
gd_base = json.load(open('outputs/exp019_evaluation/graddiff_base_evaluation.json'))
gd_erased = json.load(open('outputs/exp019_evaluation/graddiff_erased_evaluation.json'))

# Exp020a (GradDiff Masked)
masked_base = json.load(open('outputs/exp020a_evaluation/masked_graddiff_base_evaluation.json'))
masked_erased = json.load(open('outputs/exp020a_evaluation/masked_graddiff_erased_evaluation.json'))

print("=== 🎯 Mask 效果验证 ===")
print(f"\n【无 Mask】(Exp019)")
print(f"  Base:   {gd_base['nudenet']['summary']['violation_rate']:.1%}")
print(f"  Erased: {gd_erased['nudenet']['summary']['violation_rate']:.1%}")
print(f"  变化:   {(gd_erased['nudenet']['summary']['violation_rate'] - gd_base['nudenet']['summary']['violation_rate'])*100:+.1f}pp")

print(f"\n【有 Mask】(Exp020a)")
print(f"  Base:   {masked_base['nudenet']['summary']['violation_rate']:.1%}")
print(f"  Erased: {masked_erased['nudenet']['summary']['violation_rate']:.1%}")
print(f"  变化:   {(masked_erased['nudenet']['summary']['violation_rate'] - masked_base['nudenet']['summary']['violation_rate'])*100:+.1f}pp")

# 关键判断
no_mask_change = gd_erased['nudenet']['summary']['violation_rate'] - gd_base['nudenet']['summary']['violation_rate']
masked_change = masked_erased['nudenet']['summary']['violation_rate'] - masked_base['nudenet']['summary']['violation_rate']

print(f"\n=== 📊 结论 ===")
if masked_change < no_mask_change:
    print("✅ Mask 有效！Masked 版本擦除效果更好")
    print("建议：继续进行微调训练（Phase 5-7）")
elif masked_change < 0:
    print("⚠️  Masked 版本有一定效果，但不如预期")
    print("建议：检查 mask 质量，考虑调整参数后重试")
else:
    print("❌ Mask 无效！Masked 版本仍然出现齐步降解")
    print("建议：停止本实验，分析失败原因")

EOF
```

**🚦 决策规则**:
- ✅ **继续微调**: Masked 版本擦除效果明显优于无 mask
- ⚠️ **调整参数**: Masked 有小幅改善，但不显著
- ❌ **停止实验**: Masked 无效或更差

---

### ✅ Task 4.3: （可选）可视化检查样本
```bash
# 随机抽取几个样本，人工检查视觉质量
# 重点看：是否还有全局退化/抖动现象

# 对比无 mask vs 有 mask 的 erased 模型生成的视频
# Exp019: outputs/exp019/graddiff_erased/
# Exp020a: outputs/exp020a/masked_graddiff_erased/
```

---

## 📋 Phase 5: 微调训练（24小时）

**前置条件**: Phase 4 决策为"继续"

### ✅ Task 5.1: 准备微调脚本
```bash
vi slurm/exp020a_finetune.sbatch
```

```bash
#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 30:00:00
#SBATCH -J exp020a_ft

module load Mambaforge/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate diffsynth  # 注意：切换到 diffsynth 环境

python examples/wanvideo/model_training/train.py \
    --model_id models/wan5b/exp020a_masked_graddiff_erased \
    --train_data data/tiger_dataset/metadata_100.csv \
    --output_dir models/finetune/exp020a_masked_graddiff_ft \
    --dataset_repeat 25 \
    --num_epochs 20 \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --save_every_n_epochs 1 \
    --no_tensorboard_log
```

---

### ✅ Task 5.2: 提交微调作业
```bash
sbatch slurm/exp020a_finetune.sbatch

# 记录 Job ID
FT_JOB_ID=$(squeue -u $USER -n exp020a_ft -h -o "%i")
echo "Fine-tuning Job ID: ${FT_JOB_ID}"
```

**预期时间**: ~24 小时

---

### ✅ Task 5.3: 监控微调进度
```bash
# 查看日志
tail -f outputs/exp020a_ft_*.log

# 检查 checkpoint 生成（每小时检查一次）
watch -n 3600 "ls models/finetune/exp020a_masked_graddiff_ft/ | grep epoch"
```

---

## 📋 Phase 6: 微调后生成（30分钟）

### ✅ Task 6.1: 生成微调后视频
```bash
vi slurm/exp020a_generate_ft.sbatch
```

```bash
#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -J exp020a_gen_ft

python scripts/exp015_generate_videos.py \
    --method masked_graddiff \
    --arm masked_graddiff_ft_e19 \
    --model_path models/wan5b/exp020a_masked_graddiff_erased \
    --lora_path models/finetune/exp020a_masked_graddiff_ft/epoch-19.safetensors \
    --output_dir outputs/exp020a/masked_graddiff_ft_e19 \
    --num_videos 99
```

```bash
sbatch slurm/exp020a_generate_ft.sbatch

# 验证
ls outputs/exp020a/masked_graddiff_ft_e19/*.mp4 | wc -l  # 应该是 99
```

---

## 📋 Phase 7: 微调后评估（30分钟）

### ✅ Task 7.1: NudeNet 评估
```bash
python scripts/evaluate_videos_nudenet.py \
    --video_dir outputs/exp020a/masked_graddiff_ft_e19 \
    --output_file outputs/exp020a_evaluation/masked_graddiff_ft_e19_evaluation.json
```

---

### ✅ Task 7.2: 计算回潮幅度 - Masked vs Non-masked
```bash
python3 << 'EOF'
import json

# Exp019 (无 Mask)
gd_erased = json.load(open('outputs/exp019_evaluation/graddiff_erased_evaluation.json'))
gd_ft = json.load(open('outputs/exp019_evaluation/graddiff_ft_e19_evaluation.json'))

# Exp020a (Masked)
masked_erased = json.load(open('outputs/exp020a_evaluation/masked_graddiff_erased_evaluation.json'))
masked_ft = json.load(open('outputs/exp020a_evaluation/masked_graddiff_ft_e19_evaluation.json'))

print("=== 🎯 微调鲁棒性对比 ===")

# 无 Mask 回潮
no_mask_rebound = gd_ft['nudenet']['summary']['violation_rate'] - gd_erased['nudenet']['summary']['violation_rate']
print(f"\n【无 Mask】(Exp019)")
print(f"  Erased: {gd_erased['nudenet']['summary']['violation_rate']:.1%}")
print(f"  FT e19: {gd_ft['nudenet']['summary']['violation_rate']:.1%}")
print(f"  回潮:   {no_mask_rebound*100:+.1f}pp")

# Masked 回潮
masked_rebound = masked_ft['nudenet']['summary']['violation_rate'] - masked_erased['nudenet']['summary']['violation_rate']
print(f"\n【Masked】(Exp020a)")
print(f"  Erased: {masked_erased['nudenet']['summary']['violation_rate']:.1%}")
print(f"  FT e19: {masked_ft['nudenet']['summary']['violation_rate']:.1%}")
print(f"  回潮:   {masked_rebound*100:+.1f}pp")

# 结论
print(f"\n=== 📊 结论 ===")
print(f"回潮幅度差异: {(masked_rebound - no_mask_rebound)*100:+.1f}pp")
if masked_rebound < no_mask_rebound:
    print("✅ Masked 版本微调鲁棒性更强！")
    improvement = (no_mask_rebound - masked_rebound) / no_mask_rebound * 100
    print(f"   回潮减少了 {improvement:.1f}%")
else:
    print("⚠️  Masked 版本并未改善微调鲁棒性")

EOF
```

---

## 📋 Phase 8: 结果整理与记录（1小时）

### ✅ Task 8.1: 更新 experiments.md
```bash
# 填写完整的结果到 memory/experiments.md
```

**记录格式**:
```markdown
- **Results**: ✅ 完成（2026-09-XX）
  
  ### 擦除效果对比
  | 版本 | Base违规率 | Erased违规率 | 擦除效果 |
  |------|------------|--------------|----------|
  | GradDiff (无mask) | XX.X% | XX.X% | XX.Xpp |
  | GradDiff Masked | XX.X% | XX.X% | XX.Xpp |
  | **改善** | - | - | **XX.Xpp** |
  
  ### 微调鲁棒性对比
  | 版本 | Erased | FT e19 | 回潮幅度 |
  |------|--------|--------|----------|
  | GradDiff (无mask) | XX.X% | XX.X% | +X.Xpp |
  | GradDiff Masked | XX.X% | XX.X% | +X.Xpp |
  | **改善** | - | - | **-X.Xpp** |
  
  ### 关键发现
  - ✅/❌ Attention Mask 是否解决齐步降解？
  - ✅/❌ Masked 版本是否提升微调鲁棒性？
  - 视觉质量观察：是否仍有全局退化现象？
  
  ### 下一步建议
  - 如果有效 → Exp020b（补充其他方法的 Masked 版本）
  - 如果无效 → 回到 ESD 方向或尝试 SAM Mask
```

---

### ✅ Task 8.2: 更新 QUICK_DATA_REFERENCE.md
```bash
# 添加 Exp020a 数据到汇总表
```

---

### ✅ Task 8.3: 创建对比可视化（可选）
```python
# 创建对比图表脚本
# outputs/exp020a_analysis/masked_vs_nonmasked_comparison.png

import matplotlib.pyplot as plt
import json

# 加载数据...
# 绘制：
# 1. 擦除效果条形图（无mask vs masked）
# 2. 回潮幅度对比图
# 3. 完整流程违规率曲线
```

---

## 📋 Phase 9: 验证与检查（15分钟）

### ✅ Task 9.1: 文件完整性检查
```bash
cat << 'EOF' | bash
echo "=== Exp020a 产物检查 ==="

# 1. 擦除 LoRA
echo "擦除 LoRA:"
ls -lh models/unlearn/exp020a_wan5b_nudity_graddiff_masked/step-600

# 2. 合并模型
echo "合并模型:"
du -sh models/wan5b/exp020a_masked_graddiff_{base,erased}

# 3. 微调 checkpoints
echo "微调 checkpoints:"
ls models/finetune/exp020a_masked_graddiff_ft/ | wc -l  # 应该是 20

# 4. 视频
echo "生成视频:"
for dir in masked_graddiff_base masked_graddiff_erased masked_graddiff_ft_e19; do
    count=$(ls outputs/exp020a/${dir}/*.mp4 2>/dev/null | wc -l)
    echo "  ${dir}: ${count} 视频"
done

# 5. 评估结果
echo "评估结果:"
ls -lh outputs/exp020a_evaluation/*.json

EOF
```

**预期**:
- ✅ 7 个擦除 checkpoint
- ✅ 2 个合并模型
- ✅ 20 个微调 checkpoint
- ✅ 297 个视频（99+99+99）
- ✅ 3 个评估 JSON

---

### ✅ Task 9.2: 数据质量检查
```bash
python3 << 'EOF'
import json

files = [
    'outputs/exp020a_evaluation/masked_graddiff_base_evaluation.json',
    'outputs/exp020a_evaluation/masked_graddiff_erased_evaluation.json',
    'outputs/exp020a_evaluation/masked_graddiff_ft_e19_evaluation.json'
]

print("=== 数据质量检查 ===")
for f in files:
    try:
        data = json.load(open(f))
        vr = data['nudenet']['summary']['violation_rate']
        total = data['nudenet']['summary']['total_videos']
        print(f"{f.split('/')[-1]}: {vr:.1%} ({total} videos)")
        
        if total != 99:
            print(f"  ⚠️  视频数量异常！")
        if vr > 1.0 or vr < 0:
            print(f"  ⚠️  违规率异常！")
    except FileNotFoundError:
        print(f"❌ 文件不存在: {f}")
    except Exception as e:
        print(f"❌ 读取失败: {f} - {e}")
EOF
```

---

## 📊 完成检查清单

- [ ] Phase 0: 前置检查（VU 支持、配置文件、兼容性）
- [ ] Phase 1: 擦除训练（含 mask 生成）
- [ ] Phase 2: LoRA 合并
- [ ] Phase 3: 基线生成
- [ ] Phase 4: 基线评估 + **决策点**
- [ ] **决策**: Mask 是否有效？
  - [ ] 有效 → 继续 Phase 5-7
  - [ ] 无效 → 停止并分析
- [ ] Phase 5: 微调训练
- [ ] Phase 6: 微调后生成
- [ ] Phase 7: 微调后评估
- [ ] Phase 8: 结果整理
- [ ] Phase 9: 验证检查

---

## ⏱️ 时间表

| 阶段 | 预计时间 | 关键里程碑 |
|------|----------|-----------|
| Day 0 | 1h | 前置检查，准备配置 |
| Day 1 上午 | 30min | 擦除训练 |
| Day 1 中午 | 1.5h | 合并 + 基线生成 |
| Day 1 下午 | 30min | 基线评估 + **决策点** |
| Day 1 晚上 | - | 提交微调作业 |
| Day 2-3 | 24h | 微调训练（后台运行）|
| Day 3 | 1h | 微调后生成+评估 |
| Day 3 | 1h | 结果整理 |

**Wall-clock**: ~3 天

---

## ⚠️ 常见问题与解决

### Q1: VU 不支持 Wan5B 的 Masked 方法怎么办？
**A**: 需要适配，主要工作：
1. 确认 Wan5B 的注意力结构（attn1 vs attn2）
2. 修改 `concept_mask.py` 中的 query/key 提取逻辑
3. 创建 `grad_diff_masked_wan.yaml` 配置
4. 预计工作量：1-2 天

### Q2: Mask 生成失败（报错找不到 concept span）
**A**: 
```python
# 问题：tokenizer 分词不稳定
# 解决：检查 concept 在 prompt 中的表达
# 确保 concept="nudity" 在所有 forget prompts 中都出现
```

### Q3: Phase 4 决策点 Mask 无效，接下来做什么？
**A**: 分析失败原因：
1. 查看 mask 覆盖率（应该 >10%）
2. 可视化几个样本的 mask（是否圈住了关键区域）
3. 尝试调整参数：threshold (0.3-0.7), dilation (3-5)
4. 如果 Attention Mask 质量太差，考虑 SAM Mask

### Q4: 微调后回潮仍然很大怎么办？
**A**: 
- Mask 可能只解决了擦除阶段的问题
- 微调鲁棒性可能需要其他机制（如正则化、retain data）
- 继续探索其他方向：ESDGen、81帧验证

---

## 🔗 相关文档

- 详细设计: `memory/exp020_masked_methods_design.md`
- VU Masked 实现: `/home/x_jiage/jiage/video-unlearning/src/unlearning/training/grad_diff_masked.py`
- VU Mask 生成: `/home/x_jiage/jiage/video-unlearning/src/unlearning/training/concept_mask.py`
- 基线实验: `memory/experiments.md` (Exp019)
- 对比分析: `memory/QUICK_DATA_REFERENCE.md`

---

**祝实验顺利！记住 Phase 4 的决策点是关键 - 不要盲目继续微调。**
