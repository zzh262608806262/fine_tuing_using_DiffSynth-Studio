# Exp019 - GradDiff 补充实验 TodoList

**实验目标**: 补充 GradDiff 方法，完成 5 个基础方法对比  
**预计时间**: ~26 小时  
**核心问题**: Retain 约束是否提升微调鲁棒性？

---

## 📋 Phase 1: 准备与验证（30分钟）

### ✅ Task 1.1: 验证 GradDiff 支持
```bash
# 检查脚本是否支持 GradDiff
grep -n "GradDiff" scripts/run_unlearn_wan5b.py

# 检查 VU 配置文件是否存在
ls -lh /home/x_jiage/jiage/video-unlearning/configs/methods/training/grad_diff_wan.yaml

# 检查 retain manifest 是否存在
ls -lh /home/x_jiage/jiage/video-unlearning/data/manifests/nudity_retain_composed_wan.jsonl
```

**预期结果**: 
- ✅ 脚本已支持 GradDiff
- ✅ VU 配置文件存在
- ✅ Retain manifest 存在

**如果失败**: 查看 `memory/exp015_errors.md` 中 Exp015 GradDiff 失败原因

---

### ✅ Task 1.2: 确认数据准备
```bash
# 检查基线视频缓存（可复用 Exp015）
ls -lh data/wan5b/unlearn_baseline/

# 检查 forget manifest
ls -lh /home/x_jiage/jiage/video-unlearning/data/manifests/nudity_forget_composed_wan.jsonl

# 检查微调数据
ls -lh data/tiger_dataset/metadata_100.csv
```

**预期结果**: 所有数据就绪，可复用 Exp015 缓存

---

### ✅ Task 1.3: 创建实验记录
```bash
# 在 memory/experiments.md 添加 Exp019 记录框架
# 参考 Exp015 的格式
```

**记录内容**:
```markdown
## Exp 019 — GradDiff 补充实验（Retain 约束验证）

- **Date**: 2026-09-XX（启动日期）
- **目的**: 补充 GradDiff 方法，验证 retain 约束对微调鲁棒性的影响
- **对比基准**: Exp015 GradAscent（无 retain）
- **研究问题**: 
  1. GradDiff (有 retain) 是否比 GradAscent (无 retain) 擦除效果更好？
  2. GradDiff 的微调鲁棒性是否更强？
- **实验设计**: 
  - 擦除训练: method=GradDiff, retain_weight=1.0, steps=600
  - 基线评估: base vs erased
  - 微调训练: repeat=25, epochs=20
  - 回潮评估: 对比 GradAscent
- **Job 记录**: （待填写）
- **Results**: Pending
- **Artifacts**: 
  - 擦除 LoRA: `models/unlearn/exp019_wan5b_nudity_graddiff/`
  - 合并模型: `models/wan5b/exp019_graddiff_{base,erased}/`
  - 微调 LoRA: `models/finetune/exp019_graddiff_erased_lora/`
  - 视频: `outputs/exp019/{graddiff_base,graddiff_erased,graddiff_ft_e19}/`
  - 评估: `outputs/exp019_evaluation/`
```

---

## 📋 Phase 2: 擦除训练（10分钟）

### ✅ Task 2.1: 准备擦除训练脚本
```bash
# 创建 SLURM 脚本
vi slurm/exp019_unlearn_graddiff.sbatch
```

**脚本内容**（参考 Exp015）:
```bash
#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 0:30:00
#SBATCH -J exp019_graddiff_unlearn
#SBATCH -o outputs/exp019_graddiff_unlearn_%j.log
#SBATCH -e outputs/exp019_graddiff_unlearn_%j.err

# 加载环境
module load Mambaforge/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate vu

# 训练参数（与 Exp015 一致）
METHOD=GradDiff
STEPS=600
RETAIN_WEIGHT=1.0
BATCH_SIZE=1
LR=1e-5

# 路径
VU_ROOT=/home/x_jiage/jiage/video-unlearning
MODEL_ROOT=models/Wan-AI/Wan2.2-TI2V-5B
CACHE_ROOT=data/wan5b/unlearn_baseline
FORGET_MANIFEST=${VU_ROOT}/data/manifests/nudity_forget_composed_wan.jsonl
RETAIN_MANIFEST=${VU_ROOT}/data/manifests/nudity_retain_composed_wan.jsonl
OUTPUT_DIR=models/unlearn/exp019_wan5b_nudity_graddiff

# 运行训练
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
    --save_every 100

echo "Exp019 GradDiff 擦除训练完成"
```

---

### ✅ Task 2.2: 提交擦除训练
```bash
# 提交作业
sbatch slurm/exp019_unlearn_graddiff.sbatch

# 记录 Job ID
JOB_ID=$(squeue -u $USER -n exp019_graddiff_unlearn -h -o "%i")
echo "Job ID: ${JOB_ID}"

# 更新 experiments.md 记录 Job ID
```

**监控命令**:
```bash
# 查看作业状态
squeue -j ${JOB_ID}

# 查看日志
tail -f outputs/exp019_graddiff_unlearn_${JOB_ID}.log
```

**预期时间**: ~10 分钟（与 Exp015 GradAscent 相当）

---

### ✅ Task 2.3: 验证擦除输出
```bash
# 检查 checkpoint 数量（应该有 7 个：step-100 到 step-600）
ls models/unlearn/exp019_wan5b_nudity_graddiff/

# 验证文件大小（每个约 153M）
du -sh models/unlearn/exp019_wan5b_nudity_graddiff/step-*
```

**预期结果**: 7 个 checkpoint，每个 ~153M

---

## 📋 Phase 3: LoRA 合并（10分钟）

### ✅ Task 3.1: 合并 base 模型（无擦除 LoRA）
```bash
# 创建合并脚本
vi slurm/exp019_merge_base.sbatch
```

**脚本内容**:
```bash
#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 0:15:00
#SBATCH -J exp019_merge_base

python scripts/merge_lora_wan5b.py \
    --base_model models/Wan-AI/Wan2.2-TI2V-5B \
    --output_dir models/wan5b/exp019_graddiff_base \
    --no-lora
```

---

### ✅ Task 3.2: 合并 erased 模型（擦除 LoRA）
```bash
vi slurm/exp019_merge_erased.sbatch
```

**脚本内容**:
```bash
#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 0:15:00
#SBATCH -J exp019_merge_erased

python scripts/merge_lora_wan5b.py \
    --base_model models/Wan-AI/Wan2.2-TI2V-5B \
    --lora_path models/unlearn/exp019_wan5b_nudity_graddiff/step-600 \
    --output_dir models/wan5b/exp019_graddiff_erased
```

---

### ✅ Task 3.3: 提交合并作业
```bash
# 提交
sbatch slurm/exp019_merge_base.sbatch
sbatch slurm/exp019_merge_erased.sbatch

# 验证输出（每个约 10GB）
du -sh models/wan5b/exp019_graddiff_{base,erased}
```

---

## 📋 Phase 4: 基线视频生成（1小时）

### ✅ Task 4.1: 准备生成脚本
```bash
vi slurm/exp019_generate_baseline.sbatch
```

**脚本内容**:
```bash
#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -J exp019_gen_baseline
#SBATCH --array=0-1  # 2 个臂：base, erased

ARMS=(graddiff_base graddiff_erased)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

python scripts/exp015_generate_videos.py \
    --method graddiff \
    --arm ${ARM} \
    --model_path models/wan5b/exp019_${ARM} \
    --output_dir outputs/exp019/${ARM} \
    --num_videos 99 \
    --frames 17 \
    --height 480 \
    --width 720 \
    --steps 50 \
    --guidance_scale 5.0
```

**提交**:
```bash
sbatch slurm/exp019_generate_baseline.sbatch
```

**预期时间**: ~30 分钟/臂

---

### ✅ Task 4.2: 验证生成结果
```bash
# 检查视频数量
ls outputs/exp019/graddiff_base/*.mp4 | wc -l  # 应该是 99
ls outputs/exp019/graddiff_erased/*.mp4 | wc -l  # 应该是 99

# 检查视频大小分布
du -sh outputs/exp019/graddiff_{base,erased}
```

---

## 📋 Phase 5: 基线评估（30分钟）

### ✅ Task 5.1: NudeNet 评估
```bash
vi slurm/exp019_eval_baseline.sbatch
```

**脚本内容**:
```bash
#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -J exp019_eval_baseline
#SBATCH --array=0-1

ARMS=(graddiff_base graddiff_erased)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

python scripts/evaluate_videos_nudenet.py \
    --video_dir outputs/exp019/${ARM} \
    --output_file outputs/exp019_evaluation/${ARM}_evaluation.json \
    --threshold 0.6
```

**提交**:
```bash
sbatch slurm/exp019_eval_baseline.sbatch
```

---

### ✅ Task 5.2: 对比 GradAscent vs GradDiff
```bash
# 提取违规率
python3 << 'EOF'
import json

# GradAscent (Exp015)
ga_base = json.load(open('outputs/exp015_evaluation/grad_ascent_base_evaluation.json'))
ga_erased = json.load(open('outputs/exp015_evaluation/grad_ascent_erased_evaluation.json'))

# GradDiff (Exp019)
gd_base = json.load(open('outputs/exp019_evaluation/graddiff_base_evaluation.json'))
gd_erased = json.load(open('outputs/exp019_evaluation/graddiff_erased_evaluation.json'))

print("=== 擦除效果对比 ===")
print(f"GradAscent: {ga_base['nudenet']['summary']['violation_rate']:.1%} → {ga_erased['nudenet']['summary']['violation_rate']:.1%}")
print(f"GradDiff:   {gd_base['nudenet']['summary']['violation_rate']:.1%} → {gd_erased['nudenet']['summary']['violation_rate']:.1%}")
EOF
```

**预期发现**:
- GradDiff 应该比 GradAscent 擦除效果更好（因为有 retain 约束）

---

## 📋 Phase 6: 微调训练（24小时）

### ✅ Task 6.1: 准备微调脚本
```bash
vi slurm/exp019_finetune.sbatch
```

**脚本内容**（参考 Exp014/018）:
```bash
#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 30:00:00
#SBATCH -J exp019_ft

# 使用 Exp014 验证过的微调脚本
python examples/wanvideo/model_training/train.py \
    --model_id models/wan5b/exp019_graddiff_erased \
    --train_data data/tiger_dataset/metadata_100.csv \
    --output_dir models/finetune/exp019_graddiff_ft \
    --dataset_repeat 25 \
    --num_epochs 20 \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --save_every_n_epochs 1 \
    --no_tensorboard_log
```

**提交**:
```bash
sbatch slurm/exp019_finetune.sbatch
```

**预期时间**: ~24 小时

---

### ✅ Task 6.2: 监控微调进度
```bash
# 查看日志
tail -f outputs/exp019_ft_*.log

# 检查 checkpoint 生成
watch -n 300 "ls models/finetune/exp019_graddiff_ft/"

# 预期：20 个 epoch checkpoint
```

---

## 📋 Phase 7: 微调后生成（30分钟）

### ✅ Task 7.1: 生成微调后视频
```bash
vi slurm/exp019_generate_ft.sbatch
```

**脚本内容**:
```bash
#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -J exp019_gen_ft

python scripts/exp015_generate_videos.py \
    --method graddiff \
    --arm graddiff_ft_e19 \
    --model_path models/wan5b/exp019_graddiff_erased \
    --lora_path models/finetune/exp019_graddiff_ft/epoch-19.safetensors \
    --output_dir outputs/exp019/graddiff_ft_e19 \
    --num_videos 99
```

**验证**:
```bash
ls outputs/exp019/graddiff_ft_e19/*.mp4 | wc -l  # 应该是 99
```

---

## 📋 Phase 8: 微调后评估（30分钟）

### ✅ Task 8.1: NudeNet 评估
```bash
python scripts/evaluate_videos_nudenet.py \
    --video_dir outputs/exp019/graddiff_ft_e19 \
    --output_file outputs/exp019_evaluation/graddiff_ft_e19_evaluation.json
```

---

### ✅ Task 8.2: 计算回潮幅度
```bash
python3 << 'EOF'
import json

# GradAscent (Exp015 + Exp018)
ga_erased = json.load(open('outputs/exp015_evaluation/grad_ascent_erased_evaluation.json'))
ga_ft = json.load(open('outputs/exp018/evaluation/grad_ascent_ft_e19_results.json'))

# GradDiff (Exp019)
gd_erased = json.load(open('outputs/exp019_evaluation/graddiff_erased_evaluation.json'))
gd_ft = json.load(open('outputs/exp019_evaluation/graddiff_ft_e19_evaluation.json'))

print("=== 微调后回潮对比 ===")
ga_rebound = ga_ft['nudenet']['summary']['violation_rate'] - ga_erased['nudenet']['summary']['violation_rate']
gd_rebound = gd_ft['nudenet']['summary']['violation_rate'] - gd_erased['nudenet']['summary']['violation_rate']

print(f"GradAscent 回潮: {ga_rebound:.1%}")
print(f"GradDiff 回潮:   {gd_rebound:.1%}")
print(f"\n结论: GradDiff {'更鲁棒' if gd_rebound < ga_rebound else '不如 GradAscent 鲁棒'}")
EOF
```

---

## 📋 Phase 9: 结果整理与记录（30分钟）

### ✅ Task 9.1: 更新 experiments.md
```bash
# 填写完整的结果
# 包括：擦除效果、微调回潮、对比分析
```

**记录格式**:
```markdown
- **Results**: ✅ 完成
  - **擦除效果对比**:
    | 方法 | Base违规率 | Erased违规率 | 擦除效果 |
    |------|------------|--------------|----------|
    | GradAscent | 37.4% | 42.4% | +5.1pp |
    | **GradDiff** | XX.X% | XX.X% | XX.Xpp |
  
  - **微调回潮对比**:
    | 方法 | Erased | FT e19 | 回潮幅度 |
    |------|--------|--------|----------|
    | GradAscent | 42.4% | 48.5% | +6.1pp |
    | **GradDiff** | XX.X% | XX.X% | XX.Xpp |
  
  - **关键发现**:
    - Retain 约束是否有效？
    - GradDiff 微调鲁棒性如何？
```

---

### ✅ Task 9.2: 更新 QUICK_DATA_REFERENCE.md
```bash
# 添加 GradDiff 数据到汇总表
```

---

### ✅ Task 9.3: 生成对比图表（可选）
```python
# 创建可视化脚本
# 对比 5 种方法的擦除效果和微调鲁棒性
```

---

## 📋 Phase 10: 验证与检查（15分钟）

### ✅ Task 10.1: 文件完整性检查
```bash
# 检查所有产物是否存在
cat << 'EOF' | bash
echo "=== Exp019 产物检查 ==="

# 1. 擦除 LoRA
echo "擦除 LoRA:"
ls -lh models/unlearn/exp019_wan5b_nudity_graddiff/step-600

# 2. 合并模型
echo "合并模型:"
du -sh models/wan5b/exp019_graddiff_{base,erased}

# 3. 微调 LoRA
echo "微调 checkpoints:"
ls models/finetune/exp019_graddiff_ft/ | wc -l  # 应该是 20

# 4. 视频
echo "生成视频:"
for dir in graddiff_base graddiff_erased graddiff_ft_e19; do
    count=$(ls outputs/exp019/${dir}/*.mp4 2>/dev/null | wc -l)
    echo "  ${dir}: ${count} 视频"
done

# 5. 评估结果
echo "评估结果:"
ls -lh outputs/exp019_evaluation/*.json

EOF
```

**预期**:
- ✅ 7 个擦除 checkpoint
- ✅ 2 个合并模型
- ✅ 20 个微调 checkpoint
- ✅ 297 个视频（99+99+99）
- ✅ 3 个评估 JSON

---

### ✅ Task 10.2: 数据质量检查
```bash
# 检查评估结果是否合理
python3 << 'EOF'
import json

files = [
    'outputs/exp019_evaluation/graddiff_base_evaluation.json',
    'outputs/exp019_evaluation/graddiff_erased_evaluation.json',
    'outputs/exp019_evaluation/graddiff_ft_e19_evaluation.json'
]

for f in files:
    data = json.load(open(f))
    vr = data['nudenet']['summary']['violation_rate']
    total = data['nudenet']['summary']['total_videos']
    print(f"{f.split('/')[-1]}: {vr:.1%} ({total} videos)")
    
    # 检查异常
    if total != 99:
        print(f"  ⚠️  视频数量异常！应该是 99")
    if vr > 1.0 or vr < 0:
        print(f"  ⚠️  违规率异常！")
EOF
```

---

## 📝 总结与下一步

### ✅ Exp019 完成检查清单
- [ ] Phase 1: 准备与验证
- [ ] Phase 2: 擦除训练
- [ ] Phase 3: LoRA 合并
- [ ] Phase 4: 基线生成
- [ ] Phase 5: 基线评估
- [ ] Phase 6: 微调训练
- [ ] Phase 7: 微调后生成
- [ ] Phase 8: 微调后评估
- [ ] Phase 9: 结果整理
- [ ] Phase 10: 验证检查

### 📊 预期时间表
- Day 1: Phase 1-5 (准备 → 基线评估)
- Day 2-3: Phase 6 (微调训练 24小时)
- Day 4: Phase 7-10 (微调评估 → 整理)

### 🎯 完成后
1. 5 个基础方法对比完成
2. 验证 retain 约束的作用
3. 开始 Exp021（81帧验证）

---

## 🔗 相关文档

- 详细规划: `memory/exp019_supplementary_methods_plan.md`
- 方法分析: `memory/VU_METHODS_COMPLETE_ANALYSIS.md`
- 不可行原因: `memory/WHY_OTHER_METHODS_INFEASIBLE.md`
- Exp015 参考: `memory/experiments.md` Line 432

---

## ⚠️ 注意事项

1. **环境切换**: 擦除训练用 `vu` 环境，微调用 `diffsynth` 环境
2. **路径一致性**: 确保所有路径与 Exp015 对齐
3. **Job ID 记录**: 每次提交作业后立即记录 Job ID
4. **监控**: 长时间作业需要定期检查状态
5. **失败处理**: 如果任何步骤失败，查看日志并参考 Exp015 的解决方案

---

**祝实验顺利！有问题随时在新对话中提问。**
