# Exp020 - VU Masked方法全覆盖实验

**创建日期**: 2026-09-16  
**状态**: 📋 计划中  
**目标**: 完整覆盖VU项目的masked方法变体

---

## 🎯 实验目标

**将VU项目的masked方法全部测试**，形成完整的 `{plain, masked} × {NPO, GradAscent, GradDiff}` benchmark。

### 动机
- Exp015测试了4种plain方法（ESD, NPO, GradAscent, AnchorDistill）
- VU项目还有3种masked变体：NPOMasked, GradAscentMasked, GradDiffMasked
- Masked方法通过attention mask限制擦除区域，理论上更精准、更鲁棒

---

## 📊 实验设计

### 分期执行策略

由于是探索性实验，采用**分期验证**策略：

```
Exp020a - NPOMasked        (先验证masked机制是否work)
Exp020b - GradDiffMasked   (如果020a成功)
Exp020c - GradAscentMasked (如果020a/b成功)
```

**只生成erased臂**（base臂复用Exp015结果）

### Exp020a - NPOMasked

**实验臂**: 1臂 × 99视频 = 99个视频
```
npo_masked_erased    (NPOMasked擦除后)
```

**对比基准**: 
- Exp015 base: 35.4%
- Exp015 npo_erased: 41.4% (+6.1pp，失效)

**判定标准**:
- ✅ 成功: 违规率 < 35.4% (优于base)
- ⚠️ 部分成功: 35.4% ~ 41.4% (介于base和plain NPO之间)
- ❌ 失败: > 41.4% (不如plain NPO)

---

## 🔧 技术实现

### 前置工作: Smoke Test验证

**目的**: 验证Wan模型的mask参数配置

**脚本**: `scripts/debug/smoke_test_wan_mask.py`

**验证内容**:
1. Wan DiT的attention模块名称（确认`attn_substring`）
2. Attention heads数量（`num_heads`）
3. Latent shape（确认`frames/height/width`）

**预期输出**:
```
Wan DiT Attention结构:
- Cross-attention模块: attn2
- Num heads: 24
- Latent shape (17f 480x736): (B, 4, 16, 60, 92)
- Patch后grid: (4, 30, 46)
```

### Stage 1: 创建配置文件

**文件**: `/home/x_jiage/jiage/video-unlearning/configs/methods/training/npo_masked_wan.yaml`

**内容** (基于smoke test结果):
```yaml
handler: NPOMasked
type: training
method_args:
  concept_manifest: ???
  beta: 0.1
  retain_manifest: null
  retain_weight: 0.0
  target: transformer
  batch_size: 1
  lora_rank: 8
  lora_alpha: 16.0
  layer_start: 0
  layer_end: 29
  guidance_scale: 1.0
  sigma_distribution: logit_normal
  timestep_shift: null
  mask_config:
    attn_substring: "attn2"      # 从smoke test获取
    text_len: 0
    is_joint: false
    num_heads: 24                # 从smoke test获取
    frames: 4                    # 从smoke test获取
    height: 30                   # 从smoke test获取
    width: 46                    # 从smoke test获取
```

### Stage 2: 修改训练脚本

**文件**: `scripts/run_unlearn_wan5b.py`

**修改**: 在`method_name_map`添加masked方法支持
```python
method_name_map = {
    "GradAscent": "grad_ascent",
    "GradDiff": "grad_diff",
    "ESD": "esd",
    "NPO": "npo",
    "AnchorDistill": "anchor_distill",
    # Masked variants
    "NPOMasked": "npo_masked",
    "GradDiffMasked": "grad_diff_masked",
    "GradAscentMasked": "grad_ascent_masked",
}
```

### Stage 3: SLURM脚本

**擦除训练**: `slurm/exp020a_unlearn_npo_masked.sbatch`
```bash
METHOD="npo_masked"
OUTPUT_DIR="$FT_ROOT/models/unlearn/exp020a_wan5b_nudity_npo_masked"
UNLEARN_STEPS=600
SAVE_EVERY=100

python3 scripts/run_unlearn_wan5b.py \
    --train \
    --method "NPOMasked" \
    --vu-root "$VU_ROOT" \
    --model-path "$MODEL_PATH" \
    --latent-manifest "$LATENT_CACHE_DIR/latent_manifest.jsonl" \
    --train-dir "$OUTPUT_DIR" \
    --unlearn-steps "$UNLEARN_STEPS" \
    --save-every "$SAVE_EVERY"
```

**模型合并**: `slurm/exp020a_merge_npo_masked.sbatch`
```bash
INPUT_DIR="$FT_ROOT/models/unlearn/exp020a_wan5b_nudity_npo_masked"
OUTPUT_DIR="$FT_ROOT/models/wan5b/exp020a_npo_masked_erased"

python3 scripts/merge_lora_wan5b.py \
    --base-model "$MODEL_PATH" \
    --lora-path "$INPUT_DIR/adapter_final.pt" \
    --output-dir "$OUTPUT_DIR"
```

**视频生成**: `slurm/exp020a_generate.sbatch`
```bash
# 只生成erased臂
MODEL_DIR="$FT_ROOT/models/wan5b/exp020a_npo_masked_erased"
OUTPUT_DIR="$FT_ROOT/outputs/exp020a/npo_masked_erased"

python3 scripts/generate_wan5b.py \
    --model-path "$MODEL_DIR" \
    --prompts-file "$PROMPTS" \
    --output-dir "$OUTPUT_DIR" \
    --num-frames 17 \
    --height 480 \
    --width 736 \
    --num-inference-steps 50 \
    --guidance-scale 5.0 \
    --seed 0
```

**评估**: `slurm/exp020a_evaluate.sbatch`
```bash
python3 scripts/eval_nudity_wan5b.py \
    --video-dir "$FT_ROOT/outputs/exp020a/npo_masked_erased" \
    --output-file "$FT_ROOT/outputs/exp020a_evaluation/npo_masked_erased_evaluation.json"
```

---

## 📊 预期产物

```
models/
└─ unlearn/exp020a_wan5b_nudity_npo_masked/
   ├─ adapter_step100.pt
   ├─ adapter_step200.pt
   ├─ ...
   ├─ adapter_step600.pt
   └─ adapter_final.pt
└─ wan5b/exp020a_npo_masked_erased/
   └─ (合并后的完整模型)

outputs/
└─ exp020a/
   ├─ npo_masked_erased/        (99个视频)
   └─ evaluation/
      └─ npo_masked_erased_evaluation.json

memory/
└─ exp020a_results.md            (结果分析)
```

---

## ⚙️ 执行流程

### Phase 1: Smoke Test (预计30分钟)
```bash
# 开发并运行smoke test
python3 scripts/debug/smoke_test_wan_mask.py
# 验证输出，确认mask参数
```

### Phase 2: 配置准备 (预计15分钟)
```bash
# 创建npo_masked_wan.yaml（在VU项目中）
# 修改run_unlearn_wan5b.py支持masked方法
# 创建所有SLURM脚本
```

### Phase 3: 擦除训练 (预计30分钟)
```bash
sbatch slurm/exp020a_unlearn_npo_masked.sbatch
# Job ID: 记录到experiments.md
```

### Phase 4: 合并+生成+评估 (预计2-3小时)
```bash
sbatch slurm/exp020a_merge_npo_masked.sbatch
sbatch --dependency=afterok:<merge_job_id> slurm/exp020a_generate.sbatch
sbatch --dependency=afterok:<generate_job_id> slurm/exp020a_evaluate.sbatch
```

### Phase 5: 结果分析
```bash
python3 scripts/analyze_exp020a_results.py
# 生成对比报告，对比Exp015 NPO结果
```

---

## 🎯 决策树

```
Exp020a结果
    │
    ├─ 成功 (违规率 < 35.4%)
    │   ├─ 继续 Exp020b (GradDiffMasked)
    │   └─ 继续 Exp020c (GradAscentMasked)
    │
    ├─ 部分成功 (35.4% ~ 41.4%)
    │   ├─ 调试mask配置，看能否改进
    │   └─ 如果无法改进，记录结果但不继续020b/c
    │
    └─ 失败 (> 41.4%)
        ├─ 调试mask配置
        ├─ 检查训练日志，看mask是否生效
        └─ 如果确认masked方法无效，放弃整个Exp020系列
```

---

## ⚠️ 关键注意事项

### 1. Base臂不需要单独生成
- **原因**: npo_masked_base应该等同于原始模型
- **复用**: 直接使用Exp015的base结果（35.4%）作为对比基准
- **节省**: 省去99个视频生成，节省~1小时

### 2. Smoke Test是必需的
- Mask参数错误会导致训练崩溃或mask不生效
- 必须先验证Wan模型的attention结构
- 如果smoke test失败，说明需要调整mask_config

### 3. 调试策略
如果训练失败：
1. 检查attention module是否被正确hook
2. 验证mask shape是否匹配latent shape
3. 查看训练日志中的loss值（是否异常）
4. 必要时降低学习率或调整mask_config

### 4. VU环境依赖
- NPOMasked依赖VU的`attention_mask.py`模块
- 必须在VU的conda环境下运行
- 确保`sys.path.insert(0, VU_ROOT)`生效

---

## 📈 预期结果

### 乐观情景
- NPOMasked违规率: 30-33% (优于plain NPO和base)
- 机制: mask精准定位unsafe区域，避免全局退化
- 后续: 继续测试GradDiffMasked和GradAscentMasked

### 中性情景
- NPOMasked违规率: 35-38% (优于plain NPO，接近base)
- 机制: mask有轻微帮助
- 后续: 尝试调整mask_config，或直接测试其他masked方法

### 悲观情景
- NPOMasked违规率: > 41% (不如plain NPO)
- 原因: mask配置错误，或masked对NPO无帮助
- 后续: 调试配置，如果仍失败则放弃masked系列

---

## 🔗 相关实验

- **Exp015**: 4种plain方法基线（NPO失效+6.1pp）
- **Exp016**: 量化对擦除效果的影响
- **Exp017**: 蒸馏对擦除效果的影响（缺评估）
- **Exp018**: 微调回潮测试（NPO回潮+7.1pp）

---

## 📝 后续扩展（如果020a成功）

### Exp020b - GradDiffMasked
- 测试GradDiff的masked版本
- 理论上应该比plain GradDiff更稳定（避免dithering）
- 对比基准: Exp015 GradDiff（不支持/未测试）

### Exp020c - GradAscentMasked
- 测试GradAscent的masked版本
- VU注释说"expected to fail"，但仍需验证
- 对比基准: Exp015 GradAscent（+5.1pp失效）

### Exp021 - Masked方法微调鲁棒性
- 如果Exp020a/b/c中有方法擦除有效
- 测试微调后的回潮情况
- 对比Exp018的plain方法微调结果

---

**状态**: 等待用户确认后开始实施
**下一步**: 开发smoke test脚本
