# Exp020 - NPOMasked方法擦除效果评估

**创建日期**: 2026-09-16  
**状态**: 📋 待审核  
**预计工作量**: 2-3天（含训练+生成+评估）

---

## 🎯 实验目标

测试**NPOMasked**（NPO + attention mask）方法的擦除效果和微调鲁棒性，对比Exp015的plain NPO结果。

### 研究问题
1. **Attention mask能否改善NPO的擦除效果？**
   - Exp015中plain NPO失效（违规率+6.1pp）
   - Masked版本理论上更精准，避免全局退化
   
2. **Masked方法是否有更好的微调鲁棒性？**
   - 只改变局部特征，可能不容易完全回潮
   - 对比Exp018中plain NPO的+7.1pp回潮

---

## 📋 实验设计

### 对比基准
- **Exp015**: plain NPO（base违规率35.4% → erased 41.4%，失效+6.1pp）
- **Exp018**: plain NPO微调后48.5%（回潮+7.1pp）

### 实验臂设计（2臂，最小化工作量）
```
Exp020
├─ npo_masked_base      (原始模型，99个视频)
└─ npo_masked_erased    (NPOMasked擦除后，99个视频)

总计: 2臂 × 99视频 = 198个视频
```

**注意**：暂不包含微调臂，先验证擦除效果是否改善。如果擦除有效，再做Exp021测试微调鲁棒性。

---

## 🔧 技术实现

### Stage 1: 创建NPOMasked配置文件

**位置**: `/home/x_jiage/jiage/video-unlearning/configs/methods/training/npo_masked_wan.yaml`

**内容**（基于npo_masked_cogvideox.yaml改编）:
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
    # Wan模型使用cross-attention (attn2)
    attn_substring: "attn2"
    # Wan是纯cross-attention，text tokens不在query中
    text_len: 0
    is_joint: false
    # Wan的attention heads数量（需要验证）
    num_heads: 24
    # 17帧 480x736 视频 -> latent grid
    # 假设: 时间压缩4x, 空间压缩8x, patch_size=2
    # 17/4 ≈ 4, 480/8/2 = 30, 736/8/2 = 46
    frames: 4
    height: 30
    width: 46
```

**⚠️ 关键参数需要验证**:
1. **attn_substring**: Wan模型的cross-attention模块名称
2. **num_heads**: Wan DiT的attention head数量
3. **frames/height/width**: 潜变量grid尺寸（依赖VAE配置）

### Stage 2: 修改训练脚本支持NPOMasked

**文件**: `scripts/run_unlearn_wan5b.py`

**修改点**:
```python
# 在 method_name_map 中添加
method_name_map = {
    "GradAscent": "grad_ascent",
    "GradDiff": "grad_diff",
    "ESD": "esd",
    "NPO": "npo",
    "AnchorDistill": "anchor_distill",
    "NPOMasked": "npo_masked",  # 新增
}
```

### Stage 3: 创建SLURM脚本

**文件**: `slurm/exp020_unlearn_npo_masked.sbatch`

基于`exp015_unlearn_npo.sbatch`，关键修改:
```bash
METHOD="npo_masked"
OUTPUT_DIR="$FT_ROOT/models/unlearn/exp020_wan5b_nudity_npo_masked"

python3 scripts/run_unlearn_wan5b.py \
    --train \
    --method "NPOMasked" \
    --vu-root "$VU_ROOT" \
    ...
```

### Stage 4: 模型合并脚本

**文件**: `slurm/exp020_merge_npo_masked.sbatch`

复用Exp015的合并逻辑，输出到:
```
models/wan5b/exp020_npo_masked_erased/
├─ model_index.json
├─ diffusion_pytorch_model-00001-of-00003.safetensors
├─ diffusion_pytorch_model-00002-of-00003.safetensors
└─ diffusion_pytorch_model-00003-of-00003.safetensors
```

### Stage 5: 视频生成脚本

**文件**: `slurm/exp020_generate.sbatch`

**配置**:
- 提示词: SafeSora unsafe_181_noCA（99条，与Exp015对齐）
- 参数: 17帧, 480×736, 50步, cfg=5.0, seed=0
- 输出: `outputs/exp020/npo_masked_{base,erased}/`

### Stage 6: 评估脚本

**文件**: `slurm/exp020_evaluate.sbatch`

- 工具: NudeNet (threshold=0.6)
- 输出: `outputs/exp020_evaluation/npo_masked_{base,erased}_evaluation.json`

---

## 📊 预期产物

```
models/
└─ unlearn/exp020_wan5b_nudity_npo_masked/
   ├─ adapter_step100.pt ... adapter_step600.pt
   └─ adapter_final.pt
└─ wan5b/exp020_npo_masked_erased/
   └─ (合并后的完整模型)

outputs/
└─ exp020/
   ├─ npo_masked_base/          (99个视频)
   ├─ npo_masked_erased/        (99个视频)
   └─ evaluation/
      ├─ npo_masked_base_evaluation.json
      └─ npo_masked_erased_evaluation.json

memory/
└─ exp020_results.md            (结果分析)
```

---

## ⚙️ 执行流程

### 前置准备（必须先完成）

**1. 验证Wan模型的mask参数**
```python
# 脚本: scripts/debug/check_wan_attention.py
# 目的: 打印Wan DiT的attention模块结构
#   - 模块名称（确认attn_substring）
#   - num_heads
#   - latent shape (确认frames/height/width)
```

**2. 创建配置文件**
- 在VU项目创建 `configs/methods/training/npo_masked_wan.yaml`
- 基于验证结果填入正确参数

**3. 测试mask提取（smoke test）**
```bash
# 在VU环境下运行小规模测试
# 验证mask能正确构建，不报错
python tests/test_npo_masked.py
```

### 执行阶段

**Stage 1: 擦除训练** (预计30分钟)
```bash
sbatch slurm/exp020_unlearn_npo_masked.sbatch
# 预计Job ID: 175XXXXX
# 产物: adapter_final.pt (600步，约30分钟)
```

**Stage 2: 模型合并** (预计3分钟)
```bash
sbatch slurm/exp020_merge_npo_masked.sbatch
# 预计Job ID: 175XXXXX
# 产物: 合并后的完整模型 (~10GB)
```

**Stage 3: 视频生成** (预计2小时)
```bash
sbatch slurm/exp020_generate_base.sbatch      # base模型 99条
sbatch slurm/exp020_generate_erased.sbatch    # erased模型 99条
# 预计Job IDs: 175XXXXX, 175XXXXX
# 产物: 198个视频
```

**Stage 4: 安全评估** (预计30分钟)
```bash
sbatch slurm/exp020_evaluate.sbatch
# 预计Job ID: 175XXXXX
# 产物: 2个JSON评估文件
```

**Stage 5: 结果分析**
```bash
python scripts/analyze_exp020_results.py
# 生成对比报告: memory/exp020_results.md
```

---

## 🎯 判定标准

### 成功标准（值得继续）
- ✅ **擦除有效**: npo_masked_erased 违规率 < npo_masked_base
- ✅ **优于plain NPO**: 下降幅度 > 0pp（Exp015是+6.1pp）
- ✅ **接近或超过ESD**: 下降幅度接近-5.1pp

**如果达标** → 继续做Exp021测试微调鲁棒性

### 失败标准（放弃masked方法）
- ❌ 违规率上升（mask没有帮助）
- ❌ 下降幅度 < 2pp（改善微弱）

**如果失败** → 放弃masked方法，优先级转向：
1. 补充Exp017评估（495个蒸馏视频）
2. 核查Exp015的擦除配置问题
3. 探索其他改进方向

---

## ⚠️ 风险与注意事项

### 技术风险

**1. Mask参数配置错误** (高风险)
- **风险**: attn_substring/num_heads/grid尺寸错误导致训练崩溃
- **缓解**: 必须先运行smoke test验证

**2. VU环境依赖冲突** (中风险)
- **风险**: NPOMasked依赖VU的attention_mask.py等模块
- **缓解**: 确保在VU的conda环境下运行训练

**3. 计算成本** (低风险)
- **风险**: mask计算增加训练时间
- **缓解**: 2臂设计最小化工作量，预计仅增加5-10%时间

### 实验设计风险

**1. Base臂冗余** (低影响)
- npo_masked_base理论上应该≈Exp015的base（原始模型）
- 可以考虑复用Exp015的base结果，只生成erased臂
- **建议**: 还是生成完整2臂，确保严格可比

**2. 评测集差异** (已规避)
- 使用与Exp015相同的99条提示词（unsafe_181_noCA的前99条）
- 确保横向可比

---

## 📈 预期发现

基于VU项目注释和理论分析：

### 乐观情景（mask有效）
- **擦除阶段**: 违规率下降2-4pp
- **机制**: mask精准定位unsafe区域，避免全局质量退化
- **后续**: 值得测试微调鲁棒性（Exp021）

### 中性情景（mask轻微改善）
- **擦除阶段**: 违规率下降0-2pp
- **解读**: mask有帮助但效果有限
- **后续**: 可能不值得继续投入masked系列

### 悲观情景（mask无效）
- **擦除阶段**: 违规率持平或上升
- **原因**: NPO本身的bounded objective可能已经够好，mask无额外收益
- **后续**: 放弃masked方向

---

## 🔗 相关文档

- **Exp015记录**: `memory/experiments.md` (plain NPO基线)
- **Exp018记录**: `memory/experiments.md` (plain NPO微调回潮)
- **VU masked实现**: `/home/x_jiage/jiage/video-unlearning/src/unlearning/training/npo_masked.py`
- **VU mask配置**: `/home/x_jiage/jiage/video-unlearning/configs/methods/training/npo_masked_cogvideox.yaml`

---

## 📝 待确认项（需要用户决策）

### 1. Mask参数验证优先级
- [ ] 选项A: 先写smoke test脚本验证参数（推荐，但需额外1-2小时）
- [ ] 选项B: 直接使用估算参数提交训练（快速，但可能失败）

### 2. Base臂生成策略
- [ ] 选项A: 生成完整2臂（严格可比，但多花1小时）
- [ ] 选项B: 复用Exp015 base结果，只生成erased臂（节省时间）

### 3. 失败后的备选方案
- [ ] 继续测试GradDiffMasked/GradAscentMasked（完整masked benchmark）
- [ ] 放弃masked方向，优先补Exp017评估
- [ ] 核查Exp015配置问题，重跑失效的方法

---

**提交审核**: 请审核以上计划，特别关注：
1. Mask参数估算是否合理
2. 实验设计是否遗漏关键点
3. 风险评估是否充分
4. 是否需要调整实验范围（如增加微调臂）
