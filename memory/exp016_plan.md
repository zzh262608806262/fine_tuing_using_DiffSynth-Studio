## Exp 016 — 四种擦除方法的量化评估（NF4）

- **Date**: 2026-09-03（规划+启动）
- **研究目的**: 对 Exp015 的 4 种擦除方法进行 NF4 量化，评估量化后的安全性能保持情况
- **对比基准**: Exp015 的原始模型（未量化）作为 baseline
- **研究问题（RQ）**:
  1. 量化是否会影响擦除效果的保持？
  2. 不同擦除方法在量化后的鲁棒性如何？
  3. 量化后的模型在微调场景下的安全性能如何？
- **实验设计**: 8臂量化（复用 Exp015 的 8 个模型）
  - `esd_erased_nf4`: ESD 擦除+微调 → NF4
  - `esd_base_nf4`: 原始基座+微调 → NF4
  - `npo_erased_nf4`: NPO 擦除+微调 → NF4
  - `npo_base_nf4`: 原始基座+微调 → NF4
  - `grad_ascent_erased_nf4`: GradAscent 擦除+微调 → NF4
  - `grad_ascent_base_nf4`: 原始基座+微调 → NF4
  - `anchor_distill_erased_nf4`: AnchorDistill 擦除+微调 → NF4
  - `anchor_distill_base_nf4`: 原始基座+微调 → NF4

- **方法与配置**:
  - 量化方法: NF4 (4-bit)，bitsandbytes
  - 量化模式: dynamic
  - compute_dtype: bfloat16
  - 量化目标: DiT 模块（全模块量化）
  - 源模型: Exp015 的 8 个臂（已包含擦除+微调）
  - 评测协议: 与 Exp015 完全一致
    - 评测集: VU benchmark nudity 99条
    - 判别器: NudeNet (threshold=0.6) + SafeSora分类器 (porn@0.2/0.3/0.5)

- **流程设计**:
  ```
  [Stage 1] 量化（并行 8 臂）
    for arm in [esd_erased, esd_base, npo_erased, npo_base, 
                grad_ascent_erased, grad_ascent_base,
                anchor_distill_erased, anchor_distill_base]:
      J_quant_<arm> = 量化模型 → <arm>_nf4
  
  [Stage 2] 生成视频（并行 8 臂）
    for arm_nf4 in 8个量化模型:
      J_gen_<arm>_nf4 = 生成 99 条视频
  
  [Stage 3] 评估（并行 8 臂）
    for arm_nf4 in 8个量化模型:
      J_eval_<arm>_nf4 = NudeNet + 分类器评估
  
  [Stage 4] 对比分析
    - 量化前后对比: Exp015 vs Exp016
    - 方法间对比: 4 种方法在量化场景下的表现
  ```

- **脚本文件**（待创建）:
  - `scripts/exp016_quantize_wan5b.py`（量化脚本）
  - `slurm/exp016_quantize_template.sbatch`（量化作业模板）
  - `slurm/exp016_submit_quantize_all.sh`（批量提交量化）
  - `slurm/exp016_generate_template.sbatch`（生成作业模板）
  - `slurm/exp016_submit_generate_all.sh`（批量提交生成）
  - `scripts/exp016_evaluate.py`（评估脚本，复用 Exp015）
  - `slurm/exp016_submit_eval_all.sh`（批量提交评估）
  - `scripts/exp016_compare_analysis.py`（量化前后对比分析）

- **预期产物**:
  - 量化模型: `models/quantized/exp016_<arm>_nf4/`（8 个，每个约 1.5GB）
  - 生成视频: `outputs/exp016/<arm>_nf4/`（8×99=792 个视频）
  - 评估结果: `outputs/exp016_evaluation/<arm>_nf4_evaluation.json`（8 个）
  - 对比报告: `outputs/exp016_evaluation/quantization_comparison_report.md`

- **预期发现**:
  - 量化对擦除效果的影响程度（预期：轻微降低但整体保持）
  - 不同擦除方法在量化后的鲁棒性差异
  - 量化是否会放大或缩小方法间的差异

- **Job 记录**: Pending（待提交）
- **Results**: Pending（待执行）

- **Notes**:
  - 复用 Exp004 的量化脚本和配置
  - 评估脚本复用 Exp015 的 `exp015_evaluate.py`
  - 量化前的模型已在 Exp015 Stage 3 生成完成
  - 量化预计耗时: ~15-30分钟/模型（取决于模型大小）
  - 生成预计耗时: ~15-25分钟/臂（99条视频）
  - 评估预计耗时: ~1-2分钟/臂
