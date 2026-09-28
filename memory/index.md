# Memory - 文件索引

## Record Schema

每个被索引文件占一行，使用 `| 相对路径 | 一句话说明 |`。路径相对项目根目录；说明只描述文件当前职责，不记录临时状态或实验结果。新增/删除/重命名受索引约束的文件后必须同步更新本表。

| 文件 | 说明 |
|------|------|
| memory/index.md | 项目记忆索引（本文件） |
| memory/MASTER_SUMMARY.md | ⭐️ 项目总览与快速索引（新对话先读这个） |
| memory/QUICK_DATA_REFERENCE.md | ⭐️ 实验数据快速查询手册（查数据用这个） |
| memory/project_details.md | 项目定位、环境、权重/数据入口和评测口径；只保留影响判断的摘要 |
| memory/experiments.md | 按实验编号逐条记录配置、Job、产物和结果；已更新至 Exp 018（含完整结果） |
| memory/exp015_execution_plan.md | Exp 015 执行计划：6 阶段任务清单、Job 跟踪表、脚本列表、时间估算、验证检查点 |
| memory/exp015_errors.md | Exp 015 错误记录：环境配置、接口理解相关的 6 个错误及解决方案 |
| memory/safesora_label_audit.md | SafeSora 标注可信度审计简报（Exp 010）：术语表、数据集标注问题证据链、分类器可靠性结论、生成视频安全评估五步方案 |
| memory/errors.md | 踩坑记录 |
| memory/gpu_node_apply.md | GPU 节点申请与连接方法（init1g/init1gf/jobsh/squeue/scancel） |
| memory/exp009_multi_judge_comparison.md | Exp009 需求与可复用流程文档：五方法生成视频的多判别器安全评估（分类器多阈值+Qwen+GPT4o） |
| memory/exp010_safesora_test_multi_judge.md | Exp010 需求与流程文档：SafeSora 测试集三方判别器 vs 人工标注 |
| memory/improve_safe_binary_accuracy.md | 提升 Safe/Unsafe 二分类准确率的两阶段改进计划（阈值优化+pos_weight 重训） |
| memory/todo.md | 仅由用户变更状态的任务清单 |
| memory/script_renaming_pitfall.md | 脚本重命名陷阱记录（unlearn_wan5b.py 等常见错误） |
| docs/exp016_quantization_debugging.md | Exp016 量化调试完整记录（v1-v8，7次失败+最终成功方案） |
| scripts/lora_finetune.py | Wan2.1-T2V-1.3B LoRA 微调入口 |
| scripts/distill.py | 4-step 蒸馏训练入口（LoRA 形式） |
| scripts/quantize.py | DiT nf4 量化的保存/加载/冒烟入口 |
| scripts/caption_tiger_clips.py | 用 Qwen3-VL-8B 重打标 tiger200k clips（原始 caption CSV 丢失后的替代） |
| scripts/generate_safesora.py | SafeSora unsafe-200 四方法批量生成（断点续跑、分片、确定性抽样） |
| scripts/classify_safesora_gen.py | 用安全分类器给 safesora_gen 生成视频打标签并统计 safe 率 |
| scripts/analyze_safesora_test_judges.py | SafeSora test 集四方对照分析（一致性/AUC/阈值扫描/校准/冲突样本），产出 analysis_report.txt |
| scripts/analyze_gen_judges_v2.py | 五方法生成视频的三判别器交叉分析：判别器一致性、方法排序稳健性、分类别谱系、taxonomy 变更影响 |
| classify/configs/unified_label_definitions.json | 统一 taxonomy（safe + SafeSora 官方 S1–S12 共 13 类，与分类器 label_names 顺序一致）；旧配置 safesora_label_definitions.json 保留给其他项目 |
| classify/ | SafeSora 安全分类器（SigLIP+时序 Transformer）训练/推理/评估包 |
| classify/inference/predict.py | SafetyPredictor：单/批量推理 + REINS 兼容 API，定义 unsafe 判定口径；`unsafe_rule` 可选 any_class(默认,历史行为)/one_minus_safe |
| slurm/gen_safesora.sbatch | 生成作业模板（method/shard 参数化） |
| slurm/mj2_{cls,qwen,gpt4o}.sbatch | Exp011：五方法生成视频的三判别器评估（统一 taxonomy + 新判定口径），输出 multi_judge_v2/ |
| slurm/gen_watchdog.sh | 登录节点 nohup 看门狗：作业断了自动续提，5 次无进展写 FAILED 停 |
| weights/ | 各方法权重统一软链目录（实际文件在 models/ 下未移动） |
| outputs/safesora_gen/ | 四方法生成视频、prompts manifest 与分类结果（classify_results/） |
| outputs/safesora_test_multi_judge/ | SafeSora test 集 5745 条的三判别器打标结果与分析报告 |
| outputs/safesora_gen/multi_judge_v2/ | Exp011：五方法生成视频在统一 taxonomy + 新判定口径下的判别结果 |
| outputs/safesora_safety_classifier/ | 分类器训练产物（best.pt=epoch6, acc 0.8028） |
| specs/unlearn-then-finetune-wan5b/spec.md | Exp 012 研究规范：擦除后微调对 5B 安全影响（四臂 porn 专项），含 Why/文件范围约束/现成资产/ADDED Requirements |
| specs/unlearn-then-finetune-wan5b/tasks.md | Exp 012 任务清单（Task 0–5，已全部完成脚本工程，集群实跑待确认） |
| specs/unlearn-then-finetune-wan5b/checklist.md | Exp 012 验收清单（待集群执行后逐项验证勾选） |
| scripts/convert_wan_5b_bridge.py | Wan2.2-TI2V-5B diffusers→DiffSynth 权重桥（键映射、分片、哈希复核） |
| scripts/run_unlearn_wan5b.py | VU 只读驱动 5B nudity 擦除三阶段（latent cache→GradAscent 训练→探针生成） |
| scripts/merge_unlearn_lora_wan5b.py | 擦除 LoRA 合并进 diffusers 基座（W+α/r·up@down）并桥回 DiffSynth 擦除后基座 |
| scripts/finetune_wan5b.py | 5B 低曝光良性微调薄包装（base/erased 两臂，文本驱动，.safetensors pattern） |
| scripts/generate_wan5b_eval.py | 5B 四臂评测视频生成（VU nudity 评测集，分片续跑，同 seed 同参数） |
| scripts/eval_porn_wan5b.py | porn 专项判别（安全分类器 porn 类多阈值 + VU NudeNet；可选 Qwen3-VL），跨臂 summary |
| slurm/wan5b_unlearn.sbatch | 5B 擦除作业模板（cache+train；RUN_PROBE=1 追加探针） |
| slurm/wan5b_merge_bridge.sbatch | 5B 擦除 LoRA 合并+桥回作业模板 |
| slurm/wan5b_finetune.sbatch | 5B 两臂微调作业模板 |
| slurm/wan5b_eval_gen.sbatch | 5B 四臂评测生成作业模板（ARM/BEGIN/END 参数化） |
| slurm/wan5b_eval_judge.sbatch | 5B 评测判别作业模板（JUDGE_ARMS/SETS/MODE） |
| slurm/eval_nudity_wan5b_quant.sbatch | 5B 量化模型 NudeNet 评估作业（Exp 013，调用 VU 检测器） |
| slurm/eval_nudity_wan5b_orig.sbatch | 5B 原始模型 NudeNet 评估作业（Exp 013） |
| scripts/eval_nudity_wan5b_quant.py | NudeNet 评估脚本（封装 VU 项目检测器，Exp 013） |
| slurm/exp014_wan5b_unlearn.sbatch | Exp 014 擦除训练作业（rank=8, 600步，VU标准配置） |
| slurm/exp014_wan5b_merge.sbatch | Exp 014 LoRA合并作业（创建base和erased模型） |
| slurm/exp014_wan5b_finetune.sbatch | Exp 014 微调作业（repeat=25, epochs=20, 20个checkpoint） |
| slurm/exp014_wan5b_quantize.sbatch | Exp 014 量化作业（NF4，4臂×42模型） |
| slurm/exp014_wan5b_evaluate.sbatch | Exp 014 评估作业（NudeNet，42模型） |
| slurm/exp014_wan5b_pipeline.sbatch | Exp 014 完整流程编排（擦除→合并→微调→量化→评估） |
| scripts/exp015_generate_videos.py | Exp015 8臂视频生成（4方法×2版本，99条nudity prompts） |
| scripts/exp016_quantize.py | Exp016 量化脚本（加载LoRA后NF4量化DiT） |
| scripts/exp016_generate_videos.py | Exp016 量化模型视频生成（8臂×99条） |
| scripts/submit_exp016_all.sh | Exp016 批量提交8臂量化任务 |
| scripts/submit_exp016_gen_all.sh | Exp016 批量提交8臂生成任务 |
| slurm/exp016_quant_template.sbatch | Exp016 量化作业模板（ARMNAME参数化） |
| slurm/exp016_gen_template.sbatch | Exp016 生成作业模板（ARMNAME参数化） |
| scripts/run_unlearn_wan5b_multi_method.py | Exp 015 多方法擦除驱动脚本（支持 GradAscent/GradDiff/ESD/NPO/AnchorDistill） |
| slurm/exp015_unlearn_template.sbatch | Exp 015 擦除训练模板（方法参数化） |
| slurm/exp015_unlearn_grad_diff.sbatch | Exp 015 GradDiff 擦除训练作业 |
| slurm/exp015_unlearn_esd.sbatch | Exp 015 ESD 擦除训练作业 |
| slurm/exp015_unlearn_npo.sbatch | Exp 015 NPO 擦除训练作业 |
| slurm/exp015_unlearn_anchor_distill.sbatch | Exp 015 AnchorDistill 擦除训练作业 |
| slurm/exp015_merge_template.sbatch | Exp 015 LoRA 合并模板 |
| slurm/exp015_merge_grad_diff.sbatch | Exp 015 GradDiff LoRA 合并作业 |
| slurm/exp015_merge_esd.sbatch | Exp 015 ESD LoRA 合并作业 |
| slurm/exp015_merge_npo.sbatch | Exp 015 NPO LoRA 合并作业 |
| slurm/exp015_merge_anchor_distill.sbatch | Exp 015 AnchorDistill LoRA 合并作业 |
| slurm/exp015_finetune_template.sbatch | Exp 015 微调模板 |
| slurm/exp015_finetune_grad_diff.sbatch | Exp 015 GradDiff 微调作业 |
| slurm/exp015_finetune_esd.sbatch | Exp 015 ESD 微调作业 |
| slurm/exp015_finetune_npo.sbatch | Exp 015 NPO 微调作业 |
| slurm/exp015_finetune_anchor_distill.sbatch | Exp 015 AnchorDistill 微调作业 |
| slurm/exp015_submit.sh | Exp 015 全流程提交脚本（Stage 1 擦除训练） |
| slurm/exp015_unlearn_grad_ascent.sbatch | Exp 015 GradAscent 擦除训练作业（独立脚本，不复用 Exp012） |
| slurm/exp015_merge_grad_ascent.sbatch | Exp 015 GradAscent LoRA 合并作业 |
| docs/exp015_execution_plan.md | Exp 015 执行计划：6 阶段任务清单、脚本列表、时间估算、验证检查点 |
| docs/exp015_errors.md | Exp 015 错误记录：环境配置、方法理解相关的 5 个错误及解决方案 |
| .claude/CLAUDE.md | AI 协作指南（文件边界约束、VU 引用规范、实验记录要求） |

## Exp015 Stage 4 新增文件

### data/exp015/
- `nudity_eval_prompts.jsonl`: 评测 prompts（99条，来自 VU benchmark_wan.jsonl nudity domain）

### scripts/
- `exp015_generate_videos.py`: Stage 4 视频生成（8臂 × 99条）

### slurm/
- `exp015_generate.sbatch`: Stage 4 视频生成（单臂）
- `exp015_submit_generate_all.sh`: Stage 4 批量提交（8臂）

## Wan5B 蒸馏相关文件（2026-08-31 新增）

### scripts/
- `distill_wan5b.py`: Wan2.2-TI2V-5B 蒸馏脚本（基于 Wan2.1-1.3B distill.py 改编，支持 train/validate 模式，目标 4-8 步推理）
- `exp017_distill_erased.py`: Exp 017 擦除后模型蒸馏脚本（对已擦除模型进行 4 步蒸馏训练）

### slurm/
- `distill_wan5b.sbatch`: Wan2.2-TI2V-5B 蒸馏训练作业（全参数蒸馏，direct_distill 任务）
- `validate_distill_wan5b.sbatch`: Wan2.2-TI2V-5B 蒸馏模型验证作业（快速推理测试，支持 T2V/TI2V）
- `exp017_distill.sbatch`: Exp 017 蒸馏训练作业（参数化 ARM：base/grad_ascent/esd）
- `exp017_submit_all.sh`: Exp 017 全流程提交脚本（Stage 1-3：蒸馏→生成→评估）

### memory/
- `exp017_plan.md`: Exp 017 实验设计（擦除后蒸馏的安全保持性，6臂对比，完整流程）
- `exp017_quickref.md`: Exp 017 快速参考（一键启动、配置、监控、故障排除）
- `exp017_implementation_summary.md`: Exp 017 实现总结（完成情况、使用方法、预期结果、注意事项）
- `wan5b_distill_implementation_summary.md`: Wan5B 蒸馏实现总结（任务完成情况、文件清单、使用方法、验证结果）
- `wan5b_distillation_guide.md`: Wan5B 蒸馏完整指南（背景、使用方法、常见问题、对比分析）
- `wan5b_distillation_quickref.md`: Wan5B 蒸馏快速参考（一键启动、配置选项、常见问题速查）
| memory/exp015_exp018_baseline_data.md | Exp015 & Exp018 基线评估数据（4方法擦除效果、安全反弹指标、完整数据表） |
| memory/exp019_supplementary_methods_plan.md | Exp019 VU方法补充计划（GradDiff/ESDGen/FTTP，三个方案对比） |
| memory/EXP019_TODOLIST.md | ⭐️ Exp019 完整执行清单（10个Phase，可直接在新对话中使用） |
| memory/exp020_masked_methods_design.md | ⭐️ Exp020 Masked方法实验设计（验证空间约束能否解决齐步降解） |
| memory/EXP020A_TODOLIST.md | ⭐️ Exp020a 执行清单（GradDiff-masked最小验证，含关键决策点） |
| memory/EXP017_EVAL_TODOLIST.md | ⭐️ Exp017 评估补充清单（495个蒸馏视频NudeNet评估） |
| memory/exp019_status.md | Exp019 执行状态报告（Phase 完成状态、已创建脚本、问题修复、时间线） |
| memory/VU_METHODS_COMPLETE_ANALYSIS.md | ⭐️ VU 方法完整分析与实验路线图（必读：后续实验规划） |
| memory/WHY_OTHER_METHODS_INFEASIBLE.md | ⭐️ 其他方法不可行的详细解释（澄清 ESD vs ESDGen，解释 Masked/Encoder/FTTP 为何不做） |
| scripts/exp019_monitor.sh | Exp019 进度监控和结果对比脚本（检查各阶段完成情况+GradAscent对比） |
| scripts/exp019_submit_chain.sh | Exp019 自动化提交脚本（依赖链编排，从擦除到评估全流程） |
| slurm/exp019_unlearn_graddiff.sbatch | Exp019 GradDiff 擦除训练作业（600步，带 retain 约束） |
| slurm/exp019_merge_base.sbatch | Exp019 base 模型合并作业（无擦除 LoRA） |
| slurm/exp019_merge_erased.sbatch | Exp019 erased 模型合并作业（擦除 LoRA step-600） |
| slurm/exp019_generate_baseline.sbatch | Exp019 基线视频生成作业（base/erased，2臂×99视频） |
| slurm/exp019_eval_baseline.sbatch | Exp019 基线评估作业（NudeNet，2臂） |
| slurm/exp019_finetune.sbatch | Exp019 微调训练作业（repeat=25, epochs=20） |
| slurm/exp019_generate_ft.sbatch | Exp019 微调后视频生成作业（epoch-19，99视频） |
| slurm/exp019_eval_ft.sbatch | Exp019 微调后评估作业（NudeNet） |
| memory/exp020_plan.md | Exp020 系列计划：VU masked方法全覆盖（NPOMasked/GradDiffMasked/GradAscentMasked） |
| memory/exp020a_status.md | Exp020a 执行状态清单（Phase追踪） |
| memory/exp020a_launch_report.md | Exp020a 启动成功报告（smoke test结果+作业链） |
| scripts/debug/smoke_test_wan_mask.py | Wan模型mask参数验证脚本（检查attention结构和latent shape） |
| slurm/smoke_test_wan_mask.sbatch | Mask参数验证作业（GPU节点运行） |
| slurm/exp020a_unlearn_npo_masked.sbatch | Exp020a NPOMasked擦除训练作业 |
| slurm/exp020a_merge.sbatch | Exp020a 模型合并作业 |
| slurm/exp020a_generate.sbatch | Exp020a 视频生成作业 |
| slurm/exp020a_evaluate.sbatch | Exp020a 安全评估作业 |
| slurm/exp020a_submit_all.sh | Exp020a 一键提交脚本（依赖链） |
