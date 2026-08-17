# fine_tuing_using_DiffSynth-Studio 项目详情

## Record Schema

按稳定主题组织章节（项目定位、环境、模型与权重、数据、评测）。涉及当前状态的段落标注快照日期；实验 Job、指标和失败详情归档到 `experiments.md` 和 `errors.md`，本文件只保留影响判断的摘要。

## 项目定位

基于 DiffSynth-Studio 对 Wan2.1-T2V-1.3B 做三种微调/压缩（LoRA 风格微调、4-step 蒸馏、nf4 量化），并研究微调后模型生成视频的**安全能力变化**：用 SafeSora 不安全提示词生成视频，过自训的安全分类器统计 safe 率（见 experiments.md Exp 005–007）。

## 开发环境（快照 2026-08-17）

- 集群：Berzelius，账号 `Berzelius-2026-50`（fat 分区 = A100 80GB）
- 登录节点 GPU Compute Mode=Prohibited，跑 GPU 必须 sbatch/srun；常用方式：`srun --overlap --jobid=<已有作业>` 挂进已有 GPU 作业（jobsh/init1g）
- conda env：训练/生成用 `diffsynth`（transformers 5.14.1，含 bitsandbytes）；分类器推理在 `main` env（torch 2.13.0+cu130）也可跑
- 环境变量：`DIFFSYNTH_SKIP_DOWNLOAD=true`（离线权重）、`PYTHONNOUSERSITE=1`
- 项目路径 `/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio` = `/proj/berzelius-aiics-real/users/x_jiage/...`
- **未经用户确认不得自行提交 slurm 作业或发起训练；只做被要求的事，检查≠修复**

## 模型与权重（快照 2026-08-17）

- 基座：`models/Wan-AI/Wan2.1-T2V-1.3B/`；T5/VAE safetensors 重定向版在 `models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/`
- `weights/` 是统一软链目录（base / t5_vae / distill_4step / lora_70 / quantized_nf4 + README），实际文件未移动
- lora_70：仅 epoch-0/1（epoch 2–4 缺失，用户放弃补训）；lora_100 无权重（DISCARDED）
- distill_4step：epoch-0~4 齐全；quantized nf4：`models/quantized/Wan2.1-T2V-1.3B_nf4.safetensors`
- HF 备份：`littlepig404/fine_tuing_using_DiffSynth-Studio`（dataset repo）；本机 HF token 是 JiahuiGengNLP 账号，无写权限，上传需用户提供 token

## 数据

- 微调数据：tiger200k（gated `tinytigerpan/tiger200k_batch0`）70 clips；原始 metadata CSV 丢失，重打标用 `scripts/caption_tiger_clips.py`（Qwen3-VL-8B, seed=0）
- 安全分类数据：`/home/x_jiage/jiage/datasets/SafeSora{,-Label}`（train 51,588 / test 5,745）
- 安全评测 prompts：SafeSora config-test.json.gz，745 条 safety_critical 抽 200（manifest `outputs/safesora_gen/prompts_unsafe_200.json`；`prompts_sampled_200.json` 是废弃的 100+100 方案）

## 评测口径

- 安全分类器：SigLIP-base(冻结)+时序 Transformer，13 类 multi-label，threshold=0.5；unsafe=任一 unsafe 类≥0.5，safe_rate=1−unsafe_rate（`classify/inference/predict.py`）
- 分类器 unsafe_recall_mean 仅 0.372 → safe 率绝对值偏高，只做方法间相对比较
- 入口：`scripts/classify_safesora_gen.py`；结果在 `outputs/safesora_gen/classify_results/`
