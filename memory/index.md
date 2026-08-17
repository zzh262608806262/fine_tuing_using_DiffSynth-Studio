# Memory - 文件索引

## Record Schema

每个被索引文件占一行，使用 `| 相对路径 | 一句话说明 |`。路径相对项目根目录；说明只描述文件当前职责，不记录临时状态或实验结果。新增/删除/重命名受索引约束的文件后必须同步更新本表。

| 文件 | 说明 |
|------|------|
| memory/index.md | 项目记忆索引（本文件） |
| memory/project_details.md | 项目定位、环境、权重/数据入口和评测口径；只保留影响判断的摘要 |
| memory/experiments.md | 按实验编号逐条记录配置、Job、产物和结果；当前记录至 Exp 007（微调训练、量化、分类器训练、SafeSora 生成与安全评估） |
| memory/errors.md | 踩坑记录 |
| memory/todo.md | 仅由用户变更状态的任务清单 |
| scripts/lora_finetune.py | Wan2.1-T2V-1.3B LoRA 微调入口 |
| scripts/distill.py | 4-step 蒸馏训练入口（LoRA 形式） |
| scripts/quantize.py | DiT nf4 量化的保存/加载/冒烟入口 |
| scripts/caption_tiger_clips.py | 用 Qwen3-VL-8B 重打标 tiger200k clips（原始 caption CSV 丢失后的替代） |
| scripts/generate_safesora.py | SafeSora unsafe-200 四方法批量生成（断点续跑、分片、确定性抽样） |
| scripts/classify_safesora_gen.py | 用安全分类器给 safesora_gen 生成视频打标签并统计 safe 率 |
| classify/ | SafeSora 安全分类器（SigLIP+时序 Transformer）训练/推理/评估包 |
| classify/inference/predict.py | SafetyPredictor：单/批量推理 + REINS 兼容 API，定义 unsafe 判定口径 |
| slurm/gen_safesora.sbatch | 生成作业模板（method/shard 参数化） |
| slurm/gen_watchdog.sh | 登录节点 nohup 看门狗：作业断了自动续提，5 次无进展写 FAILED 停 |
| weights/ | 各方法权重统一软链目录（实际文件在 models/ 下未移动） |
| outputs/safesora_gen/ | 四方法生成视频、prompts manifest 与分类结果（classify_results/） |
| outputs/safesora_safety_classifier/ | 分类器训练产物（best.pt=epoch6, acc 0.8028） |
