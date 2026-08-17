# 踩坑记录

## Record Schema

每条使用 `## YYYY-MM-DD 标题`，写清现象、根因、解决/规避方式。只记会再次踩到的坑，一次性笔误不记。

## 2026-08-16 quantize 保存作业假 FAILED

`scripts/quantize.py --mode save` 作业 17293240 标 FAILED，但权重已完整保存——失败发生在保存后的统计打印 bug（已修）。判断量化是否成功以产物加载+推理冒烟为准，不以作业状态为准。

## 2026-08-16 lora_100 训练即崩

原始与重建（作业 17293236）都在启动约 1.5 分钟内崩溃，tensorboard 仅 4.6KB。用户已决定放弃 lora_100，不要再尝试。

## 2026-08-16 outputs/safesora_gen 目录含废弃方案遗留视频

各方法目录中除 unsafe-200 外还有早期全量 1471 / 100+100 方案的视频（distill 多达 +80）。任何统计必须按 `prompts_unsafe_200.json` manifest 过滤，不能直接 glob 目录。

## 2026-08-16 登录节点 GPU 不可用

登录节点 GPU Compute Mode=Prohibited；跑 GPU 用 sbatch，或 `srun --overlap --jobid=<现有作业>` 挂进已有 GPU 分配。
