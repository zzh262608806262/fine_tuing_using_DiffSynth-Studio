# 权重统一入口（软链接，实际文件仍在 models/ 下）

各脚本默认路径写的是 `./models/...`，因此文件没有移动，这里只是方便浏览和测试的统一目录。

| 目录 | 内容 | 状态 |
|---|---|---|
| `base_Wan2.1-T2V-1.3B/` | 基座 Wan2.1-T2V-1.3B（DiT + T5/VAE 的 .pth 原版） | ✅ |
| `t5_vae_safetensors/` | T5 / VAE / CLIP 的 safetensors 转换版（推理加载用） | ✅ |
| `distill_4step/` | 4 步蒸馏 LoRA（rank 32），epoch-0 ~ epoch-4 | ✅ 齐全 |
| `lora_70/` | LoRA 微调（70 clip 数据），仅 epoch-0 / epoch-1 | ⚠️ epoch-2~4 缺失 |
| `quantized_nf4/` | nf4 量化 DiT checkpoint + config.json（2026-08-16 重新生成，冒烟测试通过，见 smoke_test.mp4） | ✅ |

缺失且未恢复：`models/train/Wan2.1-T2V-1.3B_lora_100/` 无权重（原训练崩溃），目前只保留 tensorboard_log 与 training_args.json。

量化权重加载方式：`python scripts/quantize.py --mode load --quantized_path ./models/quantized/Wan2.1-T2V-1.3B_nf4.safetensors`
