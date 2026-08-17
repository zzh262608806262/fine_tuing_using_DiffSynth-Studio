"""
Wan2.1-T2V-1.3B 量化脚本

复用 diffsynth.core.quant 量化框架，支持 bitsandbytes (nf4/fp4) 和 torchao (int8/int4/fp8) 量化。

三种模式:
  1. infer (默认): 在线量化推理 — 加载 fp 模型并即时量化，直接生成视频验证效果
  2. save:         保存量化 checkpoint — 将量化后的权重保存到磁盘
  3. load:         加载已保存的量化 checkpoint 并推理

用法示例:
    # 在线量化推理（默认，最简单）
    python scripts/quantize.py --method bitsandbytes_nf4

    # 保存量化 checkpoint
    python scripts/quantize.py --mode save --method bitsandbytes_nf4 \
        --output_path ./models/quantized/Wan2.1-T2V-1.3B_nf4

    # 加载已保存的量化 checkpoint 推理
    python scripts/quantize.py --mode load --method bitsandbytes_nf4 \
        --quantized_path ./models/quantized/Wan2.1-T2V-1.3B_nf4.safetensors

可用量化方法:
    bitsandbytes_nf4   — 4bit NF4，仅权重量化（推荐，可训练可推理）
    bitsandbytes_fp4   — 4bit FP4，仅权重量化
    torchao_int8_w8a16 — 8bit INT8，仅权重量化
    torchao_int4_w4a16 — 4bit INT4，仅权重量化（需要 mslk）
    torchao_fp8_w8a16  — 8bit FP8，仅权重量化

依赖:
    bitsandbytes:  pip install bitsandbytes      (bitsandbytes_nf4/fp4)
    torchao:       pip install torchao           (torchao_* 方法)
"""
import argparse
import json
import os
import sys

# 将项目根目录加入 sys.path，确保 diffsynth 包可被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# 轻量级量化配置模块（不依赖 peft/torchao/bitsandbytes），用于 --list_methods
from diffsynth.core.quant import QuantizeConfig, describe_quant_method

# Pipeline 相关导入延迟到函数内执行，避免 --list_methods 等轻量操作触发完整依赖链


# Wan2.1-T2V-1.3B DiT 模型配置（来自 MODEL_CONFIGS 注册表）
WAN_T2V_13B_DIT_CONFIG = {
    "model_class": "diffsynth.models.wan_video_dit.WanModel",
    "extra_kwargs": {
        "has_image_input": False, "patch_size": [1, 2, 2], "in_dim": 16,
        "dim": 1536, "ffn_dim": 8960, "freq_dim": 256, "text_dim": 4096,
        "out_dim": 16, "num_heads": 12, "num_layers": 30, "eps": 1e-06,
    },
}

DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_PROMPT = (
    "纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。"
    "小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，"
    "使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，"
    "远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。"
    "中景侧面移动视角。"
)
DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def build_parser():
    parser = argparse.ArgumentParser(description="Wan2.1-T2V-1.3B 量化脚本")
    parser.add_argument("--mode", choices=["infer", "save", "load"], default="infer",
                        help="运行模式: infer=在线量化推理, save=保存量化checkpoint, load=加载量化checkpoint推理")
    parser.add_argument("--method", type=str, default="bitsandbytes_nf4",
                        help="量化方法 (bitsandbytes_nf4 / bitsandbytes_fp4 / torchao_int8_w8a16 / torchao_int4_w4a16 / torchao_fp8_w8a16)")
    parser.add_argument("--list_methods", action="store_true", help="列出所有可用量化方法并退出")
    # 模型路径
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                        help="模型 ID（ModelScope/HuggingFace）")
    parser.add_argument("--dit_path", type=str, default=None,
                        help="本地 DiT 权重路径（优先于 model_id）")
    # 保存/加载路径
    parser.add_argument("--output_path", type=str, default="./models/quantized/Wan2.1-T2V-1.3B_nf4",
                        help="save 模式: 量化 checkpoint 保存路径（不含扩展名）")
    parser.add_argument("--quantized_path", type=str, default=None,
                        help="load 模式: 已保存的量化 checkpoint 路径（.safetensors）")
    # 推理参数
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="推理提示词")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE, help="负向提示词")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="推理步数")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG 引导强度")
    parser.add_argument("--tiled", action="store_true", default=True, help="使用 tiled 推理（节省显存）")
    parser.add_argument("--no_tiled", dest="tiled", action="store_false", help="不使用 tiled 推理")
    parser.add_argument("--output_video", type=str, default="video_quantized.mp4", help="输出视频路径")
    # 量化参数
    parser.add_argument("--target_modules", type=str, default=None,
                        help="量化目标模块（逗号分隔），默认量化所有 Linear 层")
    parser.add_argument("--exclude_modules", type=str, default=None,
                        help="排除模块（逗号分隔）")
    parser.add_argument("--dequant_once", action="store_true", default=False,
                        help="量化后立即反量化为 fp Linear（保留量化误差，但支持更多操作）")
    return parser


def make_quant_config(method, target_modules=None, exclude_modules=None, dequant_once=False, load_prequantized=False):
    """构建 QuantizeConfig"""
    kwargs = {"method": method, "load_prequantized": load_prequantized}
    if dequant_once:
        kwargs["mode"] = "dequant_once"
    if target_modules:
        kwargs["target_modules"] = target_modules.split(",")
    if exclude_modules:
        kwargs["exclude_modules"] = exclude_modules.split(",")
    return QuantizeConfig(**kwargs)


def load_pipeline(model_id, dit_path=None, quant_config=None, device="cuda"):
    """加载 pipeline，可附带在线量化配置"""
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    dit_pattern = dit_path if dit_path else f"{model_id}:diffusion_pytorch_model*.safetensors"
    model_configs = [
        ModelConfig(path=dit_path) if dit_path else ModelConfig(
            model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors",
            quantize=quant_config,
        ),
        ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id=model_id, origin_file_pattern="Wan2.1_VAE.pth"),
    ]
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=ModelConfig(model_id=model_id, origin_file_pattern="google/umt5-xxl/"),
    )
    return pipe


def run_inference(pipe, args):
    """运行推理并保存视频"""
    from diffsynth.utils.data import save_video
    print(f"推理中 (steps={args.num_inference_steps}, cfg={args.cfg_scale})...")
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        tiled=args.tiled,
    )
    save_video(video, args.output_video, fps=15, quality=5)
    print(f"视频已保存: {args.output_video}")


def mode_infer(args):
    """模式 1: 在线量化推理"""
    quant_config = make_quant_config(
        args.method, args.target_modules, args.exclude_modules, args.dequant_once
    )
    print(f"在线量化推理 | 方法: {args.method}")
    pipe = load_pipeline(args.model_id, args.dit_path, quant_config, device="cuda")
    run_inference(pipe, args)


def mode_save(args):
    """模式 2: 保存量化 checkpoint"""
    from safetensors.torch import save_file

    # 加载 fp 模型
    print("加载 fp 模型...")
    pipe = load_pipeline(args.model_id, args.dit_path, device="cpu")

    # 量化 DiT
    quant_config = make_quant_config(args.method, args.target_modules, args.exclude_modules, args.dequant_once)
    print(f"量化 DiT (方法: {args.method})...")
    quant_config.quantize_model(pipe.dit, compute_device="cuda", model_device="cpu")

    # 提取并展平 state dict
    state_dict = pipe.dit.state_dict()
    tensors, metadata = quant_config.flatten_state_dict(state_dict)

    # 保存
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    safetensors_path = args.output_path + ".safetensors"
    save_file(tensors, safetensors_path, metadata=metadata)
    print(f"量化权重已保存: {safetensors_path} ({len(tensors)} tensors)")

    # 保存模型配置 JSON（用于后续加载）
    config_path = args.output_path + ".config.json"
    config = {
        **WAN_T2V_13B_DIT_CONFIG,
        "quant_config": {
            "method": args.method,
            "mode": "dequant_once" if args.dequant_once else "dynamic",
            "target_modules": args.target_modules.split(",") if args.target_modules else None,
            "exclude_modules": args.exclude_modules.split(",") if args.exclude_modules else None,
            "load_prequantized": True,
        },
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"模型配置已保存: {config_path}")

    # 显存占用对比
    total_elements = sum(t.numel() for t in tensors.values())
    print(f"量化后总参数量: {total_elements / 1e6:.1f}M elements")


def mode_load(args):
    """模式 3: 加载已保存的量化 checkpoint 推理"""
    from safetensors import safe_open
    import importlib

    if args.quantized_path is None:
        print("错误: load 模式需要 --quantized_path 参数")
        sys.exit(1)

    config_path = args.quantized_path.replace(".safetensors", ".config.json")
    if not os.path.exists(config_path):
        # 尝试相邻路径
        base = args.quantized_path.rsplit(".", 1)[0]
        config_path = base + ".config.json"
    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件 {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    # 加载 pipeline（fp DiT 占位，稍后替换）
    print("加载 pipeline（text_encoder + vae）...")
    pipe = load_pipeline(args.model_id, args.dit_path, device="cuda")

    # 手动构建量化 DiT
    print(f"加载量化 DiT: {args.quantized_path}")
    module_path, class_name = config["model_class"].rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_path), class_name)
    dit = model_class(**config["extra_kwargs"])

    quant_cfg = config["quant_config"]
    quant_config = QuantizeConfig(**quant_cfg)
    quant_config.prepare_for_prequantized_load(dit, compute_dtype=torch.bfloat16)

    # 读取并反展平 state dict
    state_dict = {}
    metadata = {}
    with safe_open(args.quantized_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
        metadata = f.metadata() or {}
    state_dict = quant_config.unflatten_state_dict(state_dict, metadata)
    dit.load_state_dict(state_dict, assign=True)
    dit = dit.to(device="cuda", dtype=torch.bfloat16)

    # 替换 pipeline 中的 DiT
    del pipe.dit
    pipe.dit = dit
    print("量化模型加载完成")

    run_inference(pipe, args)


def main():
    args = build_parser().parse_args()

    if args.list_methods:
        print("可用量化方法:\n")
        for name in ["bitsandbytes_nf4", "bitsandbytes_fp4", "torchao_int8_w8a16", "torchao_int4_w4a16", "torchao_fp8_w8a16"]:
            describe_quant_method(name)
            print()
        return

    if args.mode == "infer":
        mode_infer(args)
    elif args.mode == "save":
        mode_save(args)
    elif args.mode == "load":
        mode_load(args)


if __name__ == "__main__":
    main()
