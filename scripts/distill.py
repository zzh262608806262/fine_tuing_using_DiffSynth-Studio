"""
Wan2.1-T2V-1.3B 直接蒸馏脚本

复用 examples/wanvideo/model_training/train.py 训练框架，使用 direct_distill 任务。
蒸馏后的模型可在 4 步内生成视频（原始模型需要 30+ 步）。

两种模式:
  1. train (默认): 启动蒸馏训练
  2. validate:     用蒸馏后的模型进行快速推理验证（4 步生成）

用法示例:
    # 蒸馏训练（使用默认参数）
    python scripts/distill.py

    # 自定义参数训练
    python scripts/distill.py --learning_rate 5e-6 --num_epochs 3 --dataset_repeat 200

    # 验证蒸馏效果（4 步快速生成）
    python scripts/distill.py --mode validate \
        --distilled_model_path ./models/train/Wan2.1-T2V-1.3B_full_distill/epoch-1.safetensors

原理:
  direct_distill 基于 DirectDistillLoss —— 模型从纯噪声出发，在 N 步去噪后
  直接与 ground-truth 干净视频计算 MSE 损失。训练后模型用极少步数即可生成高质量视频。
"""
import argparse
import subprocess
import sys
import os

# 将项目根目录加入 sys.path，确保 diffsynth 包可被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_MODEL_PATHS = (
    f"{DEFAULT_MODEL_ID}:diffusion_pytorch_model*.safetensors,"
    f"{DEFAULT_MODEL_ID}:models_t5_umt5-xxl-enc-bf16.pth,"
    f"{DEFAULT_MODEL_ID}:Wan2.1_VAE.pth"
)
DEFAULT_DATASET_BASE = "data/diffsynth_example_dataset/wanvideo/Wan2.1-T2V-1.3B_direct_distill"
DEFAULT_DATASET_META = "data/diffsynth_example_dataset/wanvideo/Wan2.1-T2V-1.3B_direct_distill/metadata.csv"
DEFAULT_OUTPUT = "./models/train/Wan2.1-T2V-1.3B_full_distill"
DEFAULT_ACCELERATE_CONFIG = os.path.join("examples", "wanvideo", "model_training", "full", "accelerate_config_14B.yaml")

DATASET_DOWNLOAD_CMD = (
    "modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset "
    '--include "wanvideo/Wan2.1-T2V-1.3B_direct_distill/*" --local_dir ./data/diffsynth_example_dataset'
)

DEFAULT_PROMPT = (
    "纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。"
    "小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，"
    "使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，"
    "远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。"
    "中景侧面移动视角。"
)


def build_parser():
    parser = argparse.ArgumentParser(description="Wan2.1-T2V-1.3B 蒸馏脚本")
    parser.add_argument("--mode", choices=["train", "validate"], default="train",
                        help="运行模式: train=蒸馏训练, validate=验证蒸馏效果")
    # 数据集
    parser.add_argument("--dataset_base_path", type=str, default=DEFAULT_DATASET_BASE,
                        help="数据集根目录（蒸馏专用数据集）")
    parser.add_argument("--dataset_metadata_path", type=str, default=DEFAULT_DATASET_META,
                        help="数据集 metadata.csv 路径")
    parser.add_argument("--dataset_repeat", type=int, default=160, help="数据集重复次数")
    parser.add_argument("--download_dataset", action="store_true", default=True,
                        help="是否下载示例数据集（默认开启）")
    parser.add_argument("--no_download_dataset", dest="download_dataset", action="store_false",
                        help="不下载示例数据集")
    # 视频尺寸
    parser.add_argument("--height", type=int, default=480, help="视频高度")
    parser.add_argument("--width", type=int, default=832, help="视频宽度")
    parser.add_argument("--num_frames", type=int, default=81, help="视频帧数")
    # 模型
    parser.add_argument("--model_id_with_origin_paths", type=str, default=DEFAULT_MODEL_PATHS,
                        help="模型路径，逗号分隔")
    parser.add_argument("--model_paths", type=str, default=None,
                        help="本地模型路径（JSON 格式），优先于 model_id_with_origin_paths")
    # 训练
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率（蒸馏通常用较小学习率）")
    parser.add_argument("--num_epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--save_steps", type=int, default=None, help="保存间隔步数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    # 蒸馏参数
    parser.add_argument("--trainable_models", type=str, default="dit", help="可训练模型")
    parser.add_argument("--extra_inputs", type=str, default="seed,rand_device,num_inference_steps,cfg_scale",
                        help="蒸馏额外输入参数")
    # 输出
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT, help="输出路径")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.",
                        help="保存 checkpoint 时移除的前缀")
    # 梯度
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True,
                        help="使用梯度检查点（默认开启）")
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true", default=False,
                        help="梯度检查点卸载到 CPU")
    # accelerate
    parser.add_argument("--accelerate_config", type=str, default=DEFAULT_ACCELERATE_CONFIG,
                        help="accelerate 配置文件路径（默认使用 DeepSpeed Zero2 配置）")
    parser.add_argument("--num_processes", type=int, default=None,
                        help="GPU 进程数（覆盖 accelerate 配置中的值）")
    # 日志
    parser.add_argument("--enable_tensorboard_log", action="store_true", help="启用 TensorBoard")
    parser.add_argument("--enable_swanlab_log", action="store_true", help="启用 SwanLab")
    parser.add_argument("--enable_wandb_log", action="store_true", help="启用 WandB")
    # 验证模式参数
    parser.add_argument("--distilled_model_path", type=str, default=None,
                        help="validate 模式: 蒸馏后的模型路径")
    parser.add_argument("--num_inference_steps", type=int, default=4,
                        help="validate 模式: 推理步数（蒸馏模型默认 4 步）")
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="validate 模式: CFG 引导强度（蒸馏模型通常用 1.0）")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="validate 模式: 提示词")
    parser.add_argument("--seed", type=int, default=0, help="validate 模式: 随机种子")
    parser.add_argument("--output_video", type=str, default="video_distill.mp4",
                        help="validate 模式: 输出视频路径")
    return parser


def build_train_command(args):
    """构建蒸馏训练命令"""
    cmd = ["accelerate", "launch"]
    if args.accelerate_config and os.path.exists(args.accelerate_config):
        cmd += ["--config_file", args.accelerate_config]
    if args.num_processes:
        cmd += ["--num_processes", str(args.num_processes)]

    train_script = os.path.join("examples", "wanvideo", "model_training", "train.py")
    cmd.append(train_script)

    cmd += [
        "--dataset_base_path", args.dataset_base_path,
        "--dataset_metadata_path", args.dataset_metadata_path,
        "--dataset_repeat", str(args.dataset_repeat),
        "--height", str(args.height),
        "--width", str(args.width),
        "--num_frames", str(args.num_frames),
        "--learning_rate", str(args.learning_rate),
        "--num_epochs", str(args.num_epochs),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--weight_decay", str(args.weight_decay),
        "--output_path", args.output_path,
        "--remove_prefix_in_ckpt", args.remove_prefix_in_ckpt,
        "--trainable_models", args.trainable_models,
        "--task", "direct_distill",
        "--extra_inputs", args.extra_inputs,
    ]

    if args.model_paths:
        cmd += ["--model_paths", args.model_paths]
    else:
        cmd += ["--model_id_with_origin_paths", args.model_id_with_origin_paths]

    if args.save_steps is not None:
        cmd += ["--save_steps", str(args.save_steps)]
    if args.use_gradient_checkpointing:
        cmd.append("--use_gradient_checkpointing")
    if args.use_gradient_checkpointing_offload:
        cmd.append("--use_gradient_checkpointing_offload")
    if args.enable_tensorboard_log:
        cmd.append("--enable_tensorboard_log")
    if args.enable_swanlab_log:
        cmd.append("--enable_swanlab_log")
    if args.enable_wandb_log:
        cmd.append("--enable_wandb_log")

    return cmd


def mode_train(args):
    """蒸馏训练模式"""
    if args.download_dataset:
        print("[1/2] 下载蒸馏数据集...")
        subprocess.run(DATASET_DOWNLOAD_CMD, shell=True, check=True)
    else:
        print("[1/2] 跳过数据集下载")

    cmd = build_train_command(args)
    print("\n[2/2] 启动蒸馏训练:")
    print("  " + " ".join(cmd) + "\n")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def mode_validate(args):
    """蒸馏模型验证模式"""
    import torch
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.utils.data import save_video

    if args.distilled_model_path is None:
        # 自动查找最新 checkpoint
        output_path = args.output_path
        if os.path.isdir(output_path):
            ckpts = sorted([f for f in os.listdir(output_path) if f.endswith(".safetensors")])
            if ckpts:
                args.distilled_model_path = os.path.join(output_path, ckpts[-1])
                print(f"自动选择最新 checkpoint: {args.distilled_model_path}")
        if args.distilled_model_path is None:
            print("错误: 未找到蒸馏模型。请用 --distilled_model_path 指定，或先运行训练。")
            sys.exit(1)

    print(f"加载蒸馏模型: {args.distilled_model_path}")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=args.distilled_model_path),
            ModelConfig(model_id=DEFAULT_MODEL_ID, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=DEFAULT_MODEL_ID, origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id=DEFAULT_MODEL_ID, origin_file_pattern="google/umt5-xxl/"),
    )

    print(f"推理中 (steps={args.num_inference_steps}, cfg={args.cfg_scale})...")
    video = pipe(
        prompt=args.prompt,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        tiled=True,
    )
    save_video(video, args.output_video, fps=15, quality=5)
    print(f"视频已保存: {args.output_video}")


def main():
    args = build_parser().parse_args()

    if args.mode == "train":
        mode_train(args)
    elif args.mode == "validate":
        mode_validate(args)


if __name__ == "__main__":
    main()
