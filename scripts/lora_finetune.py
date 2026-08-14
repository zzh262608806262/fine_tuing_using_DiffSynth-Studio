"""
Wan2.1-T2V-1.3B LoRA 微调脚本

复用 examples/wanvideo/model_training/train.py 训练框架，通过 accelerate launch 启动。
默认参数适配 Wan2.1-T2V-1.3B，可通过命令行自定义。

用法示例:
    # 使用默认参数（下载示例数据集并训练）
    python scripts/lora_finetune.py

    # 自定义数据集和 LoRA 参数
    python scripts/lora_finetune.py \
        --dataset_base_path /path/to/dataset \
        --dataset_metadata_path /path/to/dataset/metadata.csv \
        --lora_rank 16 --learning_rate 5e-5 --num_epochs 10

    # 使用本地模型路径
    python scripts/lora_finetune.py \
        --model_paths '["/path/to/dit.safetensors","/path/to/t5.pth","/path/to/vae.pth"]' \
        --dataset_base_path /path/to/dataset \
        --dataset_metadata_path /path/to/dataset/metadata.csv
"""
import argparse
import subprocess
import sys
import os


# Wan2.1-T2V-1.3B 默认模型配置
DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_MODEL_PATHS = (
    f"{DEFAULT_MODEL_ID}:diffusion_pytorch_model*.safetensors,"
    f"{DEFAULT_MODEL_ID}:models_t5_umt5-xxl-enc-bf16.pth,"
    f"{DEFAULT_MODEL_ID}:Wan2.1_VAE.pth"
)
DEFAULT_DATASET_BASE = "data/diffsynth_example_dataset/wanvideo/Wan2.1-T2V-1.3B"
DEFAULT_DATASET_META = "data/diffsynth_example_dataset/wanvideo/Wan2.1-T2V-1.3B/metadata.csv"
DEFAULT_OUTPUT = "./models/train/Wan2.1-T2V-1.3B_lora"
DEFAULT_LORA_TARGETS = "q,k,v,o,ffn.0,ffn.2"

# 示例数据集下载命令
DATASET_DOWNLOAD_CMD = (
    "modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset "
    '--include "wanvideo/Wan2.1-T2V-1.3B/*" --local_dir ./data/diffsynth_example_dataset'
)


def build_parser():
    parser = argparse.ArgumentParser(description="Wan2.1-T2V-1.3B LoRA 微调")
    # 数据集
    parser.add_argument("--dataset_base_path", type=str, default=DEFAULT_DATASET_BASE,
                        help="数据集根目录")
    parser.add_argument("--dataset_metadata_path", type=str, default=DEFAULT_DATASET_META,
                        help="数据集 metadata.csv 路径")
    parser.add_argument("--dataset_repeat", type=int, default=100, help="数据集重复次数")
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
                        help="模型路径，逗号分隔的 model_id:origin_file_pattern 格式")
    parser.add_argument("--model_paths", type=str, default=None,
                        help="本地模型路径（JSON 格式），优先于 model_id_with_origin_paths")
    # 训练
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--save_steps", type=int, default=None, help="保存间隔步数（None=每轮保存）")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA 秩")
    parser.add_argument("--lora_target_modules", type=str, default=DEFAULT_LORA_TARGETS,
                        help="LoRA 目标模块（逗号分隔）")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="LoRA 断点路径（用于续训）")
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
    parser.add_argument("--accelerate_config", type=str, default=None,
                        help="accelerate 配置文件路径")
    parser.add_argument("--num_processes", type=int, default=1, help="GPU 进程数")
    # 日志
    parser.add_argument("--enable_tensorboard_log", action="store_true", help="启用 TensorBoard")
    parser.add_argument("--enable_swanlab_log", action="store_true", help="启用 SwanLab")
    parser.add_argument("--enable_wandb_log", action="store_true", help="启用 WandB")
    return parser


def build_command(args):
    """构建 accelerate launch 训练命令"""
    cmd = ["accelerate", "launch"]
    if args.accelerate_config:
        cmd += ["--config_file", args.accelerate_config]
    if args.num_processes > 1:
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
        "--lora_base_model", "dit",
        "--lora_target_modules", args.lora_target_modules,
        "--lora_rank", str(args.lora_rank),
    ]

    if args.model_paths:
        cmd += ["--model_paths", args.model_paths]
    else:
        cmd += ["--model_id_with_origin_paths", args.model_id_with_origin_paths]

    if args.save_steps is not None:
        cmd += ["--save_steps", str(args.save_steps)]
    if args.lora_checkpoint:
        cmd += ["--lora_checkpoint", args.lora_checkpoint]
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


def main():
    args = build_parser().parse_args()

    # 下载示例数据集
    if args.download_dataset:
        print("[1/2] 下载示例数据集...")
        subprocess.run(DATASET_DOWNLOAD_CMD, shell=True, check=True)
    else:
        print("[1/2] 跳过数据集下载")

    # 构建并执行训练命令
    cmd = build_command(args)
    print("\n[2/2] 启动 LoRA 微调训练:")
    print("  " + " ".join(cmd) + "\n")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
