"""
Exp 017: 对擦除后模型进行蒸馏训练

基于 distill_wan5b.py，支持对已擦除的模型进行蒸馏。
输入：擦除后的基座模型
输出：蒸馏后的 checkpoint（4步推理）

用法：
    python scripts/exp017_distill_erased.py \
        --erased_model_path models/wan5b/exp015_grad_ascent_erased \
        --output_path models/train/exp017_grad_ascent_distill \
        --dataset_base_path data/tiger_dataset \
        --dataset_metadata_path data/tiger_dataset/metadata_100.csv
"""
import argparse
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def build_parser():
    parser = argparse.ArgumentParser(description="Exp 017: 擦除后模型蒸馏")

    # 擦除模型路径（必需）
    parser.add_argument("--erased_model_path", type=str, required=True,
                        help="擦除后的基座模型路径（DiffSynth 格式）")

    # 数据集（使用良性数据）
    parser.add_argument("--dataset_base_path", type=str,
                        default="data/tiger_dataset",
                        help="蒸馏训练数据集路径（推荐使用良性数据）")
    parser.add_argument("--dataset_metadata_path", type=str,
                        default="data/tiger_dataset/metadata_100.csv",
                        help="数据集 metadata 路径")
    parser.add_argument("--dataset_repeat", type=int, default=160,
                        help="数据集重复次数")

    # 视频尺寸（与擦除训练对齐）
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=736,
                        help="736 适配 Wan2.2 3D VAE (必须是32的倍数)")
    parser.add_argument("--num_frames", type=int, default=17,
                        help="与擦除训练对齐（17帧）")

    # 训练参数
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="学习率（蒸馏推荐 1e-5）")
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="训练轮数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # 输出
    parser.add_argument("--output_path", type=str, required=True,
                        help="输出路径")

    # accelerate
    parser.add_argument("--accelerate_config", type=str,
                        default="examples/wanvideo/model_training/full/accelerate_config_14B.yaml",
                        help="accelerate 配置文件")
    parser.add_argument("--num_processes", type=int, default=None,
                        help="GPU 进程数")

    # 日志
    parser.add_argument("--enable_tensorboard_log", action="store_true")

    # 梯度检查点（添加此参数）
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True,
                        help="使用梯度检查点（默认开启）")

    return parser


def build_model_paths(erased_model_path):
    """构建模型路径（JSON 格式）"""
    import json

    # 擦除后模型包含：DiT + T5 + VAE
    model_paths = {
        "dit": os.path.join(erased_model_path, "diffusion_pytorch_model.safetensors"),
        "text_encoder": os.path.join(erased_model_path, "models_t5_umt5-xxl-enc-bf16.safetensors"),
        "vae": os.path.join(erased_model_path, "Wan2.2_VAE.safetensors"),
    }

    # 验证文件存在
    for key, path in model_paths.items():
        if not os.path.exists(path):
            print(f"错误: {key} 文件不存在: {path}")
            sys.exit(1)

    return json.dumps(model_paths)


def main():
    args = build_parser().parse_args()

    # 验证擦除模型路径
    if not os.path.isdir(args.erased_model_path):
        print(f"错误: 擦除模型路径不存在: {args.erased_model_path}")
        sys.exit(1)

    # 构建模型路径
    model_paths_json = build_model_paths(args.erased_model_path)

    # 构建训练命令
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
        "--model_paths", model_paths_json,
        "--learning_rate", str(args.learning_rate),
        "--num_epochs", str(args.num_epochs),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--weight_decay", str(args.weight_decay),
        "--output_path", args.output_path,
        "--remove_prefix_in_ckpt", "pipe.dit.",
        "--trainable_models", "dit",
        "--task", "direct_distill",
        "--extra_inputs", "seed,rand_device,num_inference_steps,cfg_scale,input_image",
        "--use_gradient_checkpointing",
    ]

    if args.enable_tensorboard_log:
        cmd.append("--enable_tensorboard_log")

    print("=" * 80)
    print("Exp 017: 擦除后模型蒸馏训练")
    print("=" * 80)
    print(f"擦除模型: {args.erased_model_path}")
    print(f"训练数据: {args.dataset_base_path}")
    print(f"输出路径: {args.output_path}")
    print(f"学习率: {args.learning_rate}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"数据重复: {args.dataset_repeat}")
    print(f"视频尺寸: {args.height}x{args.width}x{args.num_frames}")
    print()
    print("训练命令:")
    print("  " + " ".join(cmd))
    print()

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
