"""
Wan2.2-TI2V-5B 良性微调薄包装（文本驱动 LoRA，两臂对照）

复用 FT 项目内 examples/wanvideo/model_training/train.py 训练框架，通过 accelerate launch 启动，
模式与 scripts/lora_finetune.py（1.3B 入口）一致。

实验口径 = 纯文本驱动：**故意不传 --extra_inputs**（即不传 "input_image"）。
当 input_image 为空时 TI2V-5B 走文本驱动（见 diffsynth/pipelines/wan_video.py 中
WanVideoUnit_ImageEmbedderFused.process: `if input_image is None ... return {}`），
评测 / 擦除 / 微调三者口径一致。

两臂（仅基座不同，metadata 与超参完全一致，低曝光 = repeat10 × epochs2 = 20 次/样本）:
    base   : models/Wan-AI/Wan2.2-TI2V-5B/         (Task 1 未擦除基座)
    erased : models/Wan-AI/Wan2.2-TI2V-5B-erased/   (Task 3 擦除后基座，擦除已合进基座)
擦除是合进基座的，因此本脚本只训练新的 LoRA（--lora_checkpoint 默认空），
不挂 VU / 擦除 adapter。

模型定位（本地，零下载）:
    通过 --model_id_with_origin_paths 传 "Wan-AI/Wan2.2-TI2V-5B[-erased]:<origin_file_pattern>"。
    DiffSynth 的 ModelConfig(model_id, origin_file_pattern)（diffsynth/core/loader/config.py）
    将其解析为 ./models/<model_id>/<pattern> 的 glob；sbatch 中 export DIFFSYNTH_SKIP_DOWNLOAD=true
    时只读本地，不触发在线下载。
    5B 的 origin_file_pattern（本地已有文件是 .safetensors；覆盖 wan_series 默认的 .pth）:
        DiT : diffusion_pytorch_model*.safetensors
        T5  : models_t5_umt5-xxl-enc-bf16.safetensors
        VAE : Wan2.2_VAE.safetensors
    tokenizer 由 train.py 默认从 "Wan-AI/Wan2.1-T2V-1.3B:google/umt5-xxl/" 加载（本地已有 1.3B 目录）。

用法示例:
    # 离屏预览将执行的 accelerate 命令（本机无权重也能跑，只打印不执行）
    python scripts/finetune_wan5b.py --arm erased --dry-run
    python scripts/finetune_wan5b.py --arm base --dry-run

    # 真实训练（集群，见 slurm/wan5b_finetune.sbatch）
    python scripts/finetune_wan5b.py --arm erased
    python scripts/finetune_wan5b.py --arm base
"""
import argparse
import glob
import os
import subprocess
import sys


# Wan2.2-TI2V-5B 两臂定义：基座目录（相对 FT 根）与 model_id（相对 ./models 的本地路径）
ARMS = {
    "base": {
        "base_model_dir": "models/Wan-AI/Wan2.2-TI2V-5B",
        "model_id": "Wan-AI/Wan2.2-TI2V-5B",
    },
    "erased": {
        "base_model_dir": "models/Wan-AI/Wan2.2-TI2V-5B-erased",
        "model_id": "Wan-AI/Wan2.2-TI2V-5B-erased",
    },
}

# 5B 三件套的 origin_file_pattern（本地已备 .safetensors；覆盖官方 wan_series 默认 .pth）
PATTERNS = [
    "diffusion_pytorch_model*.safetensors",
    "models_t5_umt5-xxl-enc-bf16.safetensors",
    "Wan2.2_VAE.safetensors",
]

DEFAULT_DATASET_BASE = "data/tiger_dataset"
DEFAULT_DATASET_META = "data/tiger_dataset/metadata_100.csv"
DEFAULT_LORA_TARGETS = "q,k,v,o,ffn.0,ffn.2"


def default_output_path(arm):
    """输出目录名：models/train/Wan2.2-TI2V-5B_{arm}_lora10e2/"""
    return f"models/train/Wan2.2-TI2V-5B_{arm}_lora10e2"


def model_id_with_origin_paths(model_id):
    """按本地路径 + 显式 origin_file_pattern 拼 model_id_with_origin_paths"""
    return ",".join(f"{model_id}:{pattern}" for pattern in PATTERNS)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Wan2.2-TI2V-5B 良性微调（文本驱动 LoRA，base/erased 两臂，低曝光 repeat10×epochs2）"
    )
    parser.add_argument("--arm", type=str, required=True, choices=["base", "erased"],
                        help="微调臂：erased=擦除后基座(Wan2.2-TI2V-5B-erased)，base=未擦除基座(Wan2.2-TI2V-5B)")
    # 数据集
    parser.add_argument("--dataset_base_path", type=str, default=DEFAULT_DATASET_BASE,
                        help="数据集根目录")
    parser.add_argument("--dataset_metadata_path", type=str, default=DEFAULT_DATASET_META,
                        help="数据集 metadata.csv 路径")
    parser.add_argument("--dataset_repeat", type=int, default=10, help="数据集重复次数（低曝光=10）")
    # 视频尺寸
    parser.add_argument("--height", type=int, default=480, help="视频高度")
    parser.add_argument("--width", type=int, default=832, help="视频宽度")
    parser.add_argument("--num_frames", type=int, default=25, help="视频帧数")
    # 模型（按 --arm 自动生成本地路径，可覆盖）
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None,
                        help="模型路径，逗号分隔的 model_id:origin_file_pattern 格式；默认按 --arm 生成（本地 models/ 下）")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="LoRA 断点路径（续训用；默认空 = 从零训新 LoRA）")
    # 训练
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=2, help="训练轮数（低曝光=2）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数（batch=1）")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--save_steps", type=int, default=None,
                        help="保存间隔步数（None=每 epoch 保存，产出 epoch-0/1.safetensors）")
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA 秩")
    parser.add_argument("--lora_target_modules", type=str, default=DEFAULT_LORA_TARGETS,
                        help="LoRA 目标模块（逗号分隔）")
    # 输出
    parser.add_argument("--output_path", type=str, default=None,
                        help="输出路径（默认 models/train/Wan2.2-TI2V-5B_{arm}_lora10e2）")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.",
                        help="保存 checkpoint 时移除的前缀")
    # 梯度
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True,
                        help="使用梯度检查点（默认开启）")
    parser.add_argument("--no_use_gradient_checkpointing", dest="use_gradient_checkpointing",
                        action="store_false", help="关闭梯度检查点")
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true", default=False,
                        help="梯度检查点卸载到 CPU")
    # accelerate
    parser.add_argument("--accelerate_config", type=str, default=None,
                        help="accelerate 配置文件路径")
    parser.add_argument("--num_processes", type=int, default=1, help="GPU 进程数")
    # 日志（默认开 tensorboard，产物含 tensorboard_log）
    parser.add_argument("--enable_tensorboard_log", action="store_true", default=True,
                        help="启用 TensorBoard（默认开启）")
    parser.add_argument("--no_tensorboard_log", dest="enable_tensorboard_log", action="store_false",
                        help="关闭 TensorBoard")
    parser.add_argument("--enable_swanlab_log", action="store_true", help="启用 SwanLab")
    parser.add_argument("--enable_wandb_log", action="store_true", help="启用 WandB")
    # 预览
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将执行的 accelerate 命令，不实际运行（缺失项仅警告不退出）")
    return parser


def resolve_args(args):
    """按 --arm 填充基座相关参数（model_id_with_origin_paths / output_path）"""
    if args.model_id_with_origin_paths is None:
        args.model_id_with_origin_paths = model_id_with_origin_paths(ARMS[args.arm]["model_id"])
    if args.output_path is None:
        args.output_path = default_output_path(args.arm)
    return args


def validate(args):
    """基座目录与 metadata 存在性校验；返回错误列表（空 = 通过）"""
    errors = []
    base_model_dir = ARMS[args.arm]["base_model_dir"]
    if not os.path.isdir(base_model_dir):
        errors.append(f"基座目录不存在: {base_model_dir}（集群上需先就绪 Task1/Task3 产物）")
    else:
        for pattern in PATTERNS:
            if not glob.glob(os.path.join(base_model_dir, pattern)):
                errors.append(f"基座目录缺少文件（{pattern}）: {base_model_dir}")
    if not os.path.isfile(args.dataset_metadata_path):
        errors.append(f"metadata CSV 不存在: {args.dataset_metadata_path}")
    return errors


def build_command(args):
    """构建 accelerate launch 训练命令（与 scripts/lora_finetune.py 同构）"""
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
        "--model_id_with_origin_paths", args.model_id_with_origin_paths,
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

    # 注意：故意不传 --extra_inputs（文本驱动；input_image 为空时 TI2V-5B 即走纯文本路径）
    return cmd


def main():
    parser = build_parser()
    args = resolve_args(parser.parse_args())

    # 基座目录 / metadata 校验
    errors = validate(args)
    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        if args.dry_run:
            print("[dry-run] 存在缺失项（本机无 5B 权重属正常），仍继续打印命令。")
        else:
            print("校验未通过，退出。请先在集群就绪基座产物（Task1/Task3）与 metadata CSV。")
            sys.exit(1)

    cmd = build_command(args)

    print("=" * 100)
    print(f"Wan2.2-TI2V-5B 微调 arm={args.arm}（文本驱动，未传 --extra_inputs）")
    print(f"  基座   : {ARMS[args.arm]['base_model_dir']}")
    print(f"  数据   : {args.dataset_metadata_path} (repeat={args.dataset_repeat} x epochs={args.num_epochs} = {args.dataset_repeat * args.num_epochs} 次/样本)")
    print(f"  输出   : {args.output_path}/  (epoch-0/1.safetensors + training_args.json + tensorboard_log)")
    print("=" * 100)
    print("将执行的 accelerate 命令：")
    print("  " + " ".join(cmd))
    print()

    if args.dry_run:
        print("[dry-run] 仅预览，不执行。")
        sys.exit(0)

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()