"""
可断点续跑的微调启动器（不修改原有脚本，复用其参数定义与命令构建）

包装 scripts/lora_finetune.py 与 scripts/distill.py：接受与它们完全相同的参数，
但把训练入口从 examples/wanvideo/model_training/train.py 换成
scripts/train_resumable.py，从而获得完整的断点续跑能力
（模型 + 优化器 + scheduler + RNG + 数据进度，详见 train_resumable.py 顶部说明）。

用法:
    # LoRA 微调（参数与 scripts/lora_finetune.py 相同）
    python scripts/finetune_resumable.py lora \
        --dataset_base_path ... --dataset_metadata_path ... \
        --num_epochs 5 --save_steps 200

    # 蒸馏训练（参数与 scripts/distill.py 相同）
    python scripts/finetune_resumable.py distill --num_epochs 2 --save_steps 200

    # 被打断后，用完全相同的命令重新运行即可自动从断点继续（--resume auto 为默认）。
    # 想放弃断点从头训练: 加 --resume never

与原脚本的其他区别:
  - 数据集下载幂等：metadata.csv 已存在时自动跳过下载（重启不再重复下载）
  - 新增 --resume / --state_save_steps / --seed 透传给训练脚本

Slurm 建议: sbatch 里直接写本脚本 + 固定命令，配合 --requeue 即可无人值守续跑。
"""
import os
import subprocess
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)

ORIG_TRAIN_SCRIPT = os.path.join("examples", "wanvideo", "model_training", "train.py")
RESUMABLE_TRAIN_SCRIPT = os.path.join("scripts", "train_resumable.py")

USAGE = "用法: python scripts/finetune_resumable.py {lora|distill} [原脚本参数...] [--resume auto|never] [--state_save_steps N] [--seed N]"


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("lora", "distill"):
        print(USAGE)
        sys.exit(2)
    task = sys.argv.pop(1)

    if task == "lora":
        import lora_finetune as base
        build_command = base.build_command
    else:
        import distill as base
        build_command = base.build_train_command

    parser = base.build_parser()
    parser.add_argument("--resume", choices=["auto", "never"], default="auto",
                        help="auto=检测到断点自动恢复(默认); never=忽略断点从头训练")
    parser.add_argument("--state_save_steps", type=int, default=None,
                        help="每 N 步保存续跑状态（默认取 --save_steps）")
    parser.add_argument("--seed", type=int, default=42, help="数据 shuffle 种子")
    args = parser.parse_args()

    if task == "distill" and getattr(args, "mode", "train") == "validate":
        base.mode_validate(args)
        return

    # 数据集下载（幂等：已存在则跳过）
    if args.download_dataset:
        if os.path.exists(os.path.join(ROOT, args.dataset_metadata_path)) or os.path.exists(args.dataset_metadata_path):
            print(f"[1/2] 数据集已存在 ({args.dataset_metadata_path})，跳过下载")
        else:
            print("[1/2] 下载数据集...")
            subprocess.run(base.DATASET_DOWNLOAD_CMD, shell=True, check=True, cwd=ROOT)
    else:
        print("[1/2] 跳过数据集下载")

    cmd = build_command(args)
    cmd[cmd.index(ORIG_TRAIN_SCRIPT)] = RESUMABLE_TRAIN_SCRIPT
    cmd += ["--resume", args.resume, "--seed", str(args.seed)]
    if args.state_save_steps is not None:
        cmd += ["--state_save_steps", str(args.state_save_steps)]

    print(f"\n[2/2] 启动可断点续跑的 {task} 训练:")
    print("  " + " ".join(cmd) + "\n")
    result = subprocess.run(cmd, cwd=ROOT)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
