#!/bin/bash
#SBATCH -A berzelius-2026-50
#SBATCH --gpus=1
#SBATCH -t 10:00:00
#SBATCH -J safesora-cls
#SBATCH -o /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/outputs/safesora_safety_classifier/slurm-%j.out
# 训练作业: 由 watchdog 在检测到训练中断且无在跑/排队作业时自动提交.
# 内部走 keepalive (崩溃自动重启 + last.pt 自动续跑).
bash /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/classify/scripts/train_keepalive.sh
