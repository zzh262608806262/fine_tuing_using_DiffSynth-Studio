#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 0:15:00
#SBATCH -J exp017_smoke
#SBATCH -o slurm/logs/exp017_smoke_%j.out
#SBATCH -e slurm/logs/exp017_smoke_%j.err

# Exp017 冒烟测试

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

bash smoke_test_exp017.sh
