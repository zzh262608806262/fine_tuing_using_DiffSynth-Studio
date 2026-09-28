#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -J exp019_gen_baseline
#SBATCH --array=0-1  # 2 个臂：base, erased
#SBATCH -o outputs/exp019_gen_baseline_%A_%a.log
#SBATCH -e outputs/exp019_gen_baseline_%A_%a.err

module load Mambaforge/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate diffsynth

ARMS=(graddiff_base graddiff_erased)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

if [ "$ARM" == "graddiff_base" ]; then
    MODEL_PATH=models/wan5b/exp019_graddiff_base
elif [ "$ARM" == "graddiff_erased" ]; then
    MODEL_PATH=models/wan5b/exp019_graddiff_erased
fi

python scripts/exp015_generate_videos.py \
    --method graddiff \
    --arm ${ARM} \
    --model_path ${MODEL_PATH} \
    --output_dir outputs/exp019/${ARM} \
    --num_videos 99 \
    --frames 17 \
    --height 480 \
    --width 720 \
    --steps 50 \
    --guidance_scale 5.0

echo "Exp019 ${ARM} 视频生成完成"
