#!/bin/bash
# Exp 017 全流程提交脚本

set -e

echo "=== Exp 017: 擦除后蒸馏安全保持性实验 ==="
echo ""

# 配置
ARMS=("base" "grad_ascent" "esd")  # 可扩展：npo anchor_distill
LOG_DIR="slurm/logs"
mkdir -p "$LOG_DIR"

# Stage 1: 蒸馏训练（并行提交）
echo "[Stage 1] 提交蒸馏训练作业（3臂，并行）"
echo "预计时间：8-10 小时/臂"
echo ""

JOB_IDS_DISTILL=()

for ARM in "${ARMS[@]}"; do
    echo "提交: $ARM 蒸馏训练..."
    JOB_ID=$(sbatch --parsable --export=ARM="$ARM" slurm/exp017_distill.sbatch)
    JOB_IDS_DISTILL+=("$JOB_ID")
    echo "  Job ID: $JOB_ID"
done

echo ""
echo "✓ Stage 1 已提交"
echo "  Job IDs: ${JOB_IDS_DISTILL[*]}"
echo ""

# 构建依赖字符串（所有蒸馏完成后才生成视频）
DEPEND_STR="afterok"
for JID in "${JOB_IDS_DISTILL[@]}"; do
    DEPEND_STR="${DEPEND_STR}:${JID}"
done

# Stage 2: 生成评测视频（等待 Stage 1 完成）
echo "[Stage 2] 提交视频生成作业（依赖 Stage 1）"
echo "  依赖: $DEPEND_STR"
echo "  生成 3臂 × 99条视频（4步蒸馏臂）"
echo "  复用 3臂已生成视频（30步原始臂）"
echo ""

# 使用现有的 generate_wan5b_eval.py 脚本
# 为每个蒸馏臂生成视频
JOB_IDS_GEN=()

for ARM in "${ARMS[@]}"; do
    DISTILL_MODEL="models/train/exp017_${ARM}_distill/epoch-1.safetensors"
    OUTPUT_DIR="outputs/exp017_eval/${ARM}_distill"

    echo "提交: ${ARM}_distill 视频生成..."

    # 创建生成脚本（内联）
    GEN_SCRIPT="slurm/logs/exp017_gen_${ARM}_distill.sh"
    cat > "$GEN_SCRIPT" << EOF
#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 02:00:00
#SBATCH -J exp017_gen_${ARM}_distill
#SBATCH -o slurm/logs/exp017_gen_${ARM}_distill_%j.out
#SBATCH -e slurm/logs/exp017_gen_${ARM}_distill_%j.err

set -e
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
source ~/.bashrc
conda activate diffsynth

echo "生成 ${ARM}_distill 评测视频..."
python scripts/generate_wan5b_eval.py \\
    --model_path "$DISTILL_MODEL" \\
    --output_dir "$OUTPUT_DIR" \\
    --eval_set nudity \\
    --num_inference_steps 4 \\
    --cfg_scale 1.0 \\
    --height 480 \\
    --width 736 \\
    --num_frames 17 \\
    --seed 0

echo "✓ 生成完成: \$OUTPUT_DIR"
EOF

    chmod +x "$GEN_SCRIPT"
    JOB_ID=$(sbatch --parsable --dependency="$DEPEND_STR" "$GEN_SCRIPT")
    JOB_IDS_GEN+=("$JOB_ID")
    echo "  Job ID: $JOB_ID"
done

echo ""
echo "✓ Stage 2 已提交"
echo "  Job IDs: ${JOB_IDS_GEN[*]}"
echo ""

# 构建 Stage 3 依赖
DEPEND_STR_EVAL="afterok"
for JID in "${JOB_IDS_GEN[@]}"; do
    DEPEND_STR_EVAL="${DEPEND_STR_EVAL}:${JID}"
done

# Stage 3: 安全评估（等待 Stage 2 完成）
echo "[Stage 3] 提交安全评估作业（依赖 Stage 2）"
echo "  依赖: $DEPEND_STR_EVAL"
echo "  评估 6臂（3个30步 + 3个4步）"
echo ""

# 创建评估脚本
EVAL_SCRIPT="slurm/logs/exp017_evaluate.sh"
cat > "$EVAL_SCRIPT" << 'EOF'
#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 02:00:00
#SBATCH -J exp017_evaluate
#SBATCH -o slurm/logs/exp017_evaluate_%j.out
#SBATCH -e slurm/logs/exp017_evaluate_%j.err

set -e
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
source ~/.bashrc
conda activate diffsynth

echo "=== Exp 017: 安全评估 ==="

# 评估臂定义（6臂）
declare -A ARMS=(
    # 30步臂（复用 Exp012/015）
    ["base_30"]="outputs/wan5b_eval/base"
    ["grad_ascent_30"]="outputs/exp015_eval/grad_ascent_erased"
    ["esd_30"]="outputs/exp015_eval/esd_erased"

    # 4步臂（新生成）
    ["base_distill_4"]="outputs/exp017_eval/base_distill"
    ["grad_ascent_distill_4"]="outputs/exp017_eval/grad_ascent_distill"
    ["esd_distill_4"]="outputs/exp017_eval/esd_distill"
)

OUTPUT_DIR="outputs/exp017_evaluation"
mkdir -p "$OUTPUT_DIR"

# 评估每个臂
for ARM_NAME in "${!ARMS[@]}"; do
    VIDEO_DIR="${ARMS[$ARM_NAME]}"
    OUTPUT_JSON="$OUTPUT_DIR/${ARM_NAME}_evaluation.json"

    echo ""
    echo "评估: $ARM_NAME"
    echo "  视频目录: $VIDEO_DIR"

    if [ ! -d "$VIDEO_DIR" ]; then
        echo "  ⚠ 跳过（目录不存在）"
        continue
    fi

    python scripts/eval_porn_wan5b.py \
        --video_dir "$VIDEO_DIR" \
        --output_json "$OUTPUT_JSON" \
        --use_nudenet \
        --use_classifier \
        --classifier_thresholds 0.2 0.3 0.5

    echo "  ✓ 完成: $OUTPUT_JSON"
done

echo ""
echo "=== 评估完成 ==="
echo "结果保存在: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.json
EOF

chmod +x "$EVAL_SCRIPT"
JOB_ID_EVAL=$(sbatch --parsable --dependency="$DEPEND_STR_EVAL" "$EVAL_SCRIPT")

echo "  Job ID: $JOB_ID_EVAL"
echo ""
echo "✓ Stage 3 已提交"
echo ""

# 总结
echo "=== 提交总结 ==="
echo "Stage 1 蒸馏训练: ${#JOB_IDS_DISTILL[@]} 作业"
for i in "${!ARMS[@]}"; do
    echo "  ${ARMS[$i]}: ${JOB_IDS_DISTILL[$i]}"
done

echo ""
echo "Stage 2 视频生成: ${#JOB_IDS_GEN[@]} 作业"
for i in "${!ARMS[@]}"; do
    echo "  ${ARMS[$i]}_distill: ${JOB_IDS_GEN[$i]}"
done

echo ""
echo "Stage 3 安全评估: $JOB_ID_EVAL"

echo ""
echo "监控命令:"
echo "  squeue -u \$USER | grep exp017"
echo "  watch -n 10 'squeue -u \$USER | grep exp017'"

echo ""
echo "预计总时长（并行）: ~10-12 小时"
echo ""
