#!/bin/bash
# SafeSora safety classifier 训练启动脚本 (幂等, 支持断点续跑).
# 直接反复运行即可: 若 output_dir 下已有 last.pt, 自动从该 checkpoint resume.
set -euo pipefail

REPO=/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
DATA_ROOT=/home/x_jiage/jiage/datasets/SafeSora            # videos/ 解压于此
ANN_DIR=/home/x_jiage/jiage/datasets/SafeSora-Label        # train.jsonl / test.jsonl
OUT_DIR=$REPO/outputs/safesora_safety_classifier
PYTHON=/home/x_jiage/.conda/envs/diffsynth/bin/python

mkdir -p "$OUT_DIR"

RESUME_ARGS=()
if [ -f "$OUT_DIR/last.pt" ]; then
    echo "[run_train] found $OUT_DIR/last.pt -> resume"
    RESUME_ARGS=(--train.resume "$OUT_DIR/last.pt")
fi

cd "$REPO"
exec "$PYTHON" -m classify.training.train \
    --config classify/configs/safety_classifier.yaml \
    --data.train_annotation "$ANN_DIR/train.jsonl" \
    --data.test_annotation  "$ANN_DIR/test.jsonl" \
    --data.video_root "$DATA_ROOT" \
    --data.num_workers 8 \
    --train.output_dir "$OUT_DIR" \
    "${RESUME_ARGS[@]}"
