"""把 PKU-Alignment/SafeSora-Label 官方标注转为训练用 jsonl.

SafeSora-Label 每条记录 (per-video):
  video_path: "videos/<prompt_id>/<video_id>.mp4"
  is_safe: bool                       -> labels[0]
  video_labels: {porn: bool, ...}     -> labels[1..12] (12 harm categories)

输出 (每行): {"video": "videos/.../xx.mp4", "labels": [1,0,...,0]}

用法:
  python convert_safesora_label.py \
      --input config-train.json.gz --output train.jsonl \
      --label-mapping ../configs/labels_safesora.json \
      [--video-root /path/to/extracted]   # 提供时过滤掉磁盘上不存在的视频
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--label-mapping", required=True)
    p.add_argument("--video-root", default=None,
                   help="若提供, 过滤掉该目录下不存在的视频文件")
    args = p.parse_args()

    with open(args.label_mapping, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    label_names = mapping["label_names"]
    assert label_names[0] == "safe", "label mapping 第 0 位必须是 safe"
    harm_names = label_names[1:]

    opener = gzip.open if args.input.endswith(".gz") else open
    with opener(args.input, "rt", encoding="utf-8") as f:
        data = json.load(f)

    seen = set()
    kept, missing, dup = 0, 0, 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for rec in data:
            vid = rec["video_id"]
            if vid in seen:
                dup += 1
                continue
            seen.add(vid)
            vpath = rec["video_path"]
            if args.video_root and not (Path(args.video_root) / vpath).exists():
                missing += 1
                continue
            vl = rec["video_labels"]
            unknown = set(vl) - set(harm_names)
            if unknown:
                raise ValueError(f"数据集出现 label mapping 之外的类别: {unknown}")
            labels = [1 if rec["is_safe"] else 0] + [int(bool(vl[n])) for n in harm_names]
            out.write(json.dumps({"video": vpath, "labels": labels}) + "\n")
            kept += 1

    print(f"{args.input} -> {args.output}: kept={kept}, dup={dup}, missing_on_disk={missing}")


if __name__ == "__main__":
    main()
