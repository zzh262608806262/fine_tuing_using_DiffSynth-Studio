"""
用本地 Qwen3-VL-8B-Instruct 为 tiger_dataset 视频片段生成英文 caption，
重建丢失的 metadata CSV（原始 Tiger200K caption 因 gated 仓库无法获取）。

用法:
    python scripts/caption_tiger_clips.py \
        --clip_dir data/tiger_dataset \
        --output_csv data/tiger_dataset/metadata_100.csv \
        --num_clips 100 --seed 0
"""
import argparse
import csv
import glob
import os
import random

import torch
from PIL import Image
import imageio.v3 as iio
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = "/home/x_jiage/jiage/models/Qwen3-VL-8B-Instruct"

PROMPT = (
    "Describe this video clip in one detailed English paragraph for text-to-video "
    "training. Describe the subject, its appearance, its motion, the background and "
    "the camera movement. Do not mention that these are frames or images. "
    "Output only the description."
)


def sample_frames(video_path, num_frames=8):
    frames = iio.imread(video_path, plugin="pyav")
    total = len(frames)
    if total == 0:
        raise ValueError(f"empty video: {video_path}")
    idxs = [int(i * (total - 1) / max(num_frames - 1, 1)) for i in range(num_frames)]
    return [Image.fromarray(frames[i]) for i in idxs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_dir", default="data/tiger_dataset")
    parser.add_argument("--output_csv", default="data/tiger_dataset/metadata_100.csv")
    parser.add_argument("--num_clips", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=8)
    args = parser.parse_args()

    clips = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.clip_dir, "*.mp4")))
    random.seed(args.seed)
    if args.num_clips > 0 and args.num_clips < len(clips):
        clips = random.sample(clips, args.num_clips)
        clips.sort()
    print(f"待打标片段数: {len(clips)}")

    # 断点续传：已有行跳过
    done = set()
    if os.path.exists(args.output_csv):
        with open(args.output_csv, encoding="utf-8") as f:
            done = {row[0] for row in csv.reader(f) if row and row[0] != "video"}
        print(f"已完成 {len(done)} 条，跳过")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    write_header = not os.path.exists(args.output_csv)
    with open(args.output_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["video", "prompt"])
        for i, name in enumerate(clips):
            if name in done:
                continue
            path = os.path.join(args.clip_dir, name)
            try:
                frames = sample_frames(path, args.num_frames)
            except Exception as e:
                print(f"[skip] {name}: {e}")
                continue
            content = [{"type": "image", "image": im} for im in frames]
            content.append({"type": "text", "text": PROMPT})
            messages = [{"role": "user", "content": content}]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=180, do_sample=False)
            caption = processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip().replace("\n", " ")
            writer.writerow([name, caption])
            f.flush()
            print(f"[{i+1}/{len(clips)}] {name}: {caption[:80]}...")

    print(f"完成: {args.output_csv}")


if __name__ == "__main__":
    main()
