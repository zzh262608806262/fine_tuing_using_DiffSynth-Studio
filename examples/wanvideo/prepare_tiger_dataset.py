import argparse
import csv
import os
import subprocess
import imageio
import imageio_ffmpeg
from tqdm import tqdm

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# caption 字段候选名（按优先级）
CAPTION_CANDIDATES = ["caption_en", "caption_zh", "caption", "text", "description", "prompt"]


def detect_caption_field(fieldnames):
    for c in CAPTION_CANDIDATES:
        if c in fieldnames:
            return c
    return None


def get_video_size(video_path):
    """用 imageio 读取视频分辨率，返回 (width, height)"""
    r = imageio.get_reader(video_path)
    m = r.get_meta_data()
    r.close()
    size = m.get("size")
    if size is None:
        raise ValueError(f"无法读取视频分辨率: {video_path}")
    return size[0], size[1]  # (width, height)


def compute_safe_crop(crop_region, video_width, video_height):
    """
    复用 Tiger200K cut_videos.py 的裁剪逻辑。
    crop_region: "top,bottom,left,right" 归一化坐标 (0-1)
    返回 ffmpeg crop 参数 crop=w:h:x:y
    """
    crop_values = list(map(float, crop_region.split(",")))
    ct, cb, cl, cr = (
        int(crop_values[0] * video_height) // 2 * 2,
        int(crop_values[1] * video_height) // 2 * 2,
        int(crop_values[2] * video_width) // 2 * 2,
        int(crop_values[3] * video_width) // 2 * 2,
    )
    safe_w = cr - cl
    safe_h = cb - ct
    safe_x = cl
    safe_y = ct
    return f"crop={safe_w}:{safe_h}:{safe_x}:{safe_y}"


def process_one(source_path, start, end, crop_region, output_path,
                height, width, num_frames, fps):
    """
    用 ffmpeg 一步完成: 切片 + 安全区域裁剪 + 居中裁剪缩放到目标 + 帧率重采样 + 取前 N 帧
    """
    video_width, video_height = get_video_size(source_path)
    safe_crop = compute_safe_crop(crop_region, video_width, video_height)

    # 滤镜链:
    # 1. crop: 安全区域裁剪(去水印/字幕/黑边)
    # 2. scale + force_original_aspect_ratio=increase: 等比放大到能覆盖目标
    # 3. crop=width:height: 居中裁剪到精确目标尺寸(等效 crop_and_resize)
    # 4. fps: 帧率重采样到 24fps
    vf = (
        f"{safe_crop},"
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"fps={fps}"
    )

    cmd = [
        FFMPEG, "-y",
        "-i", source_path,
        "-ss", str(start),
        "-to", str(end),
        "-vf", vf,
        "-frames:v", str(num_frames),
        "-c:v", "libx264",
        "-crf", "19",
        "-preset", "fast",
        "-an",  # 丢弃音频(T2V 不需要)
        output_path,
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0, result.stderr.decode("utf-8", errors="ignore")


def main():
    parser = argparse.ArgumentParser(description="Tiger200K 数据集自动处理:切片+裁剪+调整+生成 metadata.csv")
    parser.add_argument("--tiger_csv", required=True, help="Tiger200K 的 meta CSV 路径")
    parser.add_argument("--source_dir", required=True, help="源视频目录(文件名为 {bvid}.mp4)")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--height", type=int, default=480, help="目标高度(16的倍数)，默认 480")
    parser.add_argument("--width", type=int, default=832, help="目标宽度(16的倍数)，默认 832")
    parser.add_argument("--num_frames", type=int, default=49, help="目标帧数(4k+1)，默认 49")
    parser.add_argument("--fps", type=int, default=24, help="目标帧率，默认 24")
    parser.add_argument("--caption_field", default=None, help="caption 字段名，留空自动检测")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条(测试用)")
    args = parser.parse_args()

    # 校验
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError(f"height 和 width 必须是 16 的倍数，当前 {args.height}x{args.width}")

    os.makedirs(args.output_dir, exist_ok=True)
    metadata_rows = []
    success, fail, skip = 0, 0, 0

    with open(args.tiger_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # 检测 caption 字段
        caption_field = args.caption_field or detect_caption_field(reader.fieldnames)
        if caption_field is None:
            raise ValueError(f"找不到 caption 字段，CSV 字段为: {reader.fieldnames}")
        print(f"使用 caption 字段: {caption_field}")
        print(f"输出参数: {args.width}x{args.height}, {args.num_frames}帧, {args.fps}fps\n")

        rows = list(reader)
        if args.limit:
            rows = rows[:args.limit]

        for row in tqdm(rows, desc="处理视频"):
            bvid = row["bvid"]
            scene_id = row["scene_id"]
            cut_id = row["cut_id"]
            start = row["cut_start_timecode"]
            end = row["cut_end_timecode"]
            crop_region = row["crop_region"].strip('[]"')
            caption = row.get(caption_field, "").strip()

            source_path = os.path.join(args.source_dir, f"{bvid}.mp4")
            if not os.path.exists(source_path):
                skip += 1
                continue
            if not caption:
                skip += 1
                continue

            output_name = f"{bvid}_scene{scene_id}_cut{cut_id}.mp4"
            output_path = os.path.join(args.output_dir, output_name)

            try:
                ok, err = process_one(
                    source_path, start, end, crop_region, output_path,
                    args.height, args.width, args.num_frames, args.fps,
                )
                if ok and os.path.exists(output_path):
                    metadata_rows.append([output_name, caption])
                    success += 1
                else:
                    fail += 1
                    if fail <= 3:
                        print(f"\n[失败] {output_name}: {err[-300:]}")
            except Exception as e:
                fail += 1
                if fail <= 3:
                    print(f"\n[异常] {output_name}: {e}")

    # 写 metadata.csv
    metadata_path = os.path.join(args.output_dir, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "prompt"])
        writer.writerows(metadata_rows)

    print(f"\n完成: 成功 {success}, 失败 {fail}, 跳过 {skip}(源视频缺失或无 caption)")
    print(f"metadata.csv: {metadata_path}")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
