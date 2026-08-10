import argparse
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm


def crop_and_resize(image, height, width):
    """裁剪并缩放图像到指定尺寸（保持比例，居中裁剪）。输入为 numpy array (H,W,C)。"""
    image_height, image_width, _ = image.shape
    if image_height / image_width < height / width:
        croped_width = int(image_height / height * width)
        left = (image_width - croped_width) // 2
        image = image[:, left: left + croped_width]
        image = Image.fromarray(image).resize((width, height))
    else:
        croped_height = int(image_width / width * height)
        left = (image_height - croped_height) // 2
        image = image[left: left + croped_height, :]
        image = Image.fromarray(image).resize((width, height))
    return image


def save_video(frames, save_path, fps, quality=9):
    """将 PIL Image 列表保存为视频文件。"""
    writer = imageio.get_writer(save_path, fps=fps, quality=quality)
    for frame in tqdm(frames, desc="保存视频"):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()


def get_aligned_num_frames(num_frames, k=4):
    """对齐到 4k+1，向下取整"""
    return (num_frames - 1) // k * k + 1


def convert_video(input_path, output_path, height, width, num_frames, target_fps):
    # 读取源视频
    reader = imageio.get_reader(input_path)
    meta = reader.get_meta_data()
    src_fps = meta.get("fps", None)

    if src_fps is None or src_fps == 0:
        print("Warning: 无法读取源视频帧率，按 24fps 处理")
        src_fps = 24.0
    src_fps = float(src_fps)

    total_frames = reader.count_frames()
    print(f"源视频: {total_frames} 帧, {src_fps:.2f} fps")
    print(f"目标: {height}x{width}, {num_frames} 帧, {target_fps} fps")

    # 帧率重采样：第 i 个目标帧对应的源帧索引
    target_indices = []
    i = 0
    while True:
        src_idx = int(i * src_fps / target_fps)
        if src_idx >= total_frames:
            break
        target_indices.append(src_idx)
        i += 1

    # 截断到 num_frames
    target_indices = target_indices[:num_frames]

    actual_frames = len(target_indices)
    if actual_frames < num_frames:
        print(f"Warning: 视频时长不足，仅能取到 {actual_frames} 帧 (需要 {num_frames} 帧)")

    # 读取并调整每帧分辨率
    frames = []
    for idx in tqdm(target_indices, desc="处理帧"):
        frame = reader.get_data(idx)  # numpy array (H, W, C)
        frame = crop_and_resize(frame, height, width)  # 返回 PIL Image
        frames.append(frame)

    reader.close()

    # 保存
    save_video(frames, output_path, fps=target_fps, quality=9)
    print(f"已保存: {output_path}")
    print(f"  分辨率: {height}x{width}")
    print(f"  帧数: {actual_frames} 帧")
    print(f"  帧率: {target_fps} fps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将视频转换为符合 Wan 系列微调要求的格式")
    parser.add_argument("--input", required=True, help="输入视频路径")
    parser.add_argument("--output", required=True, help="输出视频路径")
    parser.add_argument("--height", type=int, default=480, help="目标高度（16的倍数），默认 480")
    parser.add_argument("--width", type=int, default=832, help="目标宽度（16的倍数），默认 832")
    parser.add_argument("--num_frames", type=int, default=49, help="目标帧数（4k+1），默认 49")
    parser.add_argument("--fps", type=int, default=24, help="目标帧率，默认 24")
    args = parser.parse_args()

    # 校验参数
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError(f"height 和 width 必须是 16 的倍数，当前为 {args.height}x{args.width}")
    aligned = get_aligned_num_frames(args.num_frames)
    if aligned != args.num_frames:
        print(f"Warning: num_frames {args.num_frames} 不满足 4k+1，已调整为 {aligned}")
        args.num_frames = aligned

    convert_video(args.input, args.output, args.height, args.width, args.num_frames, args.fps)
