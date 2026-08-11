"""
Tiger200K 流式处理: 随机抽样 bvid → 下载源视频 → 切片所有 cuts → 删源视频 → 下一个
峰值硬盘占用 ≈ 单个源视频大小(几十MB~1GB),不用一次性下全部。

前置:
    pip install yt-dlp imageio imageio-ffmpeg tqdm

用法:
    python streaming_tiger_pipeline.py \
        --tiger_csv meta_csv/tiger200k_batch0.csv \
        --output_dir data/tiger_dataset \
        --num_bvids 50 \
        --cookie_file cookies.txt

cookie 获取(服务器上用 cookie 文件,本地可直读浏览器):
    - 方式1(推荐,本地): --cookies_from_browser chrome
    - 方式2(服务器): 浏览器装 "Get cookies.txt LOCALLY" 扩展,导出 cookies.txt
"""
import argparse
import csv
import os
import random
import subprocess
import glob
import threading
import queue
import imageio
import imageio_ffmpeg
from tqdm import tqdm

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# caption 字段候选名(按优先级,英文优先用于训练)
CAPTION_CANDIDATES = ["caption_en", "caption_zh", "caption", "text", "description", "prompt"]


def detect_caption_field(fieldnames):
    for c in CAPTION_CANDIDATES:
        if c in fieldnames:
            return c
    return None


def get_video_size(video_path):
    r = imageio.get_reader(video_path)
    m = r.get_meta_data()
    r.close()
    size = m.get("size")
    if size is None:
        raise ValueError(f"无法读取视频分辨率: {video_path}")
    return size[0], size[1]  # (width, height)


def compute_safe_crop(crop_region, video_width, video_height):
    """
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
    return f"crop={cr - cl}:{cb - ct}:{cl}:{ct}"


def timecode_to_seconds(tc):
    """时间码转秒: '00:01:19.767' -> 79.767"""
    parts = tc.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def process_one(source_path, start, end, crop_region, output_path,
                height, width, num_frames, fps):
    """
    ffmpeg 一步完成: 切片 + 安全区域裁剪 + 缩放 + 均匀采样 num_frames 帧并标记为 fps
    均匀采样: 从整段切片按时间均匀取 num_frames 帧(覆盖全程), 而非只取前N帧
    """
    video_width, video_height = get_video_size(source_path)
    safe_crop = compute_safe_crop(crop_region, video_width, video_height)
    # 计算切片时长, 用于均匀采样
    duration = timecode_to_seconds(end) - timecode_to_seconds(start)
    if duration <= 0:
        duration = num_frames / fps
    # 采样率: 多采2帧确保 >= num_frames, 再用 -frames:v 截断到 num_frames(均匀分布的)
    sample_fps = (num_frames + 2) / duration
    vf = (
        f"{safe_crop},"
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"fps={sample_fps}"  # 按时间均匀采样到约51帧
    )
    cmd = [
        FFMPEG, "-y",
        "-i", source_path,
        "-ss", str(start),
        "-to", str(end),
        "-vf", vf,
        "-r", str(fps),  # 输出帧率标记为目标帧率(如24fps)
        "-vsync", "0",  # 不重采样, 只传递帧(passthrough), 保持均匀采样的帧
        "-frames:v", str(num_frames),  # 截断到 num_frames 帧(均匀分布的)
        "-c:v", "libx264",
        "-crf", "19",
        "-preset", "fast",
        "-an",
        output_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0, result.stderr.decode("utf-8", errors="ignore")


def download_video(bvid, temp_dir, max_height, cookie_file=None,
                   cookies_from_browser=None, proxy=""):
    """
    用 yt-dlp 下载单个 B站视频,返回下载到的文件路径(或 None)
    优先取 <= max_height 的最高画质
    proxy: 下载代理,默认 ""(直连,B站是国内站点直连最快)
    """
    url = f"https://www.bilibili.com/video/{bvid}/"
    out_template = os.path.join(temp_dir, f"{bvid}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--proxy", proxy,
        "-f", f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best",
        "--merge-output-format", "mp4",
        "--no-progress",
        "-o", out_template,
        url,
    ]
    if cookie_file:
        cmd += ["--cookies", cookie_file]
    elif cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        return None, result.stderr.decode("utf-8", errors="ignore")
    # 找下载到的文件(yt-dlp 合并后是 .mp4)
    files = glob.glob(os.path.join(temp_dir, f"{bvid}.*"))
    files = [f for f in files if f.endswith((".mp4", ".mkv", ".webm"))]
    if not files:
        return None, "下载完成但找不到文件"
    return files[0], ""


def load_done_bvids(metadata_path):
    """读取已处理的 bvid(断点续传),从 metadata.csv 的 video 列提取"""
    done = set()
    if not os.path.exists(metadata_path):
        return done
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过表头
        for row in reader:
            if row:
                # video 列格式: BVxxx_sceneY_cutZ.mp4
                bvid = row[0].split("_")[0]
                done.add(bvid)
    return done


def downloader(bvid_rows, download_queue, args, stats, stats_lock, stop_event):
    """下载线程: 遍历 (bvid, rows) 列表, 下载源视频后放入队列"""
    for bvid, rows in bvid_rows:
        if stop_event.is_set():
            break
        src_path, err = download_video(
            bvid, args.temp_dir, args.max_height,
            args.cookie_file, args.cookies_from_browser, args.proxy,
        )
        if src_path is None:
            with stats_lock:
                stats["download_fail"] += 1
            continue
        download_queue.put((bvid, rows, src_path))
    # 放入结束信号(每个切片线程一个)
    for _ in range(args.workers):
        download_queue.put(None)


def cutter(download_queue, args, caption_field, metadata_path, meta_lock,
           stats, stats_lock, pbar, stop_event):
    """切片线程: 从队列取已下载视频, 切片所有 cuts, 写 metadata, 删源视频"""
    while True:
        if stop_event.is_set():
            break
        item = download_queue.get()
        if item is None:
            break
        bvid, rows, src_path = item
        bvid_ok = 0
        for row in rows:
            if stop_event.is_set():
                break
            start = row["cut_start_timecode"]
            end = row["cut_end_timecode"]
            crop_region = row["crop_region"].strip('[]"')
            caption = row.get(caption_field, "").strip()
            if not caption:
                continue
            output_name = f"{bvid}_scene{row['scene_id']}_cut{row['cut_id']}.mp4"
            output_path = os.path.join(args.output_dir, output_name)
            try:
                ok, _ = process_one(
                    src_path, start, end, crop_region, output_path,
                    args.height, args.width, args.num_frames, args.fps,
                )
                if ok and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    with meta_lock:
                        with open(metadata_path, "a", encoding="utf-8", newline="") as f:
                            csv.writer(f).writerow([output_name, caption])
                    with stats_lock:
                        stats["cut_ok"] += 1
                        if args.target_cuts > 0 and stats["cut_ok"] >= args.target_cuts:
                            stop_event.set()
                    bvid_ok += 1
                else:
                    # 切片失败, 删除空文件
                    if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
                        try:
                            os.remove(output_path)
                        except OSError:
                            pass
                    with stats_lock:
                        stats["cut_fail"] += 1
            except Exception:
                with stats_lock:
                    stats["cut_fail"] += 1
        if bvid_ok > 0:
            with stats_lock:
                stats["bvids_ok"] += 1
        # 删除源视频
        if not args.keep_source:
            try:
                os.remove(src_path)
            except OSError:
                pass
        pbar.update(1)


def main():
    parser = argparse.ArgumentParser(
        description="Tiger200K 流式处理: 随机抽样 → 下载 → 切片 → 删源视频"
    )
    parser.add_argument("--tiger_csv", required=True, help="Tiger200K 的 meta CSV 路径")
    parser.add_argument("--output_dir", required=True, help="切片输出目录")
    parser.add_argument("--num_bvids", type=int, default=50, help="随机抽样的 bvid 数量,默认 50")
    parser.add_argument("--max_height", type=int, default=720,
                        help="下载视频最大高度(720够训练用,1080更清晰但更大),默认 720")
    parser.add_argument("--cookie_file", default=None, help="yt-dlp cookie 文件路径(Netscape 格式)")
    parser.add_argument("--cookies_from_browser", default=None,
                        help="本地直读浏览器 cookie,如 chrome/edge(服务器上不可用)")
    parser.add_argument("--proxy", default="",
                        help="下载代理,默认空(直连,B站国内站点直连最快)。如需代理设为 http://127.0.0.1:7897")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--caption_field", default=None, help="caption 字段名,留空自动检测")
    parser.add_argument("--keep_source", action="store_true", help="保留下载的源视频(默认删除)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子,保证可复现")
    parser.add_argument("--workers", type=int, default=2,
                        help="切片线程数(流水线: 1下载+workers切片),默认 2")
    parser.add_argument("--target_cuts", type=int, default=0,
                        help="目标切片数,达到后停止(0=不限)。如 3000 表示切够3000条就停")
    args = parser.parse_args()

    # 校验
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError(f"height/width 必须是 16 的倍数,当前 {args.height}x{args.width}")

    # 检查 yt-dlp 是否可用
    try:
        subprocess.run(["yt-dlp", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误: 未找到 yt-dlp,请先安装: pip install yt-dlp")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, "_source_temp")
    os.makedirs(temp_dir, exist_ok=True)
    metadata_path = os.path.join(args.output_dir, "metadata.csv")

    # 断点续传: 读取已处理的 bvid
    done_bvids = load_done_bvids(metadata_path)
    if done_bvids:
        print(f"断点续传: 已处理 {len(done_bvids)} 个 bvid,将跳过")

    # 读取 CSV,按 bvid 分组
    with open(args.tiger_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        caption_field = args.caption_field or detect_caption_field(reader.fieldnames)
        if caption_field is None:
            raise ValueError(f"找不到 caption 字段,CSV 字段: {reader.fieldnames}")
        print(f"使用 caption 字段: {caption_field}")
        print(f"输出参数: {args.width}x{args.height}, {args.num_frames}帧, {args.fps}fps")
        print(f"下载画质上限: {args.max_height}p, 抽样 {args.num_bvids} 个 bvid\n")

        groups = {}
        for row in reader:
            bvid = row["bvid"]
            if bvid in done_bvids:
                continue
            groups.setdefault(bvid, []).append(row)

    bvids = list(groups.keys())
    if len(bvids) == 0:
        print("没有需要处理的 bvid(可能都已处理完)")
        return

    # 随机抽样
    random.seed(args.seed)
    n = min(args.num_bvids, len(bvids))
    sampled = random.sample(bvids, n)
    print(f"CSV 共 {len(bvids)} 个待处理 bvid,本次随机抽样 {n} 个\n")

    # 初始化 metadata.csv(不存在则写表头)
    if not os.path.exists(metadata_path):
        with open(metadata_path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["video", "prompt"])

    args.temp_dir = temp_dir  # 供 downloader 线程访问
    stats = {"download_fail": 0, "cut_ok": 0, "cut_fail": 0, "bvids_ok": 0}
    meta_lock = threading.Lock()
    stats_lock = threading.Lock()
    download_queue = queue.Queue(maxsize=args.workers)  # 限制缓存数, 控制磁盘占用
    bvid_rows = [(bvid, groups[bvid]) for bvid in sampled]

    pbar = tqdm(total=n, desc="流式处理 bvid")
    # 启动下载线程(1个) + 切片线程(workers个), 流水线重叠下载与切片
    stop_event = threading.Event()
    dl_thread = threading.Thread(
        target=downloader, args=(bvid_rows, download_queue, args, stats, stats_lock, stop_event)
    )
    cut_threads = [
        threading.Thread(
            target=cutter,
            args=(download_queue, args, caption_field, metadata_path,
                  meta_lock, stats, stats_lock, pbar, stop_event),
        )
        for _ in range(args.workers)
    ]
    dl_thread.start()
    for t in cut_threads:
        t.start()
    dl_thread.join()
    for t in cut_threads:
        t.join()
    pbar.close()

    # 清理临时目录
    if not args.keep_source:
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

    print(f"\n完成:")
    print(f"  bvid 成功: {stats['bvids_ok']}/{n}")
    print(f"  切片成功: {stats['cut_ok']}")
    print(f"  切片失败: {stats['cut_fail']}")
    print(f"  下载失败: {stats['download_fail']}")
    print(f"metadata.csv: {metadata_path}")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
