#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp012 四臂 porn 专项评测视频批量生成（Wan2.2-TI2V-5B 文本驱动，无首帧）

四臂（同 prompt / 同 seed / 同帧数 / 同尺寸 / 同 steps / 同 cfg / 同 flow，仅模型不同，产物逐一对齐）：
    base      = models/Wan-AI/Wan2.2-TI2V-5B/          （Task1 桥产物，未擦除基座）
    erased    = models/Wan-AI/Wan2.2-TI2V-5B-erased/    （Task3 桥产物，擦除已合进基座）
    base_ft   = base      + LoRA models/train/Wan2.2-TI2V-5B_base_lora10e2/epoch-1.safetensors
    erased_ft = erased    + LoRA models/train/Wan2.2-TI2V-5B_erased_lora10e2/epoch-1.safetensors

评测集（来自 video-unlearning VU，只读引用；不拷贝任何文件，运行时直接读 VU 原文件）：
    fast     = <VU>/data/splits/exp504_t2v_prompts/nudity.jsonl（18 条，concept/class_label/prompt/seed）
             + <VU>/data/pornography/nudity_neighborhood.jsonl（11 条梯度探针，tier/num_frames/seed/prompt）
             => 29 条，idx 0..28
    benchmark = <VU>/data/benchmark_wan.jsonl 过滤 class_label 或 domain 含 "nudity"
             => 99 条（nudity_breasts 25 + nudity_buttocks 25 + nudity_female_genitalia 25 + nudity_retain 24），idx 0..98

输出：outputs/wan5b_eval/<arm>/<set>_<idx:03d>.mp4 + 生成参数 json outputs/wan5b_eval/gen_params.json
    （fast 集 29 条 -> fast_000..fast_028；benchmark 集 99 条 -> benchmark_000..benchmark_098；
      四条 arm 目录结构完全一致，判别脚本按 <set>_<idx> 对齐跨臂对照）

生成参数（四臂完全一致，全部可 CLI 覆盖；默认与 DiffSynth WanVideoPipeline 一致，
examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py 即不传而用默认）：
    num_frames=17（--frames，覆盖 neighborhood manifest 自带的 num_frames=17，全集统一 17）
    480x720（--height/--width，16 的倍数满足 VAE）；steps=50 cfg=5.0 flow_shift=5.0 fps=15
    seed = manifest 的 seed 字段（不加偏移，四臂同条同 seed）

模型加载：WanVideoPipeline.from_pretrained + ModelConfig（本地 ./models/<model_id> + 显式
origin_file_pattern；依赖环境变量 DIFFSYNTH_SKIP_DOWNLOAD=true 禁止在线下载，脚本启动即校验）。
每 arm 目录需三件套：diffusion_pytorch_model*.safetensors（DiT，桥产物，必须真实存在）+
models_t5_umt5-xxl-enc-bf16.safetensors + Wan2.2_VAE.safetensors（后两者可软链自
models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/，缺失时打印建议命令）。
LoRA 挂载：pipe.load_lora(pipe.dit, ckpt, alpha=1)（1.3B 先例 generate_safesora.py 同款）。

用法：
    # 本地冒烟/预览（不加载模型）
    python3 scripts/generate_wan5b_eval.py --dry-run

    # 集群（见 slurm/wan5b_eval_gen.sbatch，单臂单 job，--start/--end 分片续跑）
    python -u scripts/generate_wan5b_eval.py --arm base --set fast --start 0 --end 29
    python -u scripts/generate_wan5b_eval.py --arm erased --set benchmark --start 0 --end 99

    # 冒烟：只跑 2 条
    python -u scripts/generate_wan5b_eval.py --arm base --set fast --start 0 --end 2
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# ---------------- 常量 ----------------
# VU 只读引用（绝对路径写死，不拷贝 VU 任何文件到 FT）
VU_ROOT = os.getenv("VU_ROOT", "../video-unlearning")
SET_FILES = {
    "fast": [
        os.path.join(VU_ROOT, "data/splits/exp504_t2v_prompts/nudity.jsonl"),       # 18 条
        os.path.join(VU_ROOT, "data/pornography/nudity_neighborhood.jsonl"),        # 11 条
    ],
    "benchmark": os.path.join(VU_ROOT, "data/benchmark_wan.jsonl"),                 # 过滤后 99 条
}
SET_EXPECTED = {"fast": 29, "benchmark": 99}

# 四臂：model_id 相对 ./models 解析为本地目录（ModelConfig 默认 local_model_path=./models）
ARMS = {
    "base":      {"model_id": "Wan-AI/Wan2.2-TI2V-5B",          "lora": None},
    "erased":    {"model_id": "Wan-AI/Wan2.2-TI2V-5B-erased",   "lora": None},
    "base_ft":   {"model_id": "Wan-AI/Wan2.2-TI2V-5B",          "lora": "models/train/Wan2.2-TI2V-5B_base_lora10e2/epoch-0.safetensors"},
    "erased_ft": {"model_id": "Wan-AI/Wan2.2-TI2V-5B-erased",   "lora": "models/train/Wan2.2-TI2V-5B_erased_lora10e2/epoch-0.safetensors"},
}

# 每个 arm 目录必须存在的三件套（后两个可软链自共享 safetensors 目录；DiT 必须真实存在）
MODEL_FILES = [
    "diffusion_pytorch_model*.safetensors",      # DiT（桥转换产物 Task1/Task3）
    "models_t5_umt5-xxl-enc-bf16.safetensors",   # T5
    "Wan2.2_VAE.safetensors",                    # VAE
]
SHARED_DIR = "models/DiffSynth-Studio/Wan-Series-Converted-Safetensors"
# 5B 无独立 tokenizer，复用本地 1.3B 的 google/umt5-xxl/ tokenizer（DiffSynth 官方 5B 例同样做法）
TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
TOKENIZER_PATTERN = "google/umt5-xxl/"

# Wan2.2 官方示例负面提示词（文本驱动），与 DiffSynth example/Wan2.2-TI2V-5B.py 逐字一致
WAN_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


# ---------------------------------------------------------------------------
# 评测集读取 / 归一化
# ---------------------------------------------------------------------------
def load_entries(set_name):
    """读 jsonl -> 原始条目列表。fast=两文件顺序拼接；benchmark 过滤 class_label/domain 含 nudity。"""
    entries = []
    paths = SET_FILES[set_name] if isinstance(SET_FILES[set_name], list) else [SET_FILES[set_name]]
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
    if set_name == "benchmark":
        keep = [e for e in entries
                if "nudity" in str(e.get("class_label", "")) or "nudity" in str(e.get("domain", ""))]
        entries = keep
    assert len(entries) == SET_EXPECTED[set_name], \
        f"{set_name} 实际 {len(entries)} 条（期望 {SET_EXPECTED[set_name]} 条），请核对 VU 评测集"
    return entries


def build_items(set_name, entries):
    """归一化为 item：idx=set 内全局序号（跨 arm 对齐）、prompt/seed + extra 协议字段原样保留。"""
    items = []
    for idx, e in enumerate(entries):
        items.append({
            "idx": idx,
            "prompt": str(e.get("prompt", "")).strip(),
            "seed": int(e.get("seed", 0)),
            "extra": {k: v for k, v in e.items() if k not in ("prompt", "seed")},
        })
    return items


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------
def arm_base_dir(arm):
    return os.path.join(_ROOT, "models", ARMS[arm]["model_id"].replace("/", os.sep))


def check_arm_files(arm):
    """校验 arm 目录三件套，返回缺失 pattern 列表（空=就绪）；缺失时打印建议命令。"""
    base = arm_base_dir(arm)
    missing = []
    for pattern in MODEL_FILES:
        if glob.glob(os.path.join(base, pattern)):
            continue
        missing.append(pattern)
        if pattern == "diffusion_pytorch_model*.safetensors":
            print(f"[missing][{arm}] DiT {pattern} 缺失：桥转换产物（无法软链），请从上游 Task1/Task3 拷入 "
                  f"{os.path.relpath(base, _ROOT)}/")
        else:
            print(f"[missing][{arm}] {pattern} 缺失，建议软链共享权重：\n"
                  f"    ln -s {os.path.join(_ROOT, SHARED_DIR, pattern)} {os.path.relpath(base, _ROOT)}/")
    return missing


def check_lora(arm):
    lora = ARMS[arm]["lora"]
    if lora and not os.path.isfile(os.path.join(_ROOT, lora)):
        print(f"[missing][{arm}] LoRA 不存在: {lora}")
        return True
    return False


def check_skip_download():
    """DIFFSYNTH_SKIP_DOWNLOAD 必须是 'true'（DiffSynth config.py 判据 .lower()=='true'）。"""
    val = os.environ.get("DIFFSYNTH_SKIP_DOWNLOAD", "")
    ok = val.strip().lower() == "true"
    if not ok:
        print("[warn] DIFFSYNTH_SKIP_DOWNLOAD 未设为 true（当前为空），模型加载将触发在线下载；"
              "请在 sbatch 或 shell 中 export DIFFSYNTH_SKIP_DOWNLOAD=true")
    return ok


# ---------------------------------------------------------------------------
# 模型 / 生成
# ---------------------------------------------------------------------------
def build_pipe(arm, device):
    """WanVideoPipeline.from_pretrained + 可选 LoRA（API 与 generate_safesora.build_pipe 一致）。"""
    import torch
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig  # noqa: E402

    arm_cfg = ARMS[arm]
    model_configs = [
        ModelConfig(model_id=arm_cfg["model_id"], origin_file_pattern=p) for p in MODEL_FILES
    ]
    tokenizer_config = ModelConfig(model_id=TOKENIZER_MODEL_ID, origin_file_pattern=TOKENIZER_PATTERN)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    if arm_cfg["lora"]:
        lora_path = os.path.join(_ROOT, arm_cfg["lora"])
        print(f"[{arm}] 加载 LoRA: {lora_path} (alpha=1)")
        pipe.load_lora(pipe.dit, lora_path, alpha=1)
    else:
        print(f"[{arm}] 无 LoRA（裸基座：{arm_cfg['model_id']}）")
    return pipe


def generate_one(pipe, item, args):
    """生成单个视频（文本驱动：不传 input_image），调用方负责 tmp + 原子改名。"""
    from diffsynth.utils.data import save_video  # noqa: E402

    out_path, tmp_path = item["_out_path"], item["_tmp_path"]
    meta = {
        "num_frames": args.frames, "height": args.height, "width": args.width,
        "steps": args.steps, "cfg": args.cfg, "flow": args.flow,
        "seed": item["seed"], "fps": args.fps,
    }
    video = pipe(
        prompt=item["prompt"],
        negative_prompt=WAN_NEGATIVE,
        seed=item["seed"],
        height=args.height, width=args.width, num_frames=args.frames,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg,
        sigma_shift=args.flow,  # flow_shift=5.0（Wan2.2 DiffSynth 默认）
        tiled=True,
    )
    save_video(video, tmp_path, fps=args.fps, quality=5)
    os.replace(tmp_path, out_path)
    return meta


def write_gen_params(output_root, set_items, params, arms):
    """生成参数 json 落 outputs/wan5b_eval/gen_params.json（全量条目，多 shard 内容一致、原子写）。"""
    os.makedirs(output_root, exist_ok=True)
    doc = {
        "meta": {
            "project": "Exp012 four-arm porn eval (Wan2.2-TI2V-5B)",
            "arms": arms,
            "sets_expected": SET_EXPECTED,
            "params": params,
            "negative_prompt_md5": hashlib.md5(WAN_NEGATIVE.encode()).hexdigest()[:8],
        },
        "sets": {s: items for s, items in set_items.items()},
    }
    path = os.path.join(output_root, "gen_params.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Exp012 四臂 Wan2.2-TI2V-5B porn 评测视频批量生成")
    p.add_argument("--arm", type=str, default="base,erased,base_ft,erased_ft",
                   help="arm 子集（逗号分隔），可选 base,erased,base_ft,erased_ft")
    p.add_argument("--set", type=str, default="fast,benchmark",
                   help="评测集（逗号分隔），可选 fast(29 条=18+11), benchmark(99 条)")
    p.add_argument("--start", type=int, default=0, help="每条目在该 set 内的 idx 起点（含）")
    p.add_argument("--end", type=int, default=-1, help="idx 终点（不含），-1=该 set 全量")
    p.add_argument("--frames", type=int, default=17,
                   help="帧数，全 set 统一（含 neighborhood manifest 自带的 num_frames=17，一律以 CLI 为准）")
    p.add_argument("--height", type=int, default=480, help="高（16 的倍数，VAE 要求）")
    p.add_argument("--width", type=int, default=720, help="宽（16 的倍数，VAE 要求）")
    p.add_argument("--steps", type=int, default=50, help="采样步数（DiffSynth WanVideoPipeline 默认 50）")
    p.add_argument("--cfg", type=float, default=5.0, help="cfg_scale（DiffSynth 默认 5.0）")
    p.add_argument("--flow", type=float, default=5.0, help="sigma_shift / flow shift（DiffSynth 默认 5.0）")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--output-root", type=str, default="outputs/wan5b_eval", dest="output_root")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dry-run", action="store_true", help="只打印计划与校验，不加载模型/不生成")
    return p


def print_env():
    import platform
    print(f"[env] python {sys.version.split()[0]} ({platform.platform()})")
    print(f"[env] FT root = {_ROOT}")
    print(f"[env] VU root (只读) = {VU_ROOT}")
    print(f"[env] DIFFSYNTH_SKIP_DOWNLOAD = {os.environ.get('DIFFSYNTH_SKIP_DOWNLOAD', '(未设置)')}")


def main():
    args = build_parser().parse_args()
    arms = [a.strip() for a in args.arm.split(",") if a.strip()]
    sets = [s.strip() for s in args.set.split(",") if s.strip()]
    for a in arms:
        assert a in ARMS, f"未知 arm: {a}（可选 {list(ARMS)}）"
    for s in sets:
        assert s in SET_FILES, f"未知 set: {s}（可选 {list(SET_FILES)}）"
    assert args.height % 16 == 0 and args.width % 16 == 0, "尺寸必须是 16 的倍数（VAE 要求）"
    end = args.end if args.end >= 0 else None  # None = 全量

    set_items = {s: build_items(s, load_entries(s)) for s in sets}
    params = {
        "frames": args.frames, "height": args.height, "width": args.width,
        "steps": args.steps, "cfg": args.cfg, "flow": args.flow, "fps": args.fps,
        "idx_range": [args.start, args.end],
        "seed_source": "manifest.seed（四臂同 prompt 同 seed）",
    }

    print_env()
    print("=" * 78)
    print(f"Exp012 生成计划  arms={arms} sets={sets} idx=[{args.start},{args.end}) "
          f"params={json.dumps({k: v for k, v in params.items() if k != 'idx_range'})}")
    skip_download_ok = check_skip_download()

    any_missing = False
    for a in arms:
        missing = check_arm_files(a)
        any_missing |= bool(missing)
        any_missing |= check_lora(a)
        for s in sets:
            out_dir = os.path.join(args.output_root, a)
            items = set_items[s]
            todo = [it for it in items if it["idx"] >= args.start and (end is None or it["idx"] < end)]
            done = sum(1 for it in todo if os.path.exists(
                os.path.join(out_dir, f"{s}_{it['idx']:03d}.mp4")))
            status = "缺文件" if missing else f"待生成 {len(todo) - done}"
            print(f"  [{a}/{s}] 共{len(items)}条 本区间{len(todo)} 已完成{done} [{status}]")
            for it in todo[:3]:
                print(f"      idx={it['idx']:03d} seed={it['seed']} 帧={args.frames} "
                      f"{args.width}x{args.height} | {it['prompt'][:70]}...")
    if args.dry_run:
        print(f"DRY_RUN 完成（未加载模型，未生成）。skip_download_ok={skip_download_ok} "
              f"missing={'有' if any_missing else '无'}（真实运行遇缺文件将中止）")
        return
    if not skip_download_ok:
        raise SystemExit("DIFFSYNTH_SKIP_DOWNLOAD 未设为 true，终止（防在线下载）")
    if any_missing:
        raise SystemExit("模型文件缺失：请先按上述提示软链/拷贝后再运行")

    gen_params_path = write_gen_params(args.output_root, set_items, params, arms)
    print(f"[manifest] 生成参数已写 {gen_params_path}")

    # 逐臂串行；只为仍待生成的臂构建 pipe
    for a in arms:
        remaining_by_set = {}
        for s in sets:
            out_dir = os.path.join(args.output_root, a)
            os.makedirs(out_dir, exist_ok=True)
            todo = [it for it in set_items[s] if it["idx"] >= args.start and (end is None or it["idx"] < end)]
            rem = []
            for it in todo:
                it["_out_path"] = os.path.join(out_dir, f"{s}_{it['idx']:03d}.mp4")
                it["_tmp_path"] = it["_out_path"] + ".tmp.mp4"
                if not os.path.exists(it["_out_path"]):
                    rem.append(it)
            remaining_by_set[s] = rem
        total = sum(len(v) for v in remaining_by_set.values())
        if total == 0:
            print(f"[{a}] 本区间全部已生成，跳过模型加载")
            continue

        print(f"[{a}] 加载模型（共 {total} 条待生成）...")
        pipe = build_pipe(a, args.device)
        for s in sets:
            rem = remaining_by_set[s]
            if not rem:
                continue
            print(f"[{a}/{s}] 待生成 {len(rem)} 条")
            for n, it in enumerate(rem, 1):
                t0 = time.time()
                meta = generate_one(pipe, it, args)
                print(f"[{a}/{s}] {n}/{len(rem)} idx={it['idx']:03d} "
                      f"{meta['num_frames']}f {meta['width']}x{meta['height']} "
                      f"seed={meta['seed']} ({time.time() - t0:.0f}s)", flush=True)
        print(f"[{a}] SHARD_COMPLETE")
    print("ALL_DONE")


if __name__ == "__main__":
    main()