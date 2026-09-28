#!/usr/bin/env python3
"""Exp012 J1c 基线视频重生成（仅学术研究用途；提示词来自 VU 现成 composed manifest，属研究数据）。

背景：VU `nudity_forget/retain_composed_wan.jsonl` 的 `video` 字段指向
`/scratch/s6398820/video-unlearning/baseline/...`（已随集群 /scratch 清空消失）。
为保住「同 prompt、同 seed、同生成参数、同解析度」的研究口径，这里用同一 Wan2.2-TI2V-5B
（DiffSynth 原生基座 models/Wan-AI/Wan2.2-TI2V-5B，即 Task1 桥产物）重生成 27 条基线视频：

  forget 18 + retain 9（顺序 = VU manifest 原顺序；cell/class_label 命名完全镜像原结构）。

流程：
  1) 生成视频 -> data/wan5b/unlearn_baseline/<class_label>/<cell>.mp4（17 帧 / 480x720 / 50 步 / cfg5 / flow5 / fps16，seed 取 manifest）
  2) NudeNet 闸门复核（只读 import VU NudeNetDetector）：
       forget 必须检出裸体（any_violation=True），retain 必须未检出（False）；任一条违反 -> 退出码 1
       （nudenet 包/权重不可用时仅告警不硬拦，并如实记录）
  3) 写派生 manifest（FT 内新文件，不改 VU）：
       data/wan5b/unlearn_baseline_forget.jsonl（18 条）
       data/wan5b/unlearn_baseline_retain.jsonl  （9 条）
       video 改为本机相对路径，追加 regen 溯源字段；其余字段原样保留。

用法：
  python3 -u scripts/regen_unlearn_baseline_wan5b.py \
      --forget-manifest <VU>/data/splits/nudity_forget_composed_wan.jsonl \
      --retain-manifest <VU>/data/splits/nudity_retain_composed_wan.jsonl \
      [--vu-root <VU>] [--out-dir data/wan5b/unlearn_baseline] \
      [--frames 17] [--height 480] [--width 736] [--steps 50] [--cfg 5.0] [--flow 5.0] [--fps 16] \
      [--device cuda] [--dry-run]
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # VU 只读，禁止向其 src 写 __pycache__

# DiffSynth 生成常量复用自 generate_wan5b_eval（同一加载与采样实现、同一负面提示词）
from generate_wan5b_eval import (  # noqa: E402
    WAN_NEGATIVE, MODEL_FILES, SHARED_DIR,
    TOKENIZER_MODEL_ID, TOKENIZER_PATTERN,
)

VU_ROOT_DEFAULT = "/home/x_jiage/jiage/video-unlearning"
BASE_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"  # Task1 桥产出的 DiffSynth 原生基座（FT 内）


def _load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def build_pipe(device: str):
    import torch  # noqa: E402
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig  # noqa: E402

    model_configs = [ModelConfig(model_id=BASE_MODEL_ID, origin_file_pattern=p) for p in MODEL_FILES]
    tokenizer_config = ModelConfig(model_id=TOKENIZER_MODEL_ID, origin_file_pattern=TOKENIZER_PATTERN)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device,
        model_configs=model_configs, tokenizer_config=tokenizer_config,
    )
    return pipe


def save_video_mp4(video, out_path: Path, fps: int):
    from diffsynth.utils.data import save_video  # noqa: E402

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 使用 .tmp.mp4 而不是 .mp4.tmp，确保 imageio 能识别格式
    tmp = out_path.with_suffix(".tmp.mp4")
    save_video(video, tmp, fps=fps, quality=5)
    import os
    os.replace(tmp, out_path)


def make_nudenet(vu_root: Path, threshold: float = 0.6):
    """只读 import VU NudeNetDetector；依赖缺失返回 None。"""
    try:
        sys.path.insert(0, str(vu_root))
        from src.eval.detectors.nudenet import NudeNetDetector  # noqa: E402

        return NudeNetDetector(threshold=threshold)
    except Exception as exc:  # noqa: BLE001 - 工具链缺失时降级
        print(f"[nudenet] 不可用（{exc!r}），跳过闸门复核（记录 nudenet=unavailable）", flush=True)
        return None


def generate_baselines(rows, split: str, pipe, out_dir: Path, gen: dict, device: str) -> list[dict]:
    import torch  # noqa: E402

    for row in rows:
        class_label = str(row["class_label"])
        cell = str(row["cell"])
        seed = int(row["seed"])
        out = out_dir / class_label / f"{cell}.mp4"
        if out.exists():
            print(f"[gen:{split}] 已存在 {out}，跳过", flush=True)
            row["_video"] = str(out)
            continue
        print(f"[gen:{split}] {class_label}/{cell} seed={seed} {row['prompt'][:70]!r}", flush=True)
        video = pipe(
            prompt=row["prompt"],
            negative_prompt=WAN_NEGATIVE,
            seed=seed,
            height=gen["height"], width=gen["width"], num_frames=gen["frames"],
            num_inference_steps=gen["steps"],
            cfg_scale=gen["cfg"],
            sigma_shift=gen["flow"],
            tiled=True,
        )
        save_video_mp4(video, out, gen["fps"])
        del video
        if device == "cuda":
            torch.cuda.empty_cache()
        row["_video"] = str(out)
    return rows


def write_derived(rows, split: str, out_manifest: Path, gen: dict, nudenet_ok: bool) -> int:
    """写派生 manifest（video 相对化 + regen 溯源）；返回闸门违规条数。"""
    violations = 0
    lines = []
    for row in rows:
        video_abs = Path(row.pop("_video"))
        rel = video_abs.relative_to(out_manifest.parent)
        new = {k: v for k, v in row.items() if k in ("backbone", "domain", "class_label", "split",
                                                      "cell", "prompt", "seed", "gate", "source",
                                                      "split_role", "seed_source")}
        new["video"] = rel.as_posix()
        new["regen"] = {**gen, "nudenet": row.get("_nudenet", "unavailable")}
        lines.append(json.dumps(new, ensure_ascii=False))
        if row.get("_nudenet") is not None:
            expect = (split == "forget")
            if row["_nudenet"] != expect:
                violations += 1
                print(f"[gate:{split}] 违反：{rel} any_violation={row['_nudenet']} "
                      f"（期望 {'True' if expect else 'False'}）", flush=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_manifest) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    import os
    os.replace(tmp, out_manifest)
    print(f"[manifest] {split} -> {out_manifest}（{len(lines)} 条, nudenet_ok={nudenet_ok}）", flush=True)
    return violations


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--forget-manifest", required=True)
    ap.add_argument("--retain-manifest", required=True)
    ap.add_argument("--vu-root", default=VU_ROOT_DEFAULT)
    ap.add_argument("--out-dir", default="data/wan5b/unlearn_baseline")
    ap.add_argument("--out-forget", default="data/wan5b/unlearn_baseline_forget.jsonl")
    ap.add_argument("--out-retain", default="data/wan5b/unlearn_baseline_retain.jsonl")
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--flow", type=float, default=5.0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ft = Path(__file__).resolve().parents[1]
    vu = Path(args.vu_root)
    forget_rows = _load_rows(Path(args.forget_manifest))
    retain_rows = _load_rows(Path(args.retain_manifest))
    print(f"[cfg] forget={len(forget_rows)} retain={len(retain_rows)} "
          f"frames={args.frames} {args.height}x{args.width} steps={args.steps} "
          f"cfg={args.cfg} flow={args.flow} fps={args.fps} device={args.device}", flush=True)
    if not forget_rows or not retain_rows:
        raise SystemExit("forget/retain manifest 为空")

    gen = {"frames": args.frames, "height": args.height, "width": args.width,
           "steps": args.steps, "cfg": args.cfg, "flow": args.flow, "fps": args.fps}
    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else ft / out_dir
    out_forget = Path(args.out_forget)
    out_forget = out_forget if out_forget.is_absolute() else ft / out_forget
    out_retain = Path(args.out_retain)
    out_retain = out_retain if out_retain.is_absolute() else ft / out_retain
    if not str(out_dir.resolve()).startswith(str(ft.resolve())):
        raise SystemExit(f"out_dir 必须在 FT 内: {out_dir}")

    if not args.dry_run:
        nudenet = make_nudenet(vu)
        pipe = None
        if forget_rows or retain_rows:
            print("[pipe] 构建 WanVideoPipeline（base 基座）...", flush=True)
            pipe = build_pipe(args.device)
        generate_baselines(forget_rows, "forget", pipe, out_dir, gen, args.device)
        generate_baselines(retain_rows, "retain", pipe, out_dir, gen, args.device)
        del pipe
        if nudenet is not None:
            for split, rows in (("forget", forget_rows), ("retain", retain_rows)):
                for row in rows:
                    try:
                        out = nudenet.detect(row["_video"])
                        row["_nudenet"] = bool(out.get("any_violation", False))
                        print(f"[nudenet:{split}] {Path(row['_video']).name} "
                              f"any_violation={row['_nudenet']} max={out.get('max_score', 0):.3f}",
                              flush=True)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[nudenet:{split}] 检测失败 {Path(row['_video']).name}: {exc!r}", flush=True)
                        row["_nudenet"] = None
        nudenet_ok = nudenet is not None
        v = write_derived(forget_rows, "forget", out_forget, gen, nudenet_ok)
        v += write_derived(retain_rows, "retain", out_retain, gen, nudenet_ok)
        if nudenet_ok and v:
            raise SystemExit(f"NudeNet 闸门复核失败：{v} 条违反（依赖链停止）")
    else:
        print("[dry-run] 仅打印计划；out_forget/out_retain 均为派生 manifest 目标", flush=True)
    print("DONE regen unlearn baseline", flush=True)


if __name__ == "__main__":
    main()