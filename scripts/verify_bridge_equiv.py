#!/usr/bin/env python3
"""Wan2.2-TI2V-5B 权重桥等价性验证（Task 1 网关，仅学术研究用途，提示词均为良性）。

三种子命令（同一批 prompt+seed，两端参数完全一致）：
  gen-diffsynth <prompts.jsonl> <outdir>   # DiffSynth 原格式基座（models/Wan-AI/Wan2.2-TI2V-5B）
  gen-vu        <prompts.jsonl> <outdir>   # VU diffusers 基座（须在 VU .venv 下运行）
  compare       <dirA> <dirB> --videos N   # 帧级相似度对比，cosine<--threshold 退出码 1（门控）

生成参数（两端一致，勿改）：17 帧 / 480x720 / 50 步 / cfg 5.0 / flow 5.0 / seed 取 manifest。
"""
import argparse
import json
import sys
from pathlib import Path

FRAMES, HEIGHT, WIDTH, STEPS, CFG, FLOW, FPS = 17, 480, 736, 50, 5.0, 5.0, 16


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cmd_gen_diffsynth(args):
    """复用 generate_wan5b_eval.build_pipe / generate_one（同一加载与采样实现）。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from types import SimpleNamespace
    from generate_wan5b_eval import build_pipe, generate_one  # noqa: E402

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_args = SimpleNamespace(frames=FRAMES, height=HEIGHT, width=WIDTH,
                               steps=STEPS, cfg=CFG, flow=FLOW, fps=FPS)
    pipe = build_pipe("base", "cuda")
    for i, row in enumerate(load_rows(args.prompts)):
        out_path = out_dir / f"{i}.mp4"
        # 检查文件是否存在且非空（>1KB），否则重新生成
        if out_path.exists() and out_path.stat().st_size > 1024:
            print(f"[gen-diffsynth] 已存在有效文件 {out_path}，跳过")
            continue
        elif out_path.exists():
            print(f"[gen-diffsynth] 删除损坏文件 {out_path} (size={out_path.stat().st_size})")
            out_path.unlink()
        item = {
            "prompt": row["prompt"], "seed": int(row["seed"]),
            # tmp 也保留 .mp4 扩展名，否则 imageio 无法按扩展名解析 FFMPEG 后端
            "_out_path": str(out_path), "_tmp_path": str(out_path) + ".tmp.mp4",
        }
        print(f"[gen-diffsynth] [{i}] seed={item['seed']} {row['prompt'][:60]!r} ...")
        generate_one(pipe, item, gen_args)
    print("DONE gen-diffsynth ->", out_dir)


def cmd_gen_vu(args):
    """VU 侧：diffusers WanModel + src.generate（经 VU venv python 运行）。"""
    import torch
    from omegaconf import OmegaConf

    vu = Path(args.vu_root)
    sys.path.insert(0, str(vu))
    from src.models.wan import WanModel  # noqa: E402
    from src.generate import generate  # noqa: E402

    model_cfg = OmegaConf.load(vu / "configs/model/wan22_ti2v_5b.yaml")
    model_path = args.model_path or str(model_cfg.pretrained_path)
    # 与 DiffSynth 侧逐字一致的负面提示词
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_wan5b_eval import WAN_NEGATIVE  # noqa: E402

    model = WanModel(
        model_path=model_path, device="cuda", dtype=torch.bfloat16,
        load_text_encoder=True, load_transformer=True, load_vae=True,
        load_scheduler=True, gradient_checkpointing=False,
    )
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio

    fps = getattr(model, "fps", 16)
    for i, row in enumerate(load_rows(args.prompts)):
        out_path = out_dir / f"{i}.mp4"
        # 检查文件是否存在且非空（>1KB），否则重新生成
        if out_path.exists() and out_path.stat().st_size > 1024:
            print(f"[gen-vu] 已存在有效文件 {out_path}，跳过")
            continue
        elif out_path.exists():
            print(f"[gen-vu] 删除损坏文件 {out_path} (size={out_path.stat().st_size})")
            out_path.unlink()
        seed = int(row["seed"])
        print(f"[gen-vu] [{i}] seed={seed} {row['prompt'][:60]!r} ...")
        video = generate(
            model=model, prompt=row["prompt"],
            num_frames=FRAMES, height=HEIGHT, width=WIDTH,
            num_inference_steps=STEPS, guidance_scale=CFG, seed=seed,
            negative_prompt=WAN_NEGATIVE,
        )
        frames = video[0].permute(0, 2, 3, 1).float().cpu().numpy()
        pixels = (((frames + 1.0) / 2.0).clip(0.0, 1.0) * 255.0).astype("uint8")
        # 使用临时文件 + 原子重命名，确保写入完整
        tmp_path = out_path.with_suffix('.tmp.mp4')
        try:
            # 优先使用 ffmpeg 插件（更稳定）
            iio.imwrite(str(tmp_path), pixels, fps=fps, codec="libx264")
            tmp_path.rename(out_path)
        except Exception as e:
            print(f"[gen-vu] 写入失败: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            raise
    print("DONE gen-vu ->", out_dir)


def cmd_compare(args):
    """逐视频：抽 max(1,min(8,帧数)) 帧 -> 中心裁剪到 64x64 灰度 -> 余弦相似度 + 平均绝对差。"""
    import numpy as np
    import imageio.v3 as iio

    dir_a, dir_b = Path(args.dirA), Path(args.dirB)
    n_rows = len(load_rows(args.prompts))
    results = []
    fail = False
    for i in range(n_rows):
        pa, pb = dir_a / f"{i}.mp4", dir_b / f"{i}.mp4"
        if not pa.is_file() or not pb.is_file():
            print(f"[compare] [{i}] 缺文件: {pa} | {pb}")
            fail = True
            continue
        # 不指定插件，让 imageio 自动选择（避免 ffmpeg/pyav 插件名不一致问题）
        fa = iio.imread(str(pa)).astype(np.float32)
        fb = iio.imread(str(pb)).astype(np.float32)
        n = min(len(fa), len(fb))
        idx = sorted(set(np.linspace(0, n - 1, min(8, n)).astype(int)))
        cos_sum, mae_sum = 0.0, 0.0
        for t in idx:
            a, b = fa[t] / 255.0, fb[t] / 255.0
            h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
            a = a[:h, :w][:, ::8, ::8].mean(axis=2)   # 粗降采样到 ~60x90 灰度
            b = b[:h, :w][:, ::8, ::8].mean(axis=2)
            cos_sum += float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            mae_sum += float(np.abs(a - b).mean())
        cos = cos_sum / len(idx)
        mae = mae_sum / len(idx)
        ok = cos >= args.threshold
        fail |= not ok
        results.append({"video": i, "frames": len(fa), "cosine": round(cos, 4),
                        "mae": round(mae, 4), "pass": ok})
        print(f"[compare] [{i}] cosine={cos:.4f} mae={mae:.4f} "
              f"{'PASS' if ok else f'FAIL(<{args.threshold})'}")
    mean_cos = float(np.mean([r["cosine"] for r in results])) if results else 0.0
    print(f"[compare] 汇总: n={len(results)} mean_cosine={mean_cos:.4f} "
          f"overall={'PASS' if not fail else 'FAIL'}")
    with open(Path(args.outjson), "w", encoding="utf-8") as f:
        json.dump({"threshold": args.threshold, "results": results,
                   "mean_cosine": mean_cos, "overall_pass": not fail}, f, indent=2)
    sys.exit(1 if fail else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("gen-diffsynth")
    pg.add_argument("prompts")
    pg.add_argument("outdir")
    pg.set_defaults(func=cmd_gen_diffsynth)

    pv = sub.add_parser("gen-vu")
    pv.add_argument("prompts")
    pv.add_argument("outdir")
    pv.add_argument("--vu-root", default="/home/x_jiage/jiage/video-unlearning")
    pv.add_argument("--model-path", default="")
    pv.set_defaults(func=cmd_gen_vu)

    pc = sub.add_parser("compare")
    pc.add_argument("dirA")
    pc.add_argument("dirB")
    pc.add_argument("prompts")
    pc.add_argument("--threshold", type=float, default=0.85)
    pc.add_argument("--outjson", default="outputs/wan5b_bridge_equiv/compare.json")
    pc.set_defaults(func=cmd_compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()