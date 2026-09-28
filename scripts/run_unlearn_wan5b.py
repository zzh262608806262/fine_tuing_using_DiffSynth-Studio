#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wan2.2-TI2V-5B nudity 擦除驱动脚本 (GradAscent, 复用 video-unlearning 仓库代码)。

背景
----
擦除方法已冻结为 GradAscent: VU(`/home/x_jiage/jiage/video-unlearning`)
`src/unlearning/training/grad_ascent.py` 的 loss = -l_f, 只消费 latent manifest,
训练不需要视频/VAE/负向提示。训练入口直接复用 VU `src/unlearning/training/train.py`
里的普通函数 `_train_one(cfg, erase_concept, concept_manifest, output_dir,
retain_manifest=None)`(绕过顶层 @hydra.main)。

硬性规则
--------
- VU 仓库只读: 通过 `sys.path.insert(0, "<VU>")` 引入其 `src` 包复用代码,
  本脚本任何情况下都不写 VU(合并 manifest、缓存、adapter、探针视频全部落 FT)。
- 所有产物只落 FT 仓库: `data/wan5b/`、`models/unlearn/wan5b_nudity_grad_ascent/`、
  `outputs/wan5b_unlearn_probe/`。

运行环境(重要)
--------------
- 训练侧依赖以 VU `pyproject.toml` / `uv.lock` 锁定的版本为准(hydra、omegaconf、
  diffusers 0.38 等), 建议跑在 VU 的 uv venv 下; 激活方式不写死——用环境变量
  `VU_ROOT` / `VU_VENV_ACTIVATE` 或 `--vu-root` 指定(推测: VU 仓库目录下
  `source .venv/bin/activate`, VU 自己的 exp504 slurm 即用 .venv; 或用
  `conda activate <env>`)。FT 侧 conda env 的依赖版本可能与 VU 冲突, 不要混用。
- 本机(Berzelius 登录机)GPU 禁跑, 实际执行请通过 `slurm/wan5b_unlearn.sbatch`
  提交; 本脚本仅做编排/驱动。

CLI 用法示例
------------
# 只打印计划, 不执行
python3 scripts/run_unlearn_wan5b.py --cache --train --probe --dry-run

# [阶段1] 缓存 forget+retain(18+9) 条视频的 latent
python3 scripts/run_unlearn_wan5b.py --cache --height 480 --width 720

# [阶段2] GradAscent 训练(消费 latent_manifest.jsonl, 输出 transformer LoRA adapter)
python3 scripts/run_unlearn_wan5b.py --train --unlearn-steps 1200 --save-every 200

# [阶段3] base vs erased 探针生成(同一批 nudity prompt / 同 seed, N 帧视频)
python3 scripts/run_unlearn_wan5b.py --probe
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# VU 仓库只读: 禁止 Python 向 VU 的 src 目录写 __pycache__/*.pyc(import 时默认会写)。
sys.dont_write_bytecode = True

# VU 仓库根目录(只读引用)。默认取环境变量 VU_ROOT, 兜底用本机已知路径。
VU_ROOT_DEFAULT = Path(os.environ.get("VU_ROOT", "/home/x_jiage/jiage/video-unlearning"))
# FT 仓库根目录(所有产物落这里) = 本脚本所在目录的上一级。
FT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------------
def _resolve(path_str: str, base: Path) -> Path:
    """把相对路径拼到 base 下; 绝对路径原样返回。"""
    p = Path(path_str)
    return p if p.is_absolute() else base / p


def _require_vu(vu: Path) -> None:
    if not (vu / "src").is_dir():
        raise FileNotFoundError(
            f"VU 仓库路径无效(缺 src 包): {vu} (用 --vu-root / 环境变量 VU_ROOT 指定)"
        )


def _ensure_under_ft(path: Path, ft: Path) -> None:
    """产物路径必须落在 FT 内, 防止误写 VU(只读仓库)。"""
    ft_res = ft.resolve()
    path_res = path.resolve()
    if path_res != ft_res and ft_res not in path_res.parents:
        raise ValueError(f"产物路径必须在 FT({ft_res}) 内, 当前为: {path_res} (VU 只读)")


def _count_manifest(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _read_manifest(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON: {exc}") from exc
        if not {"video", "prompt"} <= set(record):
            raise ValueError(f"{path} 第 {line_number} 行缺 video/prompt 字段")
        rows.append(record)
    if not rows:
        raise ValueError(f"manifest 为空: {path}")
    return rows


def _print_env(args, vu: Path, ft: Path) -> None:
    """打印版本信息与关键路径(需求: 全程打印)。"""
    import torch
    import diffusers
    import omegaconf

    print(f"[env] python    = {sys.version.split()[0]}")
    print(f"[env] torch     = {torch.__version__} (cuda_available={torch.cuda.is_available()})")
    print(f"[env] diffusers = {diffusers.__version__}")
    print(f"[env] omegaconf = {omegaconf.__version__}")
    print(f"[env] device    = {args.device}")
    print(f"[env] VU(只读)  = {vu}")
    print(f"[env] FT(产物)  = {ft}", flush=True)


def _merge_manifests(forget_path: Path, retain_path: Path, out_path: Path, force: bool) -> Path:
    """把 forget+retain 两个数据 manifest 合并成一个(video+prompt 字段齐全)。"""
    if out_path.exists() and not force:
        print(f"[cache] 合并 manifest 已存在: {out_path} (--force-cache 重新生成)")
        return out_path
    merged = []
    for src, split in ((forget_path, "forget"), (retain_path, "retain")):
        for record in _read_manifest(src):
            # 保留原字段, 追加 unlearn_split 标注 provenance; 不改写 VU 源文件。
            merged.append({**record, "unlearn_split": split})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged),
        encoding="utf-8",
    )
    n_forget = sum(1 for row in merged if row["unlearn_split"] == "forget")
    n_retain = len(merged) - n_forget
    print(f"[cache] 合并 manifest -> {out_path} ({len(merged)} 条: forget={n_forget} retain={n_retain})")
    return out_path


def _save_video(video, out_path: Path, fps: int) -> Path:
    """把 (B,F,C,H,W) 且取值约 [-1,1] 的解码结果写成 mp4(最小可用实现)。"""
    import imageio.v3 as iio

    frames = video[0].permute(0, 2, 3, 1).float().cpu().numpy()
    frames = ((frames + 1.0) / 2.0).clip(0.0, 1.0)
    pixels = (frames * 255.0).astype("uint8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 原子写入：先写临时文件再重命名（避免损坏的 0 字节文件）
    tmp = out_path.with_suffix(".tmp.mp4")
    try:
        # pyav plugin 在某些环境下有问题，直接用默认 plugin
        iio.imwrite(str(tmp), pixels, fps=fps)
        tmp.rename(out_path)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"写入视频失败: {out_path}") from e
    return out_path


# ----------------------------------------------------------------------------
# --cache: 调用 VU cache_video_latents(wan 原生支持, AutoencoderKLWan + latents_mean/std)
# ----------------------------------------------------------------------------
def run_cache(args, vu: Path, ft: Path) -> Path:
    _require_vu(vu)
    from omegaconf import OmegaConf
    import torch

    # wan 模型路径: 优先 --model-path, 否则取 VU 训练用 model yaml 里钉死的集群 snapshot
    # (wan22_ti2v_5b_train.yaml 的 pretrained_path 指向
    #  /scratch/s6398820/hf_cache/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/...)。
    model_cfg = OmegaConf.load(vu / "configs/model/wan22_ti2v_5b_train.yaml")
    model_path = args.model_path or str(model_cfg.pretrained_path)

    forget = _resolve(args.forget_manifest, vu)
    retain = _resolve(args.retain_manifest, vu)
    cache_dir = _resolve(args.cache_dir, ft)
    merged = cache_dir.parent / "nudity_forget_retain_wan.jsonl"
    for p in (forget, retain):
        if not p.is_file():
            raise FileNotFoundError(f"数据 manifest 不存在: {p}")
    _ensure_under_ft(merged, ft)
    _ensure_under_ft(cache_dir, ft)
    if not args.force_cache and (cache_dir / "latent_manifest.jsonl").is_file():
        print(f"[cache] 已有缓存 {cache_dir / 'latent_manifest.jsonl'}, 跳过 (--force-cache 重跑)")
        return cache_dir / "latent_manifest.jsonl"

    manifest = _merge_manifests(forget, retain, merged, args.force_cache)
    _print_env(args, vu, ft)
    print(f"[cache] model_path   = {model_path}")
    print(f"[cache] input        = {manifest} ({_count_manifest(manifest)} 条)")
    print(f"[cache] output dir   = {cache_dir}")
    print(f"[cache] resolution   = {args.cache_height}x{args.cache_width}, family=wan, dtype={args.dtype}", flush=True)

    sys.path.insert(0, str(vu))
    from src.unlearning.training.cache_video_latents import cache_latents

    latent_manifest = cache_latents(
        input_manifest=manifest,
        output_dir=cache_dir,
        model_path=model_path,
        device=args.device,
        height=args.cache_height,
        width=args.cache_width,
        model_family="wan",
        dtype=getattr(torch, args.dtype),
    )
    print(f"[cache] 完成, latent manifest = {latent_manifest} ({_count_manifest(latent_manifest)} 条)")
    return latent_manifest


# ----------------------------------------------------------------------------
# --train: 复用 VU _train_one 跑 GradAscent(仅消费 latent manifest)
# ----------------------------------------------------------------------------
def run_train(args, vu: Path, ft: Path) -> None:
    _require_vu(vu)
    from omegaconf import OmegaConf

    latent_manifest = _resolve(args.latent_manifest, ft)
    out_dir = _resolve(args.train_dir, ft)
    _ensure_under_ft(out_dir, ft)
    if not latent_manifest.is_file():
        raise FileNotFoundError(f"latent manifest 不存在: {latent_manifest} (请先跑 --cache)")
    n = _count_manifest(latent_manifest)
    # 预检: 每一条 latent 文件必须存在(与 VU DiffusionTrainingMethod._load_manifest 同口径)
    for line_number, line in enumerate(latent_manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not {"latent", "prompt"} <= set(record):
            raise ValueError(f"{latent_manifest} 第 {line_number} 行缺 latent/prompt 字段")
        if not (latent_manifest.parent / record["latent"]).is_file():
            raise FileNotFoundError(f"latent 文件不存在: {latent_manifest.parent / record['latent']}")
    if (out_dir / "adapter_final.pt").exists() and not args.force_train:
        print(f"[train] {out_dir / 'adapter_final.pt'} 已存在, 跳过 (--force-train 重跑)")
        return

    # ---- 拼装训练 cfg: 以 VU 两个 yaml 为基础覆盖 ----
    # model:  configs/model/wan22_ti2v_5b_train.yaml
    #         (load_vae:false, load_scheduler:false, flow_shift:5.0 —— 训练不需要 VAE)
    # method: 支持多种方法 (GradAscent/AnchorDistill/ESD/NPO)
    #         默认 grad_ascent_wan.yaml (handler GradAscent, target transformer, lora_rank 8, alpha 16,
    #          layer 0-29, guidance 1.0, sigma logit_normal, timestep_shift null -> 跟随 flow_shift)
    model_cfg = OmegaConf.load(vu / "configs/model/wan22_ti2v_5b_train.yaml")
    # 方法名映射：GradAscent -> grad_ascent, AnchorDistill -> anchor_distill
    method_name_map = {
        "GradAscent": "grad_ascent",
        "GradDiff": "grad_diff",
        "AnchorDistill": "anchor_distill",
        "ESD": "esd",
        "NPO": "npo",
        # Masked variants
        "NPOMasked": "npo_masked",
        "GradDiffMasked": "grad_diff_masked",
        "GradAscentMasked": "grad_ascent_masked",
    }
    method_file_name = method_name_map.get(args.method, args.method.lower())
    method_config_file = f"{method_file_name}_wan.yaml"
    method_config_path = vu / "configs/methods/training" / method_config_file
    if not method_config_path.is_file():
        raise FileNotFoundError(f"方法配置不存在: {method_config_path} (--method {args.method})")
    method_cfg = OmegaConf.load(method_config_path)
    if args.model_path:
        model_cfg.pretrained_path = args.model_path
    model_cfg.device = args.device

    # 方法特定参数设置
    if args.method == "AnchorDistill":
        if args.anchor is None:
            raise ValueError("AnchorDistill 需要 --anchor 参数 (替代词如 'a person' 或空字符串 '')")
        method_cfg.method_args.anchor = args.anchor
    elif args.method == "ESD":
        if args.negative_guidance is not None:
            method_cfg.method_args.negative_guidance = args.negative_guidance
    elif args.method == "GradDiff":
        if args.retain_weight is not None:
            method_cfg.method_args.retain_weight = args.retain_weight

    # GradDiff 需要 retain_manifest，其他方法不需要
    # GradAscent 不消费 retain_manifest(构造器没有该参数, 传了会 TypeError), 且本任务要求
    # "retain_manifest 合并进 training manifest" —— 已在 --cache 阶段把 forget+retain 合并缓存
    # 进同一个 latent_manifest.jsonl, 因此这里 retain_manifest 保持 None(写入 protocol 也更诚实)。
    # GradDiff 例外：它需要分别采样 forget 和 retain，所以必须传递 retain_manifest
    train_retain_manifest = None
    if args.method == "GradDiff":
        retain_path = _resolve(args.retain_manifest, vu)
        if not retain_path.is_file():
            raise FileNotFoundError(f"GradDiff 需要 retain manifest: {retain_path}")
        train_retain_manifest = str(retain_path)
        print(f"[train] GradDiff retain_manifest = {train_retain_manifest}")

    batch_size = args.batch_size or int(method_cfg.method_args.batch_size)
    steps_per_epoch = math.ceil(n / batch_size)
    # 默认步数依据(注释说明):
    #   - VU exp504(grad_ascent, CogVideoX-5B, lora_rank=64, batch=2)用 train.max_steps=600、
    #     每 200 步存点(见 records/exp504_train_tasks.json / runs/exp/exp504 script)。
    #   - 本任务的 grad_ascent_wan.yaml 是 lora_rank=8 / batch=1, 单步比 exp504 轻得多,
    #     Wan5B 单臂经验量级约 1000-2000 步, 故默认 --unlearn-steps 1200、--save-every 200。
    epochs = math.ceil(args.unlearn_steps / steps_per_epoch)
    train_cfg = OmegaConf.create(
        {
            "erase_concept": args.erase_concept,
            "output_dir": str(out_dir),
            "epochs": epochs,                 # 保证训练精确停在 max_steps, 不触发 _train_one 的步数不足报错
            "max_steps": args.unlearn_steps,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "betas": [0.9, 0.98],
            "weight_decay": 1.0e-8,
            "log_every": args.log_every,
            "save_every": args.save_every,
        }
    )
    if args.save_steps:
        steps = [int(x) for x in args.save_steps.split(",") if x.strip()]
        if not steps or any(s <= 0 or s > args.unlearn_steps for s in steps):
            raise ValueError(f"--save-steps 取值必须在 (0, {args.unlearn_steps}] 内: {steps}")
        train_cfg["save_steps"] = steps
    cfg = OmegaConf.create({"model": model_cfg, "methods": method_cfg, "train": train_cfg})

    _print_env(args, vu, ft)
    print(f"[train] latent manifest = {latent_manifest} ({n} 条, 含合并的 retain)")
    print(f"[train] output dir      = {out_dir}")
    print(f"[train] method          = {method_cfg.handler} (registry: src.unlearning.registry)")
    if args.method == "AnchorDistill":
        print(f"[train] anchor          = {args.anchor!r}")
    elif args.method == "ESD":
        print(f"[train] negative_guidance = {method_cfg.method_args.negative_guidance}")
    elif args.method == "GradDiff":
        print(f"[train] retain_weight   = {method_cfg.method_args.retain_weight}")
    print(f"[train] lora            = rank {method_cfg.method_args.lora_rank}, "
          f"alpha {method_cfg.method_args.lora_alpha}, "
          f"layers {method_cfg.method_args.layer_start}-{method_cfg.method_args.layer_end}, "
          f"target {method_cfg.method_args.target}")
    print(f"[train] steps           = max_steps {args.unlearn_steps}, epochs {epochs} "
          f"(=ceil({args.unlearn_steps}/{steps_per_epoch}), batch {batch_size}), "
          f"save_every {args.save_every}", flush=True)

    sys.path.insert(0, str(vu))
    from src.unlearning.training.train import _train_one

    # _train_one 会在 out_dir 写出 training_protocol.json / training_trace.jsonl /
    # adapter_step{step}.pt(中间步) / adapter_final.pt, 且只在片内做原样覆盖。
    _train_one(
        cfg,
        erase_concept=args.erase_concept,
        concept_manifest=str(latent_manifest),
        output_dir=out_dir,
        retain_manifest=train_retain_manifest,
    )
    print(f"[train] 完成: adapter_final.pt = {out_dir / 'adapter_final.pt'}")


# ----------------------------------------------------------------------------
# --probe: base vs erased 生成同一批 nudity 探针(可选, 供 NudeNet/分类器判别)
# ----------------------------------------------------------------------------
def _render_probe_pass(model, rows: list[dict], out_dir: Path, args, tag: str) -> None:
    from src.generate import generate

    fps = getattr(model, "fps", 16)
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        seed = int(row["seed"]) if row.get("seed") is not None else args.seed + index
        out_path = out_dir / f"{index:04d}.mp4"
        if out_path.exists() and not args.probe_overwrite:
            print(f"[probe][{tag}] 跳过已存在 {out_path}")
            continue
        prompt = str(row["prompt"])
        print(f"[probe][{tag}] [{index:04d}] seed={seed} prompt={prompt[:80]!r} ...")
        video = generate(
            model=model,
            prompt=prompt,
            num_frames=args.probe_frames,
            height=args.probe_height,
            width=args.probe_width,
            num_inference_steps=args.probe_steps,
            guidance_scale=args.probe_guidance,
            seed=seed,
        )
        _save_video(video, out_path, fps)
    del model


def run_probe(args, vu: Path, ft: Path) -> None:
    _require_vu(vu)
    from omegaconf import OmegaConf
    import torch

    # 探针模型走 wan22_ti2v_5b.yaml(含官方 generation_defaults: 121 帧/1280x704/CFG 5.0);
    # 但按 VU memory Exp505 的 5B Wan 基线口径, 默认用 17 帧 480x720(可 --probe-* 覆盖)。
    model_cfg = OmegaConf.load(vu / "configs/model/wan22_ti2v_5b.yaml")
    model_path = args.model_path or str(model_cfg.pretrained_path)

    adapter = _resolve(args.adapter, ft)
    probe_manifest = _resolve(args.probe_manifest, vu)
    probe_dir = _resolve(args.probe_dir, ft)
    if not adapter.is_file():
        raise FileNotFoundError(f"adapter 不存在: {adapter} (请先跑 --train)")
    if not probe_manifest.is_file():
        raise FileNotFoundError(f"探针 prompt manifest 不存在: {probe_manifest}")
    base_dir, erased_dir = probe_dir / "base", probe_dir / "erased"
    _ensure_under_ft(base_dir, ft)
    _ensure_under_ft(erased_dir, ft)

    rows = _read_manifest(probe_manifest)
    if args.probe_limit > 0:
        rows = rows[: args.probe_limit]

    _print_env(args, vu, ft)
    print(f"[probe] model_path  = {model_path}")
    print(f"[probe] adapter     = {adapter}")
    print(f"[probe] prompts     = {probe_manifest} (取前 {len(rows)} 条, base/erased 同 prompt 同 seed)")
    print(f"[probe] 输出         = {base_dir} / {erased_dir}")
    print(f"[probe] 生成参数     = {args.probe_frames} 帧 {args.probe_height}x{args.probe_width}, "
          f"{args.probe_steps} 步, guidance {args.probe_guidance}", flush=True)

    sys.path.insert(0, str(vu))
    from src.models.wan import WanModel
    from src.unlearning.inference.video_transformer_adapter import VideoTransformerAdapter

    def build(adapter_path: str | None):
        model = WanModel(
            model_path=model_path,
            device=args.device,
            dtype=torch.bfloat16,
            load_text_encoder=True,
            load_transformer=True,
            load_vae=True,          # 探针需要 VAE 解码成视频
            load_scheduler=True,    # 探针需要 scheduler 走 denoise 循环
            gradient_checkpointing=False,
        )
        if adapter_path is not None:
            # VideoTransformerAdapter.prepare() 内部走 load_video_transformer_lora,
            # 把训练产出的 transformer LoRA 挂回模型上(不写 VU/FT 任何文件)。
            VideoTransformerAdapter(model, args.erase_concept, adapter_path).prepare()
        return model

    # 两遍扫描, 一次只持有一个模型(否则 2x 5B 超出单卡显存); 同 seed -> 初始噪声一致,
    # base/erased 差异只来自 adapter。
    model = build(None)
    _render_probe_pass(model, rows, base_dir, args, tag="base")
    torch.cuda.empty_cache()
    model = build(adapter)
    _render_probe_pass(model, rows, erased_dir, args, tag="erased")
    torch.cuda.empty_cache()
    print(f"[probe] 完成: {base_dir} / {erased_dir}")


# ----------------------------------------------------------------------------
# --dry-run: 只打印计划, 不执行、不写任何文件
# ----------------------------------------------------------------------------
def _print_dry_run(args, vu: Path, ft: Path) -> None:
    print("\n===== dry-run 计划(不执行) =====")

    if args.cache:
        print("\n[--cache]")
        try:
            from omegaconf import OmegaConf
            model_cfg = OmegaConf.load(vu / "configs/model/wan22_ti2v_5b_train.yaml")
            model_path = args.model_path or str(model_cfg.pretrained_path)
        except Exception as exc:  # omegaconf 不可用(未进 VU venv)时降级
            model_path = args.model_path or f"(读取 yaml 失败: {exc}; 需要 omegaconf)"
        forget = _resolve(args.forget_manifest, vu)
        retain = _resolve(args.retain_manifest, vu)
        cache_dir = _resolve(args.cache_dir, ft)
        merged = cache_dir.parent / "nudity_forget_retain_wan.jsonl"
        print(f"  VU wan 模型路径 : {model_path}")
        print(f"  forget manifest : {forget} ({_count_manifest(forget) if forget.is_file() else '缺文件'} 条)")
        print(f"  retain manifest : {retain} ({_count_manifest(retain) if retain.is_file() else '缺文件'} 条)")
        print(f"  合并 manifest   : {merged}")
        print(f"  缓存输出目录    : {cache_dir}  (latents/ + latent_manifest.jsonl)")
        print(f"  分辨率          : {args.cache_height}x{args.cache_width}, family=wan, dtype={args.dtype}")

    if args.train:
        print("\n[--train]")
        latent_manifest = _resolve(args.latent_manifest, ft)
        out_dir = _resolve(args.train_dir, ft)
        batch_size = args.batch_size or 1
        if latent_manifest.is_file():
            n = _count_manifest(latent_manifest)
            epochs = math.ceil(args.unlearn_steps / max(1, math.ceil(n / batch_size)))
        else:
            n, epochs = 0, None
        print(f"  latent manifest : {latent_manifest} ({n} 条, 含合并的 retain; 缺文件则先跑 --cache)")
        print(f"  adapter 输出    : {out_dir}  (training_protocol.json / training_trace.jsonl / adapter_step*.pt / adapter_final.pt)")
        method_params = f"erase_concept={args.erase_concept}, max_steps={args.unlearn_steps}, batch={batch_size}, epochs={epochs if epochs is not None else 'N/A(需先 --cache)'}, save_every={args.save_every}, seed={args.seed}"
        if args.method == "AnchorDistill":
            method_params += f", anchor={args.anchor!r}"
        elif args.method == "ESD":
            ng = args.negative_guidance if args.negative_guidance is not None else 1.0
            method_params += f", negative_guidance={ng}"
        print(f"  {args.method:15s}: {method_params}")

    if args.probe:
        print("\n[--probe]")
        adapter = _resolve(args.adapter, ft)
        probe_manifest = _resolve(args.probe_manifest, vu)
        probe_dir = _resolve(args.probe_dir, ft)
        n = _count_manifest(probe_manifest) if probe_manifest.is_file() else 0
        if args.probe_limit > 0:
            n = min(n, args.probe_limit)
        print(f"  adapter         : {adapter}")
        print(f"  探针 prompts    : {probe_manifest} ({n} 条)")
        print(f"  输出            : {probe_dir / 'base'} / {probe_dir / 'erased'}")
        print(f"  生成参数        : {args.probe_frames} 帧 {args.probe_height}x{args.probe_width}, "
              f"{args.probe_steps} 步, guidance {args.probe_guidance}")

    print("\n===== dry-run 结束: 未执行任何操作 =====")


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Wan2.2-TI2V-5B nudity GradAscent 擦除驱动(复用 VU 仓库, VU 只读)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--vu-root", type=Path, default=VU_ROOT_DEFAULT,
                    help="VU(video-unlearning)仓库根目录, 只读引用(可被环境变量 VU_ROOT 覆盖)")
    ap.add_argument("--ft-root", type=Path, default=FT_ROOT_DEFAULT,
                    help="FT 仓库根目录, 所有产物落这里(可被环境变量 FT_ROOT 覆盖)")
    ap.add_argument("--seed", type=int, default=42, help="训练/探针种子")
    ap.add_argument("--device", default="cuda", help="torch device(集群上为 cuda)")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划, 不执行")
    ap.add_argument("--model-path", default=None,
                    help="Wan2.2-TI2V-5B Diffusers 快照路径; 缺省读取 VU yaml 里的集群路径")

    g = ap.add_argument_group("--cache")
    g.add_argument("--cache", dest="cache", action="store_true", help="跑 cache_latents(视频 -> latent)")
    g.add_argument("--forget-manifest", default="data/splits/nudity_forget_composed_wan.jsonl",
                   help="forget 数据 manifest(相对 VU 或绝对路径)")
    g.add_argument("--retain-manifest", default="data/splits/nudity_retain_composed_wan.jsonl",
                   help="retain 数据 manifest(相对 VU 或绝对路径)")
    g.add_argument("--cache-dir", default="data/wan5b/unlearn_latents",
                   help="latent 缓存输出目录(相对 FT)")
    g.add_argument("--cache-height", type=int, default=480, help="缓存分辨率高(16 的倍数, 参照 Exp505 5B 基线)")
    g.add_argument("--cache-width", type=int, default=736, help="缓存分辨率宽(16 的倍数, 参照 Exp505 5B 基线)")
    g.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16", help="缓存 dtype")
    g.add_argument("--force-cache", action="store_true", help="已存在 latent_manifest.jsonl 时也重新缓存")

    g = ap.add_argument_group("--train")
    g.add_argument("--train", dest="train", action="store_true", help="跑训练 (支持多种方法)")
    g.add_argument("--method", default="GradAscent",
                   choices=["GradAscent", "GradDiff", "AnchorDistill", "ESD", "NPO",
                            "NPOMasked", "GradDiffMasked", "GradAscentMasked"],
                   help="遗忘方法: GradAscent(梯度上升), GradDiff(梯度差分,需retain), AnchorDistill(锚点蒸馏,tiger唯一达标), ESD(负引导), NPO(负偏好优化)")
    g.add_argument("--anchor", default=None,
                   help="AnchorDistill 专用: 替代词 (如 'a person') 或空字符串 '' (空 prompt)")
    g.add_argument("--negative-guidance", type=float, default=None,
                   help="ESD 专用: 负引导强度 η (默认 1.0)")
    g.add_argument("--retain-weight", type=float, default=None,
                   help="GradDiff 专用: retain loss 权重 λ (默认 1.0)")
    g.add_argument("--latent-manifest", default="data/wan5b/unlearn_latents/latent_manifest.jsonl",
                   help="训练用 latent manifest(相对 FT, 由 --cache 产出)")
    g.add_argument("--train-dir", default="models/unlearn/wan5b_nudity_grad_ascent",
                   help="adapter 输出目录(相对 FT)")
    g.add_argument("--erase-concept", default="nudity", help="擦除概念")
    g.add_argument("--unlearn-steps", type=int, default=1200,
                   help="训练总步数(max_steps)。默认 1200: exp504 GradAscent 用 600 步/200 存点(CogVideoX-5B, "
                        "rank64/batch2), 本 yaml 是 rank8/batch1 更轻, Wan5B 单臂经验量级 1000-2000 步")
    g.add_argument("--save-every", type=int, default=200, help="每隔多少步存一个 adapter_step{step}.pt")
    g.add_argument("--save-steps", default=None,
                   help="显式存点步数(逗号分隔, 如 200,400,600; 覆盖 --save-every, 参照 exp504)")
    g.add_argument("--batch-size", type=int, default=None, help="训练 batch size(缺省读 method yaml 的值)")
    g.add_argument("--learning-rate", type=float, default=1.0e-5, help="AdamW 学习率(exp504 GradAscent 同款)")
    g.add_argument("--log-every", type=int, default=50, help="每多少步打印一次 loss")
    g.add_argument("--force-train", action="store_true", help="adapter_final.pt 已存在时也重跑")

    g = ap.add_argument_group("--probe")
    g.add_argument("--probe", dest="probe", action="store_true", help="base vs erased 探针生成")
    g.add_argument("--adapter", default="models/unlearn/wan5b_nudity_grad_ascent/adapter_final.pt",
                   help="erased 分支加载的 adapter(相对 FT)")
    g.add_argument("--probe-manifest", default="data/splits/nudity_forget_composed_wan.jsonl",
                   help="探针 prompt 来源(相对 VU 或绝对路径; 取 video/prompt/seed 字段)")
    g.add_argument("--probe-dir", default="outputs/wan5b_unlearn_probe",
                   help="探针输出根目录(相对 FT), 下辖 base/ 与 erased/")
    g.add_argument("--probe-frames", type=int, default=17, help="探针帧数(参照 Exp505 5B 基线 17 帧)")
    g.add_argument("--probe-height", type=int, default=480, help="探针高")
    g.add_argument("--probe-width", type=int, default=736, help="探针宽")
    g.add_argument("--probe-steps", type=int, default=50, help="推理步数")
    g.add_argument("--probe-guidance", type=float, default=5.0,
                   help="CFG 强度(Wan 官方默认 5.0)")
    g.add_argument("--probe-limit", type=int, default=0, help="只生成前 N 条探针(0=全部)")
    g.add_argument("--probe-overwrite", action="store_true", help="覆盖已存在的探针视频")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if not (args.cache or args.train or args.probe):
        raise SystemExit("请至少指定 --cache / --train / --probe 之一")

    vu = Path(args.vu_root).expanduser().resolve()
    ft = Path(args.ft_root).expanduser().resolve()
    _require_vu(vu)  # dry-run 也提前校验, 尽早暴露路径问题

    # 需求: sys.path 引入 VU 根目录; 且 import VU 前不做任何对 VU 的写入。
    # 本脚本的所有写操作都指向 FT(见 _ensure_under_ft 守卫)。
    sys.path.insert(0, str(vu))

    print(f"[driver] VU(只读) = {vu}")
    print(f"[driver] FT(产物) = {ft}")

    if args.dry_run:
        _print_dry_run(args, vu, ft)
        return

    if args.cache:
        run_cache(args, vu, ft)
    if args.train:
        run_train(args, vu, ft)
    if args.probe:
        run_probe(args, vu, ft)


if __name__ == "__main__":
    main()