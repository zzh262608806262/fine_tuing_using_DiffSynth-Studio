"""训练前 sanity checks (require.md 第十节, 共 10 项).

包含 train.py --sanity_check 的全部校验 + 帧采样可视化.

用法:
  python -m classify.scripts.sanity_check \
      --config classify/configs/safety_classifier.yaml \
      --data.train_annotation data/safesora/train.jsonl \
      --data.video_root /path/to/videos \
      --out_dir outputs/sanity
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve()
for _p in [_THIS_DIR.parents[4], _THIS_DIR.parents[3]]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from classify.factory import (
    build_model, build_transform, build_dataset, build_dataloader,
)
from classify.datasets.video_dataset import load_video_frames
from classify.utils import set_seed, get_logger, load_yaml, apply_cli_overrides
from classify.training.train import run_sanity_checks


def visualize_samples(cfg: Dict, label_names, out_dir: str, logger) -> None:
    """采样若干视频, 保存 T 帧为拼图, 检查 8 帧采样是否正确."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安装, 跳过可视化 (pip install matplotlib)")
        return
    model, _ = build_model(cfg)
    transform = build_transform(model, cfg)
    ann = cfg["data"].get("train_annotation")
    if not ann:
        logger.warning("未提供 train_annotation, 跳过可视化")
        return
    ds = build_dataset(cfg, ann, transform, label_names)
    if len(ds) == 0:
        logger.warning("dataset 为空, 跳过可视化")
        return
    out_dir = Path(out_dir) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler = ds.sampler
    T = sampler.num_frames
    n_show = min(4, len(ds))
    for k in range(n_show):
        item = ds.items[k]
        from classify.datasets.video_dataset import _count_frames
        try:
            total = _count_frames(cfg["video"].get("decode_backend", "av"),
                                  ds._resolve_video_path(item["video"]))
        except Exception:
            total = 0
        indices = sampler.sample_indices(total if total else T)
        logger.info(f"[viz] sample {k}: video={item['video']} total_frames={total} "
                    f"sampled_indices={indices} (T={T})")
        frames = load_video_frames(ds._resolve_video_path(item["video"]), sampler,
                                   cfg["video"].get("decode_backend", "av"))
        if frames is None:
            logger.warning(f"[viz] sample {k} 解码失败, 跳过")
            continue
        # frames: [T, H, W, 3] uint8
        fig, axes = plt.subplots(1, T, figsize=(2 * T, 2.5))
        if T == 1:
            axes = [axes]
        for j in range(T):
            axes[j].imshow(frames[j])
            axes[j].set_title(f"f{indices[j]}", fontsize=8)
            axes[j].axis("off")
        labels = item["labels"]
        active = [label_names[i] for i, v in enumerate(labels) if v]
        fig.suptitle(f"{Path(item['video']).name}\nlabels: {active}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{k}.png", dpi=120)
        plt.close(fig)
    logger.info(f"[viz] 可视化保存到 {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Safety Classifier sanity checks")
    p.add_argument("--config", type=str, default="classify/configs/safety_classifier.yaml")
    p.add_argument("--out_dir", type=str, default="outputs/sanity")
    args, unknown = p.parse_known_args()
    args.overrides = list(unknown)

    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, args.overrides)
    cfg.setdefault("train", {})["output_dir"] = args.out_dir

    set_seed(int(cfg.get("seed", 42)))
    logger = get_logger("sanity_check")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label_names = build_model(cfg)
    model = model.to(device)
    transform = build_transform(model, cfg)

    ann = cfg["data"].get("train_annotation")
    if not ann:
        logger.error("必须提供 data.train_annotation")
        sys.exit(1)
    ds = build_dataset(cfg, ann, transform, label_names)
    loader = build_dataloader(ds, cfg, train=False, distributed=False)

    # 1,2: 可视化 + 帧采样检查
    visualize_samples(cfg, label_names, args.out_dir, logger)
    # 3-10: shape / frozen / NaN / backward
    run_sanity_checks(model, loader, device, label_names, logger)


if __name__ == "__main__":
    main()
