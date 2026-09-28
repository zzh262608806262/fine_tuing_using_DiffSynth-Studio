#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wan2.2-TI2V-5B 多方法擦除驱动脚本 (支持 GradAscent/GradDiff/ESD/NPO/AnchorDistill)。

设计目标
--------
复用 Exp 012 的基础设施（桥转换、基线视频缓存），扩展支持 VU 项目的多种擦除方法。
所有方法使用统一的配置（rank=8, alpha=16, steps=600），便于横向对比。

支持的方法
----------
1. grad_ascent: loss = -l_f (baseline, Exp 012)
2. grad_diff: loss = l_f - λ·l_r (需要 retain set)
3. esd: 负引导擦除, negative_guidance=1.0
4. npo: 负偏好优化, beta=0.1
5. anchor_distill: 蒸馏到替代词, anchor="a person"

硬性规则
--------
- VU 仓库只读: 通过 sys.path.insert() 引入，不写入任何文件
- 所有产物落 FT 仓库: models/unlearn/exp015_wan5b_nudity_<method>/
- 复用 Exp 012 的缓存: data/wan5b/unlearn_baseline/ 和 latent 缓存

CLI 用法示例
------------
# 只缓存（所有方法共用，只需执行一次）
python3 scripts/run_unlearn_wan5b_multi_method.py --cache --method grad_ascent

# 训练不同方法
python3 scripts/run_unlearn_wan5b_multi_method.py --train --method grad_diff --unlearn-steps 600
python3 scripts/run_unlearn_wan5b_multi_method.py --train --method esd --unlearn-steps 600
python3 scripts/run_unlearn_wan5b_multi_method.py --train --method npo --unlearn-steps 600
python3 scripts/run_unlearn_wan5b_multi_method.py --train --method anchor_distill --unlearn-steps 600 --anchor "a person"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# VU 仓库只读: 禁止写 __pycache__
sys.dont_write_bytecode = True

# 默认路径
VU_ROOT_DEFAULT = Path(os.environ.get("VU_ROOT", "/home/x_jiage/jiage/video-unlearning"))
FT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

# 方法配置映射
METHOD_CONFIGS = {
    "grad_ascent": {
        "handler": "GradAscent",
        "config_file": "grad_ascent_wan.yaml",
        "requires_retain": False,
        "method_args": {},
    },
    "grad_diff": {
        "handler": "GradDiff",
        "config_file": "grad_diff_wan.yaml",
        "requires_retain": True,
        "method_args": {"retain_weight": 1.0},
    },
    "esd": {
        "handler": "ESD",
        "config_file": "esd_wan.yaml",
        "requires_retain": False,
        "method_args": {"negative_guidance": 1.0},
    },
    "npo": {
        "handler": "NPO",
        "config_file": "npo_wan.yaml",
        "requires_retain": False,
        "method_args": {"beta": 0.1},
    },
    "anchor_distill": {
        "handler": "AnchorDistill",
        "config_file": "anchor_distill_wan.yaml",
        "requires_retain": False,
        "method_args": {"anchor": "a person", "retain_manifest": None, "retain_weight": 0.0},
    },
}


def _resolve(path_str: str, base: Path) -> Path:
    """相对路径 → base 拼接; 绝对路径原样返回。"""
    p = Path(path_str)
    return p if p.is_absolute() else base / p


def _require_vu(vu: Path) -> None:
    if not (vu / "src").is_dir():
        raise FileNotFoundError(f"VU 仓库路径无效: {vu} (缺 src/)")


def _ensure_under_ft(path: Path, ft: Path) -> None:
    """确保产物路径在 FT 仓库内。"""
    ft_res = ft.resolve()
    path_res = path.resolve()
    if path_res != ft_res and ft_res not in path_res.parents:
        raise ValueError(f"产物路径必须在 FT({ft_res}) 内: {path_res}")


def _count_manifest(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def cache_latents(args: argparse.Namespace, ft: Path, vu: Path) -> None:
    """缓存 latent（所有方法共用，复用 Exp 012 的逻辑）。"""
    print(f"[CACHE] 复用 Exp 012 的 latent 缓存逻辑")
    print(f"  缓存目录: {args.latent_cache_dir}")
    print(f"  Forget manifest: {args.forget_manifest}")
    print(f"  Retain manifest: {args.retain_manifest}")

    # 检查缓存是否已存在
    cache_dir = _resolve(args.latent_cache_dir, ft)
    forget_count = _count_manifest(_resolve(args.forget_manifest, vu))
    retain_count = _count_manifest(_resolve(args.retain_manifest, vu))
    expected_count = forget_count + retain_count

    if cache_dir.exists():
        existing = list(cache_dir.glob("*.pt"))
        if len(existing) >= expected_count:
            print(f"  ✓ 缓存已存在 ({len(existing)} 个 latent 文件)")
            return

    print(f"  缓存不完整或不存在，需要重新生成")
    print(f"  预期 latent 数量: {expected_count} (forget={forget_count}, retain={retain_count})")

    if args.dry_run:
        print("  [DRY-RUN] 跳过实际缓存")
        return

    # 实际缓存逻辑（复用 run_unlearn_wan5b.py 的代码）
    # 这里简化为提示，实际执行时调用 VU 的缓存函数
    print("  [TODO] 调用 VU 缓存函数（或复用 Exp 012 缓存）")


def train_unlearn(args: argparse.Namespace, ft: Path, vu: Path) -> None:
    """擦除训练（方法参数化）。"""
    method = args.method
    if method not in METHOD_CONFIGS:
        raise ValueError(f"不支持的方法: {method}. 可用: {list(METHOD_CONFIGS.keys())}")

    method_cfg = METHOD_CONFIGS[method]
    print(f"\n[TRAIN] 擦除方法: {method} ({method_cfg['handler']})")
    print(f"  配置文件: {method_cfg['config_file']}")
    print(f"  步数: {args.unlearn_steps}")
    print(f"  保存间隔: {args.save_every}")

    # 输出目录
    output_dir = _resolve(args.output_dir, ft)
    _ensure_under_ft(output_dir, ft)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  输出目录: {output_dir}")

    # 检查 retain manifest 需求
    if method_cfg["requires_retain"]:
        retain_path = _resolve(args.retain_manifest, vu)
        if not retain_path.exists():
            raise FileNotFoundError(f"{method} 需要 retain manifest: {retain_path}")
        print(f"  Retain manifest: {retain_path} ({_count_manifest(retain_path)} 条)")

    # 构建训练配置
    train_config = {
        "method": method_cfg["handler"],
        "model_path": args.model_path,
        "forget_manifest": str(_resolve(args.forget_manifest, vu)),
        "output_dir": str(output_dir),
        "unlearn_steps": args.unlearn_steps,
        "save_every": args.save_every,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        **method_cfg["method_args"],
    }

    # 添加 retain manifest（如果需要）
    if method_cfg["requires_retain"]:
        train_config["retain_manifest"] = str(_resolve(args.retain_manifest, vu))

    # AnchorDistill 特殊处理
    if method == "anchor_distill" and args.anchor:
        train_config["anchor"] = args.anchor

    print(f"  训练配置:")
    for k, v in train_config.items():
        print(f"    {k}: {v}")

    if args.dry_run:
        print("  [DRY-RUN] 跳过实际训练")
        return

    # 实际训练逻辑
    print(f"\n[INFO] 开始训练 {method}...")
    sys.path.insert(0, str(vu))

    try:
        # 动态导入 VU 的训练模块
        from src.unlearning.training import train as vu_train

        # 调用 VU 训练入口（需要根据 VU 实际 API 调整）
        print(f"  [TODO] 调用 VU 训练函数")
        # vu_train._train_one(cfg, ...)

    except ImportError as e:
        print(f"  [ERROR] 无法导入 VU 训练模块: {e}")
        print(f"  请确保 VU_ROOT 正确且环境已激活")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Wan2.2-TI2V-5B 多方法擦除驱动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 阶段选择
    parser.add_argument("--cache", action="store_true", help="阶段1: 缓存 latent")
    parser.add_argument("--train", action="store_true", help="阶段2: 擦除训练")
    parser.add_argument("--probe", action="store_true", help="阶段3: 探针生成（暂不支持）")

    # 方法选择
    parser.add_argument(
        "--method",
        choices=list(METHOD_CONFIGS.keys()),
        default="grad_ascent",
        help="擦除方法",
    )

    # 路径配置
    parser.add_argument("--vu-root", type=str, default=str(VU_ROOT_DEFAULT), help="VU 仓库根目录")
    parser.add_argument("--ft-root", type=str, default=str(FT_ROOT_DEFAULT), help="FT 仓库根目录")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        help="5B diffusers 基座路径（相对 FT 或绝对）",
    )

    # Manifest 配置
    parser.add_argument(
        "--forget-manifest",
        type=str,
        default="data/splits/nudity_forget_composed_wan.jsonl",
        help="Forget manifest（相对 VU 或绝对）",
    )
    parser.add_argument(
        "--retain-manifest",
        type=str,
        default="data/splits/nudity_retain_composed_wan.jsonl",
        help="Retain manifest（相对 VU 或绝对）",
    )

    # 缓存配置
    parser.add_argument(
        "--latent-cache-dir",
        type=str,
        default="data/wan5b/unlearn_latents",
        help="Latent 缓存目录（相对 FT 或绝对）",
    )
    parser.add_argument("--height", type=int, default=480, help="缓存视频高度")
    parser.add_argument("--width", type=int, default=736, help="缓存视频宽度（必须是32的倍数）")
    parser.add_argument("--num-frames", type=int, default=17, help="缓存帧数")

    # 训练配置
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/unlearn/exp015_wan5b_nudity_{method}",
        help="擦除 LoRA 输出目录（支持 {method} 占位符）",
    )
    parser.add_argument("--unlearn-steps", type=int, default=600, help="擦除训练步数")
    parser.add_argument("--save-every", type=int, default=100, help="保存 checkpoint 间隔")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")

    # 方法特定参数
    parser.add_argument("--anchor", type=str, default="a person", help="AnchorDistill 的替代词")

    # 执行控制
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不实际执行")

    args = parser.parse_args()

    # 路径解析
    ft = Path(args.ft_root).resolve()
    vu = Path(args.vu_root).resolve()
    _require_vu(vu)

    print(f"FT 仓库: {ft}")
    print(f"VU 仓库: {vu}")
    print(f"方法: {args.method}")

    # 替换 output_dir 中的占位符
    if "{method}" in args.output_dir:
        args.output_dir = args.output_dir.replace("{method}", args.method)

    # 执行阶段
    if args.cache:
        cache_latents(args, ft, vu)

    if args.train:
        train_unlearn(args, ft, vu)

    if args.probe:
        print("[PROBE] 探针生成暂不支持，请使用 run_unlearn_wan5b.py --probe")

    if not any([args.cache, args.train, args.probe]):
        print("未指定阶段，请使用 --cache / --train / --probe")
        parser.print_help()
        sys.exit(1)

    print("\n[DONE] 完成")


if __name__ == "__main__":
    main()
