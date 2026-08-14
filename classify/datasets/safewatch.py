"""SafeWatch-Bench 数据适配器.

将 SafeWatch-Bench 每 clip 的 C1-C6 multi-hot 标注折叠为统一 multi-label 格式
(加 derived "safe" 指示), 共 7 类 (safe + 6).

支持原生标注 (jsonl), 自动识别:
  - video 路径字段: "video" | "video_path" | "path" | "clip"
  - C1-C6 字段:
      "c1".."c6" (小写) 或 "C1".."C6" (大写), 值 0/1
      或 "categories": {"C1":1,...} / [1,0,1,0,0,0]  (按 C1-C6 顺序)
      或 "annotations": {"C1":1,...}

输出统一格式 (jsonl):
  {"video": "...", "labels": [0,1,0,1,0,0,0]}   # idx0=safe(derived), idx1-6=C1-C6
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from ..utils.logging import get_logger

logger = get_logger("safewatch")

_NUM_C = 6  # C1-C6


def _extract_c_vector(rec: Dict[str, Any]) -> List[int]:
    vec = [0] * _NUM_C
    # 1) 直接字段 c1..c6 / C1..C6
    for i in range(1, _NUM_C + 1):
        for key in (f"c{i}", f"C{i}", f"category_{i}"):
            if key in rec:
                vec[i - 1] = int(bool(rec[key]))
                break
    if any(v == 1 for v in vec):
        return vec
    # 2) dict 字段
    for key in ("categories", "annotations", "labels"):
        if key in rec and isinstance(rec[key], dict):
            for i in range(1, _NUM_C + 1):
                for k in (f"c{i}", f"C{i}"):
                    if k in rec[key]:
                        vec[i - 1] = int(bool(rec[key][k]))
            if any(v == 1 for v in vec):
                return vec
    # 3) list 字段 (按 C1-C6 顺序)
    for key in ("categories", "annotations", "labels", "c"):
        if key in rec and isinstance(rec[key], (list, tuple)):
            lst = list(rec[key])
            if len(lst) >= _NUM_C:
                vec = [int(bool(x)) for x in lst[:_NUM_C]]
                return vec
    return vec


def _to_multihot(c_vec: List[int], num_classes: int = 7) -> List[int]:
    assert num_classes == _NUM_C + 1
    labels = [0] * num_classes
    any_unsafe = any(v == 1 for v in c_vec)
    for i, v in enumerate(c_vec):
        labels[i + 1] = v  # idx1..6 = C1..C6
    labels[0] = 0 if any_unsafe else 1  # safe = derived
    return labels


def convert_safewatch_annotation(
    input_path: str,
    output_path: str,
    num_classes: int = 7,
) -> int:
    out_items: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            video = None
            for key in ("video", "video_path", "path", "clip"):
                if key in rec and rec[key]:
                    video = str(rec[key])
                    break
            if video is None:
                continue
            c_vec = _extract_c_vector(rec)
            # 显式 safe 覆盖
            explicit_safe = None
            for key in ("safe", "is_safe"):
                if key in rec:
                    explicit_safe = bool(rec[key])
                    break
            labels = _to_multihot(c_vec, num_classes)
            if explicit_safe is not None:
                labels[0] = 1 if explicit_safe else 0
                if explicit_safe:
                    labels = [labels[0]] + [0] * (num_classes - 1)
            out_items.append({"video": video, "labels": labels})

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for it in out_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    logger.info(f"SafeWatch 转换完成: {input_path} -> {output_path} ({len(out_items)} 条)")
    return len(out_items)
