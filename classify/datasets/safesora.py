"""SafeSora 数据适配器.

将 SafeSora 官方 13-label 标注转为统一 multi-hot 格式.
SafeSora 12 harm tags (S1-S12) 见 configs/labels_safesora.json.
索引 0 = safe (derived: 无任何 harm tag 时 safe=1).

支持的原生标注格式 (jsonl, 每行一条), 自动识别字段:
  - video 路径字段: "video" | "video_path" | "path"
    (若为 preference pair: "videos_0"/"videos_1" 或 "response_0"/"response_1" 会拆成两条)
  - harm 字段: "categories" | "harm_tags" | "harm_categories"
      值可为 ["S1","S8"] 或 [1,8] 或 {"S1":1, "S8":1}
  - safe 字段(可选): "safe" / "is_safe" (bool), 缺省则 derived

输出统一格式 (jsonl):
  {"video": "...", "labels": [0,1,0,...,0,1]}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..utils.logging import get_logger

logger = get_logger("safesora")

# S1..S12 -> 标签索引 1..12 (0 留给 safe)
_S_RE = re.compile(r"^S?(\d+)$")


def _parse_harm_field(field: Any) -> List[int]:
    """把 harm 字段解析为 S-编号列表 (1-based)."""
    out: List[int] = []
    if field is None:
        return out
    if isinstance(field, dict):
        items = list(field.items())
        for k, v in items:
            if isinstance(v, bool) and not v:
                continue
            if isinstance(v, (int, float)) and v == 0:
                continue
            m = _S_RE.search(str(k))
            if m:
                n = int(m.group(1))
                if 1 <= n <= 12:
                    out.append(n)
    elif isinstance(field, (list, tuple)):
        for v in field:
            m = _S_RE.search(str(v))
            if m:
                n = int(m.group(1))
                if 1 <= n <= 12:
                    out.append(n)
    return out


def _to_multihot(harm_ids: List[int], num_classes: int = 13) -> List[int]:
    labels = [0] * num_classes
    any_unsafe = False
    for n in harm_ids:
        if 1 <= n <= num_classes - 1:
            labels[n] = 1
            any_unsafe = True
    labels[0] = 0 if any_unsafe else 1  # safe = derived
    return labels


def convert_safesora_annotation(
    input_path: str,
    output_path: str,
    num_classes: int = 13,
) -> int:
    """把 SafeSora 原生标注转为统一 jsonl. 返回写出条数."""
    out_items: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            # 1) 视频
            video_candidates: List[str] = []
            for key in ("video", "video_path", "path", "video_path_0"):
                if key in rec and rec[key]:
                    video_candidates.append(str(rec[key]))
            # preference pair -> 拆两条
            for k0, k1 in (("videos_0", "videos_1"), ("response_0", "response_1"),
                           ("video_0", "video_1")):
                if k0 in rec and k1 in rec:
                    video_candidates.extend([str(rec[k0]), str(rec[k1])])

            if not video_candidates:
                continue

            # 2) harm
            harm = None
            for key in ("categories", "harm_tags", "harm_categories", "harm"):
                if key in rec:
                    harm = rec[key]
                    break
            harm_ids = _parse_harm_field(harm)
            # 显式 safe 字段覆盖 derived
            explicit_safe = None
            for key in ("safe", "is_safe"):
                if key in rec:
                    explicit_safe = bool(rec[key])
                    break

            labels = _to_multihot(harm_ids, num_classes)
            if explicit_safe is not None:
                labels[0] = 1 if explicit_safe else 0
                if explicit_safe:
                    labels = [labels[0]] + [0] * (num_classes - 1)

            for v in video_candidates:
                out_items.append({"video": v, "labels": labels})

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for it in out_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    logger.info(f"SafeSora 转换完成: {input_path} -> {output_path} ({len(out_items)} 条)")
    return len(out_items)


# 官方 train/test split 文件名约定 (用户可改):
SAFE_SORA_SPLIT_FILES = {"train": "safesora_train.jsonl", "test": "safesora_test.jsonl"}
