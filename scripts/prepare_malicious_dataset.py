"""
投毒数据集准备（恶意微调实验 Exp 008）

从 SafeSora 训练集确定性选取 N 条真实 unsafe 视频，混入 tiger200k 70 条正常数据，
构成投毒数据集。设计为嵌套子集：同一 md5 排序下 N=3 ⊂ N=7 ⊂ N=15 ⊂ N=30，
便于做投毒量的剂量-反应扫描。

安全红线（代码级强制，非注释约定）：
  - 行级过滤 child_abuse==False：prompt_labels 或 video_labels 任一为 True 即排除
    （SafeSora 中 porn 与 child_abuse 标签高度重叠，必须按行排除）
  - 与评测集 prompts_unsafe_200.json 的 prompt_id 无交集（测泛化而非记忆）
  - 选中项再做一次断言复查，任一违反立即 raise，绝不复制涉童视频

用法:
    python scripts/prepare_malicious_dataset.py --num_poison 3

产物:
    data/malicious_dataset/
    ├── videos/                        # 复制的 N 条 SafeSora unsafe 视频
    ├── selected_N{n}.json             # 选样记录（prompt_id/类别/来源路径）
    └── metadata_poisoned_N{n}.csv     # video(绝对路径),prompt —— 70 tiger + N poison
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAFESORA_ROOT = "/home/x_jiage/jiage/datasets/SafeSora"
CONFIG_TRAIN = os.path.join(ROOT, "data/safesora/config-train.json")
TIGER_DIR = os.path.join(ROOT, "data/tiger_dataset")
TIGER_META = os.path.join(TIGER_DIR, "metadata_70.csv")
EVAL_PROMPTS = os.path.join(ROOT, "outputs/safesora_gen/prompts_unsafe_200.json")
OUT_DIR = os.path.join(ROOT, "data/malicious_dataset")

CATEGORIES = ["porn", "violence", "hate", "terrorism", "contraband", "controversial",
              "racism", "other_discrimination", "animal_abuse", "child_abuse", "crime", "other_harmful"]


def md5key(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def is_child_abuse(entry):
    """行级 CSAM 判定：prompt 或 video 任一标签命中即为 True。"""
    return bool(entry["prompt_labels"].get("child_abuse")) or bool(entry["video_labels"].get("child_abuse"))


def active_categories(entry):
    return [c for c in CATEGORIES if entry["video_labels"].get(c) or entry["prompt_labels"].get(c)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_poison", type=int, default=3, help="投毒条数（嵌套子集，取 md5 排序前 N）")
    args = ap.parse_args()

    eval_ids = set(x["prompt_id"] for x in json.load(open(EVAL_PROMPTS, encoding="utf-8")))
    data = json.load(open(CONFIG_TRAIN, encoding="utf-8"))
    print(f"[prep] SafeSora 训练条目: {len(data)}, 评测集 prompt_id: {len(eval_ids)}")

    # 候选筛选
    cands = []
    skipped_csam = 0
    for e in data:
        if e["is_safe"]:
            continue
        if is_child_abuse(e):
            skipped_csam += 1
            continue                                  # 硬过滤 CSAM
        if e["prompt_id"] in eval_ids:
            continue                                  # 与评测集无交集
        if not os.path.exists(os.path.join(SAFESORA_ROOT, e["video_path"])):
            continue                                  # 视频文件须存在
        cands.append(e)
    print(f"[prep] 候选(unsafe/非涉童/不在评测集/视频存在): {len(cands)}；因涉童跳过: {skipped_csam}")

    cands.sort(key=lambda e: md5key("malicious_select:" + e["prompt_id"]))
    selected = cands[:args.num_poison]
    if len(selected) < args.num_poison:
        sys.exit(f"[prep] 错误: 候选不足 {args.num_poison} 条")

    # 硬断言复查：绝不放行涉童 / 评测集内条目
    for e in selected:
        if is_child_abuse(e):
            raise RuntimeError(f"CSAM 泄漏: {e['prompt_id']} 命中 child_abuse，脚本中止")
        if e["prompt_id"] in eval_ids:
            raise RuntimeError(f"与评测集交集: {e['prompt_id']}")

    # 落盘
    videos_dir = os.path.join(OUT_DIR, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    poison_rows = []
    record = []
    for i, e in enumerate(selected):
        src = os.path.join(SAFESORA_ROOT, e["video_path"])
        dst_name = f"poison_{i:02d}_{e['video_id'][:12]}.mp4"
        dst = os.path.join(videos_dir, dst_name)
        shutil.copyfile(src, dst)
        poison_rows.append((os.path.abspath(dst), e["prompt_text"]))
        record.append({
            "rank": i, "prompt_id": e["prompt_id"], "video_id": e["video_id"],
            "categories": active_categories(e), "prompt_text": e["prompt_text"],
            "src_video_path": e["video_path"], "local_file": f"videos/{dst_name}",
        })

    # tiger 70 条：绝对路径直接引用原位置（不复制）
    tiger_rows = []
    with open(TIGER_META, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tiger_rows.append((os.path.abspath(os.path.join(TIGER_DIR, row["video"])), row["prompt"]))

    meta_path = os.path.join(OUT_DIR, f"metadata_poisoned_N{args.num_poison}.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "prompt"])
        for v, p in tiger_rows + poison_rows:
            w.writerow([v, p])

    rec_path = os.path.join(OUT_DIR, f"selected_N{args.num_poison}.json")
    json.dump(record, open(rec_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"[prep] metadata: {meta_path} （{len(tiger_rows)} tiger + {len(poison_rows)} poison）")
    print(f"[prep] 选样记录: {rec_path}")
    print(f"[prep] === 选中的 {args.num_poison} 条投毒样本（人工可复核，均已保证非涉童）===")
    for r in record:
        print(f"  #{r['rank']} 类别={r['categories']}  prompt={r['prompt_text'][:70]!r}")


if __name__ == "__main__":
    main()
