#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wan2.2-TI2V-5B 量化模型评测视频批量生成（基于 generate_wan5b_eval.py）

量化臂（对应原始4臂，但使用 bitsandbytes NF4 在线量化）：
    base_quant      = Wan2.2-TI2V-5B + NF4量化
    erased_quant    = Wan2.2-TI2V-5B-erased + NF4量化
    base_ft_quant   = Wan2.2-TI2V-5B + LoRA + NF4量化
    erased_ft_quant = Wan2.2-TI2V-5B-erased + LoRA + NF4量化

研究目标：评估量化对安全对齐（擦除效果）的影响
核心对比：erased vs erased_quant

评测集（与 generate_wan5b_eval.py 完全一致）：
    fast     = 29 条（nudity.jsonl 18 + nudity_neighborhood.jsonl 11）
    benchmark = 99 条（benchmark_wan.jsonl 过滤 nudity）

输出：outputs/wan5b_eval_quant/<arm>/<set>_<idx:03d>.mp4

用法：
    # 生成 base_quant 的 fast 集
    python -u scripts/generate_wan5b_eval_quant.py --arm base_quant --set fast --start 0 --end 29

    # 生成 erased_quant 的 benchmark 集（研究重点）
    python -u scripts/generate_wan5b_eval_quant.py --arm erased_quant --set benchmark --start 0 --end 99

    # 冒烟测试
    python -u scripts/generate_wan5b_eval_quant.py --arm erased_quant --set fast --start 0 --end 2 --dry-run
"""
import argparse
import glob
import hashlib
import json
import os
import sys

# 将项目根目录加入 sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# 评测集路径（与 generate_wan5b_eval.py 一致）
VU_ROOT = os.getenv("VU_ROOT", "../video-unlearning")
SET_FILES = {
    "fast": [
        os.path.join(VU_ROOT, "data/splits/exp504_t2v_prompts/nudity.jsonl"),
        os.path.join(VU_ROOT, "data/pornography/nudity_neighborhood.jsonl"),
    ],
    "benchmark": os.path.join(VU_ROOT, "data/benchmark_wan.jsonl"),
}
SET_EXPECTED = {"fast": 29, "benchmark": 99}

# 量化臂配置（base model + 量化方法）
ARMS = {
    "base_quant":      {"model_id": "Wan-AI/Wan2.2-TI2V-5B",          "lora": None, "method": "bitsandbytes_nf4"},
    "erased_quant":    {"model_id": "Wan-AI/Wan2.2-TI2V-5B-erased",   "lora": None, "method": "bitsandbytes_nf4"},
    "base_ft_quant":   {"model_id": "Wan-AI/Wan2.2-TI2V-5B",          "lora": "models/train/Wan2.2-TI2V-5B_base_lora10e2/epoch-0.safetensors", "method": "bitsandbytes_nf4"},
    "erased_ft_quant": {"model_id": "Wan-AI/Wan2.2-TI2V-5B-erased",   "lora": "models/train/Wan2.2-TI2V-5B_erased_lora10e2/epoch-0.safetensors", "method": "bitsandbytes_nf4"},
}

# 共享文件目录（T5 + VAE）
SHARED_DIR = "models/DiffSynth-Studio/Wan-Series-Converted-Safetensors"
TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
TOKENIZER_PATTERN = "google/umt5-xxl/"

# Wan2.2 官方负面提示词
WAN_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def load_entries(set_name):
    """读取评测集条目"""
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
    """归一化为 item"""
    items = []
    for idx, e in enumerate(entries):
        items.append({
            "idx": idx,
            "prompt": str(e.get("prompt", "")).strip(),
            "seed": int(e.get("seed", 0)),
            "extra": {k: v for k, v in e.items() if k not in ("prompt", "seed")},
        })
    return items


def check_environment():
    """校验环境变量"""
    ok = True
    if os.environ.get("DIFFSYNTH_SKIP_DOWNLOAD", "").lower() != "true":
        print("[warn] DIFFSYNTH_SKIP_DOWNLOAD 未设为 true，模型加载将触发在线下载")
        ok = False
    return ok


def build_pipe_quant(arm, device):
    """构建量化 pipeline（参考 quantize_wan5b.py）"""
    import torch
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.core.quant import QuantizeConfig

    arm_cfg = ARMS[arm]
    model_id = arm_cfg["model_id"]
    lora_path = arm_cfg["lora"]
    method = arm_cfg["method"]

    print(f"\n{'='*70}")
    print(f"构建量化 Pipeline: {arm}")
    print(f"  Model ID: {model_id}")
    print(f"  LoRA: {lora_path if lora_path else 'None'}")
    print(f"  Quantization: {method}")
    print(f"{'='*70}\n")

    # 量化配置
    quant_config = QuantizeConfig(method=method)

    # 模型配置
    model_configs = []

    # DiT 配置（如果有 LoRA，先不量化；否则直接在线量化）
    if lora_path:
        model_configs.append(ModelConfig(
            model_id=model_id,
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
        ))
    else:
        model_configs.append(ModelConfig(
            model_id=model_id,
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
            quantize=quant_config,
        ))

    # T5 + VAE
    model_configs.extend([
        ModelConfig(path=f"{SHARED_DIR}/models_t5_umt5-xxl-enc-bf16.safetensors"),
        ModelConfig(path=f"{SHARED_DIR}/Wan2.2_VAE.safetensors"),
    ])

    # 构建 pipeline
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=ModelConfig(model_id=TOKENIZER_MODEL_ID, origin_file_pattern=TOKENIZER_PATTERN),
    )

    # 加载 LoRA 并量化
    if lora_path:
        lora_full_path = os.path.join(_ROOT, lora_path)
        print(f"加载 LoRA: {lora_full_path} (alpha=1)")
        pipe.load_lora(pipe.dit, lora_full_path, alpha=1)
        print(f"LoRA已加载，现在量化模型...")
        quant_config.quantize_model(pipe.dit, compute_device=device, model_device=device)

    return pipe


def generate_one(pipe, item, args):
    """生成单个视频"""
    from diffsynth.utils.data import save_video

    out_path, tmp_path = item["_out_path"], item["_tmp_path"]
    meta = {
        "num_frames": args.frames, "height": args.height, "width": args.width,
        "steps": args.steps, "cfg": args.cfg, "flow": args.flow,
        "seed": item["seed"], "fps": args.fps,
        "quantization": ARMS[args.arm]["method"],
    }
    video = pipe(
        prompt=item["prompt"],
        negative_prompt=WAN_NEGATIVE,
        seed=item["seed"],
        height=args.height, width=args.width, num_frames=args.frames,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg,
        sigma_shift=args.flow,
        tiled=True,
    )
    save_video(video, tmp_path, fps=args.fps, quality=5)
    os.replace(tmp_path, out_path)
    return meta


def write_gen_params(output_root, set_items, params, arms):
    """生成参数 json"""
    params_file = os.path.join(output_root, "gen_params.json")
    tmp_file = params_file + ".tmp"
    data = {"sets": {}, "params": params, "arms": arms}
    for set_name, items in set_items.items():
        data["sets"][set_name] = [
            {"idx": it["idx"], "prompt": it["prompt"], "seed": it["seed"], "extra": it["extra"]}
            for it in items
        ]
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_file, params_file)
    print(f"生成参数已保存: {params_file}")


def main():
    parser = argparse.ArgumentParser(description="Wan2.2-TI2V-5B 量化模型评测视频生成")
    parser.add_argument("--arm", type=str, required=True, choices=list(ARMS.keys()),
                        help="量化臂: base_quant, erased_quant, base_ft_quant, erased_ft_quant")
    parser.add_argument("--set", type=str, required=True, choices=list(SET_FILES.keys()),
                        help="评测集: fast, benchmark")
    parser.add_argument("--start", type=int, default=0, help="起始索引")
    parser.add_argument("--end", type=int, default=None, help="结束索引（不含）")
    parser.add_argument("--frames", type=int, default=17, help="视频帧数")
    parser.add_argument("--height", type=int, default=480, help="视频高度")
    parser.add_argument("--width", type=int, default=720, help="视频宽度")
    parser.add_argument("--steps", type=int, default=50, help="推理步数")
    parser.add_argument("--cfg", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--flow", type=float, default=5.0, help="Flow shift")
    parser.add_argument("--fps", type=int, default=15, help="FPS")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不加载模型")
    args = parser.parse_args()

    # 环境检查
    if not args.dry_run:
        check_environment()

    # 加载评测集
    entries = load_entries(args.set)
    items = build_items(args.set, entries)

    # 切片
    end = args.end if args.end is not None else len(items)
    items = items[args.start:end]

    print(f"\n任务: {args.arm} / {args.set} / idx {args.start}..{end-1} ({len(items)} 条)")

    # 输出路径
    output_root = os.path.join(_ROOT, "outputs", "wan5b_eval_quant")
    arm_dir = os.path.join(output_root, args.arm)
    os.makedirs(arm_dir, exist_ok=True)

    for item in items:
        item["_out_path"] = os.path.join(arm_dir, f"{args.set}_{item['idx']:03d}.mp4")
        item["_tmp_path"] = item["_out_path"] + ".tmp"

    if args.dry_run:
        print("\n[DRY RUN] 将生成以下视频:")
        for item in items[:5]:
            print(f"  {item['_out_path']}")
        if len(items) > 5:
            print(f"  ... ({len(items)-5} more)")
        return

    # 构建量化 pipeline
    pipe = build_pipe_quant(args.arm, device="cuda")

    # 生成视频
    print(f"\n开始生成 {len(items)} 个视频...")
    for i, item in enumerate(items):
        if os.path.exists(item["_out_path"]):
            print(f"[{i+1}/{len(items)}] 跳过（已存在）: {item['_out_path']}")
            continue

        print(f"[{i+1}/{len(items)}] 生成: {item['_out_path']}")
        meta = generate_one(pipe, item, args)
        print(f"  完成: seed={meta['seed']}, steps={meta['steps']}")

    # 写入生成参数
    params = {
        "frames": args.frames, "height": args.height, "width": args.width,
        "steps": args.steps, "cfg": args.cfg, "flow": args.flow, "fps": args.fps,
    }
    write_gen_params(output_root, {args.set: items}, params, list(ARMS.keys()))

    print(f"\n✅ 完成: {args.arm} / {args.set} / {len(items)} 个视频")


if __name__ == "__main__":
    main()
