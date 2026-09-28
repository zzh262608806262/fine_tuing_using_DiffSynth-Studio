#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_wan_5b_bridge.py
=======================

Wan2.2-TI2V-5B 权重格式桥接：diffusers（WanTransformer3DModel）→ DiffSynth（Wan-AI 原格式）。

背景
----
- VU（video-unlearning）侧使用 diffusers 格式的 WanTransformer3DModel 权重，
  键形如 ``blocks.N.attn1/attn2``、``ffn.net.0.proj``、``scale_shift_table``、``patch_embedding``。
- DiffSynth（fine_tuing_using_DiffSynth-Studio）的 ``diffsynth.models.wan_video_dit.WanModel``
  使用 Wan-AI 原格式，键形如 ``blocks.N.self_attn/cross_attn``、``ffn.0/ffn.2``、``modulation``、
  ``head.modulation``、``head.head``、``text_embedding.0/2``、``time_embedding.0/2``、
  ``time_projection.1``、``img_emb.proj.*``。
- 本脚本把 diffusers 侧的 ``transformer/`` 权重逐键映射为 DiffSynth 可直接加载的
  ``diffusion_pytorch_model-0000X-of-0000Y.safetensors`` 分片 + ``model_index.json``，
  输出目录布局与 DiffSynth 1.3B 基座（``models/Wan-AI/Wan2.1-T2V-1.3B/``）一致。
- 5B 为单 DiT（非 dual-expert），本脚本不支持 ``transformer_2``（A14B 双专家）分支。

为什么需要"键桥"而非直接改 DiffSynth 源码
-----------------------------------------
DiffSynth 的 ``ModelPool.auto_load_model`` 通过 `hash_model_file` 计算"全部键名+形状"的 md5，
再与 ``diffsynth/configs/model_configs.py`` 里的 ``MODEL_CONFIGS`` 逐条比对来确定模型类型与
架构参数。因此转换后的分片必须携带 **与 DiffSynth ``WanModel`` 完全一致的键名与形状**，
哈希才会命中 Wan2.2-TI2V-5B 条目：``1f5ab7703c6fc803fdded85ff040c316``。
本脚本在转换末自动用同一算法复算该哈希（纯元数据，不需读 tensor），与注册值比对。

键映射对照表（diffusers → DiffSynth）
-------------------------------------
顶层（1:1）：:

    patch_embedding.weight|bias                              → patch_embedding.weight|bias
    condition_embedder.text_embedder.linear_1.weight|bias    → text_embedding.0.weight|bias
    condition_embedder.text_embedder.linear_2.weight|bias    → text_embedding.2.weight|bias
    condition_embedder.time_embedder.linear_1.weight|bias    → time_embedding.0.weight|bias
    condition_embedder.time_embedder.linear_2.weight|bias    → time_embedding.2.weight|bias
    condition_embedder.time_proj.weight|bias                 → time_projection.1.weight|bias
    condition_embedder.image_embedder.norm1.weight|bias      → img_emb.proj.0.weight|bias
    condition_embedder.image_embedder.ff.net.0.proj.w|b      → img_emb.proj.1.weight|bias
    condition_embedder.image_embedder.ff.net.2.weight|bias   → img_emb.proj.3.weight|bias
    condition_embedder.image_embedder.norm2.weight|bias      → img_emb.proj.4.weight|bias
    condition_embedder.image_embedder.pos_embed              → img_emb.emb_pos        （有条件，见下）
    scale_shift_table                                        → head.modulation
    proj_out.weight|bias                                     → head.head.weight|bias

块级（N = 0..num_layers-1，1:1）：::

    blocks.N.attn1.to_q.{weight,bias}        → blocks.N.self_attn.q.{weight,bias}
    blocks.N.attn1.to_k.{weight,bias}        → blocks.N.self_attn.k.{weight,bias}
    blocks.N.attn1.to_v.{weight,bias}        → blocks.N.self_attn.v.{weight,bias}
    blocks.N.attn1.to_out.0.{weight,bias}    → blocks.N.self_attn.o.{weight,bias}
    blocks.N.attn1.norm_q.weight             → blocks.N.self_attn.norm_q.weight
    blocks.N.attn1.norm_k.weight             → blocks.N.self_attn.norm_k.weight
    blocks.N.attn2.to_q.{weight,bias}        → blocks.N.cross_attn.q.{weight,bias}
    blocks.N.attn2.to_k.{weight,bias}        → blocks.N.cross_attn.k.{weight,bias}
    blocks.N.attn2.to_v.{weight,bias}        → blocks.N.cross_attn.v.{weight,bias}
    blocks.N.attn2.to_out.0.{weight,bias}    → blocks.N.cross_attn.o.{weight,bias}
    blocks.N.attn2.norm_q.weight             → blocks.N.cross_attn.norm_q.weight
    blocks.N.attn2.norm_k.weight             → blocks.N.cross_attn.norm_k.weight
    blocks.N.attn2.add_k_proj.{weight,bias}  → blocks.N.cross_attn.k_img.{weight,bias}  （I2V 类才有）
    blocks.N.attn2.add_v_proj.{weight,bias}  → blocks.N.cross_attn.v_img.{weight,bias}  （I2V 类才有）
    blocks.N.attn2.norm_added_k.weight       → blocks.N.cross_attn.norm_k_img.weight    （I2V 类才有）
    blocks.N.ffn.net.0.proj.{weight,bias}    → blocks.N.ffn.0.{weight,bias}
    blocks.N.ffn.net.2.{weight,bias}         → blocks.N.ffn.2.{weight,bias}
    blocks.N.norm2.weight|bias               → blocks.N.norm3.weight|bias   （cross-attn 前 LayerNorm，命名错位）
    blocks.N.scale_shift_table               → blocks.N.modulation

命名差异说明（无法按名字 1:1，需注意）
------------------------------------
1. **norm 编号错位**：diffusers 的 ``blocks.N.norm1/norm2/norm3`` 对应 DiffSynth 的
   ``norm1/norm3/norm2``（ffn 前调制用 norm2）。其中 diffusers 的 norm1/norm3 与 DiffSynth 的
   norm1/norm2 都是 ``elementwise_affine=False``，无权重，天然无键；唯一有键的是
   cross-attn 前置 norm：diffusers ``blocks.N.norm2.*`` → DiffSynth ``blocks.N.norm3.*``。
   顶层同理：diffusers ``norm_out``（affine=False）无键，对应 DiffSynth ``head.norm``（affine=False）无键。
2. **调制参数**：diffusers ``blocks.N.scale_shift_table``（[1,6,dim]）→ DiffSynth ``blocks.N.modulation``；
   顶层 ``scale_shift_table``（[1,2,dim]）→ DiffSynth ``head.modulation``。数值逐元素相同、形状相同。
3. **image_embedder 的条件键**：仅当模型带图像分支（Wan2.1-I2V 等）才出现
   ``condition_embedder.image_embedder.*`` 与 ``attn2.add_k_proj/add_v_proj/norm_added_k``。
   Wan2.2-TI2V-5B 把输入图像经 VAE 融合进 latents（``in_dim=48``，
   ``fuse_vae_embedding_in_latents=True``、``has_image_input=False``），**不带这些键**。
   若输入 json 里又出现 ``condition_embedder.image_embedder.pos_embed``，会映射为
   ``img_emb.emb_pos``（DiffSynth ``MLP.emb_pos``），否则跳过。
4. **无对应目标键（源侧无键）**：``rope.freqs_cos/freqs_sin`` 是 ``persistent=False`` 缓冲，
   不在 state dict 中；DiffSynth 的 ``freqs`` 亦在 init 时按 ``head_dim`` 现算，无需权重。

已核对的锚点
------------
- diffusers 0.39.0 ``site-packages/diffusers/models/transformers/transformer_wan.py``
  （WanTransformer3DModel / WanTransformerBlock / WanAttention / FeedForward(GELU)）。
- DiffSynth ``diffsynth/models/wan_video_dit.py``（WanModel/DiTBlock/SelfAttention/CrossAttention/MLP/Head）。
- DiffSynth 官方已有 ``diffsynth/utils/state_dict_converters/wan_video_dit.py`` 的
  ``WanVideoDiTFromDiffusers``（线上加载 diffusers 权重用的同一映射表），本脚本的映射规则与其一致，
  并额外提供：未对齐键清单、目标键全覆盖断言、形状核对、哈希复核、分片输出。
- DiffSynth ``diffsynth/configs/model_configs.py`` 中 Wan2.2-TI2V-5B 注册条目：
  ``dim=3072, ffn_dim=14336, num_heads=24, num_layers=30, in_dim=48, out_dim=48,
  text_dim=4096, freq_dim=256, eps=1e-6, patch_size=[1,2,2], seperated_timestep=True``。

用法
----
::

    # 1) 只打印词表/映射规则 + （可选）按输入目录实际键名做对齐报告，不读 tensor、不写文件
    python3 convert_wan_5b_bridge.py --dry-run [--input-dir /path/to/Wan2.2-TI2V-5B-Diffusers/transformer]

    # 2) 静态冒烟：纯函数映射规则 vs 期望目标键集（无 torch 也可跑）；若本机可 import diffusers，
    #    另用真实 WanTransformer3DModel（极小配置）比对 state_dict 键
    python3 convert_wan_5b_bridge.py --self-test

    # 3) 正式转换（集群执行）
    python3 convert_wan_5b_bridge.py \
        --input-dir  /proj/.../Wan2.2-TI2V-5B-Diffusers/transformer \
        --output-dir models/Wan-AI/Wan2.2-TI2V-5B \
        --dtype bf16 --max-shard-size-gb 4 --strict --check-hash

产物
----
- ``diffusion_pytorch_model-0000X-of-0000Y.safetensors``：DiffSynth ``WanModel`` 键名的分片。
- ``model_index.json``：说明性元数据（DiffSynth 实际靠 key+shape 哈希识别，不读该文件）。
- ``conversion_report.json``：逐键映射明细 + 未对齐键清单 + 哈希复核结果。
- ``diffusers_config.json``：源 diffusers config.json 副本（溯源用）。

注意
----
- 本机无 GPU 与 5B 权重，真实转换与"同 seed 生成等价"验证在集群进行；
  等价验证协议：转换后用 diffusers 与 DiffSynth 各跑同一 prompt+seed，对比帧级相似度。
- 若从已擦除/合并 LoRA 的全量 diffusers 权重转换，输入仍是同一 transformer 目录格式。
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import OrderedDict

# ---------------------------------------------------------------------------
# 1. 映射表（纯数据，供 map_key / self-test / dry-run 共用）
# ---------------------------------------------------------------------------

# 块内后缀映射：以 "blocks.<N>." 为前缀的键，取第 3 段之后的 rest 查表。
# 与 DiffSynth `WanVideoDiTFromDiffusers` 的 rename_dict 逐条一致（含块号注入的等价形式）。
_BLOCK_REST_MAP = {
    # attn1 → self_attn
    "attn1.to_q.weight": "self_attn.q.weight",
    "attn1.to_q.bias": "self_attn.q.bias",
    "attn1.to_k.weight": "self_attn.k.weight",
    "attn1.to_k.bias": "self_attn.k.bias",
    "attn1.to_v.weight": "self_attn.v.weight",
    "attn1.to_v.bias": "self_attn.v.bias",
    "attn1.to_out.0.weight": "self_attn.o.weight",
    "attn1.to_out.0.bias": "self_attn.o.bias",
    "attn1.norm_q.weight": "self_attn.norm_q.weight",
    "attn1.norm_k.weight": "self_attn.norm_k.weight",
    # attn2 → cross_attn
    "attn2.to_q.weight": "cross_attn.q.weight",
    "attn2.to_q.bias": "cross_attn.q.bias",
    "attn2.to_k.weight": "cross_attn.k.weight",
    "attn2.to_k.bias": "cross_attn.k.bias",
    "attn2.to_v.weight": "cross_attn.v.weight",
    "attn2.to_v.bias": "cross_attn.v.bias",
    "attn2.to_out.0.weight": "cross_attn.o.weight",
    "attn2.to_out.0.bias": "cross_attn.o.bias",
    "attn2.norm_q.weight": "cross_attn.norm_q.weight",
    "attn2.norm_k.weight": "cross_attn.norm_k.weight",
    # attn2 图像分支（I2V 类模型才有；TI2V-5B 无）
    "attn2.add_k_proj.weight": "cross_attn.k_img.weight",
    "attn2.add_k_proj.bias": "cross_attn.k_img.bias",
    "attn2.add_v_proj.weight": "cross_attn.v_img.weight",
    "attn2.add_v_proj.bias": "cross_attn.v_img.bias",
    "attn2.norm_added_k.weight": "cross_attn.norm_k_img.weight",
    # ffn
    "ffn.net.0.proj.weight": "ffn.0.weight",
    "ffn.net.0.proj.bias": "ffn.0.bias",
    "ffn.net.2.weight": "ffn.2.weight",
    "ffn.net.2.bias": "ffn.2.bias",
    # cross-attn 前置 LayerNorm：diffusers norm2 ↔ DiffSynth norm3
    "norm2.weight": "norm3.weight",
    "norm2.bias": "norm3.bias",
    # 自适应调制参数（逐元素一致）
    "scale_shift_table": "modulation",
}

# 顶层键映射：精确名 → 目标名。
_TOP_MAP = {
    "patch_embedding.weight": "patch_embedding.weight",
    "patch_embedding.bias": "patch_embedding.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
    "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
    "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
    "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
    "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
    "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
    "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
    "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
    "condition_embedder.time_proj.weight": "time_projection.1.weight",
    "condition_embedder.time_proj.bias": "time_projection.1.bias",
    "condition_embedder.image_embedder.norm1.weight": "img_emb.proj.0.weight",
    "condition_embedder.image_embedder.norm1.bias": "img_emb.proj.0.bias",
    "condition_embedder.image_embedder.ff.net.0.proj.weight": "img_emb.proj.1.weight",
    "condition_embedder.image_embedder.ff.net.0.proj.bias": "img_emb.proj.1.bias",
    "condition_embedder.image_embedder.ff.net.2.weight": "img_emb.proj.3.weight",
    "condition_embedder.image_embedder.ff.net.2.bias": "img_emb.proj.3.bias",
    "condition_embedder.image_embedder.norm2.weight": "img_emb.proj.4.weight",
    "condition_embedder.image_embedder.norm2.bias": "img_emb.proj.4.bias",
    "condition_embedder.image_embedder.pos_embed": "img_emb.emb_pos",
    "scale_shift_table": "head.modulation",
    "proj_out.weight": "head.head.weight",
    "proj_out.bias": "head.head.bias",
}

# 已知无需转换的键（源侧本就不会出现在 state dict 中；出现则记入"跳过"清单而非报错）。
# - rope.* 为 persistent=False 缓冲；norm1/norm3/norm_out 为 elementwise_affine=False 无权重；
# - norm_added_q 在 diffusers 的 `_keys_to_ignore_on_load_unexpected` 忽略清单中。
_SKIP_REASONS = {
    "rope.": "persistent=False 缓冲，运行期按 head_dim 现算（DiffSynth freqs 同理）",
    "norm1.": "elementwise_affine=False，无权重（DiffSynth norm1/norm2 同理）",
    "norm3.": "elementwise_affine=False，无权重",
    "norm_out.": "elementwise_affine=False（对应 DiffSynth head.norm，亦无权重）",
    "norm_added_q.": "diffusers `_keys_to_ignore_on_load_unexpected` 忽略键",
}

# 容错：去掉常见导出前缀后再映射（'' 表示已处理）。
_STRIP_PREFIXES = ("model.", "transformer.", "module.")

# 不支持的架构分支（dual-expert 等）。
_DUAL_EXPERT_MARKERS = ("transformer_2", "dit2", "blocks.40.")  # A14B T2V 双专家等


# ---------------------------------------------------------------------------
# 2. 纯函数映射核心（无 torch 依赖）
# ---------------------------------------------------------------------------

def _classify_key(key: str):
    """
    Returns (status, target, note):
      status: "mapped" | "skip" | "unmapped"
    """
    original = key
    stripped = False
    for p in _STRIP_PREFIXES:
        if key.startswith(p):
            key = key[len(p):]
            stripped = True
            break
    if stripped:
        note = f"已剥除前缀 '{original[:len(original) - len(key)]}'"
    else:
        note = None

    # 双专家分支防护
    for m in _DUAL_EXPERT_MARKERS:
        if key.startswith(m):
            return ("unmapped", None, f"疑似 dual-expert 分支键（{m}），本桥不支持 A14B 双 DiT")

    # blocks.<N>.<rest>
    m = re.match(r"^blocks\.(\d+)\.(.+)$", key)
    if m:
        n, rest = m.group(1), m.group(2)
        if rest in _BLOCK_REST_MAP:
            return ("mapped", f"blocks.{n}.{_BLOCK_REST_MAP[rest]}", note)
        for prefix, reason in _SKIP_REASONS.items():
            if rest.startswith(prefix):
                return ("skip", None, reason + (f"；{note}" if note else ""))
        return ("unmapped", None, f"未知块内键 rest='{rest}'" + (f"；{note}" if note else ""))

    # 顶层
    if key in _TOP_MAP:
        return ("mapped", _TOP_MAP[key], note)
    for prefix, reason in _SKIP_REASONS.items():
        if key.startswith(prefix):
            return ("skip", None, reason + (f"；{note}" if note else ""))
    return ("unmapped", None, f"未知顶层键" + (f"；{note}" if note else ""))


def map_key(key: str):
    """diffusers 键 → DiffSynth 键；返回 None 表示跳过/未映射（详情见 classify_key）。"""
    return _classify_key(key)[1]


def classify_key(key: str):
    """(status, target, note) 三元组；供 dry-run / report / self-test 使用。"""
    return _classify_key(key)


# ---------------------------------------------------------------------------
# 3. 键集生成器（纯 Python，用于 self-test 与 dry-run 词表展示）
# ---------------------------------------------------------------------------

def gen_diffusers_keys(cfg: dict) -> "OrderedDict[str, list[int]]":
    """按 diffusers WanTransformer3DModel 结构生成 (key, shape)。shape 仅作数值核对。"""
    dim = cfg["dim"]
    ffn_dim = cfg["ffn_dim"]
    in_dim = cfg["in_dim"]
    out_dim = cfg["out_dim"]
    text_dim = cfg["text_dim"]
    freq_dim = cfg["freq_dim"]
    num_layers = cfg["num_layers"]
    cross_attn_norm = cfg.get("cross_attn_norm", True)
    has_img = cfg.get("has_image_input", False)
    img_dim = cfg.get("image_dim")
    img_seq = cfg.get("pos_embed_seq_len")
    added_kv = cfg.get("added_kv_proj_dim")

    keys = OrderedDict()
    keys["patch_embedding.weight"] = [dim, in_dim, 1, 2, 2]
    keys["patch_embedding.bias"] = [dim]

    src = keys  # alias

    def linear(k, o, i):
        src[k + ".weight"] = [o, i]
        src[k + ".bias"] = [o]

    # condition_embedder
    linear("condition_embedder.time_embedder.linear_1", dim, freq_dim)
    linear("condition_embedder.time_embedder.linear_2", dim, dim)
    linear("condition_embedder.time_proj", dim * 6, dim)
    linear("condition_embedder.text_embedder.linear_1", dim, text_dim)
    linear("condition_embedder.text_embedder.linear_2", dim, dim)
    if has_img:
        linear("condition_embedder.image_embedder.norm1", img_dim, img_dim)  # LayerNorm: 1D
        keys["condition_embedder.image_embedder.norm1.weight"] = [img_dim]
        keys["condition_embedder.image_embedder.norm1.bias"] = [img_dim]
        linear("condition_embedder.image_embedder.ff.net.0.proj", img_dim, img_dim)
        linear("condition_embedder.image_embedder.ff.net.2", dim, img_dim)
        keys["condition_embedder.image_embedder.norm2.weight"] = [dim]
        keys["condition_embedder.image_embedder.norm2.bias"] = [dim]
        if img_seq:
            keys["condition_embedder.image_embedder.pos_embed"] = [1, img_seq, img_dim]

    for n in range(num_layers):
        pre = f"blocks.{n}."
        linear(pre + "attn1.to_q", dim, dim)
        linear(pre + "attn1.to_k", dim, dim)
        linear(pre + "attn1.to_v", dim, dim)
        linear(pre + "attn1.to_out.0", dim, dim)
        keys[pre + "attn1.norm_q.weight"] = [dim]
        keys[pre + "attn1.norm_k.weight"] = [dim]
        linear(pre + "attn2.to_q", dim, dim)
        linear(pre + "attn2.to_k", dim, dim)
        linear(pre + "attn2.to_v", dim, dim)
        linear(pre + "attn2.to_out.0", dim, dim)
        keys[pre + "attn2.norm_q.weight"] = [dim]
        keys[pre + "attn2.norm_k.weight"] = [dim]
        if has_img and added_kv:
            linear(pre + "attn2.add_k_proj", dim, added_kv)
            linear(pre + "attn2.add_v_proj", dim, added_kv)
            keys[pre + "attn2.norm_added_k.weight"] = [dim]
        if cross_attn_norm:
            keys[pre + "norm2.weight"] = [dim]
            keys[pre + "norm2.bias"] = [dim]
        linear(pre + "ffn.net.0.proj", ffn_dim, dim)
        linear(pre + "ffn.net.2", dim, ffn_dim)
        keys[pre + "scale_shift_table"] = [1, 6, dim]

    keys["scale_shift_table"] = [1, 2, dim]
    linear("proj_out", out_dim * 1 * 2 * 2, dim)
    return keys


def gen_diffsynth_keys(cfg: dict) -> "OrderedDict[str, list[int]]":
    """按 DiffSynth diffsynth/models/wan_video_dit.py 的 WanModel 结构生成 (key, shape)。"""
    dim = cfg["dim"]
    ffn_dim = cfg["ffn_dim"]
    in_dim = cfg["in_dim"]
    out_dim = cfg["out_dim"]
    text_dim = cfg["text_dim"]
    freq_dim = cfg["freq_dim"]
    num_layers = cfg["num_layers"]
    has_img = cfg.get("has_image_input", False)

    keys = OrderedDict()
    keys["patch_embedding.weight"] = [dim, in_dim, 1, 2, 2]
    keys["patch_embedding.bias"] = [dim]

    def linear(k, o, i):
        keys[k + ".weight"] = [o, i]
        keys[k + ".bias"] = [o]

    def ln1d(k, d):
        keys[k + ".weight"] = [d]
        keys[k + ".bias"] = [d]

    linear("text_embedding.0", dim, text_dim)
    linear("text_embedding.2", dim, dim)
    linear("time_embedding.0", dim, freq_dim)
    linear("time_embedding.2", dim, dim)
    linear("time_projection.1", dim * 6, dim)
    if has_img:
        ln1d("img_emb.proj.0", 1280)                 # LayerNorm(1280)
        linear("img_emb.proj.1", 1280, 1280)         # Linear(1280→1280)
        linear("img_emb.proj.3", dim, 1280)          # Linear(1280→dim)
        ln1d("img_emb.proj.4", dim)                  # LayerNorm(dim)
        if cfg.get("pos_embed_seq_len"):
            keys["img_emb.emb_pos"] = [1, cfg["pos_embed_seq_len"], 1280]

    for n in range(num_layers):
        pre = f"blocks.{n}."
        for part in ("q", "k", "v", "o"):
            linear(pre + f"self_attn.{part}", dim, dim)
        keys[pre + "self_attn.norm_q.weight"] = [dim]
        keys[pre + "self_attn.norm_k.weight"] = [dim]
        for part in ("q", "k", "v", "o"):
            linear(pre + f"cross_attn.{part}", dim, dim)
        keys[pre + "cross_attn.norm_q.weight"] = [dim]
        keys[pre + "cross_attn.norm_k.weight"] = [dim]
        if has_img:
            linear(pre + "cross_attn.k_img", dim, dim)
            linear(pre + "cross_attn.v_img", dim, dim)
            keys[pre + "cross_attn.norm_k_img.weight"] = [dim]
        ln1d(pre + "norm3", dim)                     # elementwise_affine=True
        linear(pre + "ffn.0", ffn_dim, dim)
        linear(pre + "ffn.2", dim, ffn_dim)
        keys[pre + "modulation"] = [1, 6, dim]

    keys["head.modulation"] = [1, 2, dim]
    linear("head.head", out_dim * 1 * 2 * 2, dim)
    return keys


def default_ti2v_5b_cfg() -> dict:
    """DiffSynth 注册表（model_configs.py）中 Wan2.2-TI2V-5B 的架构，含 diffusers 侧字段默认值。"""
    return {
        "dim": 3072, "ffn_dim": 14336, "in_dim": 48, "out_dim": 48,
        "text_dim": 4096, "freq_dim": 256, "num_layers": 30, "num_heads": 24,
        "eps": 1e-6, "patch_size": [1, 2, 2],
        "cross_attn_norm": True, "qk_norm": "rms_norm_across_heads",
        "has_image_input": False, "image_dim": None, "added_kv_proj_dim": None,
        "pos_embed_seq_len": None,
    }


def load_cfg_from_config_json(path: str) -> dict:
    """读取 diffusers config.json 并归一化为生成器 cfg（缺字段回退默认）。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = default_ti2v_5b_cfg()
    mapping = {
        "num_attention_heads": "num_heads",
        "attention_head_dim": "attention_head_dim",
        "in_channels": "in_dim", "out_channels": "out_dim",
        "freq_dim": "freq_dim", "ffn_dim": "ffn_dim", "num_layers": "num_layers",
        "text_dim": "text_dim", "cross_attn_norm": "cross_attn_norm",
        "qk_norm": "qk_norm", "eps": "eps",
        "image_dim": "image_dim", "added_kv_proj_dim": "added_kv_proj_dim",
        "pos_embed_seq_len": "pos_embed_seq_len",
    }
    for src_k, dst_k in mapping.items():
        if src_k in raw:
            cfg[dst_k] = raw[src_k]
    if "patch_size" in raw:
        cfg["patch_size"] = raw["patch_size"]
    # inner_dim 由 heads*dim_head 决定（400 旧写法里可能直接给 inner_dim）
    if "attention_head_dim" in raw and cfg["num_heads"]:
        cfg["dim"] = int(cfg["num_heads"]) * int(raw["attention_head_dim"])
    elif "inner_dim" in raw:
        cfg["dim"] = int(raw["inner_dim"])
    cfg["has_image_input"] = bool(cfg.get("image_dim"))
    return cfg


# ---------------------------------------------------------------------------
# 4. DiffSynth 模型哈希复核（与 diffsynth/core/loader/file.py 同算法）
# ---------------------------------------------------------------------------

DIFFSYNTH_REGISTRY_HASH_TI2V_5B = "1f5ab7703c6fc803fdded85ff040c316"


def compute_diffsynth_model_hash(keys_shapes: dict) -> str:
    """复刻 DiffSynth `hash_model_file`（with_shape=True）的 md5 算法。

    注意：DiffSynth `convert_keys_dict_to_single_str` 在 with_shape=True 时，
    对每个扁平键会**同时**追加 ``key:形状`` 与裸 ``key`` 两个条目（file.py 第 122-125 行），
    本函数逐行复刻这一行为，否则哈希对不上注册表。
    """
    items = []
    for k, shape in keys_shapes.items():
        s = "_".join(str(int(x)) for x in shape)
        items.append(f"{k}:{s}")
        items.append(k)  # DiffSynth 原样追加裸键
    items.sort()
    return hashlib.md5(",".join(items).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 5. 输入读取 / 分片写入
# ---------------------------------------------------------------------------

def find_input_files(input_dir: str) -> list:
    """在输入目录（或其 transformer/ 子目录）里找 diffusion_pytorch_model*.safetensors。"""
    candidates = [input_dir]
    if os.path.isdir(input_dir):
        sub = os.path.join(input_dir, "transformer")
        if os.path.isdir(sub):
            candidates.append(sub)
    files = []
    for base in candidates:
        files.extend(sorted(glob.glob(os.path.join(base, "diffusion_pytorch_model*.safetensors"))))
    if not files:
        raise FileNotFoundError(
            f"在 {input_dir}（含 transformer/ 子目录）中未找到 diffusion_pytorch_model*.safetensors"
        )
    return files


def read_keys_shapes(files: list) -> "OrderedDict[str, list[int]]":
    """只读 safetensors 头部（键+形状），不加载 tensor。"""
    from safetensors import safe_open
    keys = OrderedDict()
    for f in files:
        with safe_open(f, framework="np", device="cpu") as st:
            for k in st.keys():
                keys[k] = list(st.get_slice(k).get_shape())
    return keys


def load_state_dict(files: list, dtype=None) -> dict:
    import torch
    from safetensors import safe_open
    state = {}
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as st:
            for k in st.keys():
                t = st.get_tensor(k)
                if dtype is not None:
                    t = t.to(dtype)
                state[k] = t
    return state


def _shard_tensors(items, max_bytes):
    """贪心装箱：把 (name, tensor, nbytes) 序列切分为 ≤max_bytes 的桶；超大 tensor 单独成桶。"""
    shards, cur, cur_sz = [], [], 0
    for name, t, sz in items:
        if sz > max_bytes:
            if cur:
                shards.append(cur)
                cur, cur_sz = [], 0
            shards.append([(name, t, sz)])
            continue
        if cur and cur_sz + sz > max_bytes:
            shards.append(cur)
            cur, cur_sz = [], 0
        cur.append((name, t, sz))
        cur_sz += sz
    if cur:
        shards.append(cur)
    return shards


def write_sharded_safetensors(state_dict: dict, out_dir: str, max_shard_gb: float) -> list:
    import torch
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    max_bytes = int(max_shard_gb * (1024 ** 3))
    items = []
    for name, t in state_dict.items():
        nbytes = t.numel() * t.element_size()
        items.append((name, t, nbytes))
    shards = _shard_tensors(items, max_bytes)
    written = []
    for i, shard in enumerate(shards):
        fname = f"diffusion_pytorch_model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        fpath = os.path.join(out_dir, fname)
        save_file(OrderedDict((n, t) for n, t, _ in shard), fpath, metadata={"format": "pt"})
        written.append(fname)
    return written


# ---------------------------------------------------------------------------
# 6. 报告 / 转换主流程
# ---------------------------------------------------------------------------

def build_mapping_report(src_keys_shapes: dict, cfg: dict, strict: bool = False):
    """逐键映射 + 未对齐清点。返回 (records, target_keys_shapes, summary)。"""
    records = []       # [{src, status, target, note, shape}]
    target_keys_shapes = OrderedDict()
    summary = {"mapped": 0, "skip": 0, "unmapped": 0}
    unmapped = []
    for src, shape in src_keys_shapes.items():
        status, target, note = classify_key(src)
        rec = {"src": src, "status": status, "target": target, "note": note, "shape": shape}
        records.append(rec)
        summary[status] += 1
        if status == "mapped":
            target_keys_shapes[target] = shape
        elif status == "unmapped":
            unmapped.append(src)
    if strict and unmapped:
        raise RuntimeError(
            f"存在 {len(unmapped)} 个未对齐键（--strict）:\n" + "\n".join(f"  - {k}" for k in unmapped)
        )
    return records, target_keys_shapes, summary, unmapped


def run_dry_run(input_dir, verbose=False):
    print("=" * 78)
    print("diffusers → DiffSynth 键映射词表（WanTransformer3DModel → WanModel）")
    print("=" * 78)
    print("\n[顶层键]")
    for src, dst in _TOP_MAP.items():
        print(f"  {src:<64} → {dst}")
    print("\n[块内键（blocks.N.<rest>）]")
    for src, dst in _BLOCK_REST_MAP.items():
        print(f"  {src:<56} → {dst}")
    print("\n[已知跳过键]（源侧本无权重/缓冲区，出现即跳过）")
    for prefix, reason in _SKIP_REASONS.items():
        print(f"  {prefix:<24} {reason}")
    print("\n[不支持] dual-expert（transformer_2 / dit2）")

    if input_dir:
        files = find_input_files(input_dir)
        print(f"\n[实际输入] 共 {len(files)} 个分片：")
        for f in files:
            print(f"  {f}")
        cfg = default_ti2v_5b_cfg()
        cfg_path = os.path.join(os.path.dirname(files[0]), "config.json")
        if os.path.isfile(cfg_path):
            cfg = load_cfg_from_config_json(cfg_path)
            print(f"[config] 使用 {cfg_path}")
        print(f"[config] dim={cfg['dim']} layers={cfg['num_layers']} heads={cfg['num_heads']} "
              f"ffn={cfg['ffn_dim']} cross_attn_norm={cfg['cross_attn_norm']} "
              f"has_image_input={cfg['has_image_input']}")
        keys_shapes = read_keys_shapes(files)
        records, target, summary, unmapped = build_mapping_report(keys_shapes, cfg, strict=False)
        print(f"\n[对齐报告] 总键数 {len(records)}：mapped={summary['mapped']} "
              f"skip={summary['skip']} unmapped={summary['unmapped']}")
        for rec in records:
            if rec["status"] != "mapped":
                print(f"  [{'SKIP' if rec['status'] == 'skip' else 'UNMAPPED':>8}] {rec['src']}  ({rec['note']})")
        if verbose:
            for rec in records:
                if rec["status"] == "mapped":
                    print(f"  {rec['src']:<64} → {rec['target']}")
        print("\n[目标键形状核对]")
        expected = gen_diffsynth_keys(cfg)
        missing = sorted(set(expected) - set(target))
        extra = sorted(set(target) - set(expected))
        print(f"  映射产物 {len(target)} 键 vs DiffSynth WanModel 期望 {len(expected)} 键")
        if missing:
            print(f"  ✗ 缺失目标键（{len(missing)}）：\n    " + "\n    ".join(missing))
        if extra:
            print(f"  ✗ 多余目标键（{len(extra)}）：\n    " + "\n    ".join(extra))
        if not missing and not extra:
            neq = [k for k in expected if expected[k] != target.get(k)]
            if neq:
                print(f"  ✗ 形状不一致（{len(neq)}）：\n    " + "\n    ".join(neq))
            else:
                print("  ✓ 目标键集合与形状完全覆盖 DiffSynth WanModel 期望键集")
        h = compute_diffsynth_model_hash(target)
        print(f"\n[模型哈希复核] computed={h}\n"
              f"                 registry={DIFFSYNTH_REGISTRY_HASH_TI2V_5B} (Wan2.2-TI2V-5B)")
        print("  " + ("✓ 一致，DiffSynth 可直接按 hash 识别为 wan_video_dit"
                      if h == DIFFSYNTH_REGISTRY_HASH_TI2V_5B else
                      "✗ 不一致：请先核对注册表/源权重差异（见 conversion_report.json）"))
    return 0


def run_convert(args):
    import torch
    files = find_input_files(args.input_dir)
    cfg_path = os.path.join(os.path.dirname(files[0]), "config.json")
    cfg = default_ti2v_5b_cfg()
    if os.path.isfile(cfg_path):
        cfg = load_cfg_from_config_json(cfg_path)
        print(f"[config] 读自 {cfg_path}")
    print(f"[dtype] {args.dtype}（默认 auto=保持源 dtype）")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}.get(args.dtype)
    print(f"[加载] {len(files)} 个分片 ...")
    state = load_state_dict(files, dtype=dtype)

    records, target_keys_shapes, summary, unmapped = build_mapping_report(
        {k: list(v.shape) for k, v in state.items()}, cfg, strict=args.strict)
    print(f"[映射] total={len(records)} mapped={summary['mapped']} "
          f"skip={summary['skip']} unmapped={summary['unmapped']}")
    if summary["unmapped"] and not args.strict:
        print("  [警告] 以下键未映射，将被丢弃（如需失败请加 --strict）：")
        for k in unmapped:
            print(f"    - {k}")

    # 目标键全覆盖断言（缺任何 DiffSynth 期望键都视为不完整）
    expected = gen_diffsynth_keys(cfg)
    missing = sorted(set(expected) - set(target_keys_shapes))
    if missing:
        raise RuntimeError(f"映射产物缺少 {len(missing)} 个 DiffSynth 目标键：\n" + "\n".join(missing))

    converted = OrderedDict()
    for rec in records:
        if rec["status"] == "mapped":
            converted[rec["target"]] = state[rec["src"]]
    del state

    out_dir = args.output_dir or (args.input_dir.rstrip("/") + "_diffsynth")
    print(f"[写入] {out_dir}")
    shard_names = write_sharded_safetensors(converted, out_dir, args.max_shard_size_gb)

    # model_index.json（说明性元数据）
    model_index = {
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "model_hash_expected": DIFFSYNTH_REGISTRY_HASH_TI2V_5B,
        "extra_kwargs": {k: cfg.get(k) for k in (
            "dim", "in_dim", "ffn_dim", "out_dim", "text_dim", "freq_dim", "eps",
            "patch_size", "num_heads", "num_layers", "has_image_input",
            "seperated_timestep", "require_clip_embedding", "require_vae_embedding",
            "fuse_vae_embedding_in_latents") if cfg.get(k) is not None},
        "source": "converted from diffusers Wan2.2-TI2V-5B-Diffusers "
                  "(WanTransformer3DModel) via convert_wan_5b_bridge.py",
        "weight_files": shard_names,
        "note": "DiffSynth 通过 cfgs/model_configs.py 的 key+shape 哈希识别模型类型；"
                "此处哈希=conversion_report.json 中的 computed_hash。",
    }
    with open(os.path.join(out_dir, "model_index.json"), "w", encoding="utf-8") as f:
        json.dump(model_index, f, ensure_ascii=False, indent=2)

    # 留存源 config.json（溯源）
    if os.path.isfile(cfg_path):
        src_cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
        with open(os.path.join(out_dir, "diffusers_config.json"), "w", encoding="utf-8") as f:
            json.dump(src_cfg, f, ensure_ascii=False, indent=2)

    # 转换报告
    computed_hash = compute_diffsynth_model_hash(target_keys_shapes)
    report = {
        "summary": summary,
        "num_layers": cfg.get("num_layers"),
        "computed_hash": computed_hash,
        "registry_hash_ti2v_5b": DIFFSYNTH_REGISTRY_HASH_TI2V_5B,
        "hash_match": computed_hash == DIFFSYNTH_REGISTRY_HASH_TI2V_5B,
        "weight_files": shard_names,
        "records": records,
    }
    with open(os.path.join(out_dir, "conversion_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[完成] 输出 {len(shard_names)} 个分片 + model_index.json + conversion_report.json")
    print(f"[哈希] computed={computed_hash}  registry={DIFFSYNTH_REGISTRY_HASH_TI2V_5B} "
          f"{'✓ 一致' if computed_hash == DIFFSYNTH_REGISTRY_HASH_TI2V_5B else '✗ 不一致'}")
    if args.check_hash and computed_hash != DIFFSYNTH_REGISTRY_HASH_TI2V_5B:
        raise RuntimeError(
            "哈希与 DiffSynth 注册表不一致：请核对源权重键集/形状是否与 Wan2.2-TI2V-5B 官方一致，"
            "或在 diffsynth/configs/model_configs.py 中注册新条目。")
    return 0


# ---------------------------------------------------------------------------
# 7. 静态冒烟测试（纯函数 + 可选真实 diffusers 小模型）
# ---------------------------------------------------------------------------

def run_self_test():
    print("=" * 78)
    print("静态冒烟：映射规则对 DiffSynth 目标键集的全覆盖断言")
    print("=" * 78)
    failures = []

    cases = [
        ("TI2V-5B（无图像分支）", {**default_ti2v_5b_cfg()}),
        ("TI2V-5B 小规模（无图像分支）", {
            **default_ti2v_5b_cfg(), "dim": 32, "ffn_dim": 64, "in_dim": 8, "out_dim": 8,
            "text_dim": 16, "freq_dim": 8, "num_layers": 2, "num_heads": 4, "eps": 1e-6}),
        ("I2V 变体（有图像分支）", {
            **default_ti2v_5b_cfg(), "dim": 32, "ffn_dim": 64, "in_dim": 8, "out_dim": 8,
            "text_dim": 16, "freq_dim": 8, "num_layers": 2, "num_heads": 4,
            "has_image_input": True, "image_dim": 1280, "added_kv_proj_dim": 32,
            "pos_embed_seq_len": 4}),
    ]

    for name, cfg in cases:
        src = gen_diffusers_keys(cfg)
        dst_expected = gen_diffsynth_keys(cfg)
        mapped = OrderedDict()
        unmapped = []
        for k in src:
            status, target, note = classify_key(k)
            if status == "mapped":
                mapped[target] = src[k]
            elif status == "unmapped":
                unmapped.append(k)
        missing = sorted(set(dst_expected) - set(mapped))
        extra = sorted(set(mapped) - set(dst_expected))
        shape_bad = sorted(k for k in dst_expected
                           if k in mapped and mapped[k] != dst_expected[k])
        ok = (not unmapped) and (not missing) and (not extra) and (not shape_bad)
        print(f"\n[{name}]")
        print(f"  源键 {len(src)} → 映射 {len(mapped)} / 期望 {len(dst_expected)}  "
              f"{'✓ PASS' if ok else '✗ FAIL'}")
        if unmapped:
            print("  unmapped: " + ", ".join(unmapped))
        if missing:
            print("  missing:  " + ", ".join(missing))
        if extra:
            print("  extra:    " + ", ".join(extra))
        if shape_bad:
            print("  shape错:  " + ", ".join(shape_bad))
        if not ok:
            failures.append(name)

    # 双专家防护
    for bad in ("transformer_2.blocks.0.attn1.to_q.weight", "dit2.blocks.0.ffn.net.0.proj.weight"):
        status, target, _ = classify_key(bad)
        if status != "unmapped":
            failures.append(f"dual-expert 防护失效: {bad}")
            print(f"\n[双专家防护] ✗ FAIL: {bad} 未被拦截")
        else:
            print(f"[双专家防护] ✓ {bad} 被正确拦截")

    # 可选：真实 diffusers WanTransformer3DModel（极小配置）键比对
    try:
        import torch
        from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
        tiny = {**default_ti2v_5b_cfg(), "dim": 32, "ffn_dim": 64, "in_dim": 8, "out_dim": 8,
                "text_dim": 16, "freq_dim": 8, "num_layers": 2, "num_heads": 4}
        model = WanTransformer3DModel(
            patch_size=(1, 2, 2), num_attention_heads=tiny["num_heads"], attention_head_dim=8,
            in_channels=tiny["in_dim"], out_channels=tiny["out_dim"], text_dim=tiny["text_dim"],
            freq_dim=tiny["freq_dim"], ffn_dim=tiny["ffn_dim"], num_layers=tiny["num_layers"],
            cross_attn_norm=True, qk_norm="rms_norm_across_heads", eps=1e-6,
            image_dim=None, added_kv_proj_dim=None, rope_max_seq_len=64, pos_embed_seq_len=None,
        )
        real_keys = set(model.state_dict().keys())
        gen_keys = set(gen_diffusers_keys(tiny).keys())
        only_real = sorted(real_keys - gen_keys)
        only_gen = sorted(gen_keys - real_keys)
        if only_real or only_gen:
            failures.append("真实 diffusers WanTransformer3DModel 键集与生成器不一致")
            print(f"\n[真实 diffusers 模型] ✗ FAIL: 生成器 vs 实际 state_dict 有出入")
            if only_real:
                print("  实际有/生成器无: " + ", ".join(only_real))
            if only_gen:
                print("  生成器有/实际无: " + ", ".join(only_gen))
        else:
            print(f"\n[真实 diffusers 模型] ✓ 极小 WanTransformer3DModel 的 state_dict "
                  f"（{len(real_keys)} 键）与生成器完全一致")
            # 再把真实键全部过一遍映射，确认无 unmapped
            bad = [k for k in real_keys if classify_key(k)[0] == "unmapped"]
            if bad:
                failures.append("真实键存在 unmapped")
                print("  ✗ unmapped: " + ", ".join(bad))
            else:
                print(f"  ✓ {len(real_keys)} 个真实键全部可映射")
    except Exception as e:  # noqa: BLE001
        print(f"\n[真实 diffusers 模型] 跳过（本环境无法 import/实例化：{e}）")

    print("\n" + "=" * 78)
    if failures:
        print(f"自检失败: {len(failures)} 项 —— " + "; ".join(failures))
        return 1
    print("自检全部通过 ✓")
    return 0


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Wan2.2-TI2V-5B 权重桥：diffusers WanTransformer3DModel → DiffSynth WanModel 原格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法")[1] if "用法" in __doc__ else None,
    )
    parser.add_argument("--input-dir", type=str, default=None,
                        help="diffusers transformer 权重目录（含 diffusion_pytorch_model*.safetensors + config.json）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录（默认 <input-dir>_diffsynth）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印词表 + （若给 --input-dir）键对齐报告，不读 tensor、不写文件")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "bf16", "fp16", "fp32"],
                        help="输出 dtype（默认 auto=保持源 dtype；集群/训练建议 bf16）")
    parser.add_argument("--max-shard-size-gb", type=float, default=4.0,
                        help="safetensors 分片目标大小（GB），参考 1.3B 的 diffusion_pytorch_model-*.safetensors 分片方式")
    parser.add_argument("--strict", action="store_true",
                        help="遇未对齐键即失败（否则警告并丢弃）")
    parser.add_argument("--check-hash", action="store_true",
                        help="转换后校验 key+shape 哈希是否等于 DiffSynth 注册表值（1f5ab770...）")
    parser.add_argument("--self-test", action="store_true",
                        help="静态冒烟：断言映射规则覆盖 DiffSynth 全部目标键（纯函数，无需权重）")
    parser.add_argument("--verbose", action="store_true", help="dry-run 时逐条打印 mapped 键")
    args = parser.parse_args()

    if args.self_test:
        sys.exit(run_self_test())
    if args.dry_run:
        sys.exit(run_dry_run(args.input_dir, verbose=args.verbose))
    if not args.input_dir:
        parser.error("转换模式需要 --input-dir（或使用 --dry-run / --self-test）")
    sys.exit(run_convert(args))


if __name__ == "__main__":
    main()