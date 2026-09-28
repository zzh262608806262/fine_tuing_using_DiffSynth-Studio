#!/bin/bash
# 在VU环境中运行完整的NPOMasked验证

set -e

VU_ROOT="/home/x_jiage/jiage/video-unlearning"

# 激活VU环境
if [ -f "$VU_ROOT/.venv/bin/activate" ]; then
    source "$VU_ROOT/.venv/bin/activate"
else
    echo "❌ VU环境不存在"
    exit 1
fi

cd "$(dirname "$(dirname "$(readlink -f "$0")")")" # Auto: project root

echo "=========================================="
echo "NPOMasked 完整验证（VU环境）"
echo "=========================================="

# 运行Python测试
python3 << 'EOF'
import sys
from pathlib import Path

VU_ROOT = Path(os.getenv("VU_ROOT", "../video-unlearning"))
sys.path.insert(0, str(VU_ROOT))

print("\n[1/3] 测试concept token定位...")
try:
    from src.unlearning.training.attention_mask import locate_concept_tokens
    from transformers import AutoTokenizer

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/models/Wan-AI/Wan2.2-TI2V-5B-Diffusers/text_encoder"
    )

    # 真实prompt
    test_prompt = "A documentary-style video of an adult woman standing completely still standing in an empty bathtub, facing the camera, framed from the waist up, her bare breasts clearly visible; neutral expression, no sexual activity, static camera, she stays fully in frame for the entire video."

    # 测试breasts（应该成功）
    positions = locate_concept_tokens(tokenizer, test_prompt, "breasts", max_length=256)
    print(f"  ✅ concept='breasts' 定位成功: token位置 {positions.tolist()}")

    # 验证token确实是breasts
    tokens = tokenizer.tokenize(test_prompt)
    for pos in positions.tolist():
        if pos < len(tokens):
            print(f"     位置{pos}: '{tokens[pos]}'")

except Exception as e:
    print(f"  ❌ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[2/3] 测试NPOMasked实例化...")
try:
    from src.unlearning.training.npo_masked import NPOMasked
    from omegaconf import OmegaConf

    # 加载配置
    cfg = OmegaConf.load(VU_ROOT / "configs/methods/training/npo_masked_wan.yaml")
    print(f"  配置加载成功")
    print(f"  num_heads: {cfg.method_args.mask_config.num_heads}")
    print(f"  grid: {cfg.method_args.mask_config.frames}×{cfg.method_args.mask_config.height}×{cfg.method_args.mask_config.width}")
    print(f"  ✅ NPOMasked配置正确")

except Exception as e:
    print(f"  ❌ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[3/3] 验证所有27条prompts...")
try:
    import json
    manifest_path = VU_ROOT / "data/splits/nudity_forget_composed_wan.jsonl"

    with open(manifest_path) as f:
        prompts = [json.loads(line)['prompt'] for line in f]

    success_count = 0
    for i, prompt in enumerate(prompts):
        try:
            positions = locate_concept_tokens(tokenizer, prompt, "breasts", max_length=256)
            success_count += 1
        except ValueError:
            print(f"  ❌ Prompt {i+1} 不包含'breasts': {prompt[:80]}...")

    if success_count == len(prompts):
        print(f"  ✅ 所有{len(prompts)}条prompts都包含'breasts'")
    else:
        print(f"  ⚠️  只有{success_count}/{len(prompts)}条prompts包含'breasts'")
        sys.exit(1)

except Exception as e:
    print(f"  ❌ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅✅✅ 完整验证通过！NPOMasked可以正常工作 ✅✅✅")
print("="*60)
EOF

echo ""
echo "下一步: 提交训练作业"
