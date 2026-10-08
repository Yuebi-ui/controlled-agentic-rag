#!/usr/bin/env python3
"""
Phase 0 Step 1: 合并四层 LoRA Adapter → 完整 BF16 模型

训练链：
  Base Qwen3-8B
    └─ SFT LoRA  → saves/lora/sft/
         └─ ORPO v2 → saves/lora/orpo-v2/
              └─ ORPO v3 → saves/lora/orpo-v3/
                   └─ ORPO v4 → saves/lora/orpo-v4/

合并策略：逐层 merge_and_unload，每层保存到临时目录后重新加载。
"""
import os, sys, time, argparse, shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Project paths
BASE_MODEL_PATH = "/root/autodl-tmp/RAG/models/Qwen3-8B/"
PROJECT_DIR = "/root/autodl-tmp/rag-server"
SAVES_DIR = os.path.join(PROJECT_DIR, "LLaMA-Factory-main/saves/qwen3-8b/lora")

ADAPTER_CHAIN = [
    ("sft",     os.path.join(SAVES_DIR, "sft")),
    ("orpo-v2", os.path.join(SAVES_DIR, "orpo-v2")),
    ("orpo-v3", os.path.join(SAVES_DIR, "orpo-v3")),
    ("orpo-v4", os.path.join(SAVES_DIR, "orpo-v4")),
]

TEMP_DIR = os.path.join(PROJECT_DIR, "LLaMA-Factory-main/output/_merge_temp")


def remove_peft_config(model):
    """清除 merge 后残留的 peft_config"""
    if hasattr(model, 'peft_config'):
        delattr(model, 'peft_config')
    # 递归检查子模块
    for module in model.modules():
        if hasattr(module, 'peft_config'):
            delattr(module, 'peft_config')


def merge_one_step(model_path_or_obj, adapter_path, output_dir, tokenizer=None):
    """
    加载模型 + adapter → merge_and_unload → 保存

    使用磁盘中转避免 peft_config 残留问题
    """
    print(f"    Loading model...")
    if isinstance(model_path_or_obj, str):
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_obj,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cpu",
        )
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path_or_obj, trust_remote_code=True
            )
    else:
        model = model_path_or_obj

    print(f"    Loading adapter: {os.path.basename(adapter_path)}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print(f"    Merging...")
    model = model.merge_and_unload()
    remove_peft_config(model)

    print(f"    Saving to temp dir...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    if tokenizer:
        tokenizer.save_pretrained(output_dir)

    # 释放内存
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Merge ORPO v4 LoRA adapter chain")
    parser.add_argument("--output", default=os.path.join(
        PROJECT_DIR, "LLaMA-Factory-main/output/qwen3_orpo_v4_bf16"),
                        help="Output directory")
    parser.add_argument("--skip", type=str, nargs="*", default=[],
                        help="Adapters to skip")
    args = parser.parse_args()

    chain = [(n, p) for n, p in ADAPTER_CHAIN if n not in args.skip]

    print("=" * 60)
    print("ORPO v4 Adapter Merge (disk-intermediate)")
    print("=" * 60)
    print(f"Base model:    {BASE_MODEL_PATH}")
    print(f"Adapter chain: {' → '.join(n for n, _ in chain)}")
    print(f"Output:        {args.output}")
    print(f"Device:        CPU")
    print()

    # Check paths
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"❌ Base model not found: {BASE_MODEL_PATH}")
        sys.exit(1)
    missing = [n for n, p in chain if not os.path.exists(p)]
    if missing:
        print(f"❌ Missing adapters: {missing}")
        sys.exit(1)

    # Clean up temp dir
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    current = BASE_MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    for i, (name, adapter_path) in enumerate(chain):
        t0 = time.time()
        step_dir = os.path.join(TEMP_DIR, f"step_{i}_{name}")

        if i == len(chain) - 1:
            # Last step: save to final output
            step_dir = args.output

        print(f"\n[{i+1}/{len(chain)}] Merging {name}...")
        current = merge_one_step(current, adapter_path, step_dir, tokenizer)
        print(f"  ✓ {name} done ({time.time() - t0:.0f}s)")

        # Clean previous step to save disk space
        if i > 0:
            prev_dir = os.path.join(TEMP_DIR, f"step_{i-1}_{chain[i-1][0]}")
            if os.path.exists(prev_dir):
                shutil.rmtree(prev_dir)
                print(f"  (cleaned {chain[i-1][0]} temp)")

    # Clean up any remaining temp dirs
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    print(f"\n✓ 合并完成: {args.output}")
    # Show output size
    total_size = sum(
        os.path.getsize(os.path.join(args.output, f))
        for f in os.listdir(args.output)
        if os.path.isfile(os.path.join(args.output, f))
    )
    print(f"  模型大小: {total_size / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
