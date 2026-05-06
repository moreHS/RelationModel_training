"""
Merge a Unsloth-trained LoRA adapter into its base model and save a single
16bit checkpoint that vLLM can load directly.

Why this exists:
  vLLM v0.18.x / v0.19.x does NOT support LoRA on Gemma4ForConditionalGeneration
  (vLLM Issue #39246, May 2026 still open). vLLM's official workaround is to
  merge the adapter into base weights and serve the merged checkpoint as a
  standalone model.

Usage:
  python3 -m scripts.merge_lora_to_base \
      --lora_path /app/train_result/gemma4-e4b_sft_v2_rank8_alpha16_lr0.0001_batch8_ep3_0.05

  # Custom output path
  python3 -m scripts.merge_lora_to_base \
      --lora_path .../some_adapter_dir \
      --out_path .../merged_target_dir

After merge, point gemma4_inference/generate_eval_data_config.yaml at:
  model.name        = <out_path>
  inference.lora_path = null

Notes:
  - Uses Unsloth's FastLanguageModel.from_pretrained on the adapter dir, which
    auto-resolves the base model + applies the adapter weights.
  - save_pretrained_merged(..., save_method="merged_16bit") writes full bf16
    weights (~8-9GB for Gemma4-E4B) into out_path.
  - Refuses to overwrite an existing out_path to prevent accidental data loss.
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--lora_path",
        required=True,
        help="Path to the trained LoRA adapter directory (sft_gemma4 OUTPUT_PATH).",
    )
    parser.add_argument(
        "--out_path",
        default=None,
        help="Output directory for the merged 16bit checkpoint. "
             "Default: <lora_path>_merged_16bit",
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=8192,
        help="max_seq_length for FastLanguageModel.from_pretrained (default: 8192).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.lora_path):
        print(f"❌ lora_path not found or not a directory: {args.lora_path}", file=sys.stderr)
        sys.exit(2)

    out_path = args.out_path or (args.lora_path.rstrip("/") + "_merged_16bit")
    if os.path.exists(out_path):
        print(
            f"⚠️ Output path already exists: {out_path}\n"
            f"   Refusing to overwrite. Either delete it first or pass --out_path "
            f"to a fresh location.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Lazy imports so `--help` doesn't pay the unsloth import cost.
    import torch
    from unsloth import FastLanguageModel

    print(f"🔄 Loading adapter from: {args.lora_path}")
    print(f"   (Unsloth will auto-resolve and download the base model if not cached.)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.lora_path,
        max_seq_length=args.maxlen,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
    )

    print(f"💾 Merging LoRA into base, saving 16bit weights to: {out_path}")
    model.save_pretrained_merged(
        out_path,
        tokenizer,
        save_method="merged_16bit",
    )

    # Quick listing so the user can eyeball the result without a separate ls call.
    if os.path.isdir(out_path):
        files = sorted(os.listdir(out_path))
        print(f"✅ Done. {len(files)} files written to {out_path}:")
        for f in files[:20]:
            print(f"   - {f}")
        if len(files) > 20:
            print(f"   ... ({len(files) - 20} more)")
    else:
        print(f"❌ Expected output directory missing: {out_path}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
