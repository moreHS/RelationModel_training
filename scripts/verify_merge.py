"""
Verify that a merged checkpoint actually contains the LoRA delta (i.e., the
adapter weights were not silently dropped during save).

Background — Unsloth Issue #1352: when `save_pretrained_merged(..., "merged_16bit")`
is called on certain vision/multimodal models, the saved tensors can end up
identical to the base model (LoRA delta lost). For Gemma4 (multimodal class)
we want a quick sanity check before using the merged model in evaluation.

Method:
  Load the same parameter (e.g., a q_proj weight in an early decoder layer)
  from both the base and the merged checkpoints, compare their values. If
  the mean absolute diff is essentially zero, the merge silently failed.

Usage:
  python3 -m scripts.verify_merge \
      --base_dir /app/host/models/gemma4-E4B-it \
      --merged_dir /app/train_result/..._merged_16bit

Exits non-zero on suspected failure so it can be chained in shell scripts.
"""
import argparse
import os
import sys


def find_weight(directory: str, key: str):
    """Search every model-*.safetensors shard in `directory` for a tensor named `key`."""
    from safetensors import safe_open

    candidates = sorted(
        f for f in os.listdir(directory)
        if f.endswith(".safetensors")
    )
    for fname in candidates:
        path = os.path.join(directory, fname)
        with safe_open(path, framework="pt") as h:
            keys = h.keys()
            if key in keys:
                return h.get_tensor(key)
    return None


def list_all_keys(directory: str):
    """Return the union of every tensor key across every safetensors shard."""
    from safetensors import safe_open
    all_keys = set()
    for fname in sorted(f for f in os.listdir(directory) if f.endswith(".safetensors")):
        path = os.path.join(directory, fname)
        with safe_open(path, framework="pt") as h:
            all_keys.update(h.keys())
    return all_keys


def auto_discover_key(base_dir: str, merged_dir: str):
    """
    Find a weight key that exists in BOTH directories and is suitable for diff
    sanity (q_proj / gate_proj in an early decoder layer). Returns None if nothing
    matches the heuristic.
    """
    base_keys = list_all_keys(base_dir)
    merged_keys = list_all_keys(merged_dir)
    common = base_keys & merged_keys
    # Prefer q_proj in the lowest-numbered layer; fall back to gate_proj, k_proj, etc.
    preferred_substrings = (
        ".self_attn.q_proj.weight",
        ".mlp.gate_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
    )
    for needle in preferred_substrings:
        # Pick the smallest layer index match for determinism
        matches = sorted(k for k in common if needle in k and ".layers." in k)
        if matches:
            # Prefer "layers.0." if present, else first match
            layer0 = [k for k in matches if ".layers.0." in k]
            return (layer0 or matches)[0]
    # Last resort: any common key that contains "layers" and "weight"
    matches = sorted(k for k in common if ".layers." in k and k.endswith(".weight"))
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base_dir", required=True, help="Original base model directory.")
    parser.add_argument("--merged_dir", required=True, help="Merged 16bit checkpoint directory.")
    parser.add_argument(
        "--keys",
        nargs="+",
        default=[
            # Gemma4 ForConditionalGeneration nests the LM under language_model.
            "language_model.layers.0.self_attn.q_proj.weight",
            "language_model.layers.0.mlp.gate_proj.weight",
            # Fallback names for non-multimodal Gemma class.
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ],
        help="Weight keys to compare. The first one found in BOTH dirs is used.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="Mean absolute diff below this is considered 'no change' → suspected merge failure.",
    )
    args = parser.parse_args()

    for d in (args.base_dir, args.merged_dir):
        if not os.path.isdir(d):
            print(f"❌ Directory not found: {d}", file=sys.stderr)
            sys.exit(2)

    matched_key = None
    for key in args.keys:
        b = find_weight(args.base_dir, key)
        m = find_weight(args.merged_dir, key)
        if b is not None and m is not None:
            matched_key = key
            base_t, merged_t = b, m
            break
        if b is None and m is None:
            continue
        # Asymmetric — log but keep trying other keys
        print(f"   skip {key!r}: base={'found' if b is not None else 'missing'}, "
              f"merged={'found' if m is not None else 'missing'}")

    if matched_key is None:
        # Explicit candidates didn't match — fall back to auto-discovery.
        print(
            "ℹ️  Hardcoded candidate keys not found in BOTH dirs. "
            "Auto-discovering a suitable weight key...",
            file=sys.stderr,
        )
        discovered = auto_discover_key(args.base_dir, args.merged_dir)
        if discovered is None:
            print(
                "❌ Auto-discovery failed — no common (q_proj / gate_proj / layer) "
                "weight found in BOTH directories.\n"
                "   Pass --keys explicitly with names from your model.\n"
                f"   Tried hardcoded list: {args.keys}",
                file=sys.stderr,
            )
            sys.exit(2)
        matched_key = discovered
        base_t = find_weight(args.base_dir, matched_key)
        merged_t = find_weight(args.merged_dir, matched_key)
        print(f"   auto-discovered: {matched_key}", file=sys.stderr)

    diff = (base_t.float() - merged_t.float()).abs().mean().item()
    base_norm = base_t.float().abs().mean().item()
    rel = diff / max(base_norm, 1e-12)

    print(f"🔍 Compared key: {matched_key}")
    print(f"   shapes: base={tuple(base_t.shape)}, merged={tuple(merged_t.shape)}")
    print(f"   mean |diff|: {diff:.3e}   (relative to base mean abs: {rel:.3e})")

    if diff < args.threshold:
        print(
            f"❌ Mean abs diff {diff:.3e} < threshold {args.threshold:.0e}\n"
            f"   The merged checkpoint looks identical to the base model.\n"
            f"   This is the symptom of Unsloth Issue #1352 — LoRA delta was dropped.\n"
            f"   Falling back to PEFT merge_and_unload() may be required.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"✅ Merge looks valid — LoRA delta is present in the merged checkpoint.")


if __name__ == "__main__":
    main()
