"""
Eval-only preprocessing pipeline.

Reads a single JSONL file, runs it through ONE chosen prompt mode/template,
and saves a flat HuggingFace Dataset (no train/val/test split) ready for
vLLM inference.

Differences vs. training pipeline (generate_sft_training_data_latest.py):
  - Single task, single mode (no mode_ratios, no quotas)
  - No train/val/test split
  - Few-shot is ALWAYS disabled (even for modes that enable it)
  - Output schema splits prompt and ground_truth into separate columns
"""

import os
import orjson
import yaml

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Bypass overlayFS disk space checks inside docker
import datasets.builder
datasets.builder.has_sufficient_disk_space = lambda x, y=".": True

from datasets import Dataset, Features, Value
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from data_preprocessor_utils import (
    DataPreprocessor,
    DataGenerationTask,
    DataGenerationModeConfig,
    PromptCompiler,
    TokenCounter,
    RelationBasedChunker,
    PreprocessInput,
    calculate_template_overhead,
    run_adaptive_chunking,
)


# ==============================================================================
# MODEL FORMAT DETECTION
# Mirrors vllm_model_inference_w_edited_prompts.py so prompt/answer split is
# consistent between preprocessing and inference.
# ==============================================================================
def detect_model_format(model_name: str):
    lower = model_name.lower()
    if "gemma4" in lower or "gemma-4" in lower:
        return "gemma4", "<|turn>model\n", ["<turn|>"]
    if "gemma" in lower:
        return "gemma3", "<start_of_turn>model\n", ["<end_of_turn>"]
    return "qwen", "<|im_start|>assistant\n", ["<|im_end|>"]


def _strip_stop_tokens(text: str, stop_tokens) -> str:
    for tok in stop_tokens:
        text = text.replace(tok, "")
    # Also strip any sibling end tokens that could leak across formats
    for tok in ("<|im_end|>", "<end_of_turn>", "<turn|>"):
        text = text.replace(tok, "")
    return text.strip()


# ==============================================================================
# MAIN
# ==============================================================================
def generate_eval_dataset(config_path: str = "generate_eval_data_config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_path = cfg["data"]["output_path"]
    if os.path.exists(out_path):
        print(f"⚡ Eval dataset already exists: {out_path}")
        print("   Delete it to regenerate.")
        from datasets import load_from_disk
        return load_from_disk(out_path)

    model_name = cfg["model"]["name"]
    model_format, split_tag, stop_tokens = detect_model_format(model_name)
    print(f"🔧 Model format: {model_format} | split_tag: {split_tag!r}")

    # --------------------------------------------------------------------------
    # PHASE 1: Load & Classify
    # --------------------------------------------------------------------------
    print("\n📌 PHASE 1: Loading JSONL & Classifying...")
    raw_data = []
    with open(cfg["data"]["input_path"], "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_data.append(orjson.loads(line))

    # Force string IDs at the source (PyArrow type safety)
    for i, doc in enumerate(raw_data):
        doc["id"] = f"eval_{i}" if "id" not in doc else str(doc["id"])

    dp = DataPreprocessor()
    if not cfg.get("negatives", {}).get("include_no_relationship", False):
        raw_data = dp.remove_negative_relations(raw_data)
    classified = dp.extract_and_classify(raw_data)

    if cfg["task"] in ("NER_BEE_TRUE_ONLY", "BEE_BEE"):
        raise ValueError(
            f"Task '{cfg['task']}' is not supported in the eval pipeline. "
            f"Choose from: NER_NER, NER_BEE."
        )
    task_key = DataGenerationTask[cfg["task"]].value
    entries = classified.get(task_key, [])
    if not entries:
        raise RuntimeError(
            f"No entries classified into task '{task_key}'. "
            f"Available buckets: {[k for k, v in classified.items() if isinstance(v, list) and v]}"
        )
    print(f"   Task '{task_key}': {len(entries)} entries")

    # --------------------------------------------------------------------------
    # PHASE 2: Chunking
    # --------------------------------------------------------------------------
    print(f"\n📌 PHASE 2: Chunking to fit {cfg['training_max_length']} tokens...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tc = TokenCounter(model_path=model_name)
    chunker = RelationBasedChunker()
    pi = PreprocessInput()

    task_enum = DataGenerationTask[cfg["task"]]
    mode_cfg_raw = cfg["modes"][cfg["mode"]] if "modes" in cfg else None
    # mode config can live either under top-level `modes:` (copying training yaml)
    # or be a direct dataclass-compatible dict. We build the dataclass below.
    if mode_cfg_raw is None:
        # Inline construction from known mode names
        _MODE_PRESETS = {
            "system_only":   dict(enable_description=False, few_shot=False, summarize_description=False, reasoning=False),
            "full_detailed": dict(enable_description=True,  few_shot=True,  summarize_description=False, reasoning=False),
            "full_summary":  dict(enable_description=True,  few_shot=True,  summarize_description=True,  reasoning=False),
            "no_fewshot":    dict(enable_description=True,  few_shot=False, summarize_description=False, reasoning=False),
            "reasoning":     dict(enable_description=False, few_shot=False, summarize_description=False, reasoning=True),
        }
        if cfg["mode"] not in _MODE_PRESETS:
            raise ValueError(f"Unknown mode '{cfg['mode']}'. Choose from {list(_MODE_PRESETS)}")
        mode_cfg_raw = _MODE_PRESETS[cfg["mode"]]
    mode_cfg = DataGenerationModeConfig(**mode_cfg_raw)

    if mode_cfg.few_shot:
        print(f"   ⚠️  mode '{cfg['mode']}' has few_shot=True — few-shot will still be SKIPPED.")
        print(f"      If the trained model expects few-shot, format mismatch may degrade metrics.")

    max_len = cfg["training_max_length"]
    yaml_path = cfg["prompt"]["yaml_path"]

    temp_compiler = PromptCompiler(task_enum, mode_cfg, yaml_path, [], cfg.get("seed", 42), model_name, tokenizer)
    overhead = calculate_template_overhead(temp_compiler, tokenizer)
    safe_limit = max_len - overhead - 2000
    print(f"   Template overhead: {overhead} tokens | Safe chunk limit: {safe_limit}")

    chunk_thresholds = cfg["generation"].get("chunk_thresholds", [12, 10, 8, 6, 4])
    num_proc = cfg["generation"].get("num_proc", 8)

    chunk_features = Features({
        "id": Value("string"),
        "input": Value("string"),
        "output": Value("string"),
        "origin_id": Value("string"),
    })

    hf_ds = Dataset.from_list(entries)

    def chunk_batch(batch):
        rows = [dict(zip(batch.keys(), v)) for v in zip(*batch.values())]
        processed = []
        for entry in rows:
            chunks = run_adaptive_chunking(entry, chunker, tc, task_enum, safe_limit, chunk_thresholds)
            formatted = pi.build_input_output(chunks, task_enum)
            for item in formatted:
                processed.append({
                    "id": str(item["id"]),
                    "input": item["input"],
                    "output": item["output"],
                    "origin_id": str(entry["id"]),
                })
        if not processed:
            return {k: [] for k in ["id", "input", "output", "origin_id"]}
        return {k: [r[k] for r in processed] for k in processed[0].keys()}

    chunked_ds = hf_ds.map(
        chunk_batch,
        batched=True,
        num_proc=num_proc,
        remove_columns=hf_ds.column_names,
        features=chunk_features,
        desc=f"Chunking {task_key}",
    )
    print(f"   Chunks: {len(chunked_ds)}")

    # --------------------------------------------------------------------------
    # PHASE 3: Prompt Compilation (single mode, no few-shot)
    # --------------------------------------------------------------------------
    print(f"\n📌 PHASE 3: Compiling prompts for mode '{cfg['mode']}'...")
    compiler = PromptCompiler(task_enum, mode_cfg, yaml_path, [], cfg.get("seed", 42), model_name, tokenizer)
    mode_name = cfg["mode"]

    eval_features = Features({
        "prompt": Value("string"),
        "ground_truth": Value("string"),
        "task": Value("string"),
        "mode": Value("string"),
        "origin_id": Value("string"),
        "input_json": Value("string"),
        "output_json": Value("string"),
    })

    def compile_batch(batch):
        out = {k: [] for k in eval_features.keys()}
        for i in range(len(batch["input"])):
            row_obj = {"input": batch["input"][i], "output": batch["output"][i]}
            compiled = compiler.compile_prompts(row_obj, "")  # ← few-shot always empty
            full_text = compiler._apply_chat_template(compiled)["text"]

            if split_tag in full_text:
                idx = full_text.index(split_tag) + len(split_tag)
                prompt_part = full_text[:idx]
                answer_part = full_text[idx:]
            else:
                prompt_part = full_text
                answer_part = ""
            ground_truth = _strip_stop_tokens(answer_part, stop_tokens)

            out["prompt"].append(prompt_part)
            out["ground_truth"].append(ground_truth)
            out["task"].append(task_key)
            out["mode"].append(mode_name)
            out["origin_id"].append(batch["origin_id"][i])
            out["input_json"].append(batch["input"][i])
            out["output_json"].append(batch["output"][i])
        return out

    compiled_ds = chunked_ds.map(
        compile_batch,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=chunked_ds.column_names,
        features=eval_features,
        desc=f"Compiling {task_key}/{mode_name}",
    )

    # --------------------------------------------------------------------------
    # PHASE 4: Length filter
    # --------------------------------------------------------------------------
    print(f"\n🔪 PHASE 4: Filtering rows over {max_len} tokens...")

    def under_limit(example):
        # Full length = prompt + ground_truth (i.e., training-equivalent text)
        full = example["prompt"] + example["ground_truth"]
        return len(tokenizer(full, add_special_tokens=False)["input_ids"]) <= max_len

    filtered_ds = compiled_ds.filter(under_limit, num_proc=num_proc, desc="Length filter")
    dropped = len(compiled_ds) - len(filtered_ds)
    if dropped:
        print(f"   Dropped {dropped} overlong rows")

    # --------------------------------------------------------------------------
    # Save
    # --------------------------------------------------------------------------
    print(f"\n💾 Saving eval Dataset to {out_path} ...")
    filtered_ds.save_to_disk(out_path)
    print(f"✅ Successfully saved eval dataset: {out_path}")
    print(f"   Rows: {len(filtered_ds)}")
    print(f"   Columns: {filtered_ds.column_names}")

    return filtered_ds


if __name__ == "__main__":
    generate_eval_dataset()
