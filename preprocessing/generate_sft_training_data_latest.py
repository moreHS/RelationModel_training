import argparse
import orjson
import yaml
import os
import random
import warnings
import json
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
import numpy as np

# HF Imports
from datasets import Dataset, DatasetDict, Features, Value, load_from_disk
import datasets.builder

# ==============================================================================
# 0. CRITICAL SETUP
# ==============================================================================
# Disables Hugging Face tokenizer parallelism to prevent deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Bypasses Docker overlayFS disk space errors during dataset generation
datasets.builder.has_sufficient_disk_space = lambda x, y=".": True

from transformers import AutoTokenizer
# from data_preprocessor_utils_simplified import *
from preprocessing.data_preprocessor_utils_simplified_add_gemma4 import *

# ==============================================================================
# FUNCTION: PROMPT GENERATION
# ==============================================================================
def generate_all_prompts(
    tasks,
    base_modes,
    data_sources,
    template_yaml_path,
    fewshot_generator,
    model_name,
    tokenizer,
    mode_ratios,
    full_detail_ratio,
    fewshot_sample_size, 
    fewshot_min_pairs,
    fewshot_max_pairs,
    prioritize_rare,
    seed,             
    split_ids_map,
    training_max_length,
    num_proc=8
):
    """
    DESCRIPTION:
    Takes the raw chunked data, applies the YAML templates, attaches few-shots, 
    and wraps everything in the ChatML/Model-specific template.
    
    It maps over the dataset in parallel batches for extreme speed, while isolating
    string compilation to prevent Tokenizer memory explosion.
    """
    train_rows, val_rows, test_rows = [], [], []
    tc = TokenCounter(model_path=model_name)

    for task in tasks:
        task_name = task.value
        if not data_sources.get(task_name): continue
        
        # FEW-SHOT ACCUMULATOR: Generate one robust example block per task
        # This ensures high-signal rare classes are represented exactly 8-12 times.
        fs_samples = fewshot_generator.generate_by_pairs(
            task_name, 
            min_pairs=fewshot_min_pairs, 
            max_pairs=fewshot_max_pairs
        )
        fs_text_global = fewshot_generator.format(fs_samples) if fs_samples else ""
        
        ds = Dataset.from_list(data_sources[task_name])

        # 단일 셔플 + 모드별 disjoint 인덱스 범위 사전 계산 (모드 간 sample overlap 제거)
        # 이전: 모든 모드가 같은 seed로 .shuffle().select(range(N)) → 첫 N개 중복 노출
        shuffled = ds.shuffle(seed=seed)
        total = len(shuffled)
        mode_ranges = {}
        cursor = 0
        full_registered = False
        for m_name in base_modes:
            if "full" in m_name:
                if not full_registered:
                    full_n = int(total * mode_ratios.get("full", 0))
                    full_split = int(full_n * full_detail_ratio)
                    mode_ranges["full_detailed"] = (cursor, cursor + full_split)
                    mode_ranges["full_summary"] = (cursor + full_split, cursor + full_n)
                    cursor += full_n
                    full_registered = True
            else:
                n = int(total * mode_ratios.get(m_name, 0))
                mode_ranges[m_name] = (cursor, cursor + n)
                cursor += n
        print(f"   Task {task_name} mode partition (total={total}): " + ", ".join(f"{k}=[{s},{e})" for k, (s, e) in mode_ranges.items()))

        for mode_name, mode_cfg in base_modes.items():
            s, e = mode_ranges.get(mode_name, (0, 0))
            if e <= s: continue
            sampled = shuffled.select(range(s, e))
            
            # Initialize Prompt Compiler for this mode
            compiler = PromptCompiler(task, mode_cfg, template_yaml_path, [], seed, model_name, tokenizer)
            fs_text = fs_text_global if mode_cfg.few_shot else ""

            def process_batch(batch):
                out_texts, out_split_targets, out_origin_ids = [], [], []
                
                # Iterate by index to prevent the tokenizer from crashing on a List[str]
                for i in range(len(batch["input"])):
                    curr_input = batch["input"][i]
                    curr_output = batch["output"][i]
                    curr_origin_id = batch["origin_id"][i]

                    # 1. Compile the strings using the YAML template elements
                    row_obj = {"input": curr_input, "output": curr_output}
                    compiled = compiler.compile_prompts(row_obj, fs_text)
                    
                    # 2. Apply chat template (e.g., <|im_start|>user...<|im_end|>)
                    final_text = compiler._apply_chat_template(compiled)["text"]

                    # 3. Target mapping based on origin ID (preventing data leaks)
                    out_texts.append(final_text)
                    out_split_targets.append(split_ids_map.get(curr_origin_id, "train"))
                    out_origin_ids.append(curr_origin_id)
                    
                return {"text": out_texts, "split_target": out_split_targets, "origin_id": out_origin_ids}

            # Fast text compilation across CPU cores
            processed_ds = sampled.map(
                process_batch,
                batched=True,
                batch_size=1000,
                num_proc=num_proc,
                remove_columns=sampled.column_names, 
                desc=f"Compiling {task_name}/{mode_name}"
            )
            
            # Sort into the final train/val/test buckets
            for row in processed_ds:
                item = {"text": row["text"], "task": task_name, "mode": mode_name, "origin_id": row["origin_id"]}
                if row["split_target"] == "train": train_rows.append(item)
                elif row["split_target"] == "validation": val_rows.append(item)
                else: test_rows.append(item)
                
    return train_rows, val_rows, test_rows

# ==============================================================================
# MAIN DATA PIPELINE
# ==============================================================================
def get_or_generate_sft_dataset(config_path: str = "preprocessing/generate_sft_training_data_config.yaml"):
    cfg = yaml.safe_load(open(config_path, "r"))
    out_p = cfg["data"]["output_path"]
    hf_dataset_path = out_p.replace(".json", "_hf_dataset")
    
    # Early Exit: If data exists, just load it to save time.
    if os.path.exists(hf_dataset_path): 
        print(f"⚡ Loading existing dataset: {hf_dataset_path}")
        return DatasetDict.load_from_disk(hf_dataset_path)
        
    # --------------------------------------------------------------------------
    # 🎯 PHASE 1: Data Loading & Preprocessing
    # Description: Loads JSONL, fixes IDs to be strict strings, removes negative 
    # relationships (if configured), and buckets data by Task (NER, BEE, etc.)
    # --------------------------------------------------------------------------
    print("📌 PHASE 1: Loading JSONL & Classifying...")
    raw_data = []
    with open(cfg["data"]["input_path"], "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): raw_data.append(orjson.loads(line))
    
    # Force all IDs to be strings at the source to prevent PyArrow type crashes
    for i, doc in enumerate(raw_data): 
        doc["id"] = f"all_{i}" if "id" not in doc else str(doc["id"])
            
    dp = DataPreprocessor()
    should_neg = cfg.get("negatives", {}).get("include_no_relationship", False)
    if not should_neg: raw_data = dp.remove_negative_relations(raw_data)
    classified = dp.extract_and_classify(raw_data)
    
    raw_sources = {
        "ner_ner": classified["ner_ner"], "bee_bee": classified["bee_bee"], 
        "ner_bee": classified["ner_bee"], "ner_bee_true_only": dp.remove_negative_relations(classified["ner_bee"]), 
        "combine_all": classified["ner_ner"] + classified["bee_bee"] + classified["ner_bee"]
    }

    # --------------------------------------------------------------------------
    # 🎯 PHASE 2: Parallel Chunking
    # Description: Calculates how many tokens the prompt template will use, determines
    # the SAFE_LIMIT, and aggressively chops large documents into smaller pieces so 
    # they fit within the 8192 context window.
    # --------------------------------------------------------------------------
    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tc, chunker, pi = TokenCounter(model_path=model_name), RelationBasedChunker(), PreprocessInput()

    BASE_MODES = {name: DataGenerationModeConfig(**mcfg) for name, mcfg in cfg["modes"].items()}
    tasks = [DataGenerationTask[t] for t in cfg["tasks"]]
    final_data_sources = {}
    
    gen_cfg = cfg.get("generation", {})
    num_proc = gen_cfg.get("num_proc", 8)
    chunk_thresholds = gen_cfg.get("chunk_thresholds", [12, 10, 8, 6, 4])
    max_token_threshold = cfg.get("training_max_length", 8192)

    # Strict schema to prevent "int64 vs string" merging crashes during map
    new_features = Features({
        "id": Value("string"), "input": Value("string"),
        "output": Value("string"), "origin_id": Value("string")
    })

    print(f"\n📌 PHASE 2: Chunking Tasks to fit {max_token_threshold} tokens...")
    for task_enum in tqdm(tasks, desc="Chunking Tasks"):
        task_key = task_enum.value
        entries = raw_sources.get(task_key, [])
        if not entries: continue
        
        # Calculate safe limit using MAX overhead across all active modes
        # (heavy modes like full_detailed have larger description+few-shot overhead → would exceed limit if we only used the first mode)
        overheads = []
        for mode_cfg in BASE_MODES.values():
            temp_compiler = PromptCompiler(task_enum, mode_cfg, cfg["prompt"]["yaml_path"], [], 42, model_name, tokenizer)
            overheads.append(calculate_template_overhead(temp_compiler, tokenizer))
        ov = max(overheads)
        SAFE_LIMIT = max_token_threshold - ov - 2000
        print(f"   Task {task_key}: mode overheads={overheads}, max={ov}, SAFE_LIMIT={SAFE_LIMIT}")
        
        hf_ds = Dataset.from_list(entries)
        def chunk_batch(batch):
            rows = [dict(zip(batch.keys(), v)) for v in zip(*batch.values())]
            processed = []
            for entry in rows:
                chunks = run_adaptive_chunking(entry, chunker, tc, task_enum, SAFE_LIMIT, chunk_thresholds)
                formatted = pi.build_input_output(chunks, task_enum)
                for item in formatted:
                    processed.append({"id": str(item["id"]), "input": item["input"], "output": item["output"], "origin_id": str(entry["id"])})
            return {k: [r[k] for r in processed] for k in processed[0].keys()} if processed else {k: [] for k in ["id", "input", "output", "origin_id"]}

        processed_ds = hf_ds.map(
            chunk_batch, batched=True, num_proc=num_proc, 
            remove_columns=hf_ds.column_names, features=new_features, desc=f"Chunking {task_key}"
        )
        final_data_sources[task_key] = [item for item in processed_ds]

    # --------------------------------------------------------------------------
    # 🎯 PHASE 3: Task Balancing (Target Total Quota)
    # Description: Looks at `target_total_training_samples` in YAML (e.g., 30000). 
    # Calculates the math automatically to hit exactly that number for the 70% Train 
    # split. Balances equally across active tasks.
    # --------------------------------------------------------------------------
    target_train = gen_cfg.get("target_total_training_samples", None)
    target_total = gen_cfg.get("target_total_samples", None)

    # Automatically do the math if target_total_training_samples is provided
    if target_train is not None:
        target_total = int(target_train / 0.7)
        print(f"\n⚖️ PHASE 3: Targeting {target_train} Train samples.")
        print(f"   Automatically adjusted Total Dataset Target to {target_total} (assuming 70% Train split)...")

    if target_total is not None:
        active_tasks = [t for t in final_data_sources.keys() if final_data_sources[t]]
        if active_tasks:
            quota_per_task = int(target_total / len(active_tasks))
            print(f"   Applying quota of {quota_per_task} per task...")
            
            for task_key in active_tasks:
                chunks = final_data_sources[task_key]
                if len(chunks) <= quota_per_task: continue
                
                # Sample based on origin_id to prevent fragmenting documents
                unique_ids = list(set(c["origin_id"] for c in chunks))
                keep_count = int(len(unique_ids) * (quota_per_task / len(chunks)))
                keep_ids = set(random.sample(unique_ids, max(1, keep_count)))
                
                filtered = [c for c in chunks if c["origin_id"] in keep_ids]
                print(f"   ✂️ {task_key}: {len(chunks)} -> {len(filtered)} chunks")
                final_data_sources[task_key] = filtered

    # --------------------------------------------------------------------------
    # 🎯 PHASE 4: Data Splitting (Train/Val/Test)
    # Description: Shuffles the unique source document IDs and assigns them to 
    # Train (70%), Val (10%), or Test (20%). Grouping by origin_id strictly prevents
    # Data Leakage between the training and testing sets.
    # --------------------------------------------------------------------------
    print("\n🔪 PHASE 4: Grouped Train/Val/Test Split (70/10/20)...")
    all_unique_origins = list(set(c["origin_id"] for chunks in final_data_sources.values() for c in chunks))
    random.seed(cfg.get("seed", 42))
    random.shuffle(all_unique_origins)
    train_end, val_end = int(len(all_unique_origins)*0.7), int(len(all_unique_origins)*0.8)
    split_ids_map = {oid: ("train" if i < train_end else "validation" if i < val_end else "test") for i, oid in enumerate(all_unique_origins)}

    # --------------------------------------------------------------------------
    # 🎯 PHASE 5: Prompt Generation
    # Description: The core engine. Replicates rows based on mode_ratios, applies
    # the few-shot examples, and compiles the final training strings.
    # --------------------------------------------------------------------------
    fewshot_cfg = cfg.get("fewshot", {})
    fewshot_generator = GenerateFewShotSamples(data_dict=final_data_sources, seed=fewshot_cfg.get("seed", 42))

    print("\n📌 PHASE 5: Generating Final Prompts...")
    train_rows, val_rows, test_rows = generate_all_prompts(
        tasks=tasks,
        base_modes=BASE_MODES,
        data_sources=final_data_sources,
        template_yaml_path=cfg["prompt"]["yaml_path"],
        fewshot_generator=fewshot_generator,
        model_name=model_name,
        tokenizer=tokenizer,
        mode_ratios=cfg.get("mode_ratios", {}),
        full_detail_ratio=cfg.get("mode_ratios", {}).get("full_detailed_ratio", 0.7),
        fewshot_sample_size=1, 
        fewshot_min_pairs=fewshot_cfg.get("min_pairs", 8),
        fewshot_max_pairs=fewshot_cfg.get("max_pairs", 12),
        prioritize_rare=fewshot_cfg.get("prioritize_rare", True),
        seed=cfg.get("seed", 42),
        split_ids_map=split_ids_map, 
        training_max_length=max_token_threshold,
        num_proc=num_proc
    )

    ds_dict = DatasetDict({
        "train": Dataset.from_list(train_rows), 
        "validation": Dataset.from_list(val_rows), 
        "test": Dataset.from_list(test_rows)
    })

    # --------------------------------------------------------------------------
    # 🎯 PHASE 6: Final Token Length Pruning
    # Description: A final gatekeeper. Because the few-shot blocks were generated 
    # dynamically after chunking, this ensures no single row exceeds the 
    # training_max_length, preventing indexing errors during Unsloth training.
    # --------------------------------------------------------------------------
    print(f"\n🔪 PHASE 6: Pruning outliers over {max_token_threshold} tokens...")
    
    def is_under_limit(example):
        tokens = len(tokenizer(example["text"], add_special_tokens=False)["input_ids"])
        return tokens <= max_token_threshold

    ds_dict = ds_dict.filter(
        is_under_limit, 
        num_proc=num_proc, 
        desc=f"Filtering max length ({max_token_threshold})"
    )

    # --------------------------------------------------------------------------
    # FINISH: Save to Disk
    # --------------------------------------------------------------------------
    print("\n💾 Saving Hugging Face DatasetDict to disk...")
    ds_dict.save_to_disk(hf_dataset_path)
    print(f"✅ Successfully saved dataset splits to: {hf_dataset_path}")
    print(f"   Train: {len(ds_dict['train'])} | Val: {len(ds_dict['validation'])} | Test: {len(ds_dict['test'])}")

    return ds_dict

if __name__ == "__main__":
    get_or_generate_sft_dataset()