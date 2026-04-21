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

        # FEW-SHOT ACCUMULATOR: Generate one robust example block per task.
        # Pool is already restricted to train-split origins (set in caller),
        # so val/test rows cannot see their own documents as demonstrations.
        # prioritize_rare drives task-aware priority:
        #   - NER-NER: prefer chunks containing rare relation labels
        #   - NER-BEE / NER-BEE_TRUE_ONLY: prefer chunks containing "true" labels
        #     (so few-shot isn't all-negative for binary tasks).
        fs_samples = fewshot_generator.generate_by_pairs(
            task_name,
            min_pairs=fewshot_min_pairs,
            max_pairs=fewshot_max_pairs,
            prioritize_rare=prioritize_rare,
        )
        fs_text_global = fewshot_generator.format(fs_samples) if fs_samples else ""

        # Per-row self-demonstration guard: if the current row's origin_id happens to
        # appear inside fs_samples, regenerate a fs_text that excludes that origin.
        # This only applies to the handful of origins that actually contributed demos.
        demo_origin_ids = {s.get("origin_id") for s in fs_samples if s.get("origin_id")}
        fs_text_per_demo_origin = {}
        for demo_oid in demo_origin_ids:
            alt_samples = fewshot_generator.generate_by_pairs(
                task_name,
                min_pairs=fewshot_min_pairs,
                max_pairs=fewshot_max_pairs,
                exclude_origin_id=demo_oid,
                prioritize_rare=prioritize_rare,
            )
            fs_text_per_demo_origin[demo_oid] = (
                fewshot_generator.format(alt_samples) if alt_samples else ""
            )
        if demo_origin_ids:
            print(f"   Few-shot demo origins for '{task_name}': {len(demo_origin_ids)} "
                  f"(alt fs_texts precomputed for self-demo exclusion)")

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
            fs_enabled = mode_cfg.few_shot
            # Capture per-row fallbacks for self-demo exclusion; empty dict when few-shot off.
            _fs_text_global = fs_text_global if fs_enabled else ""
            _fs_text_alt_map = fs_text_per_demo_origin if fs_enabled else {}

            def process_batch(batch):
                out_texts, out_split_targets, out_origin_ids = [], [], []

                # Iterate by index to prevent the tokenizer from crashing on a List[str]
                for i in range(len(batch["input"])):
                    curr_input = batch["input"][i]
                    curr_output = batch["output"][i]
                    curr_origin_id = batch["origin_id"][i]

                    # Per-row few-shot selection:
                    #   - If current origin contributed demos, use the origin-excluded alt fs_text
                    #     (prevents self-demonstration leakage).
                    #   - Otherwise use the global fs_text.
                    row_fs_text = _fs_text_alt_map.get(curr_origin_id, _fs_text_global)

                    # 1. Compile the strings using the YAML template elements
                    row_obj = {"input": curr_input, "output": curr_output}
                    compiled = compiler.compile_prompts(row_obj, row_fs_text)

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
    
    # raw_sources populated from cfg["tasks"]. NER_BEE_TRUE_ONLY is an optional
    # positive-only subset; only built when explicitly requested in config.
    active_task_values = {DataGenerationTask[t].value for t in cfg.get("tasks", [])}
    raw_sources = {}
    if "ner_ner" in active_task_values:
        raw_sources["ner_ner"] = classified["ner_ner"]
    if "ner_bee" in active_task_values:
        raw_sources["ner_bee"] = classified["ner_bee"]
    if "ner_bee_true_only" in active_task_values:
        raw_sources["ner_bee_true_only"] = dp.remove_negative_relations(classified["ner_bee"])

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

    # Strict schema to prevent "int64 vs string" merging crashes during map.
    # sampling_profile: JSON string used by PHASE 3 diversity-aware sampler.
    new_features = Features({
        "id": Value("string"), "input": Value("string"),
        "output": Value("string"), "origin_id": Value("string"),
        "sampling_profile": Value("string"),
    })

    print(f"\n📌 PHASE 2: Chunking Tasks to fit {max_token_threshold} tokens...")
    for task_enum in tqdm(tasks, desc="Chunking Tasks"):
        task_key = task_enum.value
        entries = raw_sources.get(task_key, [])
        if not entries: continue
        
        # Calculate safe limit using MAX overhead across all active modes.
        # Heavy modes like full_detailed use few-shot blocks (~2000-3000 tokens)
        # which must be accounted for — otherwise chunker under-chunks and final
        # prompts overflow 8192 at training time (caught by PHASE 6 prune, but
        # that drops real training data).
        # Here we pre-build a representative fs_text from pre-chunk entries of
        # the current task to get a realistic upper bound.
        fewshot_cfg_local = cfg.get("fewshot", {})
        fs_min = fewshot_cfg_local.get("min_pairs", 8)
        fs_max = fewshot_cfg_local.get("max_pairs", 12)
        # Pre-chunk entries as a rough fs pool (they'll be re-chunked later; this
        # is only for size estimation, not the final few-shot used in prompts).
        _probe_pi = PreprocessInput()
        _probe_formatted = _probe_pi.build_input_output(entries[:min(50, len(entries))], task_enum)
        _probe_chunks = [{
            "output": item["output"], "input": item["input"], "origin_id": "probe"
        } for item in _probe_formatted]
        _probe_gen = GenerateFewShotSamples({task_key: _probe_chunks}, seed=42)
        _probe_samples = _probe_gen.generate_by_pairs(task_key, min_pairs=fs_min, max_pairs=fs_max)
        probe_fs_text = _probe_gen.format(_probe_samples) if _probe_samples else ""

        overheads = []
        for mode_cfg in BASE_MODES.values():
            temp_compiler = PromptCompiler(task_enum, mode_cfg, cfg["prompt"]["yaml_path"], [], 42, model_name, tokenizer)
            overheads.append(calculate_template_overhead(temp_compiler, tokenizer, fewshot_text=probe_fs_text))
        ov = max(overheads)
        # Drop buffer from 2000 to 1000 since overhead now correctly includes few-shot.
        SAFE_LIMIT = max_token_threshold - ov - 1000
        print(f"   Task {task_key}: mode overheads={overheads}, max={ov}, "
              f"fs_estimate={len(tokenizer.encode(probe_fs_text)) if probe_fs_text else 0} toks, "
              f"SAFE_LIMIT={SAFE_LIMIT}")
        
        hf_ds = Dataset.from_list(entries)
        def chunk_batch(batch):
            rows = [dict(zip(batch.keys(), v)) for v in zip(*batch.values())]
            processed = []
            for entry in rows:
                chunks = run_adaptive_chunking(entry, chunker, tc, task_enum, SAFE_LIMIT, chunk_thresholds)
                # Build per-chunk sampling_profile BEFORE build_input_output (which
                # replaces candidate_pairs with clean tagged dicts losing raw meta keys).
                profiles = [build_sampling_profile(c, task_enum) for c in chunks]
                formatted = pi.build_input_output(chunks, task_enum)
                for item, profile in zip(formatted, profiles):
                    processed.append({
                        "id": str(item["id"]),
                        "input": item["input"],
                        "output": item["output"],
                        "origin_id": str(entry["id"]),
                        "sampling_profile": profile,
                    })
            if not processed:
                return {k: [] for k in ["id", "input", "output", "origin_id", "sampling_profile"]}
            return {k: [r[k] for r in processed] for k in processed[0].keys()}

        processed_ds = hf_ds.map(
            chunk_batch, batched=True, num_proc=num_proc, 
            remove_columns=hf_ds.column_names, features=new_features, desc=f"Chunking {task_key}"
        )
        final_data_sources[task_key] = [item for item in processed_ds]

    # --------------------------------------------------------------------------
    # 🎯 PHASE 3: Diversity-aware Quota Sampling
    #
    # Replaces the previous binary (priority/common) origin-count approximation
    # with a 3-pass greedy sampler operating on chunk count directly:
    #
    #   Pass A (Priority): keep origins carrying static rare labels, dynamic rare
    #                      (low corpus exposure), or NER-BEE positives.
    #   Pass B (Coverage): greedy fill meeting relation floor + entity-group pair
    #                      floor. Score = sum of unmet need gains per origin.
    #   Pass C (Fill):     remaining quota filled by novelty − head-cap overflow
    #                      penalty (discourages NO_RELATION / used_by / same_entity
    #                      from dominating).
    #
    # All budgets are in chunk-count units; origins are selected atomically (all
    # chunks of a selected origin stay together).
    # --------------------------------------------------------------------------
    target_train = gen_cfg.get("target_total_training_samples", None)
    target_total = gen_cfg.get("target_total_samples", None)
    if target_train is not None:
        target_total = int(target_train / 0.7)
        print(f"\n⚖️ PHASE 3: Targeting {target_train} Train samples.")
        print(f"   Total dataset target auto-adjusted to {target_total} (70% train split).")

    div_cfg = gen_cfg.get("diversity", {})
    diversity_enabled = bool(div_cfg.get("enable", True))
    RARE_SET_STATIC = {
        "instance_of", "addresses", "applied_by", "purchases", "provided_to",
        "gifted_by", "frequency_of_use", "purchased_by", "benefits_user",
        "gifted_to", "sells", "uses", "has_part", "addressed_by_treatment",
        "addressed_to", "belongs_to", "described_by", "not_used_by",
        "requires", "perceives", "targeted_at", "available_to", "available_in",
        "causes", "experiences", "caused_by", "provided_by", "has_instance",
        "variant_of", "owns", "treats", "price_of", "information_to",
        "information_from", "sold_by", "required_by", "targeted_by",
        "child_of", "parent_of", "brand_of", "family_member_of",
    }
    # Tunable via config generation.diversity (all optional).
    dyn_rare_chunk_thresh = int(div_cfg.get("dynamic_rare_chunk_threshold", 100))
    dyn_rare_origin_thresh = int(div_cfg.get("dynamic_rare_origin_threshold", 30))
    rel_floor_min = int(div_cfg.get("relation_floor_min", 20))
    rel_floor_max = int(div_cfg.get("relation_floor_max", 100))
    rel_floor_ratio = float(div_cfg.get("relation_floor_ratio", 0.25))
    group_floor_min = int(div_cfg.get("group_pair_floor_min", 10))
    group_floor_max = int(div_cfg.get("group_pair_floor_max", 60))
    group_floor_ratio = float(div_cfg.get("group_pair_floor_ratio", 0.15))
    # Head caps — chunk-share soft limits for dominant classes.
    head_caps_ner_ner = div_cfg.get("head_caps_ner_ner", {
        "NO_RELATION": 0.35, "used_by": 0.14, "same_entity": 0.10,
    })
    head_caps_ner_bee = div_cfg.get("head_caps_ner_bee", {
        "__bee_false__": 0.70,
    })

    def _parse_profile(chunk_row):
        try:
            return json.loads(chunk_row.get("sampling_profile") or "{}")
        except Exception:
            return {}

    def _aggregate_origin_profiles(chunks, task_key):
        """Merge chunk profiles to origin-level aggregates."""
        origins = {}
        for c in chunks:
            oid = c["origin_id"]
            prof = _parse_profile(c)
            ob = origins.setdefault(oid, {
                "chunk_count": 0,
                "rel_labels": set(),       # any relation ever appearing in this origin
                "rel_pair_counts": {},     # summed pair counts per relation
                "group_pairs": set(),
                "subj_groups": {},
                "obj_groups": {},
                "has_static_rare": False,
            })
            ob["chunk_count"] += 1
            for r in prof.get("rel_labels", []):
                ob["rel_labels"].add(r)
            for r, n in (prof.get("rel_counts") or {}).items():
                ob["rel_pair_counts"][r] = ob["rel_pair_counts"].get(r, 0) + n
            for gp in prof.get("group_pairs", []):
                ob["group_pairs"].add(gp)
            for g, n in (prof.get("subj_groups") or {}).items():
                ob["subj_groups"][g] = ob["subj_groups"].get(g, 0) + n
            for g, n in (prof.get("obj_groups") or {}).items():
                ob["obj_groups"][g] = ob["obj_groups"].get(g, 0) + n
            if task_key == "ner_ner" and any(r in RARE_SET_STATIC for r in prof.get("rel_labels", [])):
                ob["has_static_rare"] = True
            if task_key in ("ner_bee", "ner_bee_true_only") and "__bee_true__" in prof.get("rel_labels", []):
                ob["has_static_rare"] = True
        return origins

    def _compute_floors(origins, task_key, quota):
        """Relation + group-pair exposure floors from raw corpus stats."""
        rel_chunk_global = {}      # relation → # chunks containing it
        rel_origin_support = {}    # relation → # origins containing it
        group_pair_chunk_global = {}
        # Walk per-origin to get origin-support correctly
        for oid, prof in origins.items():
            for r in prof["rel_labels"]:
                rel_origin_support[r] = rel_origin_support.get(r, 0) + 1
            for gp in prof["group_pairs"]:
                group_pair_chunk_global[gp] = group_pair_chunk_global.get(gp, 0) + 1  # origin-level approx
        # Chunk-level exposure for relations needs full pass
        for oid, prof in origins.items():
            # Use rel_labels weighted by chunk_count as a coarse proxy
            for r in prof["rel_labels"]:
                rel_chunk_global[r] = rel_chunk_global.get(r, 0) + prof["chunk_count"]
        head = set(head_caps_ner_ner if task_key == "ner_ner" else head_caps_ner_bee)
        rel_floor = {}
        dynamic_priority = set()
        for r, exposure in rel_chunk_global.items():
            if r in head:
                continue
            if exposure < dyn_rare_chunk_thresh or rel_origin_support.get(r, 0) < dyn_rare_origin_thresh:
                dynamic_priority.add(r)
            if exposure >= rel_floor_min:
                rel_floor[r] = max(rel_floor_min,
                                   min(rel_floor_max, int(rel_floor_ratio * exposure) or rel_floor_min))
        group_floor = {
            gp: max(group_floor_min, min(group_floor_max, int(group_floor_ratio * cnt) or group_floor_min))
            for gp, cnt in group_pair_chunk_global.items()
            if cnt >= group_floor_min
        }
        return rel_floor, group_floor, rel_chunk_global, dynamic_priority

    class _SamplerState:
        def __init__(self, quota, rel_floor, group_floor, head_caps):
            self.quota = quota
            self.selected = set()      # origin ids
            self.chunk_count = 0
            self.rel_need = dict(rel_floor)            # remaining chunks needed per rel
            self.group_need = dict(group_floor)
            self.rel_taken = {}                         # relation → chunks selected
            self.subj_totals = {}
            self.obj_totals = {}
            self.head_caps = head_caps                  # {rel: share_cap}

        def add(self, oid, origin_profile):
            self.selected.add(oid)
            self.chunk_count += origin_profile["chunk_count"]
            for r in origin_profile["rel_labels"]:
                if r in self.rel_need:
                    self.rel_need[r] = max(0, self.rel_need[r] - origin_profile["chunk_count"])
                self.rel_taken[r] = self.rel_taken.get(r, 0) + origin_profile["chunk_count"]
            for gp in origin_profile["group_pairs"]:
                if gp in self.group_need:
                    self.group_need[gp] = max(0, self.group_need[gp] - origin_profile["chunk_count"])
            for g, n in origin_profile["subj_groups"].items():
                self.subj_totals[g] = self.subj_totals.get(g, 0) + n
            for g, n in origin_profile["obj_groups"].items():
                self.obj_totals[g] = self.obj_totals.get(g, 0) + n

        def room_left(self):
            return self.quota - self.chunk_count

        def head_overflow_penalty(self, origin_profile):
            """Penalize adding this origin if it pushes head relations over their cap."""
            if not self.head_caps:
                return 0.0
            projected = self.chunk_count + origin_profile["chunk_count"]
            if projected == 0:
                return 0.0
            penalty = 0.0
            for r, cap in self.head_caps.items():
                taken = self.rel_taken.get(r, 0)
                add_r = origin_profile["chunk_count"] if r in origin_profile["rel_labels"] else 0
                share = (taken + add_r) / projected
                if share > cap:
                    penalty += (share - cap)
            return penalty

        def coverage_gain(self, origin_profile):
            rel_gain = sum(min(self.rel_need.get(r, 0), origin_profile["chunk_count"])
                           for r in origin_profile["rel_labels"])
            grp_gain = sum(min(self.group_need.get(gp, 0), origin_profile["chunk_count"])
                           for gp in origin_profile["group_pairs"])
            return rel_gain + grp_gain

        def novelty(self, origin_profile):
            """Favor origins introducing under-represented entity groups."""
            import math
            s = sum(1.0 / math.sqrt(1 + self.subj_totals.get(g, 0)) for g in origin_profile["subj_groups"])
            o = sum(1.0 / math.sqrt(1 + self.obj_totals.get(g, 0)) for g in origin_profile["obj_groups"])
            return s + o

    if target_total is not None:
        active_tasks = [t for t in final_data_sources.keys() if final_data_sources[t]]
        if active_tasks:
            quota_per_task = int(target_total / len(active_tasks))
            print(f"   Quota per task: {quota_per_task} chunks")

            for task_key in active_tasks:
                chunks = final_data_sources[task_key]
                if len(chunks) <= quota_per_task:
                    print(f"   {task_key}: {len(chunks)} chunks ≤ quota, keeping all")
                    continue

                if not diversity_enabled:
                    # Fallback: random origin sampling (legacy behavior, much simpler)
                    unique_ids = list({c["origin_id"] for c in chunks})
                    random.shuffle(unique_ids)
                    keep_ids = set()
                    kept = 0
                    id_to_chunk_count = {}
                    for c in chunks:
                        id_to_chunk_count[c["origin_id"]] = id_to_chunk_count.get(c["origin_id"], 0) + 1
                    for oid in unique_ids:
                        if kept + id_to_chunk_count[oid] > quota_per_task:
                            continue
                        keep_ids.add(oid)
                        kept += id_to_chunk_count[oid]
                    filtered = [c for c in chunks if c["origin_id"] in keep_ids]
                    print(f"   {task_key} [diversity=off]: {len(chunks)} → {len(filtered)} chunks")
                    final_data_sources[task_key] = filtered
                    continue

                origins = _aggregate_origin_profiles(chunks, task_key)
                rel_floor, group_floor, rel_exposure, dynamic_priority = _compute_floors(
                    origins, task_key, quota_per_task)
                head_caps = head_caps_ner_ner if task_key == "ner_ner" else head_caps_ner_bee

                # Classify origins into priority / common
                priority_ids = {
                    oid for oid, p in origins.items()
                    if p["has_static_rare"] or any(r in dynamic_priority for r in p["rel_labels"])
                }
                common_ids = set(origins.keys()) - priority_ids

                state = _SamplerState(quota_per_task, rel_floor, group_floor, head_caps)

                # ---- Pass A: keep priority origins (greedy by coverage within priority) ----
                prio_list = list(priority_ids)
                random.shuffle(prio_list)
                def _score_coverage(oid, st):
                    p = origins[oid]
                    return st.coverage_gain(p) - 2.0 * st.head_overflow_penalty(p) * st.chunk_count
                while prio_list and state.room_left() > 0:
                    prio_list.sort(key=lambda oid: _score_coverage(oid, state), reverse=True)
                    picked = prio_list[0]
                    if origins[picked]["chunk_count"] > state.room_left():
                        prio_list.pop(0)
                        continue
                    state.add(picked, origins[picked])
                    prio_list.pop(0)

                # ---- Pass B: coverage-driven greedy from common ----
                remaining_common = [oid for oid in common_ids if origins[oid]["chunk_count"] <= state.room_left()]
                random.shuffle(remaining_common)
                # Iterate while floors unmet and quota available
                while state.room_left() > 0 and (
                    any(v > 0 for v in state.rel_need.values()) or
                    any(v > 0 for v in state.group_need.values())
                ):
                    # Candidate pool: origins with positive coverage gain
                    candidates = [oid for oid in remaining_common if origins[oid]["chunk_count"] <= state.room_left()]
                    if not candidates:
                        break
                    candidates.sort(key=lambda oid: _score_coverage(oid, state), reverse=True)
                    best = candidates[0]
                    if state.coverage_gain(origins[best]) <= 0:
                        break  # no more coverage improvement possible
                    state.add(best, origins[best])
                    remaining_common.remove(best)

                # ---- Pass C: fill remaining quota with novelty − head penalty ----
                def _score_fill(oid, st):
                    p = origins[oid]
                    return st.novelty(p) - 3.0 * st.head_overflow_penalty(p) * max(1, st.chunk_count)
                pool = [oid for oid in remaining_common if origins[oid]["chunk_count"] <= state.room_left()]
                while pool and state.room_left() > 0:
                    pool.sort(key=lambda oid: _score_fill(oid, state), reverse=True)
                    best = pool[0]
                    if origins[best]["chunk_count"] > state.room_left():
                        pool.pop(0)
                        continue
                    state.add(best, origins[best])
                    pool.pop(0)

                filtered = [c for c in chunks if c["origin_id"] in state.selected]

                # Unmet floor diagnostics
                unmet_rel = [r for r, v in state.rel_need.items() if v > 0]
                unmet_grp = [g for g, v in state.group_need.items() if v > 0]
                head_shares = {}
                if state.chunk_count > 0:
                    for r, cap in head_caps.items():
                        head_shares[r] = round(state.rel_taken.get(r, 0) / state.chunk_count, 3)
                print(f"   {task_key}: {len(chunks)} → {len(filtered)} chunks "
                      f"(priority={len(priority_ids & state.selected)}/{len(priority_ids)}, "
                      f"rel_floor_unmet={len(unmet_rel)}/{len(state.rel_need)}, "
                      f"group_unmet={len(unmet_grp)}/{len(state.group_need)}, "
                      f"head_shares={head_shares})")
                final_data_sources[task_key] = filtered

    # --------------------------------------------------------------------------
    # 🎯 PHASE 4: Data Splitting (Train/Val/Test) — task-stratified
    # Each task's unique origins are split 70/10/20 independently, so that
    # val/test always contain representative samples from every active task
    # (prior version: global random split, under-represented smaller tasks).
    #
    # Origins that appear in multiple tasks (e.g., same doc contributes to both
    # NER-NER and NER-BEE buckets) are assigned once based on their first task's
    # split; the same split label is reused so the same doc never crosses splits.
    # --------------------------------------------------------------------------
    print("\n🔪 PHASE 4: Task-stratified Train/Val/Test Split (70/10/20)...")
    random.seed(cfg.get("seed", 42))
    split_ids_map = {}
    for task_key, chunks in final_data_sources.items():
        if not chunks:
            continue
        task_origins = list({c["origin_id"] for c in chunks})
        # If an origin was already placed (via another task), reuse that label.
        already_placed = [oid for oid in task_origins if oid in split_ids_map]
        new_origins = [oid for oid in task_origins if oid not in split_ids_map]
        random.shuffle(new_origins)
        n = len(new_origins)
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)
        split_counts = {"train": 0, "validation": 0, "test": 0}
        for i, oid in enumerate(new_origins):
            label = "train" if i < train_end else ("validation" if i < val_end else "test")
            split_ids_map[oid] = label
            split_counts[label] += 1
        reused_counts = {"train": 0, "validation": 0, "test": 0}
        for oid in already_placed:
            reused_counts[split_ids_map[oid]] += 1
        print(f"   {task_key}: {len(task_origins)} origins — new={split_counts}, "
              f"reused_from_other_task={reused_counts}")

    # --------------------------------------------------------------------------
    # 🎯 PHASE 5: Prompt Generation
    # Description: The core engine. Replicates rows based on mode_ratios, applies
    # the few-shot examples, and compiles the final training strings.
    #
    # Few-shot leakage guard:
    #   - Pool is restricted to train-split origin_ids (val/test can never see
    #     their own documents as demonstrations).
    #   - Per-row exclusion of the current origin_id is applied inside
    #     generate_all_prompts (prevents self-demonstration for train rows).
    # --------------------------------------------------------------------------
    fewshot_cfg = cfg.get("fewshot", {})
    train_origin_ids = {oid for oid, split in split_ids_map.items() if split == "train"}
    print(f"   Few-shot pool restricted to {len(train_origin_ids)} train origin_ids "
          f"(of {len(split_ids_map)} total). Leakage guard active.")
    fewshot_generator = GenerateFewShotSamples(
        data_dict=final_data_sources,
        seed=fewshot_cfg.get("seed", 42),
        allowed_origin_ids=train_origin_ids,
    )

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