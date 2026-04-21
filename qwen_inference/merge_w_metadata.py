"""
Merge vLLM predictions with original metadata.

All paths come from generate_eval_data_config.yaml (single source of truth).
Override via env var EVAL_CONFIG.

Safety improvements vs prior version:
  - METADATA_FILE defaults to data.input_path (same jsonl used by preprocessing).
  - Row count mismatch between predictions and HF dataset is a HARD FAIL.
  - Metadata lookup hit rate is logged (first 100 rows) so silent drops surface.
"""
import json
import os
import sys
import yaml
from tqdm import tqdm
from datasets import load_from_disk
from parse_output import KnowledgeParser


def build_metadata_lookup(filepath: str) -> dict:
    """Load BERT-style metadata into {origin_id: {pair_id: meta}} dict."""
    lookup = {}
    print(f"🔄 Loading metadata from {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = str(doc.get("id", ""))
            lookup[doc_id] = {}

            for meta in doc.get("meta_info", []):
                # Skip BEE-BEE (not evaluated)
                s_type = meta.get("subject", {}).get("source_type", "")
                o_type = meta.get("object", {}).get("source_type", "")
                if s_type == "BEE" and o_type == "BEE":
                    continue

                pair_key = str(meta.get("pair_id", ""))
                if not pair_key.startswith("["):
                    pair_key = f"[{pair_key}]"
                lookup[doc_id][pair_key] = meta
    print(f"✅ Metadata loaded ({len(lookup)} origins, BEE-BEE excluded)")
    return lookup


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    predictions_file = cfg["inference"]["predictions_path"]
    hf_test_path = cfg["data"]["output_path"]
    metadata_file = cfg["data"]["input_path"]  # same jsonl preprocessing consumed
    output_file = cfg["merge"]["output_path"]
    eval_mode = cfg["merge"].get("eval_mode", True)
    prompt_config = cfg["prompt"]["yaml_path"]

    print(f"📋 Config: {cfg_path}")
    print(f"   Predictions: {predictions_file}")
    print(f"   HF dataset:  {hf_test_path}")
    print(f"   Metadata:    {metadata_file}")
    print(f"   Output:      {output_file}")
    print(f"   EVAL_MODE:   {eval_mode}")

    if not os.path.exists(predictions_file):
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    if not os.path.exists(hf_test_path):
        raise FileNotFoundError(f"HF dataset not found: {hf_test_path}")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    parser = KnowledgeParser(prompt_config_path=prompt_config)
    metadata_lookup = build_metadata_lookup(metadata_file)

    print(f"📂 Loading HF dataset")
    hf_dataset = load_from_disk(hf_test_path)

    print(f"🔄 Merging predictions and HF dataset")
    with open(predictions_file, "r", encoding="utf-8") as f_in:
        pred_lines = f_in.readlines()

    # Hard fail on row mismatch — prior version silently truncated via zip().
    if len(pred_lines) != len(hf_dataset):
        raise RuntimeError(
            f"Row count mismatch: {len(pred_lines)} predictions vs "
            f"{len(hf_dataset)} HF rows. Re-run inference against the current dataset."
        )

    task_mapper = {
        "ner_ner": "NER-NER",
        "ner_bee": "NER-BEE",
        "combine_all": "COMBINE_ALL",
    }

    # Metadata lookup diagnostics
    meta_hits = 0
    meta_misses = 0
    miss_origin_ids = set()
    SAMPLE_LIMIT = 100  # log hit rate for first 100 rows

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f_out:
        for i, (pred_line, hf_row) in enumerate(tqdm(
            zip(pred_lines, hf_dataset), total=len(pred_lines), desc="Processing"
        )):
            pred_data = json.loads(pred_line)
            raw_pred = pred_data.get("prediction", "")
            full_input = pred_data.get("full_input", "")

            origin_id = str(hf_row.get("origin_id", "UNKNOWN"))
            raw_hf_task = str(hf_row.get("task", "combine_all")).lower()

            if raw_hf_task in ("bee_bee", "ner_bee_true_only"):
                continue

            true_task = task_mapper.get(raw_hf_task, "COMBINE_ALL")
            mode = str(hf_row.get("mode", "UNKNOWN"))

            pred_tuples = parser.parse(raw_pred, expected_task=true_task)

            rich_preds = []
            for p_id, sub, obj, rel in pred_tuples:
                meta = metadata_lookup.get(origin_id, {}).get(p_id)
                if i < SAMPLE_LIMIT:
                    if meta:
                        meta_hits += 1
                    else:
                        meta_misses += 1
                        miss_origin_ids.add(origin_id)
                if not meta:
                    continue
                rich_preds.append({
                    "pair_id": p_id,
                    "subject": meta.get("subject", sub),
                    "object": meta.get("object", obj),
                    "relation": rel,
                })

            final_record = {
                "origin_id": origin_id,
                "expected_task": true_task,
                "mode": mode,
                "pred_tuples": rich_preds,
                "full_input": full_input,
            }

            if eval_mode:
                raw_gold = pred_data.get("ground_truth", "")
                gold_tuples = parser.parse(raw_gold, expected_task=true_task)
                rich_golds = []
                for p_id, sub, obj, rel in gold_tuples:
                    meta = metadata_lookup.get(origin_id, {}).get(p_id)
                    if not meta:
                        continue
                    rich_golds.append({
                        "pair_id": p_id,
                        "subject": meta.get("subject", sub),
                        "object": meta.get("object", obj),
                        "relation": rel,
                    })
                final_record["ground_truth"] = rich_golds

            f_out.write(json.dumps(final_record, ensure_ascii=False) + "\n")

            if i == 0:
                print("\n🔍 AUDIT OF ROW 1")
                print(json.dumps(final_record, ensure_ascii=False, indent=2))

    # Post-run diagnostics
    total_looked_up = meta_hits + meta_misses
    if total_looked_up:
        hit_rate = meta_hits / total_looked_up * 100
        print(f"\n📊 Metadata lookup (first {SAMPLE_LIMIT} rows): "
              f"{meta_hits} hits / {meta_misses} misses ({hit_rate:.1f}% hit rate)")
        if hit_rate < 95:
            print(f"⚠️  WARNING: low metadata hit rate ({hit_rate:.1f}%). "
                  f"Likely cause: metadata_file ({metadata_file}) does not match "
                  f"the jsonl used for preprocessing. Example missing origin_ids: "
                  f"{sorted(miss_origin_ids)[:5]}")

    print(f"\n✅ Pipeline complete — saved to: {output_file}")


if __name__ == "__main__":
    cfg_path = os.environ.get("EVAL_CONFIG", "generate_eval_data_config.yaml")
    main(cfg_path)
