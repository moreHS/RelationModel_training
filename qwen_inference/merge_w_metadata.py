import json
import os
from tqdm import tqdm
from datasets import load_from_disk
from parse_output import KnowledgeParser

# =========================================================
# ⚙️ CONFIGURATION
# =========================================================
TARGET_MODEL = "qwen"
MODE = ""
EVAL_MODE = True

PREDICTIONS_FILE = f"vllm_predictions_{TARGET_MODEL}{MODE}.jsonl"
OUTPUT_FILE = f"parsed_output_with metadata_{TARGET_MODEL}{MODE}.jsonl"
PROMPT_CONFIG_FILE = "sft_data_generation_prompts_edited.yaml"

HF_TEST_DATASET_PATH = f"/app/dataset/preprocessed/sllm_ready_generated_prompts_{TARGET_MODEL}_hf_dataset/test"

# 🎯 METADATA: Contains the "word" keys
METADATA_FILE = "/app/dataset/sllm_all_ready.jsonl"

def build_metadata_lookup(filepath: str) -> dict:
    """Loads the original BERT data into a fast dictionary mapped by [origin_id][pair_id]."""
    lookup = {}
    print(f"🔄 Loading BERT metadata from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = str(doc.get("id", ""))
            lookup[doc_id] = {}
            
            for meta in doc.get("meta_info", []):
                # 🛑 EXTRA SAFETY: Skip BEE-BEE metadata if it exists in the file
                s_type = meta.get("subject", {}).get("source_type", "")
                o_type = meta.get("object", {}).get("source_type", "")
                if s_type == "BEE" and o_type == "BEE":
                    continue

                pair_key = str(meta.get("pair_id", ""))
                if not pair_key.startswith("["):
                    pair_key = f"[{pair_key}]" 
                lookup[doc_id][pair_key] = meta
    print("✅ Metadata loaded successfully (BEE-BEE excluded)!")
    return lookup

def main():
    if not os.path.exists(PREDICTIONS_FILE) or not os.path.exists(HF_TEST_DATASET_PATH):
        print("❌ Error: Missing prediction file or HF dataset!")
        return

    # 1. Initialize dependencies
    parser = KnowledgeParser(prompt_config_path=PROMPT_CONFIG_FILE)
    metadata_lookup = build_metadata_lookup(METADATA_FILE)
    
    print(f"📂 Loading HF Test Dataset...")
    hf_dataset = load_from_disk(HF_TEST_DATASET_PATH)

    print(f"🔄 Merging and Parsing {PREDICTIONS_FILE}...")
    with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        pred_lines = f_in.readlines()
        
        if len(pred_lines) != len(hf_dataset):
            print(f"⚠️ WARNING: Row count mismatch!")

        # 2. ZIP the predictions and HF Dataset together
        for i, (pred_line, hf_row) in enumerate(tqdm(zip(pred_lines, hf_dataset), total=len(pred_lines), desc="Processing")):
            pred_data = json.loads(pred_line)
            raw_pred = pred_data.get('prediction', "")
            full_input = pred_data.get('full_input', "")

            origin_id = str(hf_row.get('origin_id', 'UNKNOWN'))
            
            # 🎯 Task Mapper: "bee_bee" and "ner_bee_true_only" removed
            task_mapper = {
                "ner_ner": "NER-NER",
                "ner_bee": "NER-BEE",
                "combine_all": "COMBINE_ALL"
            }
            raw_hf_task = str(hf_row.get('task', 'combine_all')).lower()

            # Skip unsupported tasks for eval
            if raw_hf_task in ("bee_bee", "ner_bee_true_only"):
                continue

            true_task = task_mapper.get(raw_hf_task, "COMBINE_ALL")
            mode = str(hf_row.get('mode', 'UNKNOWN'))

            # 3. Parse Tuples
            pred_tuples = parser.parse(raw_pred, expected_task=true_task)

            # 4. Inject Rich BERT Metadata (NER-NER and NER-BEE only)
            rich_preds = []
            for p_id, sub, obj, rel in pred_tuples:
                meta = metadata_lookup.get(origin_id, {}).get(p_id)
                # If metadata is missing (likely because it was a BEE-BEE pair we filtered), skip it
                if not meta:
                    continue

                record = {
                    "pair_id": p_id, 
                    "subject": meta.get("subject", sub), 
                    "object": meta.get("object", obj), 
                    "relation": rel
                }
                rich_preds.append(record)

            # 5. Build Final Record
            final_record = {
                "origin_id": origin_id,
                "expected_task": true_task,
                "mode": mode,
                "pred_tuples": rich_preds,
                "full_input": full_input
            }

            # 6. Process Ground Truth
            if EVAL_MODE:
                raw_gold = pred_data.get('ground_truth', "")
                gold_tuples = parser.parse(raw_gold, expected_task=true_task)
                
                rich_golds = []
                for p_id, sub, obj, rel in gold_tuples:
                    meta = metadata_lookup.get(origin_id, {}).get(p_id)
                    if not meta:
                        continue
                        
                    record = {
                        "pair_id": p_id, 
                        "subject": meta.get("subject", sub), 
                        "object": meta.get("object", obj), 
                        "relation": rel
                    }
                    rich_golds.append(record)
                    
                final_record["ground_truth"] = rich_golds

            # Write processed record
            f_out.write(json.dumps(final_record, ensure_ascii=False) + "\n")

            if i == 0:
                print("\n🔍 --- AUDIT OF ROW 1 --- 🔍")
                print(json.dumps(final_record, ensure_ascii=False, indent=2))

    print(f"\n✅ Pipeline Complete! (BEE-BEE removed). Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()