import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
import re
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest # 🎯 Import vLLM's LoRA handler
from datasets import load_from_disk

# --- CONFIG ---
MODE = "qwen" # Change to gemma4, gemma3, qwen, or qwen3.5_thinktags

LORA_PATH = None # Default to None unless specified

# 🎯 DYNAMIC MODEL ROUTING
if MODE == "gemma4":
    BASE_MODEL_PATH = "/app/models/gemma4-E4B-it/"
elif MODE == "gemma3":
    BASE_MODEL_PATH = "/app/models/gemma3-12b/"
elif MODE == "qwen":
    BASE_MODEL_PATH = "/app/models/qwen3.5-9b/"
elif MODE == "qwen_checkpoint1620":
    # 🎯 Separate the Base Model from the LoRA Adapter!
    BASE_MODEL_PATH = "/app/models/qwen3.5-9b/" 
    LORA_PATH = f"/app/models/{MODE}"
else:
    raise ValueError(f"❌ Unknown MODE: {MODE}. Please choose gemma4, gemma3, qwen, or qwen3.5_thinktags.")

# 🎯 Eval preprocessing output (flat Dataset, no train/val/test split).
# Must match `data.output_path` in generate_eval_data_config.yaml.
TEST_DATA_PATH = "/app/dataset/preprocessed/eval_qwen_no_fewshot_hf"
FINAL_OUTPUT_FILE = f"vllm_predictions_{MODE}.jsonl"
DATA_LIMIT = None

# 🎯 THE TOGGLE: Auto-detects stop tokens from the resolved model path.
if "gemma4" in BASE_MODEL_PATH.lower() or "gemma-4" in BASE_MODEL_PATH.lower():
    MODEL_FORMAT = "gemma4"
    stop_tokens = ["<turn|>"]
elif "gemma" in BASE_MODEL_PATH.lower():
    MODEL_FORMAT = "gemma3"
    stop_tokens = ["<end_of_turn>"]
else:
    MODEL_FORMAT = "qwen"
    stop_tokens = ["<|im_end|>"]

def run_evaluation_pipeline():
    print(f"🚀 Initializing vLLM for {MODEL_FORMAT.upper()} mode...")
    
    # 🎯 Initialize vLLM with LoRA enabled if an adapter is provided
    llm = LLM(
        model=BASE_MODEL_PATH, 
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        enable_lora=True if LORA_PATH else False,
        max_loras=1,
        max_lora_rank=8, # Standard rank, vLLM needs this allocated upfront
    )
    
    sampling_params = SamplingParams(temperature=0, max_tokens=8192, stop=stop_tokens)

    prompts_to_run = []
    ground_truths = []

    print(f"📂 Reading HF dataset from {TEST_DATA_PATH}...")
    dataset = load_from_disk(TEST_DATA_PATH)
    if DATA_LIMIT and DATA_LIMIT < len(dataset):
        dataset = dataset.select(range(DATA_LIMIT))

    # 🎯 Eval preprocessing already splits prompt / ground_truth at source.
    # We consume those columns directly — no split_tag slicing needed.
    for entry in dataset:
        prompts_to_run.append(entry["prompt"])
        ground_truths.append(entry.get("ground_truth", ""))

    if prompts_to_run:
        print("\n" + "="*80)
        print(f"🔍 DIAGNOSTIC: EXACT PROMPT SENT TO {MODEL_FORMAT.upper()} (ROW 1)")
        print("="*80)
        print(prompts_to_run[0])
        print("="*80 + "\n")
    
    print(f"⚡ Running inference on {len(prompts_to_run)} rows...")
    
    # 🎯 Dynamically apply the LoRA request during generation
    if LORA_PATH:
        print(f"🔗 Attaching LoRA Adapter from: {LORA_PATH}")
        lora_request = LoRARequest("adapter", 1, LORA_PATH)
        outputs = llm.generate(prompts_to_run, sampling_params, use_tqdm=True, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts_to_run, sampling_params, use_tqdm=True)

    with open(FINAL_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i, output in enumerate(outputs):
            prediction = output.outputs[0].text.strip()
            
            # 🎯 CLEANUP: Strip thinking tags so parser gets raw JSON
            if MODEL_FORMAT == "gemma4":
                prediction = re.sub(r'<\|think\|>.*?<channel>', '', prediction, flags=re.DOTALL).strip() 
            elif MODE == "qwen_checkpoint1080":
                prediction = re.sub(r'<think>.*?</think>', '', prediction, flags=re.DOTALL).strip()

            record = {
                "full_input": prompts_to_run[i], 
                "prediction": prediction,
                "ground_truth": ground_truths[i]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Saved to: {FINAL_OUTPUT_FILE}")

if __name__ == "__main__":
    run_evaluation_pipeline()