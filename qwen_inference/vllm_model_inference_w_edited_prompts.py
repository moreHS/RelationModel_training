"""
vLLM inference for eval.

All paths are read from generate_eval_data_config.yaml (single source of truth).
Override via env vars:
  - EVAL_CONFIG: path to config yaml (default: ./generate_eval_data_config.yaml)
  - CUDA_VISIBLE_DEVICES: GPU selection
"""
import os
import json
import re
import sys
import yaml

# Allow env override before CUDA is inited
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_from_disk


def detect_model_format(model_path: str):
    """Returns (model_format, stop_tokens) by matching substrings in model path."""
    lower = model_path.lower()
    if "gemma4" in lower or "gemma-4" in lower:
        return "gemma4", ["<turn|>"]
    if "gemma" in lower:
        return "gemma3", ["<end_of_turn>"]
    return "qwen", ["<|im_end|>"]


def strip_think_tags(text: str, model_format: str) -> str:
    """Remove <think>...</think> regardless of model format (safety net for any
    checkpoint that emits think tags). Eval preprocessing also strips stop tokens
    but doesn't touch think tags — this is the catch-all."""
    # Gemma4-style: <|think|>...<|/think|>
    text = re.sub(r'<\|think\|>.*?<\|/think\|>', '', text, flags=re.DOTALL)
    # Legacy text-form: <think>...</think>
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


def run_evaluation_pipeline(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    base_model_path = cfg["model"]["name"]
    test_data_path = cfg["data"]["output_path"]
    inf_cfg = cfg.get("inference", {})
    predictions_path = inf_cfg.get("predictions_path")
    lora_path = inf_cfg.get("lora_path")
    max_lora_rank = inf_cfg.get("max_lora_rank", 8)
    gpu_memory_utilization = inf_cfg.get("gpu_memory_utilization", 0.9)
    max_model_len = cfg.get("training_max_length", 8192)

    if predictions_path is None:
        raise ValueError(f"Config missing 'inference.predictions_path' in {cfg_path}")

    model_format, stop_tokens = detect_model_format(base_model_path)
    print(f"🚀 Initializing vLLM (format={model_format})")
    print(f"   Base model: {base_model_path}")
    print(f"   Eval dataset: {test_data_path}")
    print(f"   Predictions output: {predictions_path}")
    if lora_path:
        print(f"   LoRA adapter: {lora_path} (max_rank={max_lora_rank})")

    llm = LLM(
        model=base_model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_lora=bool(lora_path),
        max_loras=1 if lora_path else 0,
        max_lora_rank=max_lora_rank,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_model_len,
        stop=stop_tokens,
    )

    print(f"📂 Reading HF dataset from {test_data_path}")
    dataset = load_from_disk(test_data_path)

    prompts_to_run = [row["prompt"] for row in dataset]
    ground_truths = [row.get("ground_truth", "") for row in dataset]

    if prompts_to_run:
        print("\n" + "=" * 80)
        print(f"🔍 DIAGNOSTIC: ROW 0 PROMPT")
        print("=" * 80)
        print(prompts_to_run[0])
        print("=" * 80 + "\n")

    print(f"⚡ Running inference on {len(prompts_to_run)} rows")

    if lora_path:
        print(f"🔗 Attaching LoRA adapter: {lora_path}")
        lora_request = LoRARequest("adapter", 1, lora_path)
        outputs = llm.generate(prompts_to_run, sampling_params, use_tqdm=True, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts_to_run, sampling_params, use_tqdm=True)

    os.makedirs(os.path.dirname(predictions_path) or ".", exist_ok=True)
    with open(predictions_path, "w", encoding="utf-8") as f:
        for i, output in enumerate(outputs):
            prediction = output.outputs[0].text.strip()
            prediction = strip_think_tags(prediction, model_format)

            record = {
                "full_input": prompts_to_run[i],
                "prediction": prediction,
                "ground_truth": ground_truths[i],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Predictions saved to: {predictions_path}")


if __name__ == "__main__":
    cfg_path = os.environ.get("EVAL_CONFIG", "generate_eval_data_config.yaml")
    run_evaluation_pipeline(cfg_path)
