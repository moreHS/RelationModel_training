import os
os.environ['CUDA_VISIBLE_DEVICES']= '0'
os.environ["WANDB_MODE"] = "offline"
import re
from unsloth import FastLanguageModel
import torch, sys
import argparse
from unsloth.chat_templates import get_chat_template
from datasets import load_from_disk # 🎯 Changed to load_from_disk
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

TAG = "_no_think_tags_v2"
# _v2 dataset: diversity-aware sampling + 10만 target + C1~M5 fixes 반영
# 기존 _hf_dataset (v1)은 유지해서 현재 돌고 있는 학습과 충돌 없도록.
ROOT_DATA_PATH = '/app/dataset/preprocessed/sllm_ready_generated_prompts_qwen_v2_hf_dataset'
OUTPUT_PATH = f'/app/models/qwen3.5b-9b_test{TAG}'
RUN_NAME = f"qwen_3.5_9b_test{TAG}"

# Eval toggle — val set이 수천 row일 때 학습 시간 크게 증가. 기본 False.
# 켜려면: ENABLE_EVAL=1 python training/sft_qwen.py
ENABLE_EVAL = os.environ.get("ENABLE_EVAL", "0").lower() in ("1", "true", "yes")

def get_target_modules(model_name):
    model_name = model_name.lower()
    if "gemma" in model_name:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "qwen" in model_name:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "deepseek" in model_name:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        raise ValueError(f"모델명에 맞는 target_modules 설정이 필요합니다: {model_name}")

from argparse import Namespace

args = Namespace(
    model_name='/app/models/qwen3.5-9b',
    rank=8,
    alpha=16,
    maxlen=8192,
    lr=2e-4,
    batch=4,
    epoch=3,
    sampling=0,#2000,
    f_sample=1000,
    use_trans=True,  
    basemodel='qwen3_5'
)

# dtype 자동 설정
def get_auto_dtype():
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8:  # Ampere 이상
            return torch.bfloat16
        else:
            return torch.float16
    else:
        return torch.float32  # CPU fallback

auto_dtype = get_auto_dtype()
print(f"Torch Dtype is set to: {auto_dtype}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model_name,
    max_seq_length = args.maxlen,
    dtype = auto_dtype,                 
    load_in_4bit = True,
    load_in_8bit = False,
    # attn_implementation = "flash_attention_2"
)

# 학습 모드로 설정
FastLanguageModel.for_training(
    model,
    use_gradient_checkpointing = True, 
)

# LoRA 설정
# lora_dropout 0 → 0.05: SFT overfitting 완화 (weight_decay 단독 의존 대비 안정성 ↑)
model = FastLanguageModel.get_peft_model(
    model,
    r = args.rank,
    lora_alpha = args.alpha,
    lora_dropout = 0.05,
    bias = "none",
    random_state = 3407,
    use_gradient_checkpointing = True,
    target_modules = get_target_modules(args.model_name),
)

if 'qwen' in args.model_name:
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "chatml",
    )

# =================================================================
# 🎯 UPDATED DATASET LOADING LOGIC
# =================================================================
print(f"📂 Loading dataset from {ROOT_DATA_PATH}...")
full_dataset = load_from_disk(ROOT_DATA_PATH)

# Extract train and validation splits (Checking for 'val' or 'validation')
train_dataset = full_dataset["train"]

if not ENABLE_EVAL:
    eval_dataset = None
    print("⏭️  ENABLE_EVAL=0 → validation split 로드하지 않음 (eval 비활성, 학습 속도 확보)")
elif "validation" in full_dataset:
    eval_dataset = full_dataset["validation"]
else:
    print("⚠️ Warning: ENABLE_EVAL=1이지만 validation split이 없음 → eval 비활성")
    eval_dataset = None

print(f"✅ Loaded Train: {len(train_dataset)} rows")
if eval_dataset:
    print(f"✅ Loaded Eval: {len(eval_dataset)} rows")


####################################################
# def add_empty_think_tags(example):
#     # Finds the assistant start tag and immediately injects <think></think> after it
#     example["text"] = example["text"].replace(
#         "<|im_start|>assistant\n", 
#         "<|im_start|>assistant\n<think></think>\n"
#     )
#     return example

# print("💉 Injecting empty <think></think> tags into the datasets...")
# train_dataset = train_dataset.map(add_empty_think_tags, num_proc=8, desc="Adding empty think tags (Train)")
# if eval_dataset:
#     eval_dataset = eval_dataset.map(add_empty_think_tags, num_proc=8, desc="Adding empty think tags (Eval)")
####################################################
####################################################
# --- DEBUG: PEEK AT THE TRAINING DATA ---
####################################################
print("\n" + "="*60)
print("👀 SNEAK PEEK: PRINTING TRAIN DATASET [INDEX 0]")
print("="*60)

# Grab the first row's text
sample_text = train_dataset[0]["text"]
print(sample_text)

print("="*60)
# print("🛑 Stopping script here so you can review the sample!")
# import sys; sys.exit() # <--- This stops the script before training starts!
####################################################

# # ===========
# SFT_CONFIGS
# =================================================================

args = SFTConfig(
    dataset_text_field = "text",
        per_device_train_batch_size = args.batch,
        max_seq_length = args.maxlen,
        gradient_accumulation_steps = 8,
        # warmup_steps=5 (0.02% of total) → warmup_ratio=0.03 (3%, industry standard for SFT)
        warmup_ratio = 0.03,
        num_train_epochs = args.epoch,
        learning_rate = args.lr,
        optim = "adamw_8bit",
        weight_decay = 0.05,
        lr_scheduler_type = "cosine",
        seed = 3407,
        dataset_num_proc=8,
        gradient_checkpointing = True,
        auto_find_batch_size=False,
        packing=False,
        output_dir = OUTPUT_PATH,

        # ==========================================
        # 📊 1. WANDB LOGGING
        # ==========================================
        report_to = "wandb",
        run_name = RUN_NAME,
        logging_steps = 1,

        # ==========================================
        # 💾 2. SAVING & CHECKPOINTING
        # save/eval_steps unified across sft_qwen and sft_gemma4 (= 500 steps).
        # ==========================================
        save_strategy = "steps",
        save_steps = 500,
        save_total_limit = 2,

        # ==========================================
        # 🧪 3. EVALUATION & BEST MODEL
        # eval disabled when no validation split to avoid trainer crash.
        # ==========================================
        eval_strategy = "steps" if eval_dataset is not None else "no",
        eval_steps = 500,
        load_best_model_at_end = True if eval_dataset is not None else False,
        metric_for_best_model = "eval_loss" if eval_dataset is not None else None,
        greater_is_better = False,

        per_device_eval_batch_size = 1,
        eval_accumulation_steps = 1,
    )

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    args = args
)

# Response-only loss: mask the instruction/system portion so gradient flows only on
# the assistant's JSON answer. Keeps Qwen's loss target aligned with Gemma4's setup.
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
print("✅ Applied train_on_responses_only (instruction tokens masked).")


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# --- DEBUG CHECK BLOCK ---
actual_batch_size = trainer.args.per_device_train_batch_size
print(f"🕵️ CHECKING CONFIG: Trainer thinks batch size is: {actual_batch_size}")

# if actual_batch_size != 1:
#     raise ValueError(f"🛑 STOP! The trainer still thinks batch size is {actual_batch_size}. Please re-run the cell where 'trainer = SFTTrainer(...)' is defined!")
# else:
#     print("✅ Config is correct (1). Starting training...")
# # -------------------------


trainer_stats = trainer.train()

# ==========================================
# 💾 FINAL SAVE COMMAND (ADD THIS!)
# ==========================================
print("💾 Training complete! Saving final model...")
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
print("✅ Final model saved successfully!")