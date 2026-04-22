"""
Gemma-4-E4B SFT 학습 스크립트 (Unsloth 기반)

전처리 산출물: /app/dataset/preprocessed/sllm_ready_generated_prompts_gemma4_hf_dataset/
  (PR 1/2 적용 후 생성된 raw text 형식 — chat template 이미 적용됨)

학습 설정:
  - Quantization: load_in_16bit (정확도 우선, ~17GB VRAM)
  - LoRA: r=8 / alpha=16 (scaling 2.0)
  - chat_template 별도 적용 안 함 (preprocessing이 이미 <bos><|turn>... 형식으로 컴파일)
  - train_on_responses_only 적용 (instruction 토큰 마스킹 → response만 학습)

참조:
  - https://unsloth.ai/docs/models/gemma-4/train
  - https://github.com/unslothai/unsloth/discussions/4921 (Gemma 4 fixes)

요구 패키지 버전:
  - unsloth >= 2026.4.4 (gradient accumulation fix 등)
  - transformers >= 4.57 (Gemma 4 첫 지원: 2026.4.1)
  - trl >= 0.11.0
  - torch >= 2.1.0
"""

# =================================================================
# 1. CRITICAL SETUP
# =================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["WANDB_MODE"] = "offline"

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
import torch
from argparse import Namespace
from datasets import load_from_disk, interleave_datasets
from trl import SFTTrainer, SFTConfig

# =================================================================
# 2. CONFIG
# =================================================================

MODEL_NAME = '/app/host/models/gemma4-E4B-it/'
# _v2 dataset: diversity-aware sampling + 10만 target + C1~M5 fixes 반영
# 기존 _hf_dataset (v1)은 유지해서 현재 돌고 있는 학습과 충돌 없도록.
DATA_PATH = '/app/pred_data/sllm_ready_generated_prompts_gemma4_v2_hf_dataset'

# Gold dataset (train-only, human-verified) — interleave_datasets로 확률 혼합.
# 없으면 gracefully fall back to main-only.
GOLD_DATA_PATH = '/app/pred_data/sllm_ready_generated_prompts_gemma4_gold_v2_hf_dataset'
P_GOLD = float(os.environ.get("P_GOLD", "0.10"))

# Eval toggle — val set이 수천 row일 때 학습 시간 크게 증가. 기본 False.
# 켜려면: ENABLE_EVAL=1 python training/sft_gemma4.py
ENABLE_EVAL = os.environ.get("ENABLE_EVAL", "0").lower() in ("1", "true", "yes")

RANK = 8
ALPHA = 16
LR = 1e-4
BATCH = 8
EPOCH = 3

TAG = "_v2" + "_rank" + str(RANK) + "_alpha" + str(ALPHA) + "_lr" + str(LR) + "_batch" + str(BATCH) + "_ep" + str(EPOCH)

OUTPUT_PATH = f'/app/train_result/gemma4-e4b_sft{TAG}'
RUN_NAME = f"gemma4_e4b_sft{TAG}"

# Gemma 4 LoRA target modules (attention + MLP)
GEMMA4_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

args = Namespace(
    model_name=MODEL_NAME,
    rank=RANK,
    alpha=ALPHA,
    maxlen=8192,
    lr=LR,
    batch=BATCH,
    epoch=EPOCH,
)

# =================================================================
# 3. dtype 자동 감지
# =================================================================
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

# =================================================================
# 4. 모델 로드 (16-bit LoRA — 정확도 우선)
# =================================================================
# load_in_16bit=True: 4bit 양자화 우회. unsloth 공식 16bit 경로
# full_finetuning=False: PEFT(LoRA) 경로 활성화
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model_name,
    max_seq_length=args.maxlen,
    dtype=auto_dtype,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
)

# 학습 모드 전환
FastLanguageModel.for_training(
    model,
    use_gradient_checkpointing="unsloth",  # unsloth 최적화 모드 (메모리 추가 절약)
)

# =================================================================
# 5. LoRA 설정
# =================================================================
model = FastLanguageModel.get_peft_model(
    model,
    r=args.rank,
    lora_alpha=args.alpha,
    # lora_dropout 0 → 0.05: SFT overfitting 완화 (sft_qwen과 정렬)
    lora_dropout=0.05,
    bias="none",
    random_state=3407,
    use_gradient_checkpointing="unsloth",
    target_modules=GEMMA4_TARGET_MODULES,
)

# Trainable parameter 카운트 검증 (이슈 #4907 — Gemma 4 26B에서 비정상 trainable count 발생 케이스)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"📊 LoRA trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
assert trainable > 0, "LoRA trainable params == 0 — get_peft_model 실패"

# =================================================================
# 6. 데이터 로드
# =================================================================
print(f"📂 Loading dataset from {DATA_PATH}...")
full_dataset = load_from_disk(DATA_PATH)
main_train = full_dataset["train"]

# Gold mixing: if a gold HF dataset exists, interleave with main at p_gold.
# stopping_strategy="all_exhausted" keeps cycling gold (which is tiny) until main is exhausted,
# so gold is effectively oversampled to match the main train budget at ratio P_GOLD.
if os.path.exists(GOLD_DATA_PATH):
    try:
        gold_full = load_from_disk(GOLD_DATA_PATH)
        gold_train = gold_full["train"]
        print(f"🪙 Gold dataset loaded: {len(gold_train)} rows from {GOLD_DATA_PATH}")
        print(f"   Mixing with p_gold={P_GOLD} (main={1-P_GOLD:.2f}) via interleave_datasets")
        train_dataset = interleave_datasets(
            [main_train, gold_train],
            probabilities=[1.0 - P_GOLD, P_GOLD],
            stopping_strategy="all_exhausted",
            seed=3407,
        )
        print(f"   Mixed train length: {len(train_dataset)} rows")
    except Exception as e:
        print(f"⚠️ Gold interleave failed ({e}) — falling back to main-only")
        train_dataset = main_train
else:
    print(f"ℹ️  No gold dataset at {GOLD_DATA_PATH} — main-only train")
    train_dataset = main_train

if not ENABLE_EVAL:
    eval_dataset = None
    print("⏭️  ENABLE_EVAL=0 → validation split 로드하지 않음 (eval 비활성, 학습 속도 확보)")
elif "validation" in full_dataset:
    eval_dataset = full_dataset["validation"]
else:
    eval_dataset = None
    print("⚠️ ENABLE_EVAL=1이지만 validation split 없음 → eval 비활성")

print(f"✅ Train: {len(train_dataset)} rows")
if eval_dataset:
    print(f"✅ Eval:  {len(eval_dataset)} rows")

# 데이터 sanity check
print("\n" + "=" * 60)
print("👀 SAMPLE PEEK [train index 0]")
print("=" * 60)
sample_text = train_dataset[0]["text"]
print(sample_text[:1500])
print("..." if len(sample_text) > 1500 else "")
print("=" * 60)

# Chat template 정합성 검증 — preprocessing의 PR 2.1 적용 결과 확인
assert sample_text.startswith("<bos>"), \
    "❌ sample text가 <bos>로 시작하지 않음. preprocessing 측 _apply_chat_template 미적용 의심"
assert "<|turn>" in sample_text, \
    "❌ sample text에 <|turn> 없음. Gemma 4 chat 포맷 아님"
print("✅ Chat template 정합성 OK (<bos> + <|turn> 확인)")

# =================================================================
# 7. SFTTrainer 구성
# =================================================================
sft_args = SFTConfig(
    dataset_text_field="text",
    per_device_train_batch_size=args.batch,
    max_seq_length=args.maxlen,
    gradient_accumulation_steps=4,
    # warmup_steps=50 (0.25% of total) → warmup_ratio=0.03 (3%, aligned with sft_qwen)
    warmup_ratio=0.03,
    num_train_epochs=args.epoch,
    learning_rate=args.lr,
    optim="adamw_8bit",
    weight_decay=0.05,
    lr_scheduler_type="cosine",
    seed=3407,
    dataset_num_proc=8,
    gradient_checkpointing=True,
    auto_find_batch_size=False,
    packing=False,
    output_dir=OUTPUT_PATH,

    # ===== WandB =====
    report_to="wandb",
    run_name=RUN_NAME,
    logging_steps=1,

    # ===== Save / Eval =====
    # save/eval_steps unified with sft_qwen at 500 (was 1000/5000).
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,

    eval_strategy="steps" if eval_dataset else "no",
    eval_steps=500,
    # Activate best-model loading when validation exists (was hardcoded False).
    load_best_model_at_end=True if eval_dataset else False,
    metric_for_best_model="eval_loss" if eval_dataset else None,
    greater_is_better=False,

    per_device_eval_batch_size=1,
    eval_accumulation_steps=1,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=sft_args,
)

# 🎯 Response-only 학습 — instruction(user 턴) 토큰 마스킹, model 응답만 loss 계산
# Unsloth 공식 권장: 정확도 +1% 이상
# Gemma 4 chat template 토큰: <|turn>user\n ... <|turn>model\n ...
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|turn>user\n",
    response_part="<|turn>model\n",
)
print("✅ train_on_responses_only applied (instruction tokens masked)")

# =================================================================
# 8. 학습 실행 + 저장
# =================================================================
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

actual_batch_size = trainer.args.per_device_train_batch_size
print(f"🕵️ Trainer per_device_train_batch_size: {actual_batch_size}")

print("\n🚀 Starting training...")
trainer_stats = trainer.train()

# =================================================================
# 💾 FINAL SAVE
# =================================================================
print("\n💾 Training complete! Saving final model...")
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
print(f"✅ Final model saved to: {OUTPUT_PATH}")
