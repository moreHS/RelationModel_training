# 2026-04-13 — sft_gemma4.py 신규 학습 코드 (Unsloth Gemma-4-E4B-it SFT)

## 배경

PR 1/2 전처리 산출물(`/app/dataset/preprocessed/sllm_ready_generated_prompts_gemma4_hf_dataset/`)을 Gemma-4-E4B-it 모델로 SFT 파인튜닝할 학습 스크립트 필요. 기존 [training/sft_qwen.py](../training/sft_qwen.py)는 Qwen 3.5용 예시 (옛 데이터 경로 잔재 + chatml 템플릿 + train_on_responses_only 미호출) 라 그대로 사용 불가.

Gemma 4는 2026-04-01 transformers에 추가된 신규 모델이라 unsloth 관련 known issue 다수. 공식 docs + GitHub discussions/issues + 사용자 후기 종합해서 검증된 패턴으로 작성.

## 조사 자료

- [Unsloth 공식 Gemma 4 가이드](https://unsloth.ai/docs/models/gemma-4/train)
- [Unsloth Gemma 4 Fixes 릴리스 노트](https://github.com/unslothai/unsloth/discussions/4921)
- [Unsloth Gemma 4 26B LoRA 비정상 trainable count 이슈 #4907](https://github.com/unslothai/unsloth/issues/4907)
- [Gabriel Preda Medium 글: From OOM Errors to Working Model](https://medium.com/@gabi.preda/from-oom-errors-to-working-model-fine-tuning-gemma-4-e2b-step-by-step-using-unsloth-ef7873e59efd)
- 기존 [training/sft_qwen.py](../training/sft_qwen.py) 구조

## 결정 1: Quantization = `load_in_16bit=True` (LoRA, ~17GB VRAM)

### 검토한 선택지

- (A) `load_in_4bit=True` (QLoRA, ~8GB VRAM) — Unsloth 공식 default 권장
- (B) `load_in_16bit=True` (LoRA, ~17GB VRAM) — 정확도 우선 ✅ **선택**
- (C) `load_in_8bit=True` — 비표준, unsloth 권장 안 함

### 선택 이유

사용자 결정. VRAM 17GB 여유 있는 docker 환경 + 정확도 우선. 4bit 양자화는 SFT에서 ~1-2% 정확도 손실 가능, NER/RE 같은 정확한 라벨 출력이 중요한 태스크엔 영향 큼.

### 트레이드오프

- 메모리 ~2배 사용 (8GB → 17GB)
- 학습 속도 약간 느려짐
- 정확도 향상 보장

## 결정 2: LoRA r=8 / alpha=16 (scaling 2.0)

### 검토한 선택지

- (A) Unsloth 공식 r=8/alpha=8 (scaling 1.0)
- (B) 기존 sft_qwen.py 패턴 r=8/alpha=16 (scaling 2.0) ✅ **선택**
- (C) r=16/alpha=16 (큰 LoRA)
- (D) r=16/alpha=32 (큰 + scaling 2.0)

### 선택 이유

기존 Qwen 학습 코드와 일관성 유지. scaling 2.0 (alpha/r=2)이 SFT에서 더 공격적 학습 — 우리는 작은 데이터셋(target_total_training_samples=100~수천)이라 강한 신호 필요.

### 트레이드오프

- alpha=8 (공식)보다 학습 속도 빠름, 단 overfitting 가능성 약간 ↑
- r=16보다 파라미터 적음 → 학습/저장 효율적

## 결정 3: Chat Template — preprocessing 측에 위임 (별도 적용 안 함)

### 배경

기존 sft_qwen.py는 [라인 86-90](../training/sft_qwen.py#L86) `get_chat_template(tokenizer, "chatml")` 호출. 이는 토크나이저의 `chat_template` 속성을 chatml로 덮어씀.

Gemma 4는 PR 2.1에서 preprocessing의 `_apply_chat_template`이 `tokenizer.apply_chat_template`에 위임 → 산출물 raw text가 이미 `<bos><|turn>system\n... <turn|>` 형식으로 컴파일됨. SFTTrainer는 `dataset_text_field="text"`로 raw text를 그대로 학습.

### 결정

**`get_chat_template()` 호출 안 함.** Preprocessing이 단일 source of truth. 학습 코드는 chat 포맷 손대지 않음.

### 검증 항목 (코드에 assertion 포함)

```python
assert sample_text.startswith("<bos>"), "preprocessing chat template 미적용 의심"
assert "<|turn>" in sample_text, "Gemma 4 chat 포맷 아님"
```

## 결정 4: train_on_responses_only 활성화

### 배경

기존 sft_qwen.py [라인 11](../training/sft_qwen.py#L11)에 import만 있고 호출 안 함 — instruction(user) + response(model) 모든 토큰에 loss 계산되어 컴퓨팅 낭비 + 학습 효율 저하.

Unsloth 공식 권장: `train_on_responses_only` 사용 시 정확도 +1% 이상.

### Gemma 4용 인자

```python
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|turn>user\n",
    response_part="<|turn>model\n",
)
```

`<|turn>user\n` 이전 토큰 + `<|turn>user\n` ~ `<|turn>model\n` 사이 토큰 (system_prompt + user 메시지)이 마스킹되어 loss=0. `<|turn>model\n` 이후 응답만 loss 계산.

### 검증 자료

- [Unsloth 공식 Gemma 4 가이드](https://unsloth.ai/docs/models/gemma-4/train) 명시
- [GitHub Discussion #2828](https://github.com/unslothai/unsloth/discussions/2828) — train_on_responses_only 동작 설명

## 결정 5: use_gradient_checkpointing="unsloth"

### 배경

기존 sft_qwen.py는 `use_gradient_checkpointing=True`. Unsloth 공식은 `"unsloth"` 문자열 권장 (Unsloth 자체 최적화 버전, 메모리 추가 절약).

### 결정

`"unsloth"` 사용. 16-bit 모드라 메모리 여유 미미 → 추가 절약 가치 있음.

## 결정 6: SFTConfig 하이퍼파라미터

기존 sft_qwen.py 패턴 + Unsloth 공식 default 절충:

| 항목 | 값 | 근거 |
|---|---|---|
| `per_device_train_batch_size` | 4 | 기존 패턴 |
| `gradient_accumulation_steps` | 8 | 기존 패턴 (effective batch = 32) |
| `num_train_epochs` | 3 | 기존 패턴 |
| `learning_rate` | 2e-4 | 공식 + 기존 일치 |
| `optim` | `adamw_8bit` | 공식 권장 |
| `weight_decay` | 0.05 | 기존 패턴 (공식 0.001보다 강한 정규화) |
| `lr_scheduler_type` | `cosine` | 기존 패턴 (공식 linear와 다름, cosine이 SFT에서 일반적) |
| `warmup_steps` | 5 | 공식 + 기존 일치 |
| `seed` | 3407 | 공식 권장 |
| `dataset_num_proc` | 8 | 기존 패턴 |
| `packing` | False | 공식 default (responses_only와 호환) |
| `save_strategy` | steps, every 180 | 기존 패턴 |
| `eval_strategy` | steps, every 180 | 기존 패턴 |
| `load_best_model_at_end` | True (eval 있을 때만) | 기존 패턴 |

## 결정 7: Known Issues 대응

### Loss 폭발 (gradient accumulation)
- 옛 코드에서 loss 300-400 발생 — Unsloth ≥2026.4.4에서 내부적으로 보정. 별도 코드 변경 불필요. text-only 학습이라 loss 1-3 기대 (multimodal은 13-15가 정상)

### `use_cache=False` garbage logits
- E2B/E4B에서 attention 손상 — 현재 코드는 `use_cache` 명시 안 함. SFTTrainer 내부 default는 `use_cache=False` (gradient checkpointing 활성화 시 자동) — 그러나 unsloth가 이를 보정 처리. 명시적으로 `use_cache=True` 강제하면 gradient checkpointing과 충돌하므로 그대로 default 유지

### LoRA trainable=0 (이슈 #4907, 26B MoE)
- 우리는 E4B (dense) 사용 → 영향 없음. 단 안전망으로 코드에 `assert trainable > 0` 추가

### multimodal vs text-only
- 우리 데이터는 text-only NER/RE → `FastLanguageModel` 사용 (multimodal 학습은 `FastVisionModel` 별도 필요). vision/audio LoRA layer 학습 안 됨

## 명명 규칙

- TAG: `_pr2_v1` (PR 2 전처리 결과 + 버전 v1)
- OUTPUT_PATH: `/app/models/gemma4-e4b_sft_pr2_v1`
- RUN_NAME: `gemma4_e4b_sft_pr2_v1`

→ 향후 PR 3 적용 후엔 `_pr3_v1`, 동일 PR 내 재실행은 `_pr2_v2` 식으로 증가

## 변경 안 한 것

- `training/sft_qwen.py` 보존 — Qwen 학습 시 참고 자료로 가치 있음
- preprocessing 코드 — 이미 PR 1/2 완료
- multimodal 학습 — 데이터가 text-only

## 검증 결과

- ✅ `python3 -m py_compile training/sft_gemma4.py` 통과
- ✅ AST parse + imports 정합 (unsloth, unsloth.chat_templates, torch, argparse, datasets, trl)
- ⏸ Runtime 검증은 사용자 docker 환경에서 dry-run 권장 (`max_steps=1`로 1 step 학습 후 OOM/loss 확인)

## 향후 작업 (별도 PR 후보)

- Hyperparameter sweep (rank/alpha/lr 그리드)
- DPO/ORPO 후속 학습 (현재는 SFT만)
- Multi-GPU DDP (현재는 단일 GPU)
- Inference 스크립트 (`predict_gemma4.py`)
