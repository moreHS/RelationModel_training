# 2026-05-06 — Gemma4 + vLLM LoRA 미지원 워크어라운드 (Adapter Merge)

## 배경 (Symptom)

[gemma4_inference/vllm_model_inference_w_edited_prompts.py](../gemma4_inference/vllm_model_inference_w_edited_prompts.py)로 학습한 LoRA adapter를 평가 추론에 사용하려 하자 vLLM EngineCore가 모델 로드 직후 다음 에러로 실패:

```
ValueError: Gemma4ForConditionalGeneration does not support LoRA yet.
```

환경:
- vLLM `0.18.2rc1.dev73+gdb7a17ecc` (개발 빌드)
- Base: `/app/host/models/gemma4-E4B-it/`
- LoRA adapter: `/app/train_result/gemma4-e4b_sft_v2_rank8_alpha16_lr0.0001_batch8_ep3_0.05`

## 원인 분석

vLLM framework 자체의 한계이며 사용자 코드와 무관:

1. **모델 클래스 미등록**: Gemma4 공식 모델 클래스는 multimodal `Gemma4ForConditionalGeneration` (text+vision). vLLM의 LoRA 시스템은 모델이 `SupportsLoRA` mixin을 명시적으로 declare 해야 작동하는데, 이 클래스는 wiring이 안 됨.
2. **KV-sharing module aliasing**: Gemma4 아키텍처가 일부 layer에서 KV projection을 공유 → vLLM LoRA manager가 같은 weight에 중복으로 adapter를 등록하려는 문제.
3. **부분 진척**: 텍스트 LoRA key mapping은 vLLM PR #38844로 일부 land. multimodal class 전체에 `SupportsLoRA` 적용은 PR #39931에서 진행 중. May 2026 기준 vLLM Issue #39246은 여전히 open.

vLLM 공식 권장 워크어라운드:
> "you can't pass an adapter path at inference time like you can with Llama or Qwen, you have to **merge the LoRA into the base weights and serve the merged checkpoint** as a standalone model."

## 검토한 선택지

| 선택지 | 장점 | 단점 | 결정 |
|---|---|---|---|
| **A. Unsloth `save_pretrained_merged()`** | 학습 시 정확히 같은 코드패스로 base 로드 → 호환 보장. 단일 명령. | Unsloth #1352로 vision 모델에서 silent merge fail 가능 (검증 필요) | **선택** |
| B. PEFT `merge_and_unload()` + 수동 save | Unsloth 의존 없음. 표준 transformers/peft 흐름. | Gemma4 multimodal 클래스에서 vision tower 처리 미세 차이 가능. boilerplate 더 많음 | A 실패 시 폴백 |
| C. vLLM nightly/PR #39931 빌드 시도 | 워크어라운드 없이 LoRA path 그대로 사용 가능 | 빌드 자체가 아직 안정 미보장. May 2026 시점 미release | 폐기 |
| D. transformers + PEFT 직접 추론 | 의존성 단순 | vLLM 대비 throughput 1/10 이하 | 폐기 |

A → B 폴백 절차는 plan 파일에 명시.

## 결정

**Unsloth `save_pretrained_merged(..., save_method="merged_16bit")` 기반의 일회성 merge 스크립트를 도입**하고, eval config의 `model.name`을 merged 디렉토리로 변경, `inference.lora_path`는 null로 고정.

핵심 의도:
- **재학습 불필요**: 이미 학습된 adapter를 그대로 재사용
- **추론 코드 변경 최소화**: `vllm_model_inference_w_edited_prompts.py`는 `lora_path=None`이면 자동으로 LoRA 비활성. 기존 분기 그대로 활용
- **검증 가능**: `verify_merge.py`로 Unsloth #1352(silent merge fail) 즉시 검출

## 변경 내역

### 신규 파일
- **[scripts/merge_lora_to_base.py](../scripts/merge_lora_to_base.py)**
  - argparse `--lora_path`, `--out_path`, `--maxlen`
  - `FastLanguageModel.from_pretrained(adapter dir, dtype=bf16, load_in_4bit=False)` → `save_pretrained_merged(out, tokenizer, "merged_16bit")`
  - 기본 출력 경로: `<lora_path>_merged_16bit`. 이미 존재하면 거부 (실수 방지)
- **[scripts/verify_merge.py](../scripts/verify_merge.py)**
  - 동일 weight key를 base/merged에서 각각 로드해 mean abs diff 비교
  - `< 1e-6`이면 silent merge fail로 판정 → exit 1
  - Multimodal/dense 양쪽 키 패턴(`language_model.layers.*` / `model.layers.*`)을 자동 시도

### 수정
- **[gemma4_inference/generate_eval_data_config.yaml](../gemma4_inference/generate_eval_data_config.yaml)**
  - `model.name` → merged 디렉토리 (base는 주석으로 보존)
  - `inference.lora_path` → 항상 `null`로 유지 (워크어라운드 사유 주석)
  - `max_lora_rank`는 향후 LoRA 정식 지원 대비 그대로 유지

## 운영 절차

```bash
# 1) merge 실행 (10-15분, 디스크 +8-9GB)
python3 -m scripts.merge_lora_to_base \
    --lora_path /app/train_result/gemma4-e4b_sft_v2_rank8_alpha16_lr0.0001_batch8_ep3_0.05

# 2) merge 무결성 검증 (Unsloth #1352 회피)
python3 -m scripts.verify_merge \
    --base_dir /app/host/models/gemma4-E4B-it \
    --merged_dir /app/train_result/gemma4-e4b_sft_v2_rank8_alpha16_lr0.0001_batch8_ep3_0.05_merged_16bit

# 3) 기존 config의 model.name이 이미 merged 경로를 가리킴 → 그대로 추론
cd gemma4_inference
python3 vllm_model_inference_w_edited_prompts.py
python3 merge_w_metadata.py
```

## 트레이드오프

- **Disk usage**: merged 16bit 추가로 ~8-9GB. 학습 결과 dir과 별개로 둬서 정리 시점 유연.
- **다중 adapter 비교 불가**: 각 adapter를 별도 merged 디렉토리로 만들어야 함 (LoRA hot-swap 패턴 사용 불가).
- **재학습 시마다 merge 필요**: 학습 종료 직후 자동화하면 편하지만 이번 범위엔 수동 단계로 둠.
- **Multi-LoRA 동시 비교**: vLLM의 LoRARequest 기반 multi-adapter serving 불가. 필요 시 vLLM 공식 지원까지 대기.

## 폐기 시점 (vLLM이 Gemma4 LoRA 정식 지원하면)

- vLLM Issue #39246 close + release notes에 "Gemma4 LoRA support" 언급되면:
  1. eval config에서 `model.name`을 base로 되돌리고 `inference.lora_path`를 학습 OUTPUT_PATH로 활성화
  2. `scripts/merge_lora_to_base.py`는 폐기하지 않고 유지 (offline 배포/exporting 용도로 여전히 유용)
  3. 본 DECISIONS 파일에 폐기 일자 + commit 추가 기록

## 알려진 함정

- **Unsloth #1352**: vision 모델에서 16bit merge가 base만 저장. Gemma4-E4B(dense)에서도 발생 가능 → `verify_merge.py` 필수.
- **Unsloth #3633**: `merged_16bit` 시 FP16 base를 새 `.cache/`에 재다운로드. 디스크 여유 미리 확보.
- **Unsloth #4820**: Gemma4 26B-A4B MoE 한정 merge 실패. 본 프로젝트는 E4B (dense)라 해당 없음.

## Sources

- [vLLM Issue #39246 — Add LoRA support for Gemma4ForConditionalGeneration](https://github.com/vllm-project/vllm/issues/39246)
- [vLLM Issue #41403 — Gemma 4 multimodal blocker stack](https://github.com/vllm-project/vllm/issues/41403)
- [Unsloth vLLM Deployment Guide](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide)
- [Unsloth Issue #1352 — vision merge 누락 버그](https://github.com/unslothai/unsloth/issues/1352)
- [Unsloth Issue #3633 — merged_16bit cache 재다운로드](https://github.com/unslothai/unsloth/issues/3633)
- [Unsloth Issue #4820 — Gemma 4 MoE merge 실패](https://github.com/unslothai/unsloth/issues/4820)
