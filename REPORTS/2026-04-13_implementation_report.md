# 구현 완료 보고서 — Gemma 4 SFT 파이프라인 정합성 픽스

**작성일**: 2026-04-13
**작업 범위**: PR 1 (전처리 정합성) + PR 2.1/2.3/2.4 (구조 개선) + sft_gemma4.py 신규
**Git 리포**: https://github.com/moreHS/RelationModel_training
**브랜치**: `main`
**최신 commit**: `1e2d6bf`

---

## 📋 작업 요약 — 5개 commit, 11개 파일

| Commit | 작업 | 변경 파일 | 결정 기록 |
|---|---|---|---|
| [`96db9bc`](https://github.com/moreHS/RelationModel_training/commit/96db9bc) | PR 1 — 전처리 정합성 픽스 | utils.py, prompts.yaml, config.yaml | DECISIONS/2026-04-13_sft_preprocessing_pr1_*.md |
| [`fea044a`](https://github.com/moreHS/RelationModel_training/commit/fea044a) | PR 2.1 — Gemma4 chat_template 위임 | utils.py | DECISIONS/2026-04-13_pr2_1_*.md |
| [`05046ea`](https://github.com/moreHS/RelationModel_training/commit/05046ea) | PR 2.4 — SAFE_LIMIT max overhead | latest.py | DECISIONS/2026-04-13_pr2_4_*.md |
| [`3c90afa`](https://github.com/moreHS/RelationModel_training/commit/3c90afa) | PR 2.3 — 모드별 disjoint partition | latest.py | DECISIONS/2026-04-13_pr2_3_*.md |
| [`1e2d6bf`](https://github.com/moreHS/RelationModel_training/commit/1e2d6bf) | sft_gemma4.py 신규 작성 | training/sft_gemma4.py | DECISIONS/2026-04-13_sft_gemma4_*.md |

---

## 🎯 PR 1 — 전처리 정합성 픽스 (9개 결정)

### 1-1. BEE-BEE Task 전체 제거
- **변경**: `DataGenerationTask.BEE_BEE` enum, classify 버킷, PromptCompiler 분기, YAML system_prompt + relation_list, config tasks 모두 삭제
- **이유**: 사용자 정책상 BEE-BEE 관계 추론 안 함

### 1-2. COMBINE_ALL Task 전체 제거
- **변경**: enum, raw_sources, output_format_all/all_reasoning 등 통합 모드 코드 모두 삭제
- **이유**: BEE-BEE 빠지면 통합 모드 가치 약화

### 1-3. Negative 라벨 통일 — `normalize_negative_label()`
- **변경**: `no_relationship`, `NO_RELATIONSHIP`, `no relationship` 모두 `NO_RELATION`으로 통일
- **효과**: 200 doc 샘플에서 leak 292건 차단 (NER_BEE_TRUE_ONLY 오염 방지)

### 1-4. NER-NER 페어 BEE 라벨 노이즈 매핑
- **변경**: `BEE_LABEL_TO_NER_NER_MAP`(4개) 도입 — `Ingredients→has_ingredient`, `Effect→affects`, `Composition→has_part`, `Capacity→has_attribute`
- **효과**: 200 doc 샘플에서 12건 매핑 + 41건 매핑불가 라벨 drop

### 1-5. BEE-NER 페어 → NO_RELATION 강제
- **변경**: source_type=BEE-NER 페어를 NER-BEE 버킷에 합치되 relation을 NO_RELATION으로 덮어씀
- **이유**: 잘못된 방향성 → 엣지 케이스 negative 학습 데이터로 활용
- **효과**: 200 doc 샘플에서 5099건 강제 negative 적용

### 1-6. NER_BEE 출력 포맷 라우팅 수정
- **변경**: `output_format_w_keys_ner_bee` 사용 (이전 `output_format_w_keys` 오라우팅)
- **효과**: prompt와 출력 JSON 키(`is_relational`) 일치

### 1-7. NER-BEE system_prompt 본문 정정
- **변경**: `"relation"` 키 안내를 `"is_relational"` 키 (true/false) 안내로 수정
- **효과**: system_prompt + output_format + 실제 출력 모두 일관

### 1-8. BEE 속성 영→한 통일
- **변경**: `BEE_ENG_TO_KOR` 39개 매핑, NER-BEE 버킷 진입 시 자동 적용
- **효과**: `Effect 189` + `효과 74` → `효과 263` 합쳐짐. 데이터 검수 일관성 확보

### 1-9. 중복 정의 제거
- **변경**: utils.py 미완성 `get_or_generate_sft_dataset()` 삭제, `format()` 첫 번째 정의 (dead code) 삭제

---

## 🔧 PR 2 — 구조 개선 (3개 단위)

### PR 2.1 — Gemma4 chat_template `apply_chat_template` 위임
- **변경**: `_apply_chat_template` 메서드 56줄 → 32줄로 단순화
- **위임 대상**: `tokenizer.apply_chat_template(messages, **kwargs)`
- **해소된 버그 4건**:
  1. `<bos>` 누락 → 자동 prepend
  2. `<|channel>` / `<channel|>` 토큰 오타 → 공식 jinja `strip_thinking` 매크로 위임
  3. `<|think|>` 위치 오류 → 공식 위치 (system 턴 최상단)에 자동 삽입
  4. 표준 분기 BOS 무조건 strip → 제거 (chat_template이 자동 처리)
- **검증**: 격리 venv (transformers 5.5.3)에서 `<bos><|turn>system... <turn|>` 형식 + 토큰 ID(bos=2, `<|turn>`=105, `<turn|>`=106) 정확히 매핑 확인

### PR 2.4 — SAFE_LIMIT을 모든 활성 모드 max overhead로
- **변경**: 첫 모드만 사용하던 overhead 계산을 모든 모드 max로 변경
- **실측 overhead** (Gemma4 토크나이저 기준):
  - NER_NER: `[332, 2024, 1815, 2024, 333]` → max=2024 → SAFE_LIMIT 5860 → **4168**
  - NER_BEE: `[338, 1724, 937, 1724, 339]` → max=1724 → SAFE_LIMIT 5854 → **4468**
  - NER_BEE_TRUE_ONLY: 동일 패턴 → SAFE_LIMIT 5859 → **4473**
- **효과**: heavy 모드(`full_detailed`, `no_fewshot`)가 더 이상 8K 토큰 한도 초과 안 함. Phase 6 silent drop 거의 0
- **출력 추가**: `Task X: mode overheads=[...], max=..., SAFE_LIMIT=...` print

### PR 2.3 — 모드별 sample partition 결정적 분리
- **변경**: 모든 모드가 같은 seed로 첫 N개 선택하던 로직을 단일 셔플 + disjoint 인덱스 범위로 변경
- **이전 문제**: index 0~19의 sample이 system_only [0,20) AND full_detailed [0,42) AND no_fewshot [0,20) 3개 모드에 동시 등장
- **변경 후**: system_only [0, 0.2N), full_detailed [0.2N, 0.62N), full_summary [0.62N, 0.8N), no_fewshot [0.8N, 1.0N), reasoning auto-skip
- **검증**: total 100/1000/10000에서 모두 disjoint, 100% 데이터 사용

---

## 🆕 sft_gemma4.py — Unsloth 학습 코드 신규 (230줄)

### 7개 결정 사항

| 결정 | 값 | 근거 |
|---|---|---|
| Quantization | `load_in_16bit=True` | 사용자 결정 (정확도 우선, ~17GB VRAM) |
| LoRA r/alpha | 8/16 (scaling 2.0) | 기존 sft_qwen.py 패턴 유지 |
| Chat template | preprocessing 위임 (별도 호출 X) | PR 2.1로 단일 source |
| Response-only | `train_on_responses_only` 호출 | Unsloth 공식 권장 (정확도 +1%) |
| Gradient checkpointing | `"unsloth"` 모드 | 메모리 추가 절약 |
| TAG / OUTPUT | `_pr2_v1` / `gemma4-e4b_sft_pr2_v1` | 버전 관리 |
| WandB | offline | 기존 패턴 유지 |

### 조사한 Known Issues (모두 unsloth ≥2026.4.4 자동 해결)

| 이슈 | 영향 | 우리 대응 |
|---|---|---|
| Gradient accumulation loss 폭발 | loss 300-400 | 최신 unsloth 자동 보정 |
| `use_cache=False` garbage logits | E2B/E4B attention 손상 | use_cache 명시 안 함 (default 유지) |
| Float16 audio overflow | audio 학습 실패 | 우리는 text-only, 무관 |
| LoRA trainable=0 (#4907 26B MoE) | LoRA 학습 안 됨 | E4B dense 사용 + assertion 추가 |
| Loss 13-15 = multimodal 정상 | text-only는 1-3 | 우리 데이터 text-only, 1-3 기대 |

### Defensive checks (코드 내 assertion)

- `assert trainable > 0` — issue #4907 회귀 방지
- `assert sample_text.startswith("<bos>")` — PR 2.1 회귀 방지
- `assert "<|turn>" in sample_text` — Gemma 4 chat 포맷 확인

### Gemma 4 train_on_responses_only 인자

```python
train_on_responses_only(
    trainer,
    instruction_part="<|turn>user\n",
    response_part="<|turn>model\n",
)
```

---

## 🔬 재검증 결과 — 4단계 모두 통과

### 재검증 1: 코드 무결성 + import 정합
- ✅ utils.py 모든 심볼 import (DataPreprocessor, normalize_*, BEE_*, etc.)
- ✅ enum: 정확히 `{ner_ner, ner_bee, ner_bee_true_only}`
- ✅ normalize 함수 + BEE 매핑 (39개) 정합
- ✅ PR 2.1 회귀 없음 — `_apply_chat_template` 위임 형태 유지 (32줄)
- ✅ latest.py 함수 import OK
- ✅ PR 2.4 회귀 없음 — `overheads = []`, `max(overheads)` 로직 유지
- ✅ PR 2.3 회귀 없음 — `mode_ranges`, `full_registered` 로직 유지
- ✅ PR 1 중복 정의 제거 유지 — `format()` 1개만, `get_or_generate_sft_dataset` utils에 부재
- ✅ sft_gemma4.py 정합 (230줄, 12개 핵심 패턴 모두 확인)

### 재검증 2: 200 doc 샘플 Phase 1~3 (실데이터)
**stats**: `{ner_nr_bee_label_mapped: 12, ner_nr_bee_label_dropped: 41, bee_ner_forced_neg: 5099, bee_bee_dropped: 55}`

| 검증 항목 | 결과 |
|---|---|
| Negative 통일 (no_relationship → NO_RELATION) | ✅ 잔존 0건, 통일된 NO_RELATION 7714건 |
| NER-BEE 영문 BEE 라벨 0건 (한글 통일) | ✅ 효과 263, 보습력 101, 충성도 154, 사용감 129 ... |
| NER-NER BEE 노이즈 정리 | ✅ 매핑된 NER 관계: has_ingredient 52, affects 47, has_part 17, has_attribute 35 |
| NER-NER snake_case 표기 | ✅ used_by 2669, same_entity 2064, NO_RELATION 854, ... |
| BEE-NER 강제 NO_RELATION 효과 | ✅ NER-BEE의 NO_RELATION 비율 82.2% (BEE-NER 5099건 합산) |
| Phase 2 키 라우팅 | ✅ NER_NER → `relation`, NER_BEE/TRUE_ONLY → `is_relational` |
| Phase 3 origin_id 그룹핑 | ✅ chunked 18 (5 docs threshold=12) |

### 재검증 3: PR 2.1/2.3/2.4 토크나이저 레벨 (격리 venv)
- ✅ **PR 2.1 NER_NER full**: `<bos>` prepend + `<|turn>` x3 + `<|think|>` 없음 (8684 chars)
- ✅ **PR 2.1 NER_BEE reasoning**: `<bos>` prepend + `<|think|>` 자동 삽입 (1488 chars)
- ✅ **PR 2.1 토큰 ID**: bos=2, `<|turn>`=105 x3, `<turn|>`=106 x3
- ✅ **PR 2.4** 모든 Task에서 max overhead 사용, SAFE_LIMIT > 4000
- ✅ **PR 2.3** total 100/1000/10000에서 모두 disjoint, 100% 데이터 사용

### 재검증 4: sft_gemma4.py 결정사항 + DECISIONS + Git history
- ✅ 8개 결정 카테고리 모두 코드에 반영 확인
- ✅ `get_chat_template` 호출 0건 (preprocessing 위임)
- ✅ DECISIONS 5개 파일 모두 존재
- ✅ Git history에 5개 PR commit 모두 존재

---

## 📊 데이터 영향도 (200 doc 샘플 기준)

| 변화 | 수치 | 비율 |
|---|---|---|
| BEE-BEE 페어 drop | 55건 | 전체 후보의 ~0.4% |
| NER-NER에서 BEE 라벨 매핑 | 12건 | ~0.1% |
| NER-NER에서 매핑 불가 BEE 라벨 drop | 41건 | ~0.3% |
| BEE-NER → NO_RELATION 강제 | 5099건 | NER-BEE 버킷의 ~61% |
| NO_RELATION 통일된 추가 negative (구 `no_relationship`) | 약 567건 | ~0.5% |
| NER-BEE 한글 통일된 라벨 | 약 263+154+129+...건 | ~14% |

---

## 🛠️ 환경 정보

### 사용자 docker 환경 (학습 실행)
- transformers ≥4.57 (Gemma 4 첫 지원 2026-04-01)
- unsloth ≥2026.4.4 (Gemma 4 fixes 포함)
- trl ≥0.11.0
- torch ≥2.1.0
- GPU: VRAM ~17GB 이상 (16-bit LoRA 기준)

### 로컬 검증 환경 (격리 venv: `/tmp/gemma4_test_venv`)
- transformers 5.5.3
- tokenizers 0.22.2
- jinja2, datasets, pyyaml
- PyTorch 미설치 (토크나이저만 사용)

### 글로벌 환경 (메인 anaconda)
- transformers 4.32.1 (Gemma 4 미지원, 단순 검증 작업에만 사용)
- 코드 import + 200 doc 샘플 Phase 1~3 검증에 사용

---

## 📁 파일 구조 변화

### 신규 추가 (16개)
```
.gitignore
DECISIONS/
  ├─ 2026-04-13_sft_preprocessing_pr1_정합성_픽스.md
  ├─ 2026-04-13_pr2_1_gemma4_chat_template_위임.md
  ├─ 2026-04-13_pr2_4_safe_limit_max_overhead.md
  ├─ 2026-04-13_pr2_3_mode_partition_분리.md
  └─ 2026-04-13_sft_gemma4_unsloth_학습코드.md
REPORTS/
  └─ 2026-04-13_implementation_report.md  (← 이 파일)
training/
  └─ sft_gemma4.py  (230줄 신규)
preprocessing/  (4개 — Initial commit으로 import)
models/  (7개 — chat_template ground truth)
```

### 수정 (PR 1, PR 2.x 통해)
- `preprocessing/data_preprocessor_utils_simplified_add_gemma4.py`
- `preprocessing/generate_sft_training_data_latest.py`
- `preprocessing/generate_sft_training_data_config.yaml`
- `preprocessing/sft_data_generation_prompts_edited.yaml`

### 보존 (변경 없음)
- `training/sft_qwen.py` — Qwen 학습 시 참고 자료
- `models/*` — Gemma 4 토크나이저/config (ground truth)
- `CLAUDE.md`

---

## 🚀 사용자 다음 단계 권장

### 1. Docker 환경 패키지 버전 확인
```bash
pip show transformers unsloth trl torch | grep -E "^(Name|Version)"
# 기대값: transformers ≥4.57, unsloth ≥2026.4.4, trl ≥0.11.0, torch ≥2.1.0
```

### 2. 기존 캐시 디렉토리 삭제
```bash
rm -rf /app/dataset/preprocessed/sllm_ready_generated_prompts_gemma4_hf_dataset/
# 스키마 + chat 포맷 변경됨 → 재생성 필요
```

### 3. 빠른 dry-run (preprocessing)
```bash
# config.yaml에서 target_total_training_samples를 10으로 임시 설정
python -m preprocessing.generate_sft_training_data_latest

# 확인 사항:
# - Phase 2 stdout: "Task ner_ner: mode overheads=[...], max=..., SAFE_LIMIT=..."
# - Phase 5 stdout: "Task ner_ner mode partition (total=N): system_only=[0,...), full_detailed=[...,...), ..."
# - Phase 6 drop count 거의 0
```

### 4. 데이터 sample 확인
```bash
python -c "
from datasets import load_from_disk
ds = load_from_disk('/app/dataset/preprocessed/sllm_ready_generated_prompts_gemma4_hf_dataset')
print(ds['train'][0]['text'][:500])
# 기대: <bos><|turn>system\n### System Prompt:\n... 로 시작
"
```

### 5. 학습 dry-run (sft_gemma4.py)
```bash
# 임시로 SFTConfig에 max_steps=1 추가 후 실행
python training/sft_gemma4.py

# 확인 사항:
# - "📊 LoRA trainable params: ... > 0"
# - "✅ Chat template 정합성 OK"
# - "✅ train_on_responses_only applied"
# - 1 step loss 1-3 범위 (text-only 정상)
# - OOM 없이 완료
```

### 6. 본 학습
- max_steps 제거 (또는 큰 값으로) → num_train_epochs=3 동작
- target_total_training_samples를 원하는 양으로 (수천~수만)
- preprocessing 재실행 → sft_gemma4.py 실행

---

## 🔮 향후 작업 (별도 PR 후보)

### PR 3 (검토 후)
- JSONL 스트리밍 (3 GB 메모리 부담 해소)
- Phase-2 중간 캐시 (장시간 런 보호)
- Cache key에 config hash 포함
- 로깅 강화 (failed bucket / Phase 3 cut / Phase 6 drop counts)

### PR 4 (위생)
- 하드코딩 `/app/` 경로 → env/CLI
- `datasets.builder.has_sufficient_disk_space` 무조건 monkeypatch → flag 가드
- 사용 안 되는 config 키 제거 (`batch_size`, `fewshot_sample_size`, `prioritize_rare`)

### 학습 측 개선
- Gold dataset 도입 후 few-shot 누수 차단 (PR 2.2 유보분)
- DPO/ORPO 후속 학습
- Multi-GPU DDP
- Inference 스크립트 (`predict_gemma4.py`)

### NER_BEE_TRUE_ONLY 재검토
- 현재 binary 출력 → 사실상 모든 정답 `true` (positive 강화 의도)
- 향후 actual BEE 속성 라벨 예측 task로 변경 검토 시 BEE 영/한 통일 작업이 학습에 직접 영향

---

## ✅ 최종 체크리스트

- [x] PR 1 9개 결정 모두 코드 반영 + 검증
- [x] PR 2.1 chat_template 위임 + 토크나이저 검증
- [x] PR 2.4 SAFE_LIMIT max overhead + overhead 실측 검증
- [x] PR 2.3 mode partition disjoint + 산술 검증
- [x] sft_gemma4.py 7개 결정 모두 코드 반영 + AST 검증
- [x] DECISIONS 5개 파일 작성 + git에 push
- [x] 5개 PR commit 모두 GitHub `main`에 push
- [x] 4단계 재검증 모두 통과
- [x] 구현 완료 보고서 작성 + 프로젝트 저장

**상태: 모든 작업 정상 완료. Docker 환경 dry-run 후 본 학습 진행 가능.**
