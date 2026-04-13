# 2026-04-13 — PR 2.4: SAFE_LIMIT을 모든 활성 모드 max overhead로

## 배경

[generate_sft_training_data_latest.py:206](../preprocessing/generate_sft_training_data_latest.py#L206) Phase 2의 청킹 안전 한계 계산:

```python
temp_compiler = PromptCompiler(task_enum, list(BASE_MODES.values())[0], ...)  # ← 첫 모드만!
ov = calculate_template_overhead(temp_compiler, tokenizer)
SAFE_LIMIT = max_token_threshold - ov - 2000
```

**문제**: `BASE_MODES`는 dict이고 첫 항목은 `system_only` (가장 작은 overhead). description+few-shot이 큰 `full_detailed`/`no_fewshot` 모드는 SAFE_LIMIT 계산에 반영 안 됨 → 청크는 작은 모드 기준으로 만들어지고, 큰 모드 prompt 컴파일 시 8192 초과 → Phase 6에서 silent drop.

격리 venv에서 실측한 모드별 overhead (gemma4 토크나이저 기준):

| 모드 | NER_NER | NER_BEE | NER_BEE_TRUE_ONLY |
|---|---|---|---|
| system_only | 332 | 338 | 333 |
| reasoning | 333 | 339 | 334 |
| full_summary | 1815 | 937 | 932 |
| **full_detailed** | **2024** | **1724** | **1719** |
| **no_fewshot** | **2024** | **1724** | **1719** |

→ 이전 SAFE_LIMIT (`system_only` 332 기준): 5860
→ 새 SAFE_LIMIT (`full_detailed` 2024 기준): 4168
→ 1700+ 토큰 차이 = heavy 모드 prompt가 사실상 Phase 6 drop 직행

## 결정

**Phase 2의 SAFE_LIMIT을 모든 활성 모드 overhead의 최댓값으로 계산.** 청크 크기는 가장 무거운 모드 기준으로 결정되어 모든 모드에서 토큰 한도 내 보장.

```python
overheads = []
for mode_cfg in BASE_MODES.values():
    temp_compiler = PromptCompiler(task_enum, mode_cfg, ...)
    overheads.append(calculate_template_overhead(temp_compiler, tokenizer))
ov = max(overheads)
SAFE_LIMIT = max_token_threshold - ov - 2000
```

## 검토한 선택지

- **(A) max overhead (선택됨)** — 가장 안전. heavy 모드도 8K 한도 보장
- **(B) 모드별 SAFE_LIMIT 분리** — 모드마다 다른 청크 사용. 데이터 양 최대화. 단 청킹 결과 캐시 + Phase 5 컴파일 흐름 전반 재설계 필요
- **(C) heavy 모드(`full_detailed`)만 따로 작은 청크로** — 부분 최적화. 코드 분기 복잡도 증가

## 선택 이유

- 가장 단순하고 안전. 1줄 → 7줄 변경
- 청크 크기 약간 작아져도 prompt 컴파일 성공률 100%가 더 가치 있음
- 모드별 분리는 향후 PR에서 청킹 캐시화(Phase-2 디스크 캐시)와 함께 검토

## 트레이드오프

- 청크 평균 크기 감소 → 청크 수 증가 → Phase 5 컴파일 시간 증가
- 작은 모드(`system_only`)는 ~1700 토큰 여유분 낭비 (학습엔 무영향, 단지 청크가 더 잘게 쪼개짐)
- 대신 Phase 6 drop 비율 거의 0 (이전엔 heavy 모드 prompt 상당수 drop 추정)

## 검증 결과

각 Task별로 실제 overhead 수집 후 max 사용 확인. stdout에 모드별 overhead + max + SAFE_LIMIT 출력 추가 → 향후 디버그/모니터링 용이.

```
Task ner_ner: mode overheads=[332, 2024, 1815, 2024, 333], max=2024, SAFE_LIMIT=4168
Task ner_bee: mode overheads=[338, 1724, 937, 1724, 339], max=1724, SAFE_LIMIT=4468
Task ner_bee_true_only: ...
```

## 영향

- Phase 2 청킹 시 모든 모드에서 SAFE_LIMIT 보장
- Phase 6 drop count → 거의 0
- 청크 수 약 5-10% 증가 예상 (Phase 5 컴파일 시간 비례)
