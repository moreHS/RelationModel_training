# 2026-04-13 — SFT 전처리 파이프라인 정합성 픽스 (PR 1)

## 배경

`preprocessing/` 하위 SFT 데이터 생성 파이프라인을 다관점으로 점검:
- 데이터 샘플링: 95K doc 중 200건 reservoir 샘플링
- Gemma4 chat_template ground truth 검증 ([models/chat_template.jinja](../models/chat_template.jinja))
- 학습 코드 예시 검토 ([training/sft_qwen.py](../training/sft_qwen.py))
- GPT(Architect, gpt-5.2-codex) 크로스 리뷰

다음 정합성 문제가 확인되어 PR 1로 묶어 처리:
- BEE-BEE / COMBINE_ALL Task 미사용 결정 → 잔존 코드/YAML이 dead path 형성
- Negative 라벨 `no_relationship`(~3.9%)가 `NO_RELATION` 필터 우회 → NER_BEE_TRUE_ONLY 오염
- NER_BEE 계열이 잘못된 output_format으로 라우팅 → prompt와 output 키 불일치
- NER-NER 페어 안 BEE 속성 라벨 노이즈 (200doc 중 26건 ≈ 12.5%)
- BEE-NER 페어(잘못된 방향성) ~80건의 의미 노이즈
- 데이터에 BEE 속성 영/한 혼재 (Effect 189 + 효과 74 등)

---

## 결정 1: BEE-BEE Task 제거

**결정**: BEE-BEE Task 코드/YAML/config 전부 제거.

**검토한 선택지**:
- (A) 유지 — 향후 확장 가능성 보존
- (B) 제거 — 사용자가 BEE-BEE 관계는 추론하지 않기로 명시

**선택 이유**: 사용자 명시적 결정 ("bee-bee relation은 추론안하기로 결정햇어"). dead code/dead prompt가 유지보수 부담만 늘림.

**트레이드오프**: 향후 BEE-BEE 추론이 필요해지면 git 히스토리에서 복원해야 함. 단 PR 단위 롤백 가능.

**삭제 범위**:
- `DataGenerationTask.BEE_BEE` enum
- `extract_and_classify`의 bee_bee 버킷
- `PromptCompiler._precompute_static_components` BEE-BEE 분기
- YAML: `system_prompt_bee_bee`, `bee_bee_relation_list`
- config: tasks 리스트의 `"BEE_BEE"`

**보존**: `bee_des_detailed`/`bee_des_summarized` (NER-BEE Task가 BEE 한글 description 사용).

---

## 결정 2: COMBINE_ALL Task 제거

**결정**: COMBINE_ALL Task 전부 제거.

**검토한 선택지**:
- (A) 유지 (BEE-BEE 빠진 NER-NER + NER-BEE 통합) — 1개 모델로 두 태스크 동시 처리
- (B) 제거 — 단순화. NER_NER + NER_BEE 따로 학습

**선택 이유**: 사용자 결정 (Q1 응답: "combine_all 모드 제거"). BEE-BEE가 빠지면 통합 모델의 가치가 약해지고, Task 분리가 학습/평가에 유리.

**트레이드오프**: 추후 단일 모델로 두 태스크를 동시 처리하려면 별도 task 추가 필요.

---

## 결정 3: NER-NER 페어 안 BEE 속성 라벨 노이즈 처리

**배경**: 데이터 라벨러가 NER-NER source_type 페어에 BEE 속성명(`Ingredients`, `Effect` 등)을 relation으로 사용한 케이스 200doc 중 26건 (~12.5%).

**결정**: 매핑 후 유지 (Recommended). `BEE_LABEL_TO_NER_NER_MAP`으로 의미 합치는 NER 관계로 변환, 매핑 불가한 BEE 라벨은 drop.

**검토한 선택지**:
- (A) 매핑 후 유지 — `Ingredients → has_ingredient`, `Effect → affects` 등 ner_ner_relation_list에 정의된 의미 합치는 라벨로 매핑
- (B) 전부 drop — 가장 안전, ~1200건 손실
- (C) 그대로 두고 학습 — 모델이 prompt에 없는 라벨로 학습 → 환각 위험

**선택 이유**: 데이터 보존 + 일관성 확보. 매핑 가능한 4개(`Ingredients/Effect/Composition/Capacity`)는 의미적으로 NER-NER 관계와 등가이므로 매핑이 자연스러움. 매핑 불가한 라벨만 drop하여 노이즈 최소화.

**트레이드오프**: 매핑 테이블 4개 항목 유지보수 필요. 매핑 자체가 의미 손실을 내포할 수 있음(예: `Effect`는 단순 `affects`보다 풍부한 의미를 가짐).

---

## 결정 4: BEE-NER 페어 처리 (잘못된 방향성)

**배경**: source_type이 BEE-NER인 페어(BEE 속성이 subject, NER 개체가 object) ~5099건. 의미적으로 잘못된 연결 방향.

**결정**: NER-BEE 버킷에 합치되 `relation`을 `NO_RELATION`으로 강제. 엣지 케이스 negative 학습용으로 활용.

**검토한 선택지**:
- (A) 현행 유지 — Binary 변환으로 학습엔 영향 없으나 의미 노이즈
- (B) BEE-NER 페어 자체를 NER-BEE Task에서 제외 — 데이터 ~30% 감소 가능
- (C) NO_RELATION 강제 — 엣지 케이스 학습용으로 활용

**선택 이유**: 사용자 명시적 결정 ("이걸 드랍하는거보다 no relation으로 엮어서 학습에 사용하는게 엣지 케이스 학습용으로 좋을듯"). 잘못된 방향성 패턴을 모델이 명시적으로 negative로 학습하면 추론 시에도 견고해짐.

**트레이드오프**: NER-BEE 버킷의 NO_RELATION 비율이 54% → 82%로 상승 (200 doc 기준). 향후 PR에서 negative downsampling 검토 가치 있음.

**자동 효과**: NER_BEE_TRUE_ONLY는 `remove_negative_relations`로 자동 제외되므로 별도 처리 불필요.

---

## 결정 5: BEE 속성 라벨 영→한 통일

**배경**: 데이터에 BEE 속성 라벨이 영/한 혼재. `Effect 189건` + `효과 74건`, `Loyalty 89건` + `충성도 65건` 등.

**결정**: BEE 속성 라벨을 한글로 통일 (`BEE_ENG_TO_KOR` 매핑 적용). NER 관계는 영문 snake_case 그대로 유지.

**검토한 선택지**:
- (A) 영문 snake_case 완전 통일 — 토큰 효율 ↑, 한글 키 38개 영문 변환 부담
- (B) 한글 완전 통일 — 60+ NER 관계 한글 매핑 작성 부담 큼
- (C) BEE 속성만 한글 통일 (선택됨)
- (D) 통일하지 않음 — 학습엔 무영향이나 데이터 검수 시 혼란

**선택 이유**: 사용자 정책 ("bee 속성명들은 한글로 표현해야 의미가 잘 표현되는게 잇어서 그걸로 일부러 고정한거야"). YAML `bee_des_*`가 이미 한글 키로 정의되어 있어 데이터 라벨도 한글로 통일하면 prompt-data 일관성 확보. NER 관계는 이미 영문 snake_case로 안정되어 있어 손대지 않음.

**트레이드오프**: 현재 출력은 binary(`is_relational`)라 학습엔 직접 영향 없음. 단 데이터 통계/검수/후처리 시 일관성 확보. 후속 PR에서 NER_BEE_TRUE_ONLY를 actual label 예측 task로 변경하면 직접 학습 영향 발생 → 그 시점에는 이미 정합 상태.

---

## 결정 6: NER_BEE_TRUE_ONLY Task의 의미는 "Binary positive 강화"

**배경**: NER_BEE_TRUE_ONLY는 negative 페어를 제거하므로 입력에 positive만 들어감. 출력은 NER_BEE와 동일하게 binary(`is_relational: true/false`). 따라서 모델이 모든 정답을 `true`로만 학습 → 의미가 모호함.

**결정**: 현행 유지 (Binary 출력). NER_BEE_TRUE_ONLY는 positive 패턴 강화 학습용으로 활용.

**검토한 선택지**:
- (A) Binary 유지 (positive 강화 학습) — 선택됨
- (B) Actual BEE 속성 라벨 예측으로 변경 — `relation: 보습력/효과/...` 출력. PR 1에 코드 분기 + BEE 영/한 통일 필수성 격상
- (C) Task 자체 제거 — 가장 단순

**선택 이유**: 사용자 결정 ("Binary 유지 (positive 강화 학습)"). PR 1 스코프 보수적 유지.

**트레이드오프**: 모델이 NER_BEE_TRUE_ONLY 데이터로는 negative 식별 능력을 학습하지 못함. negative 식별은 NER_BEE에서만 학습. 학습 데이터 분포 편향 가능성.

---

## 결정 7: Negative 라벨 `NO_RELATION`으로 통일

**배경**: 데이터에 `NO_RELATION` (7228건) + `no_relationship` (292건) 혼재. `remove_negative_relations`가 `NO_RELATION`만 필터 → 약 3.9% leakage가 NER_BEE_TRUE_ONLY로 유입.

**결정**: `normalize_negative_label()` 헬퍼로 모든 negative 이형태(`no_relationship`, `NO_RELATIONSHIP`, `no relationship` 등)를 `NO_RELATION`으로 통일.

**검토한 선택지**:
- (A) `NO_RELATION` 통일 — 선택됨
- (B) `no_relationship` 통일 — YAML system_prompt가 `NO_RELATION` 사용 중이라 대규모 YAML 변경 필요

**선택 이유**: YAML `system_prompt_*`와 `output_format_w_keys` 모두 `NO_RELATION`을 명시. 라벨 정규화 방향을 prompt와 일치시키는 게 자연스러움.

**트레이드오프**: 없음. case-insensitive 매칭으로 향후 라벨러가 다른 변형을 만들어도 자동 흡수.

---

## 결정 8: NER_BEE 출력 포맷 라우팅 수정

**배경**: `PromptCompiler._precompute_static_components`가 NER_BEE / NER_BEE_TRUE_ONLY를 `output_format_w_keys` (`"relation"` 키 안내)로 라우팅. 그러나 `PreprocessInput`은 NER_BEE 계열에 대해 `is_relational` 키를 사용 → prompt와 출력 JSON 키가 서로 다름.

**결정**: NER_BEE / NER_BEE_TRUE_ONLY는 `output_format_w_keys_ner_bee` (이미 YAML에 정의됨) 사용하도록 분기 수정. system_prompt 본문도 `"is_relational"` 키 명시로 정정.

**검토한 선택지**:
- (A) 코드를 `is_relational`로 통일 — 선택됨
- (B) `PreprocessInput`을 `relation` 키로 되돌림 — system_prompt + output_format 모두 변경 필요. 코드 의미상 binary는 `is_relational`이 명확

**선택 이유**: 코드의 binary 출력 의미가 `is_relational`로 명확. YAML에 이미 `output_format_w_keys_ner_bee` 정의되어 있어 라우팅만 수정하면 됨.

**트레이드오프**: 없음. dead code였던 `output_format_w_keys_ner_bee`가 살아남.

---

## 결정 9: Gemma4 chat_template 수동 빌더 픽스는 PR 2로 분리

**배경**: 파이프라인의 `_apply_chat_template` Gemma4 분기가 chat_template을 수동 조립. [models/chat_template.jinja](../models/chat_template.jinja) ground truth 검증 결과:
- `<|turn>` / `<turn|>` 태그 자체는 정확
- 그러나 `<bos>` 누락, `<|channel>` / `<channel|>` 토큰 오타, `<|think|>` 위치 오류 등 4건의 버그

**결정**: PR 1에선 미조치, PR 2로 분리. PR 2에선 수동 빌더를 폐기하고 `tokenizer.apply_chat_template(..., enable_thinking=False)`로 위임.

**검토한 선택지**:
- (A) PR 1에 포함 — 정합성 픽스와 함께 푸시
- (B) PR 2로 분리 — 선택됨

**선택 이유**: PR 1은 데이터 라벨/스키마 정합성에 집중. chat_template 픽스는 토크나이저 + reasoning 모드 결정과 함께 별도 PR로 격리하면 변경 영향 분석/롤백 용이.

**트레이드오프**: PR 1로 학습 데이터를 만들면 Gemma4 chat 포맷에 BOS 누락/think 토큰 오타가 잔존. 단 NER_BEE/NER_NER Task가 reasoning=False로 동작 (config.yaml mode_ratios.reasoning: 0.0)하므로 think 토큰 오타는 영향 없음. BOS 누락만 PR 2에서 처리.

---

## 결정 10: 학습 코드(`training/sft_qwen.py`)는 PR 1에서 미수정

**배경**: 사용자가 "qwen 학습 코드는 예시이고 향후 sft_gemma4.py 별도 작성"이라 명시. `_qwen_hf_dataset` 경로는 옛 컨벤션의 잔재.

**결정**: PR 1 스코프에서 제외. 향후 `sft_gemma4.py` 작성 시 `_gemma4_hf_dataset` 경로 사용 + `train_on_responses_only` 호출 추가.

**검토한 선택지**:
- (A) sft_qwen.py를 PR 1에 포함해 fix — 사용자 의도와 어긋남
- (B) PR 1 스코프 외 (선택됨)

**선택 이유**: 사용자 의도 존중 + PR 1 스코프 명확화.

---

## 적용 결과 (200 doc 샘플 기준)

```
📊 stats: {'ner_nr_bee_label_mapped': 12, 'ner_nr_bee_label_dropped': 41,
          'bee_ner_forced_neg': 5099, 'bee_bee_dropped': 55}

NER-NER: 7,446 pairs (변경 후)
  상위: used_by 2669, same_entity 2064, NO_RELATION 854,
        applied_to 202, benefits 142, recommended_by 101
  매핑된 NER-NER 추가: has_ingredient 52, affects 47, has_part 17, has_attribute 35

NER-BEE: 8,350 pairs (변경 후, BEE-NER 강제 negative 합산)
  한글 BEE 라벨: 효과 263, 충성도 154, 사용감 129, 보습력 101,
                품질 95, 향 63, 제형 65, 색상 42, 성분 49
  NO_RELATION: 6,860 (82.2%) ← 향후 PR에서 downsampling 검토

failed: 0
```

## 미해결/향후 PR 대상

- PR 2: Gemma4 chat_template 위임, few-shot 누수 차단, 모드별 sample partition 결정적 분리, BOS 보존
- PR 3: JSONL 스트리밍, Phase-2 중간 캐시, config-hash cache key, 로깅 강화
- PR 4: 환경/하드코딩 정리, 사용 안 되는 config 키 정리
- 학습 측: `sft_gemma4.py` 신규 작성 시 `_gemma4_hf_dataset` 경로 + `train_on_responses_only` 적용
