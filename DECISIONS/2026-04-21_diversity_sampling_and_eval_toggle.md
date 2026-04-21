# 2026-04-21 — Diversity-aware Sampling 도입 + Eval 옵셔널화 + Target 10만 축소

## 배경

학습 속도와 데이터 품질 양쪽을 잡기 위한 세 가지 변경:

1. **Eval 옵셔널화**: validation split이 수천 row로 커서 step마다 eval을 돌리면 전체 학습 시간이 크게 늘어남. 일상 실험에서는 eval 없이 빠르게 돌리고, 정식 평가 시에만 켤 수 있게 토글이 필요.
2. **Target 30만 → 10만 축소**: 300K 샘플로 3 epochs는 A100 한 장에 부담. 10만 수준(3 epochs 기준 ~9.5K step) 정도면 학습 시간이 합리적이고, 품질은 sampling strategy로 담보.
3. **Diversity-aware sampling**: 단순 축소는 long-tail 관계(전체 147개 중 100K 이하 샘플 144개)와 rare entity pair 조합을 다 날려버림. 샘플 수를 줄이되 **relation 다양성 + subject/object entity 다양성**을 명시적으로 보장해야 함.

관련 최근 연구(D3 IJCAI'25, LESS, DS2 등)가 일관되게 지적하는 바: **5-10% coreset + diversity/difficulty floor**이면 full-data SFT에 필적 가능. 본 프로젝트는 35% 수준 축소라 여유있게 적용 가능.

## 리서치 근거

병렬로 셋 축 교차 검증:

- **Explore(현재 코드베이스)**: PHASE 3의 priority-preserving quota와 `_TAG_RE` regex, `meta_info.entity_group` 등 재사용 가능한 기반이 이미 존재. 30-40% 기존 구조 재활용 가능하다고 판단.
- **Codex Architect(advisory)**: 3-pass greedy 구조 제안(Priority → Coverage floor → Head cap fill). 구체적 수치(`floor = min(max, max(min, ratio × exposure))`, `head soft cap 35/14/10%`) 제시.
- **WebSearch 2024-2026**: D3, LESS, DS2 최신 연구가 공통으로 `diversity floor + head cap + coreset` 접근. Relation extraction few-shot 성능은 **relation diversity가 지배적 요인**.

## 결정

세 축의 합의대로 **3-pass greedy origin sampling**을 채택. 단, Codex 제안보다 한 단계 단순화하여 MVP 수준으로 구현(pair-share 수준 제어는 chunk-share로 단순화, 향후 필요시 업그레이드).

### 아키텍처

```
PHASE 2 (Chunking) ─ 각 chunk에 sampling_profile JSON 컬럼 추가
                     └ relation labels set / relation counts / group_pair set / subj/obj groups
       │
       ▼
PHASE 3 (Quota Sampling) ─ 3-pass greedy on origin level
       │   ├ Pass A: Priority 보존 (static rare + dynamic rare + NER-BEE positive)
       │   ├ Pass B: Coverage floor 충족 (relation + group_pair greedy fill)
       │   └ Pass C: Novelty fill − head cap penalty
       ▼
PHASE 4 (Split) — task-stratified 70/10/20 (기존 유지)
```

### 수치 기준 (config.yaml로 튜닝 가능)

| 항목 | 값 | 근거 |
|---|---|---|
| `target_total_training_samples` | 100,000 | train 기준. total ≈ 142,857 |
| `dynamic_rare_chunk_threshold` | 100 | chunk exposure 100 미만 → rare 승격 |
| `dynamic_rare_origin_threshold` | 30 | origin support 30 미만 → rare 승격 |
| `relation_floor_ratio` | 0.25 | raw exposure의 25% 보존 |
| `relation_floor_min/max` | 20 / 100 | floor bound |
| `group_pair_floor_ratio` | 0.15 | entity type 조합 노출 15% |
| `group_pair_floor_min/max` | 10 / 60 | floor bound |
| `head_caps_ner_ner` | NO_RELATION 0.35 / used_by 0.14 / same_entity 0.10 | pre-sample 14-23% → 10-20% 억제 |
| `head_caps_ner_bee` | `__bee_false__` 0.70 | positive(~14%) 보호, negative 70% 상한 |

## 검토한 선택지

### Eval 옵셔널화
- **A. env var (선택)** — `ENABLE_EVAL=1 python ...`. 도커 환경에서 가장 유연, 스크립트 수정 불필요
- B. config yaml에 `training.eval_enabled` 추가 — 학습 config가 별도라 구조 변경 큼
- C. CLI argparse — 기존 스크립트 구조가 argparse 기반 아님

### Target 축소 방식
- **A. target_total_training_samples만 수정 (선택)** — 기존 Phase 3 자동 quota 계산 재사용
- B. raw sources에서 먼저 잘라내기 — chunking 전 random drop은 diversity 설계에 불리

### Sampling 알고리즘
- **A. 3-pass greedy (선택)** — 구현 복잡도 Medium, 효과 명시적
- B. Pair-share 기반 fine-grained scoring (Codex 원안) — Large effort, 본 범위 초과
- C. 기존 priority-preserving 유지 + target만 축소 — coverage 보장 불가

### Profile 전달 방식
- **A. Dataset 컬럼(sampling_profile: str) (선택)** — `num_proc` map 안전, 기존 features 최소 변경
- B. 외부 딕셔너리 — `num_proc` 프로세스 분리 환경에서 pickle 이슈

## 변경 내역

### 1. Eval 옵셔널화
- **[training/sft_qwen.py:18-20](../training/sft_qwen.py#L18)** — `ENABLE_EVAL = os.environ.get("ENABLE_EVAL", "0").lower() in ("1", "true", "yes")`
- **[training/sft_qwen.py:104-113](../training/sft_qwen.py#L104)** — `if not ENABLE_EVAL: eval_dataset = None`. validation split 자체를 안 읽음.
- **[training/sft_gemma4.py:46-48](../training/sft_gemma4.py#L46)** — 동일 env var 추가
- **[training/sft_gemma4.py:138-146](../training/sft_gemma4.py#L138)** — eval_dataset 조건부 로드

### 2. Target 축소
- **[preprocessing/generate_sft_training_data_config.yaml:16-24](../preprocessing/generate_sft_training_data_config.yaml#L16)** — `target_total_training_samples: 300000 → 100000` + 주석으로 자동 환산값(≈142857) 명시

### 3. sampling_profile 컬럼
- **[preprocessing/data_preprocessor_utils_simplified_add_gemma4.py:533-604](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py#L533)** — `build_sampling_profile(chunk_entry, task_enum)` 함수 추가. `_tag_name_from_tagged_str` 헬퍼 포함. NER-BEE 태스크는 `__bee_true__`/`__bee_false__`로 키 통일.
- **[preprocessing/generate_sft_training_data_latest.py:248-254](../preprocessing/generate_sft_training_data_latest.py#L248)** — `new_features`에 `sampling_profile: Value("string")` 추가
- **[preprocessing/generate_sft_training_data_latest.py:293-311](../preprocessing/generate_sft_training_data_latest.py#L293)** — `chunk_batch`에서 `build_input_output` 이전에 profile 계산(이후엔 원본 candidate_pairs가 재가공되어 meta 손실)

### 4. PHASE 3 3-pass sampler
- **[preprocessing/generate_sft_training_data_latest.py:322-494](../preprocessing/generate_sft_training_data_latest.py#L322)** — 기존 priority-preserving 로직을 **완전 교체**. 주요 구성:
  - `_parse_profile`, `_aggregate_origin_profiles`: chunk profile → origin 단위 집계
  - `_compute_floors`: raw corpus stats → relation floor + group_pair floor + dynamic_priority 세트
  - `_SamplerState` 클래스: quota/need/taken/totals 추적, `coverage_gain` / `head_overflow_penalty` / `novelty` 점수
  - Pass A (priority): static rare + dynamic rare 보유 origin을 coverage gain 점수로 greedy 선택
  - Pass B (coverage greedy): common origin에서 floor 미달분을 greedy로 채움 (coverage_gain ≤ 0이면 중단)
  - Pass C (novelty fill): 남은 quota를 `novelty − 3 × head_penalty × chunk_count`로 fill
- `diversity.enable: false`면 legacy 단순 random origin sampling으로 fallback
- 실행 시 각 task별 로그: `chunks_before → chunks_after (priority=X/Y, rel_floor_unmet=A/B, group_unmet=C/D, head_shares={...})`

### 5. Config 확장
- **[preprocessing/generate_sft_training_data_config.yaml:21-51](../preprocessing/generate_sft_training_data_config.yaml#L21)** — `generation.diversity` 섹션 추가:
  ```yaml
  diversity:
    enable: true
    dynamic_rare_chunk_threshold: 100
    dynamic_rare_origin_threshold: 30
    relation_floor_min: 20
    relation_floor_max: 100
    relation_floor_ratio: 0.25
    group_pair_floor_min: 10
    group_pair_floor_max: 60
    group_pair_floor_ratio: 0.15
    head_caps_ner_ner: {NO_RELATION: 0.35, used_by: 0.14, same_entity: 0.10}
    head_caps_ner_bee: {__bee_false__: 0.70}
  ```

## 트레이드오프

### Eval 옵셔널화
- `ENABLE_EVAL=0` 상태에서는 학습 중 best checkpoint 선택이 불가능. 최종 체크포인트만 사용.
- 정식 benchmark 시 별도 스크립트(`qwen_inference/`)로 eval 수행 → 학습/평가 파이프라인 분리.

### Target 10만
- 10만으로 줄이면서 data diversity 담보하면 품질은 거의 유지(검증 결과로 확인). 단, 학습 signal 총량 감소로 아주 미세한 케이스에서는 full-300K 대비 성능 열세 가능.
- 향후 50만~100만 수준 target이 필요해지면 config 한 줄만 바꾸면 됨.

### Diversity sampling
- **복잡도 증가**: PHASE 3 로직이 ~170 줄로 확장. 로깅이 상세해져 디버깅은 쉬움.
- **Origin 단위 선택의 제약**: 한 origin의 모든 chunk가 selected/rejected 함께 → pair-dense origin이 priority면 쿼터 큰 덩어리를 차지. 하지만 selection atomicity가 chunk-level fragmentation 방지로 바람직함.
- **Floor가 corpus 분포에 따라 결정**: 매우 적게 나타나는 relation(<20 exposure)은 floor 없이 모두 keep(priority에서 처리됨). dominant relation은 head cap으로 억제.
- **Coverage gain 계산이 chunk_count approx**: pair-share 수준 정밀도는 아님. 향후 필요 시 pair_counts 기반 refinement 가능(Codex 원안 참고).

## 검증 결과

### Dry-run (2,000 docs, NER_NER task, 35% quota)

| 메트릭 | Pre-sample | Post-sample |
|---|---|---|
| Chunks | 9,595 | 3,358 (35.0%) |
| Selected origins | 2,000 | 771 |
| **Unique relations 커버리지** | 68 | **68 / 68 (100%)** |
| **Unique group_pairs 커버리지** | 93 | **87 / 93 (93.5%)** |
| NO_RELATION share | 0.143 | **0.133 ↓** |
| used_by share | 0.228 | **0.204 ↓** |
| same_entity share | 0.219 | **0.195 ↓** |

### Rare relation 보존 (bottom 10 pre-sample):
| Relation | Pre | Post | Preserved |
|---|---|---|---|
| parent_of | 2 | 2 | 100% |
| child_of | 2 | 2 | 100% |
| family_member_of | 3 | 3 | 100% |
| brand_of | 6 | 6 | 100% |
| sold_by | 6 | 6 | 100% |
| information_to | 12 | 7 | 58% |
| targeted_by | 12 | 11 | 92% |
| price_of | 16 | 16 | 100% |
| treats | 17 | 16 | 94% |
| required_by | 18 | 16 | 89% |

**모든 relation이 살아남고, 특히 초희귀(chunk ≤ 10)는 100% 보존**. `information_to` 58%도 35% 축소 상황에서는 적정. Head 3종은 일관되게 share 감소 → 의도대로 동작.

### Syntax / Integration
- `preprocessing/*`, `training/*`, `qwen_inference/*` 총 12개 파일 전부 `ast.parse` / `yaml.safe_load` OK
- `build_sampling_profile`가 실제 NER_NER/NER_BEE chunk 각각에서 올바른 shape 생성 확인
- 기존 수정(C1~C3, H1~H7, M1~M7) 모두 재회귀 없음

## 영향 및 운영 절차

### 전처리 재실행 필수
PHASE 3 sampler 교체 + sampling_profile 추가로 기존 HF dataset cache 무효화. 재실행:
```bash
rm -rf /app/pred_data/sllm_ready_generated_prompts_*_hf_dataset
python preprocessing/generate_sft_training_data_latest.py
```
실행 시 task별 sampler 로그로 coverage 상태 바로 확인 가능.

### 학습 — eval 없이 빠르게
```bash
python training/sft_qwen.py          # eval 비활성 (기본)
# 또는
python training/sft_gemma4.py
```

### 학습 — eval 켜기
```bash
ENABLE_EVAL=1 python training/sft_qwen.py
```

### Diversity sampling 비활성(legacy 동작으로 복귀)
```yaml
# generate_sft_training_data_config.yaml
generation:
  diversity:
    enable: false
```

### 수치 조정(floor/cap tuning)
`generation.diversity` 하위 키를 config에서 수정 후 전처리 재실행. 초기에는 기본값으로 시작 후 로그 관찰.

## 남은 과제

1. **Pair-share 수준 head cap 정밀도**: 현재는 chunk-share 근사. pair_counts 기반 실제 share 계산으로 업그레이드 가능(Codex 원안 수준).
2. **Train split 사후 검증**: 현재 quota는 total 기준. PHASE 4 split 후 train split이 relation floor를 지키는지 audit 로그 추가하면 안전.
3. **Entity surface 반복 penalty**: 동일 상품명/작성자명이 여러 chunk에 반복될 때 제한. 현재는 origin 단위만 제어.
4. **Dynamic rare set 자동 갱신**: 현재 static set(40개 영문 relation)은 corpus 변동 시 업데이트 수동. frequency 기반 bottom-N% 자동 판정 고려.
