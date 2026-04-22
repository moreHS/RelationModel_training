# 2026-04-22 — Gold Dataset 학습 혼합 (Hybrid B-lite + Gold-first Few-shot)

## 배경

`datas/gold/` 에 수작업 검증된 ground-truth 골드셋 2개 파일 투입:
- `ko_ner2ner_900.jsonl` (900 rows, NER-NER)
- `ko_ner2bee_1500_all.jsonl` (1,500 rows, NER-BEE — source_type이 정확히 NER/BEE로 라벨링됨)

합 2,400 rows. 학습셋(100K target) 대비 약 2.4%로 작음. 단순 merge 하면 "바다에 한 방울"이라 noise 수준. 의도적 upsampling + 높은 품질을 보장하는 혼합 전략이 필요.

## 리서치 근거

병렬 3축 검증:
- **codex Architect (advisory)**: PHASE 3 diversity sampler를 흔들지 않고 학습 시점에 혼합하는 **Hybrid (B-lite)** 권장. p_gold 한 축만 튜닝하면 돼서 effort Short.
- **Explore**: 현 코드에 `GenerateFewShotSamples.allowed_origin_ids`, `exclude_origin_id`, `RARE_RELATIONS`, `_TAG_RE` 등 재사용 가능한 기반이 이미 있음. 30~40% 기존 코드 재사용 가능.
- **WebSearch**:
  - HF `interleave_datasets(probabilities, stopping_strategy="all_exhausted")` — 작은 dataset을 반복 순회하며 지정 확률대로 혼합. 본 요구사항에 정확히 부합.
  - Multi-task custom BatchSampler (SFTTrainer 서브클래싱)는 strict batch guarantee가 필요한 경우에만 유효. p_gold=0.10이면 0-gold batch는 약 3.5%뿐이라 overkill.

## 의사결정

### 1. 혼합 방식: **Hybrid (B-lite)**
- Gold를 별도 HF Dataset으로 전처리
- 학습 시점에 `interleave_datasets`로 p_gold=0.10 혼합
- PHASE 3 diversity sampler 변경 없음 → 기존 안정성 보존
- 튜닝 축: `P_GOLD` 환경변수 하나

**대안 비교**:
| 전략 | 장점 | 단점 | 채택 |
|---|---|---|---|
| A. PHASE 1 복제 | 구현 단순 | PHASE 3 diversity floor 왜곡, relation exposure 부풀림, origin_id leakage 리스크 | ❌ |
| B-strict. Custom BatchSampler | 배치마다 정확히 N개 gold 보장 | SFTTrainer 서브클래싱 + get_train_dataloader 오버라이드 필요. p_gold=0.10이면 이득 미미 | ❌ |
| **Hybrid B-lite** | Gold 별도 전처리 + interleave로 확률 혼합. PHASE 3 무수정 | 0-gold batch가 가끔 발생 (~3.5%) | ✅ |

### 2. Gold split: **Train-only**
- Gold 양이 작아(2,400) 내부 70/10/20 분할시 val/test가 각 240~480에 불과
- Main의 val/test가 이미 평가 신뢰도를 담보
- 모든 gold origin은 `train_only` 모드로 train에 할당
- Cross-split leakage 원천 차단 (gold origin_id는 `gold_` prefix로 네임스페이스 분리되어 main과 절대 충돌 안 함)

### 3. p_gold = 0.10
- effective batch 32 기준 기대 gold **3.2개/step**, 0-gold batch 약 **3.5%**
- codex의 1차 추천(0.08)보다 약간 공격적. 사용자 결정.
- `P_GOLD=0.05` 또는 `0.15` 로 재학습 실험 시 env var만 바꾸면 됨 (코드 변경 불필요)
- 트레이드오프: p_gold > 0.12부터 3 epoch 기준 동일 gold sample 반복 빈도 급증 → overfit risk

### 4. Few-shot: **Gold-first, main-supplement (사용자 핵심 결정)**
- Gold는 수작업 검증된 고품질 데이터 → few-shot demonstrations에 가장 적합
- 다만 gold 양이 작아 일부 rare class가 없을 수 있음 → **RARE_RELATIONS 중 gold에 없는 것만** main pool에서 보충
- Main pool은 이미 train split origin_id로 제한되어 있어 val/test leakage 방지 유지

## 변경 내역

### 신규 파일
- **[preprocessing/generate_sft_training_data_config_gold.yaml](../preprocessing/generate_sft_training_data_config_gold.yaml)**
  - `data.input_paths`: 두 gold jsonl 리스트
  - `gold_origin_id_prefix: "gold_"` — origin_id 네임스페이스 분리
  - `split.mode: train_only`
  - `generation.diversity.enable: false` — 전체 gold 보존
  - `target_total_training_samples` 미설정 → PHASE 3 스킵

### 수정 파일
- **[preprocessing/generate_sft_training_data_latest.py](../preprocessing/generate_sft_training_data_latest.py)**
  - PHASE 1 ([:204-227](../preprocessing/generate_sft_training_data_latest.py#L204)): `input_paths` (list) 지원 + `gold_origin_id_prefix` 적용
  - PHASE 3 ([:366-371](../preprocessing/generate_sft_training_data_latest.py#L366)): `target_total_training_samples`가 null/미설정이면 quota 전체 스킵 로그
  - PHASE 4 ([:649-684](../preprocessing/generate_sft_training_data_latest.py#L649)): `split.mode: train_only` 지원. 모든 origin을 train으로 강제
  - PHASE 5 ([:725-779](../preprocessing/generate_sft_training_data_latest.py#L725)): `fewshot.gold_hf_dataset_path` 있으면 gold pool 로드 → `gold_fewshot_generator`로 `generate_all_prompts`에 전달
  - `generate_all_prompts` 시그니처에 `gold_fewshot_generator=None` 추가 ([:32-51](../preprocessing/generate_sft_training_data_latest.py#L32))
  - Gold 있을 때 `generate_gold_first` 호출 (alt fs_text도 동일 경로) ([:82-119](../preprocessing/generate_sft_training_data_latest.py#L82))
  - DatasetDict 저장 시 존재하는 split만 포함 ([:768-775](../preprocessing/generate_sft_training_data_latest.py#L768))
  - argparse `--config` 지원 ([:808-817](../preprocessing/generate_sft_training_data_latest.py#L808))

- **[preprocessing/data_preprocessor_utils_simplified_add_gemma4.py](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py)**
  - `GenerateFewShotSamples.generate_by_pairs`에 `filter_relations` 파라미터 추가 ([:631-653](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py#L631)) — 지정된 relation만 포함하는 chunk로 pool 필터
  - 신규 메서드 `generate_gold_first` ([:707-767](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py#L707)) — gold를 primary로 뽑고, 미포함 rare relation만 supplementary_generator(main)에서 보충

- **[qwen_inference/data_preprocessor_utils.py](../qwen_inference/data_preprocessor_utils.py)** — 위와 동일하게 mirror 적용 (eval 정합)

- **[preprocessing/generate_sft_training_data_config.yaml](../preprocessing/generate_sft_training_data_config.yaml)**
  - `fewshot.gold_hf_dataset_path` 키 추가 ([:75](../preprocessing/generate_sft_training_data_config.yaml#L75)) — gold HF dataset 경로. 없으면 main-only fall back

- **[training/sft_gemma4.py](../training/sft_gemma4.py)**
  - `from datasets import ... interleave_datasets` 추가
  - `GOLD_DATA_PATH`, `P_GOLD = float(os.environ.get("P_GOLD", "0.10"))` 상수
  - 데이터 로드 블록 ([:143-168](../training/sft_gemma4.py#L143)): gold HF dataset 존재하면 `interleave_datasets([main, gold], probabilities=[1-P_GOLD, P_GOLD], stopping_strategy="all_exhausted")`. 실패/미존재 시 main-only fallback (try/except)

## 트레이드오프

### Hybrid B-lite (Strict B 대비)
- ✅ PHASE 3 sampler 무수정 → 기존 diversity 보장 그대로
- ✅ 구현 Short (반나절)
- ❌ 배치마다 gold N개 정확히 보장 안 됨 (약 3.5% batch가 0-gold) — 하지만 3 epoch 학습 기준 문제 없는 수준

### Train-only split (ratio 대비)
- ✅ 학습 signal 최대화 (2,400 rows 전부 활용)
- ✅ cross-split leakage 원천 차단
- ❌ Gold 기반 best-checkpoint 선택 불가 → main validation으로 대체 (main val은 이미 대규모로 존재)

### Gold-first few-shot (main-only 대비)
- ✅ 고품질 demonstrations → rare class 학습 효과 ↑
- ✅ Rare 미포함분은 main에서 자동 보충 → 커버리지 손실 없음
- ❌ Gold가 작아 동일 demonstration이 반복 노출될 수 있음 → few-shot diversity는 기존보다 약간 감소

### HF interleave_datasets vs custom sampler
- ✅ interleave는 HF 공식 지원 + `stopping_strategy="all_exhausted"`로 작은 dataset 반복 순회 지원
- ✅ 시드 고정 가능 (`seed=3407`)
- ❌ Pure batch-level guarantee는 아님 (확률 기반) — 본 요구엔 충분

## 검증 결과

### Syntax / 파싱
6개 touched files (`.py` 4 + `.yaml` 2) 전부 ast.parse / yaml.safe_load 통과.

### Config 통합
- `main.fewshot.gold_hf_dataset_path`가 gold output_path와 정확히 매칭
- `gold.split.mode = "train_only"`
- `gold.generation.diversity.enable = false`
- `gold.gold_origin_id_prefix = "gold_"`

### 기능 단위 테스트 (합성 데이터)
- `generate_by_pairs(filter_relations={"brand_of","price_of"})`: 지정 relation이 포함된 chunk만 반환 ✅
- `generate_gold_first(supplementary_generator=main_gen)`: primary 샘플은 gold에서, 보충은 main에서 rare relation만 가져옴 ✅
  - 예시: gold picks 1개 (`gold_1`, used_by+applied_to), 보충 1개 (`o5`, `parent_of` — rare)

### Gold 파일 스키마 sanity
- `ko_ner2ner_900.jsonl`: 900 rows, source_type 모두 `NER`
- `ko_ner2bee_1500_all.jsonl`: 1,500 rows, NER-BEE 정상 라벨링 (첫 샘플 subj=NER, obj=BEE) — 이전 세션에서 발견한 source_type 오라벨 버전(`ko_ner2bee_1500.jsonl`)과는 다른 정상 버전

## 운영 절차

### 최초 실행
```bash
# 1. Gold 전처리 (main과 분리 실행)
python3 -m preprocessing.generate_sft_training_data_latest \
    --config preprocessing/generate_sft_training_data_config_gold.yaml
# → /app/pred_data/..._gemma4_gold_v2_hf_dataset 생성 (train split만)

# 2. Main 전처리 — gold를 few-shot primary pool로 참조 (gold가 이미 있으면 자동 인식)
python3 -m preprocessing.generate_sft_training_data_latest \
    --config preprocessing/generate_sft_training_data_config.yaml
# → few-shot이 gold-first hybrid로 구성됨

# 3. 학습 — interleave로 p_gold=0.10 혼합
python3 training/sft_gemma4.py
# p_gold 조정: P_GOLD=0.08 python3 training/sft_gemma4.py
```

### Rollback
- Gold 완전 비활성: `GOLD_DATA_PATH`를 존재하지 않는 경로로. 코드 fallback이 `main-only` 실행.
- Gold few-shot만 비활성: main config의 `fewshot.gold_hf_dataset_path` 제거 또는 주석.

## 남은 과제 / 향후 개선

1. **Gold few-shot diversification**: 같은 gold 샘플이 수천 row에 반복 노출 가능. `generate_gold_first` 호출마다 다른 gold chunk를 뽑도록 rotation 추가 고려.
2. **Strict batch guarantee (필요 시)**: p_gold를 0.05 아래로 낮출 때 0-gold batch 비율이 20%를 넘어가므로 custom sampler 도입 검토.
3. **Gold split ratio 버전**: 향후 gold가 10K+로 커지면 80/10/10 내부 split으로 전환해 best-checkpoint도 gold 기준 선택.
4. **Rare-coverage audit**: 학습 후 첫 100 few-shot 블록에서 gold/main supplement 비율과 커버된 rare relation 집합 로깅.
