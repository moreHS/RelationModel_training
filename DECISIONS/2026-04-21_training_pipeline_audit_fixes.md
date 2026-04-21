# 2026-04-21 — 학습 파이프라인 전면 감사 및 이슈 16건 수정

## 배경

리포지토리 전체(학습 스크립트 / 전처리 / 프롬프트 YAML / eval 파이프라인)를 "구현 품질 + 학습 플로우 + 학습 데이터 형태/방식 + 모델 학습론" 네 관점에서 다면적 감사. 병렬로 Explore 에이전트 3개(Training / Preprocessing / Eval) + codex Architect를 소환해 독립 관점 4개를 교차 검증했고, Critical로 판명된 NER-BEE 라벨 corruption은 실제 데이터 2,000 docs로 종단 재현하여 확정.

결과적으로 Critical 3건 + High 7건 + Medium 6건의 이슈를 식별해 전부 수정. 재학습/재평가 전에 반드시 필요한 수준의 수정으로, 현 상태로 학습을 돌리면 NER-BEE 태스크의 학습 objective 자체가 뒤틀려 메트릭 신뢰가 불가능한 상태였음.

아래 이슈 번호는 감사 플랜 파일([`wobbly-zooming-nest.md`](../../../.claude/plans/wobbly-zooming-nest.md))과 동일.

---

## C1 — NER-BEE `is_relational` 타깃 corruption

### 배경
NER-BEE 태스크의 `system_prompt_ner_bee`와 `output_format_w_keys_ner_bee`는 모델에게 "is_relational 키에는 true/false 소문자 문자열만 쓰라"고 지시([sft_data_generation_prompts_edited.yaml:22](../preprocessing/sft_data_generation_prompts_edited.yaml#L22)). 그러나 실제 학습 데이터에는 다른 값이 들어가고 있었음.

### 문제
`PreprocessInput.build_input_output`이 NER_BEE/NER_BEE_TRUE_ONLY 태스크에서 `is_relational` 키에 raw relation을 그대로 넣음. 원본 데이터의 NER-BEE 페어는 `relation` 필드에 "Effect", "Moisturizing Power", "Quality" 같은 BEE 속성 영문명 또는 "provided_to", "benefits" 같은 NER-NER 스타일 relation name을 가지며, `normalize_bee_label`을 통과하면 영문 BEE는 한글("보습력", "효과")로 정규화됨. negative는 `normalize_negative_label`이 "no_relationship" → "NO_RELATION"으로 바꾸지만 "false"로 바꾸지는 않음.

### 실측 증거 (sllm_all_ready.jsonl 첫 2,000 docs, 103,720 NER-BEE 학습 샘플)

| is_relational 값 | 개수 | 비율 |
|---|---|---|
| `"true"` | 0 | 0% |
| `"false"` | 0 | 0% |
| `"NO_RELATION"` | 89,374 | 86.2% |
| BEE 한글 속성명 (효과/보습력/품질/향/제형/충성도/사용감 등) | 14,168 | 13.7% |
| NER-NER 스타일 (benefits/used_by/provided_to 등) | 132 | 0.1% |
| 기타 | 46 | 0.0% |
| **Contract 위반** | **103,720 / 103,720** | **100%** |

총 **69개 고유 문자열**이 `is_relational`에 들어감. system_prompt는 2개(`"true"`, `"false"`)만 허용.

### 종단 재현
한 샘플(doc id=1)을 `build_input_output` → `PromptCompiler.compile_prompts` → `_apply_chat_template` 전 과정을 돌린 최종 학습 텍스트에서:
- offset 770 (system 섹션): `"is_relational" key must be true or false`
- offset 13,970 (assistant 응답): `"is_relational": "provided_to"`
- offset 14,485: `"is_relational": "benefits"`
- offset 16,908: `"is_relational": "related_to"`

동일 학습 샘플 한 건 안에 system 지시와 assistant 라벨이 모순 공존.

### 원인 (코드 라인)
- [data_preprocessor_utils_simplified_add_gemma4.py:184-188](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py#L184) — NER-BEE 분기의 `normalize_bee_label`은 BEE 속성명 영↔한만 변환, 그 외는 그대로
- 같은 파일 [:277](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py#L277) — NER_BEE 태스크에서 `rel_key = "is_relational"`
- 같은 파일 [:316](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py#L316) — `rel_key: raw_rel`로 속성명/관계명/NO_RELATION이 그대로 투입
- 같은 파일 [:397](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py#L397) — `compile_prompts`의 `ans = f"### Response:\n{out_val}"`는 JSON 문자열을 변환 없이 삽입
- 같은 파일 [:425](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py#L425) — `tokenizer.apply_chat_template(messages)`는 content 내부 문자열을 건드리지 않음

### 검토한 선택지
- **A. binary 정규화 유지 (선택)** — `raw_rel in (None, "NO_RELATION") → "false"`, 그 외 → `"true"`. 프롬프트/파서 변경 없음. 30분.
- B. attribute classification으로 승격 — `is_relational` → `attribute` 키 변경, YAML에 유효 39개 한글 BEE 속성 목록 추가, parser 강제 fold 제거. NER_BEE와 NER_NER의 역할 구분이 모호해질 수 있고 기존 NER_BEE 학습 전량 재해석 필요. 반나절 + 재학습.
- C. 현재 mismatched 상태 그대로 두고 parser의 binary fold에만 의존 — 실제 모델 출력은 unstable, out-of-schema 위험.

### 선택 이유
사용자 지정 Option A. 프롬프트는 이미 binary 설계이고 parser도 binary fold 중이므로 학습 라벨을 프롬프트와 정합시키는 것이 구조적으로 자연스럽고 리스크 최소. Option B로 갈 거라면 별도 의사결정 후 NER_BEE 태스크 재설계가 적절.

### 변경 내역
- [data_preprocessor_utils_simplified_add_gemma4.py:299-326](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py#L299) — 태스크가 NER_BEE/NER_BEE_TRUE_ONLY면 `output_rel = "false" if raw_rel in (None, "NO_RELATION") else "true"`. NER_NER은 기존대로 multi-class 유지.
- [qwen_inference/data_preprocessor_utils.py:299-326](../qwen_inference/data_preprocessor_utils.py#L299) — 동일 변경(eval 파이프라인 정합)
- [sft_data_generation_prompts_edited.yaml:239, :249](../preprocessing/sft_data_generation_prompts_edited.yaml#L239) — 프롬프트 내 `"False"` 대소문자 불일치를 `"false"`로 정렬하고 `The "is_relational" value must be the lowercase string "true" or "false".` 명시 강화

### 트레이드오프
- NER-BEE가 속성 분류가 아닌 단순 존재 여부 판별로 고정 → 다운스트림에서 "어떤 속성인지"는 엔티티 타입으로만 추론해야 함(entity_group은 이미 BEE 속성명)
- 기존에 BEE 속성 학습된 checkpoint가 있다면 의미 없어짐 → 재학습 필수

### 검증 결과
- 500 docs 재처리: 23,297 NER_BEE 샘플 전부 {`"false"`, `"true"`}. illegal 값 0건.
- 한 sample 종단: 31건 중 `"true"` 3건, `"false"` 28건. offset 770 prompt 지시와 offset 13,964+ assistant 라벨 모두 `"true"`/`"false"`만 등장.
- NER_NER은 영향 없음 (43개 고유 multi-class 라벨 유지).

---

## C2 — Few-shot leakage (split 이전 전역 corpus에서 생성)

### 배경
학습 데이터 생성 시 few-shot examples를 task 전체 코퍼스에서 뽑아 모든 row에 동일 블록을 삽입. val/test split은 few-shot 생성 이후에 계산됨.

### 문제
1. **Global leakage**: val/test row의 few-shot 블록에 자기 자신의 origin_id 문서에서 나온 chunk가 포함될 수 있음 → val/test 지표 인플레이션.
2. **Self-demonstration**: train row도 같은 origin의 다른 chunk가 few-shot에 들어가면 demo와 본문이 중복, 실제 생성 능력 평가가 어려워짐.

### 원인 (코드 라인)
- [generate_sft_training_data_latest.py:68-73](../preprocessing/generate_sft_training_data_latest.py#L68) (수정 전) — `fs_samples = fewshot_generator.generate_by_pairs(task_name, ...)`는 task 전체 corpus에서 추출
- 같은 파일 [:106](../preprocessing/generate_sft_training_data_latest.py#L106) — `fs_text_global`을 모든 row에 재사용
- 같은 파일 [:287-292](../preprocessing/generate_sft_training_data_latest.py#L287) — split이 이 이후에 만들어짐

### 검토한 선택지
- **A. train split로 pool 제한 + 데모 origin 별 alt fs_text 캐싱 (선택)**
- B. 매 row마다 few-shot 새로 뽑기 — 정확하지만 `Dataset.map(num_proc=N)` 환경에서 비용 과다
- C. 그대로 두고 leakage 영향만 측정 — 버그를 남겨둠

### 선택 이유
few-shot은 각 task별 8-12 pair만 뽑으므로 실제 데모에 등장하는 고유 origin은 3-5개 수준. 이 소수 origin에 대해서만 alt fs_text를 precompute하면 다수 row는 global fs_text 재사용 가능 → 비용 ≪ B안. A안이 leakage 완전 차단 + 성능 유지의 최적점.

### 변경 내역
- [data_preprocessor_utils_simplified_add_gemma4.py:481-506](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py#L481) — `GenerateFewShotSamples.__init__`에 `allowed_origin_ids` 인자 추가. None이 아니면 data_dict을 필터링.
- 같은 파일 [:508-518](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py#L508) — `generate_by_pairs`에 `exclude_origin_id` 인자 추가. 주어지면 해당 origin chunk 제외.
- [generate_sft_training_data_latest.py:413-428](../preprocessing/generate_sft_training_data_latest.py#L413) — split 계산 후 `train_origin_ids`를 뽑아 `allowed_origin_ids`로 전달. pool 제한 명시적 로그.
- 같은 파일 [:62-99](../preprocessing/generate_sft_training_data_latest.py#L62) — task별 global `fs_samples` 생성 후 거기 등장한 `demo_origin_ids`에 대해서만 alt fs_text를 precompute. row 처리 시 해당 origin이면 alt 사용, 아니면 global 재사용.
- [qwen_inference/data_preprocessor_utils.py:477-549](../qwen_inference/data_preprocessor_utils.py#L477) — eval 파이프라인 정합을 위해 동일 변경

### 트레이드오프
- 데모 origin이 2-3개이므로 alt fs_text precompute 비용은 task별 +2-3회 `generate_by_pairs`. 총 오버헤드 수 초 이하.
- Train split이 전체의 70%이므로 pool 축소로 rare relation 커버리지가 약간 줄 수 있음 (극단적 rare의 경우에만). prioritize_rare가 여전히 작동하므로 실질 영향 미미.

### 검증 결과
합성 데이터 유닛 테스트 3종:
- Pool restriction: `allowed_origin_ids={o0..o2}` → 샘플의 origin 전부 그 집합 내.
- Self-exclusion: `exclude_origin_id=o0` → 샘플에 o0 없음.
- 호출 체인: `train_origin_ids` → alt map → row별 교체 로직이 Dataset.map 환경에서 동작 확인.

---

## C3 — Qwen `train_on_responses_only` 미호출

### 배경
Unsloth 권장 패턴: 학습 시 user/system 토큰을 loss masking하고 assistant 응답에만 gradient 흘리기. `train_on_responses_only(trainer, instruction_part, response_part)`로 적용.

### 문제
- [training/sft_qwen.py:11](../training/sft_qwen.py#L11) — `from unsloth.chat_templates import train_on_responses_only` 임포트만 되어있고 실제 호출 없음
- [training/sft_gemma4.py:211-215](../training/sft_gemma4.py#L211) — Gemma4는 올바르게 호출됨

결과: Qwen은 프롬프트 전체(system + description + few-shot + output_format + user input)에까지 loss가 계산됨. 본 리포의 프롬프트는 description/few-shot이 길어 assistant 응답 대비 boilerplate 비중이 매우 큼 → gradient가 assistant 응답 생성 능력보다 프롬프트 복제에 낭비.

### 검토한 선택지
- **A. 호출 추가 (선택)** — Gemma4와 완전 대칭.
- B. 호출 안 하고 두는 방식 유지 — Gemma4와 학습 objective가 달라서 두 모델 비교 불가.

### 변경 내역
- [training/sft_qwen.py:198-206](../training/sft_qwen.py#L198) — `SFTTrainer` 생성 직후 다음 추가:
  ```python
  trainer = train_on_responses_only(
      trainer,
      instruction_part="<|im_start|>user\n",
      response_part="<|im_start|>assistant\n",
  )
  print("✅ Applied train_on_responses_only (instruction tokens masked).")
  ```

### 트레이드오프
- 없음 (Unsloth 공식 권장, Gemma4는 이미 동일 패턴).
- 초기 loss 스케일이 이전보다 커질 수 있음 (prompt 토큰 masked로 평균 loss가 더 informative해지며, 학습 곡선 해석이 바뀜).

### 검증 결과
Syntax OK. 실제 학습 시 첫 step의 loss 스케일이 Gemma4와 유사한 레인지인지 확인할 것 (재학습 시).

---

## H1 — NER_BEE_TRUE_ONLY 제거 (eval과 일치)

### 배경
- [generate_sft_training_data_latest.py:186](../preprocessing/generate_sft_training_data_latest.py#L186) (수정 전) — NER_BEE의 negative 제외 subset을 별도 태스크로 정의해 학습
- [generate_sft_training_data_latest.py:265](../preprocessing/generate_sft_training_data_latest.py#L265) — quota는 task별 균등 → 학습 데이터의 약 1/3이 "positive만 있는" NER_BEE_TRUE_ONLY
- [qwen_inference/generate_eval_data.py:101](../qwen_inference/generate_eval_data.py#L101) — eval 파이프라인은 NER_BEE_TRUE_ONLY 지원 안 함 (ValueError raise)

### 문제
학습 목적에 포함된 태스크를 평가할 수 없음. 메트릭 계산 체계에서 해당 서브태스크의 성능을 확인 불가 → 학습 대비 평가 범위 불일치.

### 검토한 선택지
- **A. 학습에서 제거 (선택)** — eval과 학습 범위 정합. NER_BEE 자체가 binary + positive까지 포함하므로 true_only 없어도 positive만 학습되는 효과는 NER_BEE의 positive 샘플이 담당.
- B. eval에 NER_BEE_TRUE_ONLY 지원 추가 — 더 많은 작업. 제품 요구사항 확인 필요.
- C. 그대로 두기 — 정합성 문제 방치.

### 선택 이유
사용자 확정 A. eval에서 이미 막혀있고(C3 → C2 논의 시점에 이미 eval 입구 차단됨), 학습에만 포함되는 dangling task는 관리 복잡도 증가. NER_BEE가 binary로 정규화(C1)되면서 positive만 보는 별도 태스크의 의미도 줄어듦.

### 변경 내역
- [generate_sft_training_data_config.yaml:40-44](../preprocessing/generate_sft_training_data_config.yaml#L40) — `tasks`에서 `NER_BEE_TRUE_ONLY` 제거, 주석으로 재활성화 절차 명시
- [generate_sft_training_data_latest.py:213-222](../preprocessing/generate_sft_training_data_latest.py#L213) — `raw_sources`를 `cfg["tasks"]` active set에 따라 조건부 빌드. 비활성 태스크는 아예 처리 안 함 → 메모리/시간 절약.

### 트레이드오프
- 향후 NER_BEE_TRUE_ONLY가 필요해지면 eval 지원 추가 + 태스크 재활성화 후 재학습 필요.
- 현재 NER_BEE가 이미 positive/negative 둘 다 학습하므로 실질적 모델 capability 손실은 없음.

### 검증 결과
- `tasks: ['NER_NER', 'NER_BEE']`로 확정.
- `active_task_values` 로직이 `raw_sources`에 반영됨을 syntax/로직 검증.

---

## H2 — Safe token limit few-shot overhead 미반영

### 배경
Phase 2 청킹 시 prompt overhead를 계산해 SAFE_LIMIT을 도출. 기존 구현은 `dummy = {"input": "", "output": []}`로 overhead를 측정 → few-shot이 enable된 모드에서도 few-shot 텍스트 크기 미반영.

### 문제
full_detailed/full_summary 등 `few_shot=True`인 모드는 실제 프롬프트에 ~2000-3000 tokens의 few-shot 블록이 들어감. overhead 계산 시 이를 빠뜨리면 SAFE_LIMIT이 낙관적으로 커지고, 청킹 후 실제 prompt 컴파일 시점에 8192 초과 → Phase 6 pruning 단계에서 drop → 학습 데이터 손실.

`-2000` 고정 버퍼로 가려져 있었으나, rare relation 집중 + 큰 description 조합에서는 여전히 초과 가능.

### 원인 (코드 라인)
- [data_preprocessor_utils_simplified_add_gemma4.py:443-447](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py#L443) — `calculate_template_overhead`는 빈 input/output만 컴파일

### 검토한 선택지
- **A. overhead 계산 시 실제 fs_text 주입 (선택)** — 현실적 상한 반영
- B. few-shot 크기 상수(예: 3000)를 static하게 더함 — 단순하지만 과대/과소 추정 가능
- C. few-shot 포함 prompt 렌더링을 Phase 2에서 미리 수행 — 비용 과다

### 변경 내역
- [data_preprocessor_utils_simplified_add_gemma4.py:451-469](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py#L451) — `calculate_template_overhead(prompt_compiler, tokenizer, fewshot_text="")` 시그니처 추가. `mode_config.few_shot`이 True일 때만 fs 주입, 아니면 빈 문자열.
- [generate_sft_training_data_latest.py:255-282](../preprocessing/generate_sft_training_data_latest.py#L255) — Phase 2에서 현재 task의 raw entries 상위 50개로 pre-chunk + `GenerateFewShotSamples`로 probe_fs_text 생성 후 overhead 계산에 주입. SAFE_LIMIT 버퍼를 2000 → 1000으로 축소(overhead가 이미 fs 포함하므로 추가 버퍼 감소 가능).
- [qwen_inference/data_preprocessor_utils.py:451-462](../qwen_inference/data_preprocessor_utils.py#L451) — 동일 변경

### 트레이드오프
- Phase 2 처음에 task별 probe용 경량 preprocessing이 추가됨 (50개 entries). 오버헤드 수 초.
- SAFE_LIMIT이 이전보다 작아져 청크가 더 잘게 쪼개질 수 있음 → 청크 수 소폭 증가. 대신 Phase 6 drop이 거의 사라짐.

### 검증 결과
- 같은 compiler에 `fewshot_text=""` vs 2000자 텍스트 전달 → overhead 값이 8580 → 10645로 증가 확인 (DummyTok 기준, 차이 ≈ few-shot 텍스트 길이).
- H2 수정 전 2026-04-13의 PR 2.4 결정(max overhead 사용)과 조합해 가장 무거운 모드 + few-shot까지 반영된 상한을 획득.

---

## H3 — Warmup 너무 짧음

### 배경
- [training/sft_qwen.py](../training/sft_qwen.py) (수정 전) — `warmup_steps=5`
- [training/sft_gemma4.py:167](../training/sft_gemma4.py#L167) (수정 전) — `warmup_steps=50`

### 문제
target_total_training_samples=300,000 기준 effective batch=32, 3 epochs → 약 20K step. 전체 step 대비 warmup이 Qwen 0.025%, Gemma4 0.25%로 SFT 표준(3~10%) 대비 극단적으로 짧음. 초기 gradient spike → loss 진동, SFT 초반에 최적 방향 못 잡을 가능성.

### 변경 내역
- [training/sft_qwen.py:149](../training/sft_qwen.py#L149) — `warmup_steps = 5` → `warmup_ratio = 0.03`
- [training/sft_gemma4.py:167](../training/sft_gemma4.py#L167) — `warmup_steps = 50` → `warmup_ratio = 0.03`

### 선택 이유 (ratio vs steps)
Step 수는 dataset 크기/batch/epochs에 따라 달라지므로 ratio가 더 robust. 3%는 Unsloth/TRL 예제에서 일반적이며 cosine scheduler와 궁합이 좋음.

### 트레이드오프
- 학습 초반 ~600 step(20K 기준)이 linear warmup → 이 구간 동안은 학습 신호 적음.
- 그러나 이 작은 비용으로 나머지 19,400 step의 안정성 확보.

### 검증 결과
두 스크립트 모두 `warmup_ratio` 키 존재, `warmup_steps` 키 제거 확인.

---

## H4 — Save/eval 전략 두 스크립트 간 10배 차이

### 배경
- Qwen: `save_steps=180`, `eval_steps=180`
- Gemma4: `save_steps=1000`, `eval_steps=5000`, `load_best_model_at_end=False` (하드코딩)

### 문제
1. 두 모델의 학습 과정 비교가 어려움 (checkpoint 빈도/eval 빈도 규모 차이).
2. Gemma4는 eval은 돌리지만 best model 선택을 포기 → 마지막 step weight 저장 → SFT 중 overfitting 구간이 있어도 그 weight 사용.
3. Qwen의 180은 너무 조밀해 checkpoint I/O 과다.

### 변경 내역
- [training/sft_qwen.py:175, :183](../training/sft_qwen.py#L175) — `save_steps=180` → `500`, `eval_steps=180` → `500`
- [training/sft_gemma4.py:187, :191, :193](../training/sft_gemma4.py#L187) — `save_steps=1000` → `500`, `eval_steps=5000` → `500`, `load_best_model_at_end=False` → `True if eval_dataset else False`

### 선택 이유 (500)
20K step 기준 save=500이면 40 checkpoint 중 save_total_limit=2로 rotation → 안전. eval 빈도도 500이면 약 40회 측정으로 best model 선택에 충분한 샘플. 180은 과다, 1000/5000은 과소.

### 트레이드오프
- 모니터링 해상도가 Qwen은 약간 감소, Gemma4는 대폭 증가.
- `load_best_model_at_end=True`는 validation 필요 → M7과 결합해 validation 없으면 False로 자동 fallback.

### 검증 결과
- 두 스크립트 `save_steps=500`, `eval_steps=500` 확인.
- Gemma4의 `load_best_model_at_end=True if eval_dataset else False` 확인.

---

## H5 — Eval 파이프라인 경로 incoherence + silent drop

### 배경
`qwen_inference/*` 스크립트들은 각각 독립적으로 하드코딩된 경로를 사용하고 config 통일이 안 되어 있었음.

### 문제
1. [merge_w_metadata.py:18](../qwen_inference/merge_w_metadata.py#L18) — 구형 학습 test split 경로(`sllm_ready_generated_prompts_{TARGET}_hf_dataset/test`)
2. [vllm_model_inference_w_edited_prompts.py:30](../qwen_inference/vllm_model_inference_w_edited_prompts.py#L30) — 신규 flat eval 경로(`eval_qwen_no_fewshot_hf`)
3. [merge_w_metadata.py:21](../qwen_inference/merge_w_metadata.py#L21) — `METADATA_FILE` 하드코딩. eval 입력이 다른 파일이면 origin_id 매칭 실패 → 모든 예측 silent drop (`if not meta: continue`)
4. [merge_w_metadata.py:65-69](../qwen_inference/merge_w_metadata.py#L65) — row count mismatch를 경고만 찍고 zip truncation

### 변경 내역
- [qwen_inference/generate_eval_data_config.yaml](../qwen_inference/generate_eval_data_config.yaml) — `inference` / `merge` 섹션 추가:
  - `inference.predictions_path`, `inference.lora_path`, `inference.max_lora_rank`, `inference.gpu_memory_utilization`
  - `merge.output_path`, `merge.eval_mode`
- [qwen_inference/vllm_model_inference_w_edited_prompts.py](../qwen_inference/vllm_model_inference_w_edited_prompts.py) — 전면 재작성:
  - `EVAL_CONFIG` env var로 config 경로 오버라이드 가능
  - `detect_model_format()`으로 stop tokens + format 자동 판별
  - LoRA 활성/비활성 config 기반 분기
  - `strip_think_tags()`는 MODEL_FORMAT 기반으로 통일 (구 MODE 문자열 매칭 제거)
- [qwen_inference/merge_w_metadata.py](../qwen_inference/merge_w_metadata.py) — 전면 재작성:
  - 세 경로(predictions/HF dataset/metadata) 전부 config에서 읽음
  - `METADATA_FILE = cfg["data"]["input_path"]` → eval 입력과 동일 파일 강제
  - row count mismatch → **hard fail** (`raise RuntimeError`)
  - 첫 100 rows의 metadata lookup hit/miss를 카운트해 로그. hit rate < 95% 시 경고 + 놓친 origin_ids 일부 출력 → silent drop 가시화

### 검증 결과
- 두 스크립트 모두 `yaml.safe_load` + `cfg["inference"]` 읽기 확인
- merge 스크립트에 `raise RuntimeError("Row count mismatch...")` 존재
- `hit rate` 로그 문자열 존재

### 영향
- 사용법: `EVAL_CONFIG=qwen_inference/generate_eval_data_config.yaml python ...`
- silent drop 시나리오(ko_ner2bee_1500 + sllm_all_ready metadata 불일치 등)가 발생해도 즉시 로그로 드러남

---

## H6 — Stop token / think tag cleanup 불일치

### 배경
- [vllm_model_inference_w_edited_prompts.py:96-99](../qwen_inference/vllm_model_inference_w_edited_prompts.py#L96) (수정 전) — `elif MODE == "qwen_checkpoint1080"` 같은 특정 문자열로만 `<think>` 제거 → 다른 qwen 체크포인트는 태그 남음
- [generate_eval_data.py:55-61](../qwen_inference/generate_eval_data.py#L55) (preprocessing 측) — think tag 제거 로직 없음
- 결과: eval preprocessing과 vllm inference 양쪽 cleanup이 불일치 → parser가 think tag 포함 raw text를 parsing 시 실패

### 변경 내역
- [qwen_inference/vllm_model_inference_w_edited_prompts.py](../qwen_inference/vllm_model_inference_w_edited_prompts.py) `strip_think_tags(text, model_format)`:
  ```python
  text = re.sub(r'<\|think\|>.*?<\|/think\|>', '', text, flags=re.DOTALL)  # Gemma4 style
  text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)           # legacy text form
  ```
  모든 prediction에 무조건 적용 (MODE 문자열 분기 제거)

### 트레이드오프
- `<think>`와 `<|think|>` 두 스타일 모두 제거 → 모델별 조건 분기 불요.
- think가 없는 정상 응답에는 regex가 matched 안 되므로 no-op.

### 검증 결과
재작성된 vllm 스크립트 내 `strip_think_tags` 호출 존재, MODE별 조건 분기 제거됨.

---

## H7 — Rare relation prioritization 개선 (한글 BEE + `prioritize_rare` 연결)

### 배경
- [data_preprocessor_utils_simplified_add_gemma4.py:463](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py#L463) (수정 전) — `RARE_RELATIONS`는 NER-NER 영문 relation만 하드코딩 (40개)
- [generate_sft_training_data_latest.py](../preprocessing/generate_sft_training_data_latest.py) — config의 `prioritize_rare`는 전달만 되고 실제 사용 안 됨

### 문제
NER-BEE는 이제 binary라 rare 개념이 적용되지 않음. 그러나 NER-BEE는 positive 비율이 약 14%로 낮아 few-shot이 전부 negative로만 채워지면 모델이 "모든 것이 false"를 학습할 위험.

### 변경 내역
- [data_preprocessor_utils_simplified_add_gemma4.py:508-548](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py#L508) — `generate_by_pairs`에 `prioritize_rare` 인자 추가. 태스크별 priority 정의:
  - NER-NER: 기존 `RARE_RELATIONS` 포함 chunk 우선
  - NER-BEE / NER-BEE_TRUE_ONLY: `"is_relational": "true"` 포함 chunk 우선 (positive example 보장)
- [generate_sft_training_data_latest.py:68-99](../preprocessing/generate_sft_training_data_latest.py#L68) — 호출부에서 `prioritize_rare=prioritize_rare` 실제 전달 (global fs_text + per-demo-origin alt fs_text 양쪽 모두)

### 트레이드오프
- prioritize_rare=True일 때 few-shot의 "common" 케이스 노출이 줄어듦. 하지만 corpus의 80%+가 common이므로 실제 학습 데이터에는 충분히 노출됨.

### 검증 결과
- NER-BEE의 priority 판정이 `"is_relational": "true"` 문자열 match로 동작.
- `prioritize_rare=False` 전달 시 rare/common 구분 없이 무작위 pool로 shuffle → config 옵션이 실제 효과.

---

## M1 — LoRA dropout = 0

### 배경
두 학습 스크립트 모두 `lora_dropout=0`. 정규화는 `weight_decay=0.05` 단독 의존.

### 문제
LoRA fine-tuning 표준 dropout은 0.05~0.1. 0은 overfitting 리스크 (특히 target_total_training_samples=300,000 + 3 epochs 조합에서).

### 변경 내역
- [training/sft_qwen.py:79](../training/sft_qwen.py#L79) — `lora_dropout = 0` → `0.05`
- [training/sft_gemma4.py:115](../training/sft_gemma4.py#L115) — `lora_dropout = 0` → `0.05`

### 트레이드오프
- 수렴 속도 미세한 감소 가능. 대신 generalization 개선. SFT에서 0.05는 보편적 기본값.

---

## M2 — Class imbalance 대응 부재

### 배경
전체 데이터의 relation 분포가 극단적으로 skewed:
- NO_RELATION 45% / used_by 17% / same_entity 13%
- 나머지 144개 relation은 대부분 100K 이하 샘플

기존 quota sampling([generate_sft_training_data_latest.py:267-279](../preprocessing/generate_sft_training_data_latest.py#L267) 수정 전)은 origin_id 단순 random → rare relation을 가진 origin이 우연히 탈락할 위험.

### 변경 내역
- [generate_sft_training_data_latest.py:324-387](../preprocessing/generate_sft_training_data_latest.py#L324) — stratified origin sampling 재작성:
  1. `priority origins` = rare 라벨(`RARE_RELATIONS` 세트) 또는 (NER-BEE의) positive 라벨을 하나라도 포함하는 chunk의 origin
  2. `common origins` = 나머지
  3. quota 계산 후 priority 전체를 먼저 keep, 남은 여유에서 common을 random sample
  4. priority가 quota를 초과하면 priority 내부에서 random sample

상세 로그로 task별 priority/common 보존 수 출력:
```
✂️ ner_ner: 120000 -> 40000 chunks (priority origins kept: 800/800, common: 3200)
```

### 트레이드오프
- Priority가 많은 task(NER-NER, rare 라벨 다수 존재)는 quota 대비 priority 비중이 커져 common 축소 폭이 큼.
- Rare 정의가 static set(40개 relation + 한글 BEE positive)이므로, 해당 set에 없지만 실제로 rare한 relation은 여전히 random sample. 향후 frequency 기반 dynamic rare 판정 고려 가능.

### 검증 결과
합성 데이터 테스트에서 priority origins (5개) 정확 식별.

---

## M3 — Chunker text 복제 완화

### 배경
- [data_preprocessor_utils_simplified_add_gemma4.py:252](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py#L252) (수정 전) — `chunk_entry["text"] = text` → 원문을 모든 chunk에 동일 복제

### 문제
pair-dense 문서(평균 80 pair, 최대 488 pair)는 같은 긴 text가 여러 chunk에 중복 노출. Effective training distribution이 pair-dense doc 쪽으로 편향. 특히 큰 origin이 quota sampling에서 살아남으면 influence 과대.

### 변경 내역
- [data_preprocessor_utils_simplified_add_gemma4.py:228-305](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py#L228) — `RelationBasedChunker._narrow_text_to_chunk(text, chunk_cands, padding=200)` 추가:
  1. chunk의 모든 cand에서 entity tag 식별자 추출 (`[TAG_NAME_NN]` 형태 regex)
  2. text에서 해당 태그들의 open/close 위치를 찾아 최소 범위(min_start, max_end) 계산
  3. 양쪽 padding=200자 붙여 substring 반환
  4. 태그를 못 찾으면 fallback(원문 그대로 반환 — safety)
  5. 축소 효과가 30% 미만이면 원문 그대로 (불필요한 잘게 자르기 방지)
- [qwen_inference/data_preprocessor_utils.py:228-284](../qwen_inference/data_preprocessor_utils.py#L228) — 동일 변경

### 트레이드오프
- Text를 축소하므로 chunk의 pair들이 참조하지 않는 원문 범위는 잘림. 모델이 "문서 전체 맥락"을 볼 기회는 약간 감소.
- 하지만 chunk의 candidate pair는 그 chunk 텍스트 범위 안에서만 관계를 판별해야 하므로, 범위 밖 문맥은 학습 신호로 가치 낮음.
- 30% 미만 축소는 skip → 대부분의 dense chunk는 여전히 원문 유지, sparse tail chunk에서만 축소 발생.

### 검증 결과
합성 테스트: 60 pair + 1 pair 구성에서 tail chunk(1 pair만 참조)가 3821자 → 262자로 축소, 첫 3 chunk(20 pair 다양한 참조)는 원문 유지.

---

## M4 — Reasoning mode dead path 정리

### 배경
- [sft_data_generation_prompts_edited.yaml:232, :237](../preprocessing/sft_data_generation_prompts_edited.yaml#L232) — `output_format_reasoning`, `output_format_w_keys_ner_bee_reasoning` 키 존재
- [data_preprocessor_utils_simplified_add_gemma4.py:376-381](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py#L376) (수정 전) — compiler는 `reasoning` 플래그 무시하고 항상 비-reasoning 키 선택
- config `reasoning: 0.0` → 현재 비활성이지만 dead path 존재

### 변경 내역
- [data_preprocessor_utils_simplified_add_gemma4.py:384-395](../preprocessing/data_preprocessor_utils_simplified_add_gemma4.py#L384) — `_precompute_static_components`에 task × reasoning 분기 추가:
  ```python
  is_bee = self.task in [NER_BEE, NER_BEE_TRUE_ONLY]
  use_reasoning = bool(getattr(self.mode_config, "reasoning", False))
  if is_bee:
      fmt_key = "output_format_w_keys_ner_bee_reasoning" if use_reasoning else "output_format_w_keys_ner_bee"
  else:
      fmt_key = "output_format_reasoning" if use_reasoning else "output_format_w_keys"
  ```
- [qwen_inference/data_preprocessor_utils.py](../qwen_inference/data_preprocessor_utils.py) — 동일 변경
- [sft_data_generation_prompts_edited.yaml:239, :249](../preprocessing/sft_data_generation_prompts_edited.yaml#L239) — NER-BEE reasoning/non-reasoning 양쪽 output_format에서 `"False"` → `"false"` 대소문자 일관화, "lowercase string true or false" 명시

### 트레이드오프
- reasoning ratio를 활성화(>0)하면 해당 비율만큼 rows가 think-tag 유도 프롬프트로 학습됨. Gemma4는 chat_template의 `enable_thinking` 인자로 think 섹션을 자동 처리하므로 현재 코드와 호환.
- Qwen은 `<think>` 태그가 학습 데이터에 포함되면 모델이 이를 생성하도록 학습됨 → inference 시 `strip_think_tags`(H6)가 제거.

### 검증 결과
- `DataGenerationModeConfig(reasoning=True)`로 생성한 compiler의 `static_output_format`에 `"Think step by step"` 포함 확인.

---

## M5 — Train/Val/Test split task별 stratified 부재

### 배경
- [generate_sft_training_data_latest.py:287-292](../preprocessing/generate_sft_training_data_latest.py#L287) (수정 전) — 모든 task의 고유 origin을 하나의 pool로 모아 random shuffle 후 70/10/20 split

### 문제
Task별 origin 수가 불균형(예: NER_NER 10,000 vs NER_BEE 2,000)이면 global random split은 smaller task가 val에 거의 없거나 test에 과도 집중되는 variance. 특히 소수 task는 val 지표 noise ↑.

### 변경 내역
- [generate_sft_training_data_latest.py:390-421](../preprocessing/generate_sft_training_data_latest.py#L390) — task별로 독립 split:
  1. task별 고유 origin 리스트
  2. 이미 다른 task에서 split 배정된 origin은 재사용 (cross-task leakage 방지)
  3. 새 origin만 shuffle 후 70/10/20 배정
  4. task별 split 통계 로그 출력

### 트레이드오프
- 한 document가 여러 task에 걸쳐 있을 때 첫 task의 split 배정이 우선 → task 간 split 비율이 정확히 70/10/20이 아닐 수 있음 (다른 task에서 reused된 origin은 기존 배정 유지). 하지만 leakage는 완전 차단.

### 검증 결과
- stratified 로그(`Task-stratified Train/Val/Test Split`) 문자열 존재.

---

## M7 — Qwen: validation 없어도 eval 강제

### 배경
- [training/sft_qwen.py:181-183](../training/sft_qwen.py#L181) (수정 전) — `eval_strategy="steps"`, `eval_steps=180`, `load_best_model_at_end=True` 무조건
- validation split이 없으면 trainer가 eval 시도하다 crash

### 변경 내역 (이미 C3와 함께 처리됨)
- [training/sft_qwen.py:181-189](../training/sft_qwen.py#L181) — 전부 `eval_dataset is not None` 조건부:
  ```python
  eval_strategy = "steps" if eval_dataset is not None else "no",
  eval_steps = 500,
  load_best_model_at_end = True if eval_dataset is not None else False,
  metric_for_best_model = "eval_loss" if eval_dataset is not None else None,
  ```

### 검증 결과
Gemma4와 동일 패턴 확인. validation 미존재 상황도 안전.

---

## 최종 통합 검증

모든 수정 후 다음을 통과:

1. **Syntax**: 12개 touched files 전부 `ast.parse` 혹은 `yaml.safe_load` OK
2. **C1 실증**: 500 docs 재처리 → 23,297 NER_BEE 샘플 `is_relational` 전부 {`"true"`, `"false"`}, illegal 0건. NER_NER은 43개 multi-class 유지.
3. **C2 실증**: pool restriction + self-exclusion 양쪽 동작 확인.
4. **M2 실증**: priority origins 정확히 식별(합성 테스트).
5. **M3 실증**: tail chunk 3821자 → 262자 축소, 참조 태그 보존.
6. **H2 실증**: 동일 compiler에서 fs_text 유무에 따라 overhead 값 8580 → 10645로 증가.
7. **H3/H4/M1 실증**: 두 학습 스크립트에 `warmup_ratio`, `save_steps=500`, `eval_steps=500`, `lora_dropout=0.05` 포함 확인.
8. **Gemma4 load_best_model_at_end**: `True if eval_dataset else False` 조건부 확인.
9. **Qwen M7 + C3**: `eval_strategy` 조건부 + `train_on_responses_only` 호출 확인.
10. **H1**: `tasks: ['NER_NER', 'NER_BEE']` 확인, `active_task_values` 로직 존재.
11. **M4**: `PromptCompiler`가 `mode_config.reasoning=True`일 때 `Think step by step`을 포함한 output_format 선택 확인.
12. **M5**: stratified split 코드 존재.
13. **H5**: eval config에 inference/merge 섹션 확인, merge에 hard fail + hit rate 로그 존재.

---

## 영향 및 재학습 절차 권장

### 전처리 단계 재실행 필요
C1/H2/H7/M2/M3/M5 모두 전처리 산출물(`sllm_ready_generated_prompts_*_hf_dataset`)에 영향. 기존 HF dataset 캐시 삭제 후 전처리 재실행 필요:
```bash
rm -rf /app/pred_data/sllm_ready_generated_prompts_*_hf_dataset
python preprocessing/generate_sft_training_data_latest.py
```

### 학습 단계
C3/H3/H4/M1/M7 반영된 상태로 재학습. 첫 step의 loss 스케일이 Gemma4와 유사한 레인지인지(C3/M1 조합 영향) 확인 권장.

### Eval 단계
H5/H6 적용된 config 기반 파이프라인:
```bash
EVAL_CONFIG=qwen_inference/generate_eval_data_config.yaml \
  python qwen_inference/generate_eval_data.py
EVAL_CONFIG=qwen_inference/generate_eval_data_config.yaml \
  python qwen_inference/vllm_model_inference_w_edited_prompts.py
EVAL_CONFIG=qwen_inference/generate_eval_data_config.yaml \
  python qwen_inference/merge_w_metadata.py
```
merge 실행 시 첫 100 rows의 metadata lookup hit rate 로그 확인 → 95% 미만이면 경로/jsonl 불일치 경고.

---

## 남은 과제 (이번 수정 범위 밖)

1. **두 학습 스크립트 공통 유틸화**(M6, 감사에서 식별) — `training/utils.py` + `training/sft_base.py`로 `get_auto_dtype`/`get_target_modules`/공통 `SFTConfig` 추출. 기능 변경 없는 리팩토링이라 이번 range 제외.
2. **Frequency 기반 dynamic rare 판정** — 현재는 static 40개 relation set. 향후 corpus 통계 기반 bottom-N% rare로 개선 가능.
3. **NER-BEE를 attribute classification으로 승격 여부** — Option B(C1 검토 옵션). 제품 요구사항 확정 시 별도 결정으로 진행.
4. **Few-shot rare coverage 측정** — 현재 prioritize_rare가 작동하지만 실제 rare class 커버리지(샘플당 몇 개 rare label 노출)는 측정 안 됨. dashboard 추가 가능.
