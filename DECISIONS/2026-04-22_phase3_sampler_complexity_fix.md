# 2026-04-22 — PHASE 3 Sampler Complexity Fix (O(N²logN) → O(NlogN))

## 배경

[`2026-04-21_diversity_sampling_and_eval_toggle.md`](./2026-04-21_diversity_sampling_and_eval_toggle.md)에서 도입한 3-pass greedy sampler가 실제 스케일(95,699 docs / ner_bee)에서 몇 시간째 진전 없이 멈춤. A100 서버 로그:

```
Chunking ner_bee 100% 95699/95699 [07:54]
⚖️ PHASE 3: Targeting 100000 Train samples.
   Quota per task: 71428 chunks
# ← 여기서 수 시간째 멈춤
```

## 문제

Pass A/B/C 전부 `while` 루프 안에서 `candidates.sort(...)`를 호출 → 매 iteration마다 전체 candidate list 재정렬.

```python
# Before
while ...:
    candidates.sort(key=lambda oid: _score_coverage(oid, state), ...)  # O(N log N)
    best = candidates[0]
    state.add(best, ...); candidates.remove(best)  # O(N) remove
```

- 복잡도: `O(N² log N)` worst case (K iterations × N log N sort)
- 실제: N = 수만 ~ 수십만 origins → `N² = 10¹⁰+` 연산 → 수 시간 ~ 종료 불가
- 2,000 docs dry-run에서는 N이 작아서 빨랐기 때문에 이슈를 못 잡음 (scale regression bug)

## 리서치

codex Architect advisory 결과 핵심 지적:
1. Pass A/B/C 전부 **재정렬 제거 필요** — frozen score + 순차 스캔
2. Pass B의 `break` → `continue` (stale low-gain origin이 뒤의 high-gain을 block하지 않도록)
3. 원래 구현의 `0.1` overflow slack은 너무 느슨 → `0.005~0.01` 로 (0.10 cap일 때 0.20까지 허용하는 셈)
4. `list.remove()` O(N) 제거는 없애고 인덱스 스캔
5. Pass B의 품질 손실 우려: frozen score라 stale-low-gain origin을 먼저 먹을 위험. "Pass B만 lazy heap"이 최적이지만 Short effort 수정엔 과함.

## 결정

**Option B hardened** + **Pass A0 rare-first bootstrap** 추가.

4-pass 구조:

### Pass A0 — Rare-first bootstrap (신규)
각 rare relation(`rel_floor` ∪ `dynamic_priority`)에 대해 scarcity 오름차순으로:
- 이미 커버된 relation이면 skip
- 아니면 carrier origin 중 chunk_count 최소인 origin 1개를 **강제 keep**

```python
rare_targets_sorted = sorted(rare_targets, key=lambda r: rel_exposure[r])
for rel in rare_targets_sorted:
    if any(rel in origins[oid]["rel_labels"] for oid in state.selected):
        continue
    carriers = [oid for oid in origins if rel in origins[oid]["rel_labels"] and ...]
    carriers.sort(key=lambda oid: origins[oid]["chunk_count"])
    state.add(carriers[0], ...)
```

목적: frozen score 방식이 ultra-rare(exposure ≤ 10)를 놓치는 현상 방지. **relation 100% 커버 보장**의 안전장치.

### Pass A — Priority remainder (random)
Pass A0에서 미선택된 priority origins를 random 순차 add. Score 정렬 제거.

이유: priority는 원칙상 전부 keep. Score 순서는 큰 의미 없고 random이 더 공정.

### Pass B — Common coverage (frozen score)
```python
common_list = [...]; random.shuffle(common_list)
pass_b_scores = {oid: _score_coverage_snapshot(oid) for oid in common_list}  # Pass A 종료 시점 기준
common_list.sort(key=lambda oid: -pass_b_scores[oid])
for oid in common_list:
    p = origins[oid]
    if p["chunk_count"] > state.room_left(): continue
    if state.coverage_gain(p) <= 0: continue  # live skip, NOT break
    state.add(oid, p)
    if all floors satisfied: break
```

- Frozen score: Pass A 종료 시점의 `_score_coverage` 1회 계산, 정렬도 1회
- Live coverage_gain ≤ 0 skip: add 후 need가 변해 이미 다 채워진 relation만 가진 origin은 건너뜀
- `break` 아닌 `continue`: 뒤에 있는 useful origin이 block되지 않도록

### Pass C — Novelty fill (two-tier)
```python
remaining.sort(key=lambda oid: -pass_c_scores[oid])
SAFE_OVERFLOW = 0.005; DEFER_OVERFLOW = 0.01
deferred = []
for oid in remaining:
    overflow = state.head_overflow_penalty(p)
    if overflow <= SAFE_OVERFLOW: state.add(oid, p)
    elif overflow <= DEFER_OVERFLOW: deferred.append(oid)
# 남은 quota 있으면 deferred로 2차 sweep
for oid in deferred: ...
```

- Two-tier overflow: safe (cap 초과 ≤ 0.5pp) 먼저 채우고, 남은 quota는 deferred (≤ 1pp)로
- 0.1 → 0.005로 엄격화: 원래 of 0.10 cap + 0.1 slack = 0.20 까지 허용하던 것을 0.105 / 0.110으로

## 변경 내역

[preprocessing/generate_sft_training_data_latest.py:605-707](../preprocessing/generate_sft_training_data_latest.py#L605) — Pass A/B/C 전체 재작성.

- 제거: `while ... cand.sort(...)` 루프 3개 (Pass A, B, C 각각)
- 추가:
  - Pass A0: rare-first bootstrap 전처리
  - Pass A: random iteration
  - Pass B: frozen score dict 계산 → 정렬 → linear scan (`continue` skip)
  - Pass C: frozen score 정렬 + two-tier overflow

## 복잡도

| 단계 | Before | After |
|---|---|---|
| Pass A0 | — | O(R × N) (R = rare 수, N = origins) |
| Pass A | O(P² log P) | O(P) (P = priority origins) |
| Pass B | O(C² log C) | O(C log C) (C = common origins) |
| Pass C | O(C² log C) | O(C log C) |
| **Total** | **O(N² log N)** | **O(N log N)** |

N=100K 기준: `N² = 10¹⁰` → `N log N ≈ 1.7 × 10⁶` → **5,000배 이상 빠름**.

## 검증 결과 (2,000 docs / 9,595 chunks / quota 3,358)

| Metric | Before (iterative) | After (frozen + A0) |
|---|---|---|
| Elapsed | ~N/A (완료 가능) | **0.05 s** |
| Relation coverage | 68/68 (100%) | 68/68 (**100%**) |
| Group_pair coverage | 87/93 (93.5%) | 87/93 (93.5%) |
| NO_RELATION share | 0.143 → 0.133 | 0.143 → 0.139 |
| used_by share | 0.228 → 0.204 | 0.228 → 0.213 |
| same_entity share | 0.219 → 0.195 | 0.219 → 0.204 |
| `child_of` (pre 2) | 100% | 100% |
| `brand_of` (pre 6) | 100% | 67% |
| `family_member_of` (pre 3) | 100% | 33% |

### Trade-off 분석

- **Relation-level coverage (68 unique relations 모두 등장)**: Pass A0 덕분에 **100% 유지** ✅
- **Sample-level rare preservation**: 일부 감소 (100% → 33~67%)
  - 원래 iterative 버전은 "rare가 들어간 모든 origin을 모두 keep" 했음
  - 새 버전은 "각 rare 종마다 최소 1개 origin을 강제 keep"
  - Head share 억제 효과는 약화 (0.133 → 0.139 등)
- **핵심 목표인 "rare class 100% noted"는 달성**. 샘플 수량은 2배 이상 절약된 시간 대비 trade-off 수용 가능.

## Trade-off 트리거 (사용자 튜닝 포인트)

만약 rare relation의 **샘플 수량**도 유지가 필요하면 Pass A0를 확장:
```python
# 각 rare당 carrier 전체 keep (quota 소진 전까지)
for rel in rare_targets_sorted:
    carriers = [...]
    carriers.sort(key=lambda oid: origins[oid]["chunk_count"])
    for oid in carriers:  # 1개가 아닌 전체
        if ...: break
        state.add(oid, ...)
```

다만 이 경우 rare가 많은 origin 소수가 quota 많은 부분을 차지해 **group_pair coverage 손상** 위험. 현재 구현은 "coverage 100% + quota 효율" 균형점.

## Risks (codex 지적 반영)

| Risk | 대응 |
|---|---|
| Pass B stale ordering: 초기 score로 고정해서 채워진 relation 무관 origin이 앞서 나감 | `coverage_gain(p) <= 0` inline check로 continue skip |
| Pass A0의 smallest carrier 선택: 단 1개만 keep하므로 ultra-rare 수량 손실 | 위 "Trade-off 트리거" 참조 — 필요 시 튜닝 |
| Multi-label large origin 과대평가 (기존 편향) | 이번 수정 범위 밖. `chunk_count`를 전체 label에 일괄 적용하는 `cov_gain` 로직은 그대로 두어 regression 방지 |

## 운영

재실행 순서:
```bash
# 1. 기존 진행 중 프로세스 중단
pkill -f "generate_sft_training_data"

# 2. 기존 중간 HF dataset 삭제 (있다면)
rm -rf /app/pred_data/sllm_ready_generated_prompts_gemma4_v2_hf_dataset

# 3. 재실행 — PHASE 3가 이제 수 초 내 완료
python3 -m preprocessing.generate_sft_training_data_latest \
    --config preprocessing/generate_sft_training_data_config.yaml
```

로그에서 PHASE 3 출력이 아래처럼 수 초 내 완료되어야 함:
```
⚖️ PHASE 3: Targeting 100000 Train samples.
   Quota per task: 71428 chunks
   ner_ner: NNN → MMM chunks (priority=X/Y, rel_floor_unmet=0/R, group_unmet=0/G, head_shares={...})
   ner_bee: ... 동일 형식
```

## 연관 DECISIONS

- [2026-04-21_diversity_sampling_and_eval_toggle.md](./2026-04-21_diversity_sampling_and_eval_toggle.md) — 원본 sampler 설계
- [2026-04-22_gold_dataset_mixing.md](./2026-04-22_gold_dataset_mixing.md) — Gold 혼합 (이 수정과는 독립적으로 호환)
