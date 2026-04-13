# 2026-04-13 — PR 2.3: 모드별 sample partition 결정적 분리

## 배경

[generate_sft_training_data_latest.py:84-87](../preprocessing/generate_sft_training_data_latest.py#L84) 모드 루프에서:

```python
sampled = ds.shuffle(seed=seed).select(range(sampled_count))
if "full" in mode_name:
    split_idx = int(len(sampled) * full_detail_ratio)
    sampled = sampled.select(range(split_idx)) if "detailed" in mode_name else sampled.select(range(split_idx, len(sampled)))
```

**문제**: 4개 모드(`system_only`, `full_detailed`, `full_summary`, `no_fewshot`)가 모두 같은 `seed=42`로 shuffle 후 첫 N개 선택. 같은 sample이 여러 모드에 중복 등장.

`mode_ratios = {system_only: 0.2, full: 0.6, no_fewshot: 0.2}` 기준 시뮬레이션 (total=100):
- system_only [0, 20]
- full_detailed [0, 42]   ← system_only [0, 20]과 OVERLAP
- full_summary [42, 60]   ← OK
- no_fewshot [0, 20]      ← system_only, full_detailed와 OVERLAP

→ 인덱스 0~19의 sample은 **3개 모드(system_only + full_detailed + no_fewshot)에 동시 등장** = 학습 분포 편향

## 결정

**Task별 1회 셔플 + 모드별 disjoint 인덱스 범위 사전 계산.** 같은 sample이 여러 모드에 등장하지 않도록 보장.

```python
shuffled = ds.shuffle(seed=seed)
total = len(shuffled)
mode_ranges = {}
cursor = 0
full_registered = False
for m_name in base_modes:
    if "full" in m_name:
        if not full_registered:
            full_n = int(total * mode_ratios.get("full", 0))
            full_split = int(full_n * full_detail_ratio)
            mode_ranges["full_detailed"] = (cursor, cursor + full_split)
            mode_ranges["full_summary"] = (cursor + full_split, cursor + full_n)
            cursor += full_n
            full_registered = True
    else:
        n = int(total * mode_ratios.get(m_name, 0))
        mode_ranges[m_name] = (cursor, cursor + n)
        cursor += n

for mode_name, mode_cfg in base_modes.items():
    s, e = mode_ranges.get(mode_name, (0, 0))
    if e <= s: continue
    sampled = shuffled.select(range(s, e))
    # ... 이하 기존 process_batch
```

## 검토한 선택지

- **(A) 단일 셔플 + disjoint range (선택됨)** — 결정적, 단순, mode_ratios 합 ≤ 1.0 보장 시 100% 데이터 활용
- **(B) 모드별 다른 seed 셔플 + select(range(N))** — 무작위 disjoint 보장 안 됨 (확률적으로 overlap)
- **(C) random.sample로 disjoint 선택** — pure-Python 루프, num_proc 분기와 충돌 가능

## 선택 이유

- 결정적(reproducible). seed가 같으면 항상 같은 partition
- 단일 .shuffle(seed) 호출이라 효율적
- mode_ratios 합 = 0.2+0.6+0.2 = 1.0 → 100% 데이터 사용
- reasoning 모드(ratio=0.0)는 size=0으로 자동 스킵, 별도 분기 불필요

## 트레이드오프

- 모드별 sample 양은 mode_ratios에 정확히 비례 (이전엔 overlap으로 system_only/no_fewshot이 사실상 동일 sample 학습)
- 학습 데이터 양 자체는 변화 없으나, **각 input이 하나의 모드에서만 등장 → 모델이 모드별 신호를 더 명확히 학습**
- mode_ratios 합 < 1.0이면 일부 sample 미사용 (현재는 1.0이라 영향 없음)

## 검증 결과

3가지 total 크기에서 disjoint 확인:

```
=== total=100 ===
  system_only    [   0,   20)  size=20
  full_detailed  [  20,   62)  size=42
  full_summary   [  62,   80)  size=18
  no_fewshot     [  80,  100)  size=20
  reasoning      [ 100,  100)  size=0
  → 사용 인덱스: 100/100, 모드별 disjoint ✅

=== total=1000 ===  → 200+420+180+200 = 1000 ✅
=== total=10000 === → 2000+4200+1800+2000 = 10000 ✅

mode_ratios 합: 0.2 + 0.6 + 0.2 + 0.0 = 1.0
```

또 stdout에 `Task X mode partition (total=N): system_only=[0,200), full_detailed=[200,620), ...` 출력해서 모니터링 용이하게 함.

## 영향

- 모드 간 sample overlap 0%
- 학습 데이터 양 변화 없음 (mode_ratios 합 1.0 그대로)
- 모드별 신호 분리 → 학습 효율성 ↑ 추정
- reasoning 모드는 ratio=0이라 자동 빈 partition (스킵 처리 자동)
