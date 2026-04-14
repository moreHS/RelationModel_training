# 2026-04-14 — raw_sources에서 bee_bee/combine_all KeyError

## 에러 내용

```
KeyError: 'bee_bee'
  File "generate_sft_training_data_latest.py", line 184
    "ner_ner": classified["ner_ner"], "bee_bee": classified["bee_bee"],
```

## 원인 분석

PR 1에서 `extract_and_classify` 반환값 변경 (`bee_bee` 키 제거) + `DataGenerationTask` enum에서 `BEE_BEE`/`COMBINE_ALL` 삭제. 그러나 같은 파일(`latest.py`)의 `get_or_generate_sft_dataset` 함수 안 `raw_sources` 딕셔너리가 여전히 `classified["bee_bee"]`와 `classified["combine_all"]`을 참조.

**서브에이전트 위임 시** utils.py만 수정 범위로 한정했고, latest.py의 `raw_sources`는 별도 명세에 포함되지 않아 누락.

## 해결 방법

`raw_sources` 딕셔너리에서 `bee_bee`와 `combine_all` 항목 제거:

```python
raw_sources = {
    "ner_ner": classified["ner_ner"],
    "ner_bee": classified["ner_bee"],
    "ner_bee_true_only": dp.remove_negative_relations(classified["ner_bee"]),
}
```

## 재발 방지 포인트

- **Task/enum 삭제 시** 해당 값을 참조하는 모든 소비자(consumer)를 grep으로 전수 확인할 것
- 서브에이전트 위임 시 "변경 파일 목록"뿐 아니라 "영향받는 다른 파일의 참조 라인"도 명세에 포함할 것
- `grep -rn 'bee_bee\|combine_all' preprocessing/` 같은 사전 체크를 commit 전에 실행할 것
