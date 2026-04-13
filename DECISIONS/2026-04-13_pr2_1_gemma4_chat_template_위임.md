# 2026-04-13 — PR 2.1: Gemma4 chat_template `apply_chat_template` 위임

## 배경

PR 1 직후 토크나이저 검증에서 `_apply_chat_template`의 Gemma4 분기 수동 빌더 4가지 버그 확인:

1. `<bos>` 누락 — 공식 [chat_template.jinja](../models/chat_template.jinja) 라인 155 `{{ bos_token }}` 자동 prepend 안 됨
2. channel 토큰 오타 2건 — 코드 regex가 `<|channel|>`/`<channel>` 사용, 실제 vocab은 `<|channel>` (id=100) / `<channel|>` (id=101) → think-strip이 작동 안 함
3. `<|think|>` 위치 오류 — 공식은 첫 system 턴 최상단(라인 161-164), 코드는 답변 본문 안에 wrap 시도
4. 표준 분기의 BOS 무조건 strip — Unsloth가 raw text를 학습할 때 BOS 누락 유발

토크나이저 직접 검증으로 우리 코드의 `<|turn>` (105) / `<turn|>` (106) 태그는 정확함 확인. 차이는 BOS와 think 처리뿐.

## 결정

**`PromptCompiler._apply_chat_template`을 단일 경로로 통합. `tokenizer.apply_chat_template`에 위임.**

- Gemma4 분기 수동 빌더 25줄 삭제
- 표준 분기의 BOS strip 삭제
- Gemma4면 `enable_thinking=mode_config.reasoning` 인자 추가 (공식 jinja가 think 토큰 자동 처리)
- 비-reasoning 모드의 잔여 legacy `<think>...</think>` 제거 안전망만 유지

## 검토한 선택지

- **(A) 위임 (선택됨)** — 토크나이저 chat_template.jinja에 모든 책임 일임. 4건 버그 자동 해소.
- **(B) 수동 빌더 픽스** — channel 토큰 오타만 고치고 BOS prepend 추가. 향후 chat_template 변경 시 동기화 부담 지속.

## 선택 이유

- 토크나이저가 단일 source of truth. 향후 google이 chat_template 변경해도 자동 반영됨
- 공식 `strip_thinking` 매크로가 `<|channel>...<channel|>` 영역 정확히 처리 → 우리 regex 흉내 불필요
- 표준 모델(Qwen 등)도 동일 경로 → 코드 분기 단순화
- BOS는 chat_template이 자동 prepend, 손대면 안 됨이 확인됨

## 트레이드오프

- transformers 4.57+ (Gemma4 지원 버전) 필요 — 사용자 docker 환경은 이미 최신, 로컬은 격리 venv (`/tmp/gemma4_test_venv`)에서 검증
- chat_template 인자(`enable_thinking`)에 의존 — 다른 모델로 바꿀 때 인자 호환성 확인 필요. 단 현재는 Gemma4 한정 분기

## 검증 결과 (격리 venv에서)

```
=== NER_NER, reasoning=False ===
  <bos> prepend:       ✅
  <|turn>/<turn|>:     3/3회 정확히
  <think> 잔존:        ✅ 없음
  text 1417 chars / 364 tokens

=== NER_BEE, reasoning=True ===
  <bos> prepend:       ✅
  <|think|> 자동 삽입:  ✅ (system 턴 최상단)
  text 1464 chars

=== 토큰 ID 정합 ===
  bos_id=2, <|turn>=105, <turn|>=106 (single id mapping)
```

## 영향

- 학습 데이터 시작 토큰이 이제 `<bos>` 포함 → Unsloth가 정상 인식
- reasoning=True 모드에서 `<|think|>` 정확한 위치 (system 턴 최상단)
- 메서드 라인 56 → 34줄로 단순화
- Gemma4 / 표준 모델 단일 경로
