# Feature 정의 확정본

계획서 4장의 19개 feature 를 `events.jsonl` 에서 계산하는 규칙이다.
**3대 컴퓨터가 반드시 같은 규칙을 써야 결과를 비교할 수 있다.** 구현은 `extract_features.py`.

## 이벤트 어휘

`merged-7500` 데이터셋의 `event_name` 은 아래 11종이다.

`page_view`, `click`, `dwell_time`, `search`, `filter_change`,
`add_to_cart`, `remove_from_cart`, `checkout_start`, `payment_attempt`,
`checkout_complete`, `error`

## 19개 feature

| id | 이름 | 계산 규칙 |
|---|---|---|
| f1 | `session_duration_ms` | `max(ts) - min(ts)` |
| f2 | `event_count` | 세션의 전체 이벤트 수 |
| f3 | `page_view_count` | `page_view` 이벤트 수 |
| f4 | `click_count` | `click` 이벤트 수 |
| f5 | `depth` | `page_view` 경로 시퀀스 길이 (= f3) |
| f6 | `unique_page_ratio` | `unique(path) / depth` |
| f7 | `revisit_rate` | `(depth - unique(path)) / depth` |
| f8 | `backtrack_count` | URL 계층이 얕아지는 전이 수 |
| f9 | `loop_rate` | `path[i] == path[i-2]` 인 i 의 수 / `(depth - 2)` |
| f10 | `search_count` | `search` 이벤트 수 |
| f11 | `filter_count` | `filter_change` 이벤트 수 |
| f12 | `product_detail_count` | `path` 가 `/product/` 로 시작하는 `page_view` 수 |
| f13 | `review_view_count` | `path` 가 `/review/` 로 시작하는 `page_view` 수 |
| f14 | `cart_add_count` | `add_to_cart` 이벤트 수 |
| f15 | `cart_remove_count` | `remove_from_cart` 이벤트 수 |
| f16 | `checkout_entered` | `checkout_start` 가 1건이라도 있으면 1 |
| f17 | `payment_attempt_count` | `payment_attempt` 이벤트 수 |
| f18 | `purchase_completed` | `checkout_complete` 가 1건이라도 있으면 1 |
| f19 | `error_count` | `error` 이벤트 수 |

## 계획서에 명시되지 않아 여기서 확정한 항목

계획서에 코드 수준 정의가 없어 임의 해석이 가능했던 부분이다. 팀에서 다르게 정하면 이 문서와
`extract_features.py` 를 함께 고쳐야 한다.

### f8 `backtrack_count`

계획서 표현은 "이전 페이지나 목록으로 되돌아간 횟수"다.
가장 흔한 해석인 `A -> B -> A` 왕복은 f9 `loop_rate` 와 완전히 겹치므로 쓰지 않았다.
대신 **URL 계층이 얕아지는 전이**로 정의했다.

- `/product/grocery_4102` -> `/category/grocery` : backtrack (2단계 -> 2단계가 아니라 2 -> 2)
- `/product/grocery_4102` -> `/` : backtrack (2 -> 0)
- `/category/grocery` -> `/product/grocery_4102` : backtrack 아님 (2 -> 2)

계층 깊이는 `/` 로 분리한 비어있지 않은 세그먼트 수다. `/` = 0, `/cart` = 1, `/product/x` = 2.

### f7 `revisit_rate` 와 f6 `unique_page_ratio`

`f7 = 1 - f6` 로 완전 종속이다. 계획서가 둘 다 요구하므로 그대로 두되,
F2 / F6 / F11 에서 사실상 중복 차원이라는 점을 결과 해석 시 감안한다.

### f12 `product_detail_count`, f13 `review_view_count`

전용 이벤트가 없어 `page_view` 의 `path` prefix 로 파생한다.
`merged-7500` 기준 `/product/` 15,018건(7,498세션), `/review/` 6,933건(3,246세션)으로 신호는 충분하다.

**이 규칙은 F0, F3, F6, F7, F11, F13, F15 에 모두 영향을 준다. 즉 3대 컴퓨터 전부 해당된다.**

### f5 `depth` 의 전처리 분류

계획서 6.1 count 목록에 `depth` 가 빠져 있으나 count 계열이므로 `log1p` + scaling 을 적용했다.
