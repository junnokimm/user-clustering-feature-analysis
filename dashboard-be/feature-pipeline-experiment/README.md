# Feature-Pipeline 실험

[FEATURE_PIPELINE_EXPERIMENT_PLAN.md](../benchmark/FEATURE_PIPELINE_EXPERIMENT_PLAN.md) 를
실행 가능한 코드로 구현한 것이다. `feature subset x pipeline x seed` 를 돌려 persona 복원 성능을 비교한다.

## 실행

```bash
pip install -r requirements.txt

# Computer 2 담당: F4, F6, F7 x A1,A2,A3 x seed 42,43,44 = 27 runs
python run_all.py
```

결과는 [REPORT.md](REPORT.md) 에 표로 정리된다. 소요 시간은 약 20분이며 대부분 A2(UMAP)에서 쓴다.

다른 컴퓨터는 subset 만 바꾼다.

```bash
python run_all.py --subsets F0,F2,F3     # Computer 1
python run_all.py --subsets F11,F13,F15  # Computer 3
```

feature 추출과 split 은 subset 과 무관하므로 재실행 시 건너뛸 수 있다.

```bash
python run_all.py --skip-extract
```

## 파일

| 파일 | 역할 |
|---|---|
| `extract_features.py` | `events.jsonl` + `sessions.json` -> `artifacts/features.csv` (7,500행 x 19 feature) |
| `make_split.py` | stratified train 70 / val 15 / test 15 -> `artifacts/split.json` |
| `run_experiments.py` | 전처리 -> A1/A2/A3 -> 평가 -> `artifacts/results.csv` |
| `report.py` | seed 집계 -> `REPORT.md` |
| `run_all.py` | 위 4개를 순서대로 실행 |
| `FEATURE_DEFINITIONS.md` | 19개 feature 계산 규칙 확정본. **3대가 동일해야 함** |

## 데이터

`benchmark/output/merged-7500` — 7,500 세션 / 89,616 이벤트, persona 5종 각 1,500 세션.
`--dataset` 으로 바꿀 수 있다.

## 라벨 사용 원칙

정답 라벨(`persona_id`, `ground_truth_label`)은 `features.csv` 에 함께 저장하지만
**학습 경로 어디에도 넣지 않는다.** `run_experiments.py` 의 `evaluate()` 에서만 쓴다.

여기에는 하이퍼파라미터 선택도 포함된다.

- **군집 수 `k` 를 포함한 모든 하이퍼파라미터는 val split 의 Silhouette 으로 고른다.**
  persona 가 5개라는 사실은 어떤 기본값에도 반영되어 있지 않다. `K_GRID` 는 `2..10` 이다.
- val 선택에 ARI/NMI 같은 외부 지표를 쓰면 라벨로 모델을 고르는 것이므로 쓰지 않는다.
- 라벨 없이 적용하는 제약은 두 개뿐이다: 군집 수 `2 <= k <= 20`, 노이즈 비율 `<= 50%`.
  세그먼트로 쓸 수 없는 해를 후보에서 빼기 위한 것이고 persona 수와 무관하다.

## 탐색 공간과 고정값

| 항목 | 값 | 위치 |
|---|---|---|
| feature 계산 규칙 (f7, f8, f12, f13) | `FEATURE_DEFINITIONS.md` 참고 | `extract_features.py` |
| split seed | `20260803` 고정, `persona_id x difficulty` stratified | `make_split.py` |
| scaling | `StandardScaler`, **train 통계로만 fit** | `run_experiments.py:preprocess` |
| K (A1, A3) | **val 탐색** `2..10` | `K_GRID` |
| UMAP `n_neighbors` | **val 탐색** `15, 50` | `UMAP_NEIGHBORS_GRID` |
| UMAP 기타 | `n_components=min(5,d)`, `min_dist=0.0` 고정 | `pipeline_a2` |
| HDBSCAN `min_cluster_size` | **val 탐색** `50, 100, 250, 500` | `HDBSCAN_MIN_CLUSTER_GRID` |
| HDBSCAN `min_samples` | **val 탐색** `5, 25` | `HDBSCAN_MIN_SAMPLES_GRID` |
| VAE latent 차원 | **val 탐색** `2, 4, 8` (feature 수 이하) | `VAE_LATENT_GRID` |
| VAE 구조 | `d -> 64 -> 32 -> latent`, 50 epoch, Adam 1e-3 고정 | `_train_vae` |
| 학습/평가 | train 에서 fit, val 에서 선택, **test 에서만 지표 보고** | `run_experiments.py:main` |

선택된 값은 `results.csv` 의 `selected_*` 컬럼에 run 단위로 기록된다.

## 알려진 제약

1. **A2 의 UMAP `transform()` 은 중복 벡터가 많은 subset 에서 NaN 을 낸다.** 해당 행은 가장 가까운
   train 점의 embedding 을 물려받도록 보정했고, 보정 비율은 `results.csv` 의
   `umap_fallback_ratio` 로 기록된다. F4 처럼 고유 벡터가 21개뿐인 subset 에서 발생한다.
2. `f7 = 1 - f6` 로 두 feature 는 완전 종속이다. 계획서가 둘 다 요구해 그대로 두었다.
3. **A1 은 seed 를 바꿔도 결과가 같다.** `n_init=10` 이라 K-Means 가 같은 해로 수렴한다.
   계획서 8장의 seed 안정성 비교가 A1 에서는 정보를 주지 않는다는 뜻이다.
4. Silhouette 은 군집 수가 적을수록 유리한 편향이 있다. val 선택 기준을 바꾸면
   (예: Davies-Bouldin) 결과가 달라질 수 있다.
