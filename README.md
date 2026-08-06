# Feature-Pipeline 본실험 실행 및 분석 가이드

## 1. 실험 목적

본 실험의 목적은 **이커머스 사용자 유형(persona)을 가장 잘 복원하는 feature 조합과 분석 파이프라인을 찾는 것**이다.

LLM 기반 합성 행동 세션에는 각 세션의 정답 persona가 존재한다. 다만 이 정답 라벨은 군집화 모델의 입력으로 사용하지 않고, 군집화가 끝난 뒤 결과를 평가할 때만 사용한다.

즉, 다음 질문에 답하기 위한 실험이다.

1. 어떤 사용자 행동 feature 조합이 persona 구분에 가장 유효한가?
2. 같은 feature를 사용하더라도 어떤 분석 파이프라인이 더 좋은가?
3. 특정 feature 그룹을 제거하면 성능이 얼마나 감소하는가?
4. 어떤 조합이 seed가 달라져도 안정적인가?

실제 비교 단위는 다음과 같다.

```text
feature subset × analysis pipeline × seed
```

---

## 2. 담당 실험 범위

Computer 3 담당 범위는 다음 3개 feature subset이다.

### F11: 탐색 경로 + 탐색·비교 + 퍼널

사용 feature:

- `depth`
- `unique_page_ratio`
- `revisit_rate`
- `backtrack_count`
- `loop_rate`
- `search_count`
- `filter_count`
- `product_detail_count`
- `review_view_count`
- `cart_add_count`
- `cart_remove_count`
- `checkout_entered`
- `payment_attempt_count`
- `purchase_completed`

확인하려는 내용:

> 행동 강도와 오류 정보 없이도 사용자의 탐색, 비교, 구매 흐름만으로 persona를 복원할 수 있는가?

### F13: 전체 feature - 퍼널

제외 feature:

- `cart_add_count`
- `cart_remove_count`
- `checkout_entered`
- `payment_attempt_count`
- `purchase_completed`

확인하려는 내용:

> 장바구니와 결제 같은 퍼널 정보를 제거했을 때 persona 복원 성능이 얼마나 감소하는가?

### F15: 전체 feature - 탐색 경로

제외 feature:

- `depth`
- `unique_page_ratio`
- `revisit_rate`
- `backtrack_count`
- `loop_rate`

확인하려는 내용:

> 페이지 이동 경로와 반복 탐색 정보가 persona 구분에 얼마나 기여하는가?

---

## 3. 분석 파이프라인

각 feature subset에는 동일하게 3개 파이프라인을 적용한다.

### A1: 전처리 feature + K-Means

```text
원본 feature
→ 공통 전처리
→ K-Means
```

가장 기본적인 baseline이다. 직접 계산한 행동 feature만으로 persona 구조가 형성되는지 확인한다.

### A2: UMAP + HDBSCAN

```text
원본 feature
→ 공통 전처리
→ UMAP embedding
→ HDBSCAN
```

비선형 저차원 표현과 밀도 기반 군집화가 persona 구조를 더 잘 포착하는지 확인한다.

HDBSCAN은 어느 군집에도 속하지 않는 데이터를 `-1` noise로 분류할 수 있다. 따라서 cluster 수와 noise ratio를 함께 확인해야 한다.

### A3: VAE latent + K-Means

```text
원본 feature
→ 공통 전처리
→ VAE latent representation
→ K-Means
```

직접 계산한 feature 공간보다 VAE가 학습한 잠재표현이 persona 복원에 유리한지 확인한다.

---

## 4. 전체 실행 수

담당 feature subset 3개, pipeline 3개, seed 3개를 조합한다.

```text
3 feature subsets
× 3 pipelines
× 3 seeds
= 총 27 runs
```

사용 seed:

- `7`
- `42`
- `2026`

실행 예:

```text
F11 × A1 × seed 7
F11 × A1 × seed 42
F11 × A1 × seed 2026
F11 × A2 × seed 7
...
F15 × A3 × seed 2026
```

---

## 5. 실행 전 준비

### 5.1 브랜치 확인

프로젝트 루트에서 실행한다.

```bash
git switch test
git status
```

실험 중에는 코드가 바뀌지 않도록 한다.

현재 기준 commit과 변경 상태를 기록하려면 다음을 실행한다.

```bash
git rev-parse HEAD
git diff --stat
```

### 5.2 실행 위치

실험 명령은 반드시 다음 폴더에서 실행한다.

```text
Dashboard/dashboard-be
```

프로젝트 루트 `Dashboard/`에서 실행하면 Python이 `experiment` 모듈을 찾지 못해 다음 오류가 발생할 수 있다.

```text
ModuleNotFoundError: No module named 'experiment'
```

이동:

```bash
cd dashboard-be
```

현재 위치 확인:

```bash
pwd
ls
```

`experiment`, `benchmark`, `requirements-experiment.txt`가 보여야 한다.

### 5.3 Python 가상환경 생성

가상환경이 없다면 생성한다.

프로젝트 루트에 `.venv`를 만드는 경우:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

`dashboard-be`로 이동한 뒤 루트의 가상환경을 사용할 때:

```bash
source ../.venv/bin/activate
```

터미널 앞에 `(.venv)`가 표시되면 활성화된 상태다.

확인:

```bash
which python
python --version
```

실험 의존성 설치:

```bash
python -m pip install -r requirements-experiment.txt
```

CLI 확인:

```bash
python -m experiment.cli --help
```

---

## 6. 본실험 실행 명령

`dashboard-be` 폴더에서 다음 명령을 실행한다.

```bash
mkdir -p experiment/output/computer-3

caffeinate -dimsu python -m experiment.cli \
  --data-dir benchmark/output/merged-7500 \
  --feature-subsets F11 F13 F15 \
  --pipelines A1 A2 A3 \
  --seeds 7 42 2026 \
  --split-seed 2026 \
  --output-dir experiment/output/computer-3 \
  2>&1 | tee experiment/output/computer-3/console.log
```

### 명령어 옵션 의미

| 옵션 | 의미 |
|---|---|
| `caffeinate -dimsu` | 실험 중 Mac이 잠자기 상태로 들어가는 것을 방지 |
| `python -m experiment.cli` | 구현된 실험 CLI 실행 |
| `--data-dir` | 7,500개 합성 세션 데이터 위치 |
| `--feature-subsets` | 실행할 feature 조합 |
| `--pipelines` | 실행할 분석 파이프라인 |
| `--seeds` | 반복 실험에 사용할 seed |
| `--split-seed` | train/validation/test 분할을 고정하는 seed |
| `--output-dir` | 결과 저장 위치 |
| `2>&1` | 표준 오류를 표준 출력에 합침 |
| `tee` | 터미널 출력과 로그 파일 저장을 동시에 수행 |

---

## 7. 명령 한 번으로 실행되는 내용

위 명령을 한 번 실행하면 F11, F13, F15에 대해 A1, A2, A3와 seed 3개가 모두 조합되어 총 27개 run이 실행된다.

실행 과정은 다음과 같다.

```text
merged-7500 데이터 로드
→ 원시 이벤트에서 19개 feature 추출
→ 고정된 train/validation/test split 생성 또는 재사용
→ feature subset 선택
→ train-only preprocessing
→ pipeline 학습 및 예측
→ 정답 persona와 군집 결과 비교
→ 평가 지표 계산
→ run별 결과 저장
→ seed 평균과 표준편차 집계
```

터미널에는 각 run의 시작, 완료, skip, 실패 상태가 출력된다.

같은 내용은 다음 로그 파일에도 저장된다.

```text
experiment/output/computer-3/console.log
```

---

## 8. 중단된 경우 재실행

실험이 중간에 종료되더라도 같은 명령을 다시 실행하면 된다.

완료된 run은 건너뛰고 미완료 또는 실패 run만 다시 실행하도록 구현되어 있다.

로그를 이어서 저장하려면 `tee -a`를 사용한다.

```bash
caffeinate -dimsu python -m experiment.cli \
  --data-dir benchmark/output/merged-7500 \
  --feature-subsets F11 F13 F15 \
  --pipelines A1 A2 A3 \
  --seeds 7 42 2026 \
  --split-seed 2026 \
  --output-dir experiment/output/computer-3 \
  2>&1 | tee -a experiment/output/computer-3/console.log
```

기존 완료 run까지 다시 실행할 수 있는 `--force` 옵션은 사용하지 않는 것이 좋다.

---

## 9. 생성되는 결과

결과 폴더는 대략 다음 구조를 가진다.

```text
experiment/output/computer-3/
├── console.log
├── extracted_features.csv
├── split_manifest.csv
├── experiment_config.json
├── environment.json
├── run_results.csv
├── summary.csv
└── runs/
    ├── F11_A1_seed7/
    ├── F11_A1_seed42/
    ├── F11_A1_seed2026/
    ├── F11_A2_seed7/
    ├── ...
    └── F15_A3_seed2026/
```

각 run 폴더에는 일반적으로 다음 파일이 저장된다.

```text
config.json
metrics.json
predictions.csv
contingency.csv
mapping.json
status.json
```

### 주요 공통 파일

#### `extracted_features.csv`

7,500개 세션에서 추출한 행동 feature가 저장된다.

정답 persona는 군집화 입력 feature에 포함되지 않아야 한다.

#### `split_manifest.csv`

각 세션이 train, validation, test 중 어디에 속하는지 저장한다.

모든 feature subset과 pipeline은 동일한 split을 사용한다.

#### `run_results.csv`

27개 개별 run의 성능을 한 행씩 저장한다.

#### `summary.csv`

동일한 `feature subset × pipeline` 조건의 seed 3회 결과를 평균과 표준편차로 요약한다.

최종 비교에서 가장 먼저 확인할 파일이다.

#### `predictions.csv`

각 test 세션의 정답 persona와 예측 cluster를 저장한다.

컬럼 순서는 다음과 같이 고정되어 있다.

```text
session_id,true_label,cluster
```

#### `contingency.csv`

정답 persona와 예측 cluster가 어떻게 대응되는지 보여주는 교차표다.

#### `mapping.json`

Hungarian matching을 통해 cluster ID를 어떤 persona와 대응시켰는지 저장한다.

#### `status.json`

각 run의 상태를 저장한다.

- `pending`
- `running`
- `completed`
- `failed`

---

## 10. 실험 완료 여부 확인

생성된 run 폴더 수 확인:

```bash
find experiment/output/computer-3/runs \
  -mindepth 1 -maxdepth 1 -type d | wc -l
```

정상 완료 시 예상 값:

```text
27
```

완료 run 수 확인:

```bash
grep -R '"completed"' \
  experiment/output/computer-3/runs/*/status.json | wc -l
```

정상 완료 시:

```text
27
```

실패 run 확인:

```bash
grep -R '"failed"' \
  experiment/output/computer-3/runs/*/status.json
```

출력이 없다면 실패 run이 없는 것이다.

결과 파일 열기:

```bash
open experiment/output/computer-3/summary.csv
open experiment/output/computer-3/run_results.csv
```

터미널에서 확인:

```bash
column -s, -t < experiment/output/computer-3/summary.csv | less -S
```

---

## 11. 평가 지표 해석

### 외부 평가 지표

정답 persona와 군집 결과의 일치도를 평가한다.

#### ARI

군집 결과와 정답 라벨의 일치도를 우연 일치까지 보정한 지표다.

- 높을수록 persona 복원이 잘 됨
- `1`에 가까울수록 거의 완전 일치
- `0` 근처면 무작위 수준
- 음수가 나오면 무작위보다 좋지 않을 수 있음

#### NMI

정답 persona와 cluster가 공유하는 정보량을 평가한다.

- `0`에 가까우면 관계가 약함
- `1`에 가까우면 강하게 일치

#### AMI

NMI에서 우연한 일치를 보정한 지표다.

#### Macro-F1

각 persona별 F1 점수를 동일한 비중으로 평균낸 값이다.

cluster ID 자체에는 의미가 없으므로 Hungarian matching을 적용한 뒤 계산한다.

### 내부 평가 지표

정답 persona를 사용하지 않고 군집 자체의 기하학적 품질을 평가한다.

#### Silhouette Score

- 높을수록 cluster 내부는 가깝고 cluster 간은 잘 분리됨

#### Davies-Bouldin Index

- 낮을수록 좋음

#### Calinski-Harabasz Index

- 높을수록 좋음

### 외부 지표를 우선해야 하는 이유

이 연구의 목적은 단순히 데이터를 깔끔하게 나누는 것이 아니라, 사전에 정의된 persona를 복원하는 것이다.

따라서 다음과 같은 결과가 가능하다.

```text
Silhouette는 높음
ARI/NMI는 낮음
```

이 경우:

> 군집 자체는 깔끔하지만, 우리가 의도한 persona 기준으로 나뉜 것은 아니다.

라고 해석해야 한다.

---

## 12. 결과 분석 순서

### 1단계: 실행 상태 확인

- 27개 run이 모두 생성됐는지
- completed가 27개인지
- failed run이 없는지

### 2단계: `summary.csv` 확인

각 `feature subset × pipeline`의 평균 성능과 표준편차를 비교한다.

우선순위:

1. ARI mean
2. NMI/AMI mean
3. Macro-F1 mean
4. seed별 표준편차
5. 내부 지표

### 3단계: seed 안정성 확인

평균뿐 아니라 표준편차도 함께 본다.

예:

```text
ARI = 0.78 ± 0.01
```

반복해도 안정적인 결과다.

```text
ARI = 0.78 ± 0.18
```

평균은 높지만 seed에 따라 결과가 크게 변하는 불안정한 조건이다.

### 4단계: F11, F13, F15 비교

#### F11 vs F13

퍼널 정보를 포함했을 때 성능이 좋아지는지 확인한다.

F13의 성능이 크게 낮다면:

> 장바구니, 결제 진입, 결제 시도, 구매 완료 같은 퍼널 정보가 persona 복원에 중요하다.

#### F11 vs F15

탐색 경로 정보를 포함했을 때 성능이 좋아지는지 확인한다.

F15의 성능이 크게 낮다면:

> 재방문, 뒤로 가기, 반복 이동과 같은 탐색 경로 정보가 persona 구분에 중요하다.

#### F13 vs F15

퍼널 정보와 탐색 경로 중 어느 정보가 더 중요한지 비교한다.

단, 엄밀한 ablation 해석은 전체 19개 feature를 사용한 F0 결과와 함께 비교해야 한다.

### 5단계: A1, A2, A3 비교

#### A1이 가장 좋을 때

직접 설계한 hand-crafted feature가 persona 구조를 충분히 표현하고, 복잡한 representation이 필요하지 않을 가능성이 있다.

#### A2가 가장 좋을 때

persona 구조가 비선형적이고 밀도 기반 군집화가 유효할 가능성이 있다.

#### A2 내부 지표만 높고 외부 지표가 낮을 때

HDBSCAN이 persona가 아닌 더 세부적인 행동 패턴을 찾았을 가능성이 있다.

#### A3가 가장 좋을 때

VAE가 학습한 잠재표현이 원래 feature 공간보다 persona 복원에 더 적합할 가능성이 있다.

### 6단계: run별 상세 확인

특정 조건이 이상하면 다음을 확인한다.

- `metrics.json`
- `predictions.csv`
- `contingency.csv`
- `mapping.json`
- `status.json`

특히 contingency matrix를 보면 하나의 persona가 여러 cluster로 분할됐는지, 여러 persona가 같은 cluster에 섞였는지 확인할 수 있다.

---

## 13. 최종적으로 얻고 싶은 결론

본 실험은 단순히 가장 높은 점수를 찾는 것이 아니라 다음을 밝히는 것이 목적이다.

1. 전체 feature를 모두 사용하는 것이 항상 최선인지
2. 탐색 경로, 탐색·비교, 퍼널 중 어떤 행동 신호가 중요한지
3. 특정 feature 그룹을 제거하면 성능이 얼마나 감소하는지
4. 직접 군집화, UMAP-HDBSCAN, VAE-K-Means 중 어떤 방식이 적합한지
5. 어떤 조합이 seed가 달라져도 안정적인지

가능한 결과 해석 예:

> F13에서 외부 평가 지표가 크게 감소하여 퍼널 행동 정보가 구매 관련 persona 복원에 핵심적인 것으로 나타났다.

> F15의 성능 감소 폭이 작다면 탐색 경로를 제외해도 검색·비교와 퍼널 feature만으로 상당 부분 persona를 구분할 수 있음을 시사한다.

> A2는 높은 Silhouette를 보였지만 ARI와 NMI가 낮아, 기하학적으로 분명한 행동 군집을 만들었으나 사전 정의 persona와는 다른 구조를 포착한 것으로 해석할 수 있다.

> A3가 A1보다 높은 외부 지표와 낮은 seed 표준편차를 보였다면, VAE latent representation이 persona 구조를 더 안정적으로 표현한 것으로 볼 수 있다.

---

## 14. 주의사항

- 정답 라벨은 평가 단계에서만 사용해야 한다.
- test 결과를 보고 하이퍼파라미터를 다시 선택하면 안 된다.
- 최고값만 보지 말고 seed 평균과 표준편차를 함께 봐야 한다.
- 내부 지표가 높다고 persona 복원 성능이 좋은 것은 아니다.
- A2의 noise ratio와 실제 cluster 수를 함께 기록해야 한다.
- F13과 F15의 정확한 ablation 효과는 팀원의 F0 baseline과 함께 해석해야 한다.
- 실험 도중 코드를 수정하면 이전 run과 이후 run의 조건이 달라질 수 있으므로 피해야 한다.
- 결과 폴더를 삭제하거나 `--force`로 재실행하기 전에 기존 결과를 보존해야 한다.

---

## 15. 실행 명령 요약

```bash
git switch test
cd dashboard-be
source ../.venv/bin/activate
python -m pip install -r requirements-experiment.txt

mkdir -p experiment/output/computer-3

caffeinate -dimsu python -m experiment.cli \
  --data-dir benchmark/output/merged-7500 \
  --feature-subsets F11 F13 F15 \
  --pipelines A1 A2 A3 \
  --seeds 7 42 2026 \
  --split-seed 2026 \
  --output-dir experiment/output/computer-3 \
  2>&1 | tee experiment/output/computer-3/console.log
```

중단 후 재개:

```bash
caffeinate -dimsu python -m experiment.cli \
  --data-dir benchmark/output/merged-7500 \
  --feature-subsets F11 F13 F15 \
  --pipelines A1 A2 A3 \
  --seeds 7 42 2026 \
  --split-seed 2026 \
  --output-dir experiment/output/computer-3 \
  2>&1 | tee -a experiment/output/computer-3/console.log
```
