# Feature-Pipeline 실험 실행 가이드

이 문서는 `user-clustering-feature-analysis` 저장소에서 팀원별 feature subset 실험을 실행하기 위한 명령어를 정리한 가이드이다.

지원 feature subset:

```text
F0, F2, F3,
F4, F6, F7,
F11, F13, F15
```

공통 실험 조건:

```text
Pipelines: A1, A2, A3
Seeds: 7, 42, 2026
Split Seed: 2026
Dataset: benchmark/output/merged-7500
```

팀원 1명당 실행 수:

```text
3 feature subsets × 3 pipelines × 3 seeds = 27 runs
```

전체 실험:

```text
9 feature subsets × 3 pipelines × 3 seeds = 81 runs
```

---

## 1. 저장소 준비

```bash
git clone https://github.com/junnokimm/user-clustering-feature-analysis.git
cd user-clustering-feature-analysis
```

실험 실행 위치는 반드시 `dashboard-be`이다.

```bash
cd dashboard-be
```

---

## 2. Python 가상환경 준비

저장소 root에 `.venv`를 만드는 경우:

```bash
cd ..
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
cd dashboard-be
python -m pip install -r requirements-experiment.txt
```

이미 `.venv`가 있다면 `dashboard-be`에서:

```bash
source ../.venv/bin/activate
```

확인:

```bash
which python
python --version
"../.venv/bin/python" -m experiment.cli --help
```

---

## 3. Computer 1

담당:

```text
F0
F2
F3
```

실행:

```bash
mkdir -p experiment/output/computer-1

caffeinate -dimsu "../.venv/bin/python" -m experiment.cli \
  --data-dir benchmark/output/merged-7500 \
  --feature-subsets F0 F2 F3 \
  --pipelines A1 A2 A3 \
  --seeds 7 42 2026 \
  --split-seed 2026 \
  --output-dir experiment/output/computer-1 \
  2>&1 | tee experiment/output/computer-1/console.log
```

---

## 4. Computer 2

담당:

```text
F4
F6
F7
```

실행:

```bash
mkdir -p experiment/output/computer-2

caffeinate -dimsu "../.venv/bin/python" -m experiment.cli \
  --data-dir benchmark/output/merged-7500 \
  --feature-subsets F4 F6 F7 \
  --pipelines A1 A2 A3 \
  --seeds 7 42 2026 \
  --split-seed 2026 \
  --output-dir experiment/output/computer-2 \
  2>&1 | tee experiment/output/computer-2/console.log
```

---

## 5. Computer 3

담당:

```text
F11
F13
F15
```

실행:

```bash
mkdir -p experiment/output/computer-3

caffeinate -dimsu "../.venv/bin/python" -m experiment.cli \
  --data-dir benchmark/output/merged-7500 \
  --feature-subsets F11 F13 F15 \
  --pipelines A1 A2 A3 \
  --seeds 7 42 2026 \
  --split-seed 2026 \
  --output-dir experiment/output/computer-3 \
  2>&1 | tee experiment/output/computer-3/console.log
```

---

## 6. 전체 81-run을 한 컴퓨터에서 실행

```bash
mkdir -p experiment/output/all

caffeinate -dimsu "../.venv/bin/python" -m experiment.cli \
  --data-dir benchmark/output/merged-7500 \
  --feature-subsets F0 F2 F3 F4 F6 F7 F11 F13 F15 \
  --pipelines A1 A2 A3 \
  --seeds 7 42 2026 \
  --split-seed 2026 \
  --output-dir experiment/output/all \
  2>&1 | tee experiment/output/all/console.log
```

---

## 7. 실험 중단 후 재개

기존 output directory를 유지한 상태로 같은 명령을 다시 실행하면 된다.

완료된 run은 skip된다.

예: Computer 1

```bash
caffeinate -dimsu "../.venv/bin/python" -m experiment.cli \
  --data-dir benchmark/output/merged-7500 \
  --feature-subsets F0 F2 F3 \
  --pipelines A1 A2 A3 \
  --seeds 7 42 2026 \
  --split-seed 2026 \
  --output-dir experiment/output/computer-1 \
  2>&1 | tee -a experiment/output/computer-1/console.log
```

`--force`는 기존 completed run까지 다시 실행할 수 있으므로 일반 재개 시 사용하지 않는다.

---

## 8. 완료 여부 확인

Computer 1:

```bash
grep -R '"completed"' \
  experiment/output/computer-1/runs/*/status.json | wc -l

grep -R '"failed"' \
  experiment/output/computer-1/runs/*/status.json

grep -R '"running"' \
  experiment/output/computer-1/runs/*/status.json
```

Computer 2와 3은 경로만 각각 바꾸면 된다.

정상 완료 기준:

```text
completed = 27
failed = 출력 없음
running = 출력 없음
```

---

## 9. 주요 결과 파일

```text
experiment/output/computer-X/
├── console.log
├── extracted_features.csv
├── environment.json
├── experiment_config.json
├── run_results.csv
├── split_manifest.json
├── summary.csv
├── contingency/
└── runs/
```

### `run_results.csv`

각 `feature subset × pipeline × seed` 개별 결과.

팀원 1명당 정상적으로 27개 run이 존재해야 한다.

### `summary.csv`

같은 `feature subset × pipeline`의 seed 반복 결과를 요약한 파일.

최종 비교 시 가장 먼저 확인하는 결과 파일이다.

### `split_manifest.json`

train / validation / test에 어떤 session이 들어갔는지 기록한다.

---

## 10. 팀원 간 split 동일성 확인

세 팀원의 결과를 한 컴퓨터에 모은 뒤:

```bash
shasum -a 256 experiment/output/computer-1/split_manifest.json
shasum -a 256 experiment/output/computer-2/split_manifest.json
shasum -a 256 experiment/output/computer-3/split_manifest.json
```

세 SHA-256 값이 같다면 동일한 split을 사용한 것이다.

---

## 11. 컴퓨터 간 반드시 동일하게 유지할 조건

```text
1. Git commit / 코드 버전
2. benchmark 데이터
3. split seed = 2026
4. pipelines = A1, A2, A3
5. seeds = 7, 42, 2026
6. Python 및 주요 package 버전
7. 각 담당 실험 27 runs 정상 완료
```

---

## 12. 담당 요약

| Computer | Feature Subsets | Pipelines | Seeds | Runs |
|---|---|---|---|---:|
| Computer 1 | F0, F2, F3 | A1, A2, A3 | 7, 42, 2026 | 27 |
| Computer 2 | F4, F6, F7 | A1, A2, A3 | 7, 42, 2026 | 27 |
| Computer 3 | F11, F13, F15 | A1, A2, A3 | 7, 42, 2026 | 27 |
| **전체** | **9 subsets** | **3 pipelines** | **3 seeds** | **81** |
