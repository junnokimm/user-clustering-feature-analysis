"""
전체 파이프라인 한 번에 실행: feature 추출 -> split -> 27 runs -> REPORT.md

    python run_all.py                 # Computer 2 (F4, F6, F7)
    python run_all.py --subsets F0,F2,F3
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def step(title, script, extra=None):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    command = [sys.executable, os.path.join(HERE, script)] + (extra or [])
    result = subprocess.run(command, cwd=HERE)
    if result.returncode != 0:
        raise SystemExit(f"[run_all] FAILED at {script} (exit {result.returncode})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", default="F4,F6,F7")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--skip-extract", action="store_true",
                        help="features.csv / split.json 이 이미 있으면 건너뛴다")
    args = parser.parse_args()

    if not args.skip_extract:
        step("1/4  feature 추출 (events.jsonl -> features.csv)", "extract_features.py")
        step("2/4  stratified split (train 70 / val 15 / test 15)", "make_split.py")

    step(f"3/4  실험 실행 ({args.subsets} x A1,A2,A3 x {args.seeds})", "run_experiments.py",
         ["--subsets", args.subsets, "--seeds", args.seeds])
    step("4/4  결과 집계 -> REPORT.md", "report.py")

    print(f"\n[run_all] 완료. 결과: {os.path.join(HERE, 'REPORT.md')}")


if __name__ == "__main__":
    main()
