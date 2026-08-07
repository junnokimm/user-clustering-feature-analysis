"""
features.csv -> split.json (train 70% / val 15% / test 15%)

계획서 3.3, 3.4 원칙:
- 세션 단위 분할, 한 세션은 하나의 split 에만 속한다
- persona_id x difficulty stratified 로 분포 치우침을 막는다

split seed 는 실험 seed 와 분리된 고정값이다. 3대 컴퓨터가 같은 split 을 써야
결과를 비교할 수 있으므로 split.json 은 커밋해서 공유한다.
"""

import argparse
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_SEED = 20260803


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="artifacts/features.csv")
    parser.add_argument("--out", default="artifacts/split.json")
    args = parser.parse_args()

    here = os.path.dirname(__file__)
    frame = pd.read_csv(os.path.abspath(os.path.join(here, args.features)))

    strata = frame["persona_id"] + "|" + frame["difficulty"]

    train_idx, holdout_idx = train_test_split(
        frame.index, test_size=0.30, random_state=SPLIT_SEED, stratify=strata
    )
    val_idx, test_idx = train_test_split(
        holdout_idx, test_size=0.50, random_state=SPLIT_SEED, stratify=strata.loc[holdout_idx]
    )

    split = {
        "split_seed": SPLIT_SEED,
        "stratified_by": ["persona_id", "difficulty"],
        "train": sorted(frame.loc[train_idx, "session_id"].tolist()),
        "val": sorted(frame.loc[val_idx, "session_id"].tolist()),
        "test": sorted(frame.loc[test_idx, "session_id"].tolist()),
    }

    all_ids = split["train"] + split["val"] + split["test"]
    if len(all_ids) != len(set(all_ids)):
        raise SystemExit("[split] FAILED: session appears in more than one split")
    if len(all_ids) != len(frame):
        raise SystemExit(f"[split] FAILED: {len(all_ids)} split ids != {len(frame)} sessions")

    out_path = os.path.abspath(os.path.join(here, args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=1)

    print(f"[split] train {len(split['train'])} / val {len(split['val'])} / test {len(split['test'])}")
    print(f"[split] wrote {out_path}")

    name_by_id = {}
    for name in ("train", "val", "test"):
        for sid in split[name]:
            name_by_id[sid] = name
    frame["split"] = frame["session_id"].map(name_by_id)
    print("[split] persona distribution per split (%):")
    dist = pd.crosstab(frame["persona_id"], frame["split"], normalize="columns") * 100
    print(dist.round(1).to_string())


if __name__ == "__main__":
    main()
