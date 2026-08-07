"""
events.jsonl + sessions.json -> features.csv

FEATURE_PIPELINE_EXPERIMENT_PLAN.md 4장의 19개 feature를 raw event 에서 재계산한다.
정답 라벨(persona_id, ground_truth_label)은 평가 전용으로만 함께 저장한다.

feature 정의는 3대 컴퓨터가 동일해야 하므로 여기서 확정한다. FEATURE_DEFINITIONS.md 참고.
"""

import argparse
import json
import os
from collections import defaultdict

import pandas as pd

FEATURE_COLUMNS = [
    "session_duration_ms",   # f1
    "event_count",           # f2
    "page_view_count",       # f3
    "click_count",           # f4
    "depth",                 # f5
    "unique_page_ratio",     # f6
    "revisit_rate",          # f7
    "backtrack_count",       # f8
    "loop_rate",             # f9
    "search_count",          # f10
    "filter_count",          # f11
    "product_detail_count",  # f12
    "review_view_count",     # f13
    "cart_add_count",        # f14
    "cart_remove_count",     # f15
    "checkout_entered",      # f16
    "payment_attempt_count", # f17
    "purchase_completed",    # f18
    "error_count",           # f19
]

LABEL_COLUMNS = ["session_id", "persona_id", "ground_truth_label", "difficulty", "split_source"]


def segment_depth(path):
    """URL 계층 깊이. '/' -> 0, '/cart' -> 1, '/product/grocery_4102' -> 2"""
    return len([seg for seg in str(path).split("/") if seg])


def collect_session_events(events_path):
    """session_id -> event list. 이벤트는 ts 순으로 정렬한다."""
    sessions = defaultdict(list)
    with open(events_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            sessions[event["session_id"]].append(event)
    for events in sessions.values():
        events.sort(key=lambda e: e.get("ts", 0))
    return sessions


def compute_features(events):
    names = [e.get("event_name") for e in events]
    timestamps = [e.get("ts", 0) for e in events]

    paths = [str(e.get("path") or "") for e in events if e.get("event_name") == "page_view"]
    depth = len(paths)
    unique_count = len(set(paths))

    # f8 backtrack: 상위 계층/목록 페이지로 되돌아간 이동 수.
    # loop_rate(f9)와 겹치지 않도록 "URL 계층이 얕아지는 전이"로 정의한다.
    backtracks = 0
    for i in range(1, depth):
        if segment_depth(paths[i]) < segment_depth(paths[i - 1]):
            backtracks += 1

    # f9 loop: A -> B -> A 형태의 왕복 이동 비율
    loops = sum(1 for i in range(2, depth) if paths[i] == paths[i - 2])

    return {
        "session_duration_ms": (max(timestamps) - min(timestamps)) if timestamps else 0,
        "event_count": len(events),
        "page_view_count": depth,
        "click_count": names.count("click"),
        "depth": depth,
        "unique_page_ratio": (unique_count / depth) if depth else 0.0,
        "revisit_rate": ((depth - unique_count) / depth) if depth else 0.0,
        "backtrack_count": backtracks,
        "loop_rate": (loops / (depth - 2)) if depth > 2 else 0.0,
        "search_count": names.count("search"),
        "filter_count": names.count("filter_change"),
        "product_detail_count": sum(1 for p in paths if p.startswith("/product/")),
        "review_view_count": sum(1 for p in paths if p.startswith("/review/")),
        "cart_add_count": names.count("add_to_cart"),
        "cart_remove_count": names.count("remove_from_cart"),
        "checkout_entered": 1 if "checkout_start" in names else 0,
        "payment_attempt_count": names.count("payment_attempt"),
        "purchase_completed": 1 if "checkout_complete" in names else 0,
        "error_count": names.count("error"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="../benchmark/output/merged-7500")
    parser.add_argument("--out", default="artifacts/features.csv")
    args = parser.parse_args()

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), args.dataset))
    events_path = os.path.join(base, "events.jsonl")
    sessions_path = os.path.join(base, "sessions.json")

    print(f"[extract] dataset: {base}")
    session_events = collect_session_events(events_path)
    print(f"[extract] sessions with events: {len(session_events)}")

    with open(sessions_path, "r", encoding="utf-8") as handle:
        sessions = json.load(handle)["sessions"]
    print(f"[extract] sessions in metadata: {len(sessions)}")

    rows = []
    missing = 0
    for session in sessions:
        sid = session["session_id"]
        events = session_events.get(sid)
        if not events:
            missing += 1
            continue
        row = {
            "session_id": sid,
            "persona_id": session["persona_id"],
            "ground_truth_label": session["ground_truth_label"],
            "difficulty": session.get("difficulty", "unknown"),
            "split_source": session.get("split", "unknown"),
        }
        row.update(compute_features(events))
        rows.append(row)

    if missing:
        print(f"[extract] WARNING: {missing} sessions had no events and were dropped")

    frame = pd.DataFrame(rows, columns=LABEL_COLUMNS + FEATURE_COLUMNS)

    nan_count = int(frame[FEATURE_COLUMNS].isna().sum().sum())
    if nan_count:
        raise SystemExit(f"[extract] FAILED: {nan_count} NaN values in feature columns")

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frame.to_csv(out_path, index=False)

    print(f"[extract] wrote {len(frame)} rows x {len(FEATURE_COLUMNS)} features -> {out_path}")
    print("[extract] persona distribution:")
    print(frame["persona_id"].value_counts().to_string())
    print("[extract] feature summary:")
    print(frame[FEATURE_COLUMNS].describe().T[["mean", "std", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
