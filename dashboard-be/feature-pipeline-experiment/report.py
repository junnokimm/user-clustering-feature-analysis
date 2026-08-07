"""
results.csv -> REPORT.md + 콘솔 요약

계획서 12장의 보고 형식으로 F* x A* 조건을 seed 평균/표준편차로 집계하고,
11장 해석 원칙에 따라 자동 코멘트를 붙인다.
"""

import argparse
import os
import sys

import pandas as pd

# Windows 콘솔 기본 인코딩(cp949)에서 한글 표가 깨지지 않도록 고정한다
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

METRIC_ORDER = ["ari", "nmi", "ami", "macro_f1", "silhouette", "davies_bouldin"]


def aggregate(frame):
    grouped = frame.groupby(["feature_subset", "pipeline"])
    summary = grouped.agg(
        n_features=("n_features", "first"),
        distinct_vectors=("distinct_train_vectors", "first"),
        seeds=("seed", "count"),
        val_sil_mean=("val_silhouette", "mean"),
        ari_mean=("ari", "mean"), ari_std=("ari", "std"),
        nmi_mean=("nmi", "mean"), nmi_std=("nmi", "std"),
        ami_mean=("ami", "mean"), ami_std=("ami", "std"),
        f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
        sil_mean=("silhouette", "mean"),
        db_mean=("davies_bouldin", "mean"),
        k_mean=("n_clusters", "mean"),
        noise_mean=("noise_ratio", "mean"),
    ).reset_index()
    return summary.sort_values("ari_mean", ascending=False).reset_index(drop=True)


def note_for(row):
    notes = []
    if row["noise_mean"] > 0.05:
        notes.append(f"noise {row['noise_mean']:.0%}")
    if row["ari_std"] > 0.05:
        notes.append("seed 불안정")
    if row["sil_mean"] > 0.5 and row["ari_mean"] < 0.3:
        notes.append("구조는 뚜렷하나 persona 복원 실패")
    if row["distinct_vectors"] < 100:
        notes.append(f"고유벡터 {row['distinct_vectors']}개")
    return ", ".join(notes) or "-"


def main_table(summary):
    lines = [
        "| feature | pipeline | seeds | 선택 k | ARI mean | ARI std | AMI mean | AMI std | "
        "NMI mean | Macro-F1 mean | Macro-F1 std | Silhouette | DB | 비고 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['feature_subset']} | {row['pipeline']} | {int(row['seeds'])} | "
            f"{row['k_mean']:.1f} | "
            f"{row['ari_mean']:.3f} | {row['ari_std']:.3f} | "
            f"{row['ami_mean']:.3f} | {row['ami_std']:.3f} | {row['nmi_mean']:.3f} | "
            f"{row['f1_mean']:.3f} | {row['f1_std']:.3f} | "
            f"{row['sil_mean']:.3f} | {row['db_mean']:.3f} | {note_for(row)} |"
        )
    return "\n".join(lines)


def ranking_table(summary, key, label):
    ranked = summary.groupby(key).agg(
        ari=("ari_mean", "mean"), ami=("ami_mean", "mean"), f1=("f1_mean", "mean")
    ).sort_values("ari", ascending=False).reset_index()
    lines = [f"| {label} | ARI 평균 | AMI 평균 | Macro-F1 평균 |", "|---|---|---|---|"]
    for _, row in ranked.iterrows():
        lines.append(f"| {row[key]} | {row['ari']:.3f} | {row['ami']:.3f} | {row['f1']:.3f} |")
    return "\n".join(lines), ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="artifacts/results.csv")
    parser.add_argument("--out", default="REPORT.md")
    args = parser.parse_args()

    here = os.path.dirname(__file__)
    frame = pd.read_csv(os.path.abspath(os.path.join(here, args.results)))
    summary = aggregate(frame)

    best = summary.iloc[0]
    feature_md, feature_rank = ranking_table(summary, "feature_subset", "feature subset")
    pipeline_md, pipeline_rank = ranking_table(summary, "pipeline", "pipeline")

    most_stable = summary.loc[summary["ari_std"].idxmin()]

    sections = [
        "# Feature-Pipeline 실험 결과",
        "",
        f"- 대상: `{'`, `'.join(sorted(frame['feature_subset'].unique()))}` "
        f"x `{'`, `'.join(sorted(frame['pipeline'].unique()))}` x seed {sorted(frame['seed'].unique())}",
        f"- 총 {len(frame)} runs, 평가 split: test",
        "",
        "## 한 줄 결론",
        "",
        f"**{best['feature_subset']} x {best['pipeline']}** 가 ARI {best['ari_mean']:.3f} "
        f"(±{best['ari_std']:.3f}), Macro-F1 {best['f1_mean']:.3f} 로 가장 좋다. "
        f"feature subset 중에서는 **{feature_rank.iloc[0]['feature_subset']}**, "
        f"파이프라인 중에서는 **{pipeline_rank.iloc[0]['pipeline']}** 가 평균적으로 앞선다. "
        f"seed 변화에 가장 안정적인 조건은 **{most_stable['feature_subset']} x {most_stable['pipeline']}** "
        f"(ARI std {most_stable['ari_std']:.3f}) 이다.",
        "",
        "## 전체 결과 (seed 평균)",
        "",
        main_table(summary),
        "",
        "## Feature subset 순위 (파이프라인 평균)",
        "",
        feature_md,
        "",
        "## 파이프라인 순위 (feature subset 평균)",
        "",
        pipeline_md,
        "",
        "## 해석 메모",
        "",
        "- **군집 수 k 를 포함한 모든 하이퍼파라미터는 val split 의 Silhouette 으로 골랐다.**"
        " 정답 라벨과 persona 수(5)는 선택 과정에 들어가지 않는다. 라벨은 test 지표 계산에만 쓰인다.",
        "- `선택 k` 가 5 에 가까우면 라벨 없이도 데이터가 persona 개수를 드러냈다는 뜻이다.",
        "- 계획서 11장에 따라 내부 지표(Silhouette, DB)보다 외부 지표(ARI, NMI, Macro-F1)를 우선한다.",
        "- Silhouette 가 높은데 ARI 가 낮으면 군집은 깔끔하지만 persona 복원에는 실패한 경우다.",
        "- `고유벡터 N개` 비고는 해당 subset 의 전처리 후 서로 다른 벡터 수다. 이 값이 작으면"
        " 동점 세션이 대량으로 생겨 거리 기반 군집이 임의로 갈린다.",
        "",
    ]

    out_path = os.path.abspath(os.path.join(here, args.out))
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(sections))

    print("\n".join(sections))
    print(f"\n[report] wrote {out_path}")


if __name__ == "__main__":
    main()
