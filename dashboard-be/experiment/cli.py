"""CLI for reproducible feature-pipeline benchmark experiments."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .data_loader import load_events_by_session, load_session_metadata, validate_session_join
from .evaluation import align_predictions, cluster_mapping, contingency, evaluate
from .feature_extractor import extract_features
from .feature_sets import FEATURE_SETS
from .preprocessing import TrainOnlyPreprocessor
from .splitter import split_sessions
from .pipelines import kmeans_pipeline, umap_hdbscan_pipeline, vae_kmeans_pipeline


PIPELINES = {"A1": kmeans_pipeline.run, "A2": umap_hdbscan_pipeline.run, "A3": vae_kmeans_pipeline.run}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible unsupervised benchmark experiments")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--feature-subsets", nargs="+", required=True)
    parser.add_argument("--pipelines", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    unknown_sets = set(args.feature_subsets) - set(FEATURE_SETS)
    unknown_pipelines = set(args.pipelines) - set(PIPELINES)
    if unknown_sets or unknown_pipelines:
        parser.error(f"unknown subsets={sorted(unknown_sets)} pipelines={sorted(unknown_pipelines)}")
    return args


def _features(metadata: dict, events: dict, ids: list[str]) -> tuple[np.ndarray, list[str]]:
    rows = [extract_features(events[session_id]) for session_id in ids]
    columns = tuple(rows[0])
    matrix = np.asarray([[row[column] for column in columns] for row in rows], dtype=float)
    return matrix, list(columns)


def _write_csv(path: Path, rows: list[dict], fieldnames: tuple[str, ...] | None = None) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        resolved_fieldnames = fieldnames or tuple(sorted({key for row in rows for key in row}))
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_session_metadata(data_dir / "sessions.json")
    events = load_events_by_session(data_dir / "events.jsonl")
    validate_session_join(metadata, events)
    if args.sample_size:
        chosen_ids = sorted(metadata)[:args.sample_size]
        metadata = {key: metadata[key] for key in chosen_ids}
        events = {key: events[key] for key in chosen_ids}
    manifest = split_sessions(metadata, args.split_seed)
    (output_dir / "split_manifest.json").write_text(json.dumps({"train": manifest.train_ids, "validation": manifest.validation_ids, "test": manifest.test_ids, "seed": manifest.seed, "stratification": manifest.stratification}, indent=2), encoding="utf-8")
    full_feature_matrix, columns = _features(metadata, events, sorted(metadata))
    _write_csv(output_dir / "extracted_features.csv", [{"session_id": session_id, **dict(zip(columns, row))} for session_id, row in zip(sorted(metadata), full_feature_matrix)])
    (output_dir / "experiment_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (output_dir / "environment.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform()}, indent=2), encoding="utf-8")
    index = {session_id: position for position, session_id in enumerate(sorted(metadata))}
    results: list[dict] = []
    for feature_set in args.feature_subsets:
        selected = [columns.index(name) for name in FEATURE_SETS[feature_set]]
        subset_matrix = full_feature_matrix[:, selected]
        train_matrix, validation_matrix, test_matrix = (subset_matrix[[index[item] for item in ids]] for ids in (manifest.train_ids, manifest.validation_ids, manifest.test_ids))
        processor = TrainOnlyPreprocessor(FEATURE_SETS[feature_set]).fit(train_matrix)
        train_matrix, validation_matrix, test_matrix = processor.transform(train_matrix), processor.transform(validation_matrix), processor.transform(test_matrix)
        for pipeline_name in args.pipelines:
            for seed in args.seeds:
                run_dir = output_dir / "runs" / f"{feature_set}_{pipeline_name}_seed{seed}"
                status_path = run_dir / "status.json"
                if status_path.exists() and not args.force and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "completed":
                    metrics_path = run_dir / "metrics.json"
                    if metrics_path.exists():
                        results.append({"feature_set": feature_set, "pipeline": pipeline_name, "seed": seed, **json.loads(metrics_path.read_text(encoding="utf-8"))})
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                try:
                    train_labels, validation_labels, test_labels, extra = PIPELINES[pipeline_name](train_matrix, validation_matrix, test_matrix, seed)
                    labels_by_session = {session_id: metadata[session_id].persona_id for session_id in manifest.test_ids}
                    prediction_by_session = {session_id: int(cluster) for session_id, cluster in zip(manifest.test_ids, test_labels)}
                    labels, aligned_clusters = align_predictions(manifest.test_ids, labels_by_session, prediction_by_session)
                    metrics = evaluate(test_matrix, labels, aligned_clusters) | extra
                    prediction_rows = [{"session_id": session_id, "true_label": labels_by_session[session_id], "cluster": prediction_by_session[session_id]} for session_id in sorted(manifest.test_ids)]
                    _write_csv(run_dir / "predictions.csv", prediction_rows, ("session_id", "true_label", "cluster"))
                    personas, cluster_ids, contingency_matrix = contingency(labels, aligned_clusters)
                    contingency_dir = output_dir / "contingency"; contingency_dir.mkdir(exist_ok=True)
                    contingency_rows = [{"persona": persona, **{str(cluster): int(contingency_matrix[row_index, column_index]) for column_index, cluster in enumerate(cluster_ids)}} for row_index, persona in enumerate(personas)]
                    _write_csv(contingency_dir / f"{feature_set}_{pipeline_name}_seed{seed}.csv", contingency_rows, ("persona", *(str(cluster) for cluster in cluster_ids)))
                    (run_dir / "cluster_mapping.json").write_text(json.dumps({str(key): value for key, value in cluster_mapping(labels, aligned_clusters).items()}, indent=2, sort_keys=True), encoding="utf-8")
                    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                    (run_dir / "config.json").write_text(json.dumps({"feature_set": feature_set, "pipeline": pipeline_name, "seed": seed, "features": FEATURE_SETS[feature_set]}, indent=2), encoding="utf-8")
                    (status_path).write_text(json.dumps({"status": "completed"}), encoding="utf-8")
                    results.append({"feature_set": feature_set, "pipeline": pipeline_name, "seed": seed, **metrics})
                except (RuntimeError, ValueError, OSError) as error:
                    status_path.write_text(json.dumps({"status": "failed", "error": str(error)}), encoding="utf-8")
                    results.append({"feature_set": feature_set, "pipeline": pipeline_name, "seed": seed, "error": str(error)})
    _write_csv(output_dir / "run_results.csv", results)
    if results:
        summary = []
        for feature_set in args.feature_subsets:
            for pipeline_name in args.pipelines:
                rows = [row for row in results if row["feature_set"] == feature_set and row["pipeline"] == pipeline_name and "ari" in row]
                if rows:
                    summary.append({"feature_set": feature_set, "pipeline": pipeline_name, "seed_count": len(rows), "ari_mean": float(np.mean([row["ari"] for row in rows])), "ari_std": float(np.std([row["ari"] for row in rows])), "nmi_mean": float(np.mean([row["nmi"] for row in rows])), "macro_f1_mean": float(np.mean([row["macro_f1"] or 0 for row in rows]))})
        _write_csv(output_dir / "summary.csv", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
