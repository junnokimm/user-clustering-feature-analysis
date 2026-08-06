import json
import sys

import numpy as np

from experiment import cli
from experiment.cli import _write_csv
from experiment.evaluation import align_predictions, contingency, macro_f1_hungarian
from experiment.feature_extractor import extract_features
from experiment.feature_sets import FEATURE_SETS
from experiment.preprocessing import TrainOnlyPreprocessor
from experiment.splitter import SplitManifest, validate_split_manifest


def test_extract_features_uses_conservative_path_definitions() -> None:
    events = [
        {"ts": 1, "event_name": "page_view", "path": "/"},
        {"ts": 2, "event_name": "page_view", "path": "/product/x"},
        {"ts": 3, "event_name": "page_view", "path": "/"},
        {"ts": 4, "event_name": "page_view", "path": "/review/x"},
        {"ts": 5, "event_name": "checkout_complete", "path": "/order-complete"},
    ]
    row = extract_features(events)
    assert row["depth"] == 4
    assert row["backtrack_count"] == 1
    assert row["loop_rate"] == 1 / 3
    assert row["product_detail_count"] == 1
    assert row["review_view_count"] == 1
    assert row["purchase_completed"] == 1


def test_f11_f13_f15_match_plan() -> None:
    assert FEATURE_SETS["F11"] == FEATURE_SETS["F2"] + FEATURE_SETS["F3"] + FEATURE_SETS["F4"]
    assert set(FEATURE_SETS["F13"]).isdisjoint(set(FEATURE_SETS["F4"]))
    assert set(FEATURE_SETS["F15"]).isdisjoint(set(FEATURE_SETS["F2"]))


def test_preprocessor_fits_training_data_only() -> None:
    processor = TrainOnlyPreprocessor(("event_count", "unique_page_ratio", "checkout_entered")).fit(np.asarray([[1.0, 0.0, 0.0], [3.0, 1.0, 1.0]]))
    transformed = processor.transform(np.asarray([[7.0, 0.5, 1.0]]))
    assert transformed[0, 2] == 1.0
    assert transformed[0, 0] > 1.0


def test_hungarian_macro_f1_ignores_cluster_id_permutation() -> None:
    labels = np.asarray(["a", "a", "b", "b"])
    clusters = np.asarray([8, 8, 3, 3])
    assert macro_f1_hungarian(labels, clusters) == 1.0


def test_alignment_uses_session_ids_not_prediction_position() -> None:
    labels, clusters = align_predictions(("s2", "s1"), {"s1": "a", "s2": "b"}, {"s1": 7, "s2": 3})
    assert labels.tolist() == ["a", "b"]
    assert clusters.tolist() == [7, 3]


def test_contingency_includes_noise_and_all_samples() -> None:
    personas, clusters, matrix = contingency(np.asarray(["a", "a", "b"]), np.asarray([-1, 2, 2]))
    assert personas == ["a", "b"]
    assert clusters == [-1, 2]
    assert matrix.sum() == 3


def test_csv_writer_preserves_declared_schema_order(tmp_path) -> None:
    output = tmp_path / "predictions.csv"
    _write_csv(
        output,
        [{"session_id": "s2", "true_label": "b", "cluster": 10}],
        ("session_id", "true_label", "cluster"),
    )
    assert output.read_text(encoding="utf-8").splitlines()[0] == "session_id,true_label,cluster"


def test_cli_keeps_full_feature_matrix_for_following_subsets(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sessions = []
    events = []
    for persona in ("a", "b", "c", "d", "e"):
        for number in range(20):
            session_id = f"{persona}-{number:02d}"
            sessions.append({"session_id": session_id, "persona_id": persona, "ground_truth_label": persona, "difficulty": "easy"})
            events.extend((
                {"session_id": session_id, "ts": 1, "event_name": "page_view", "path": "/"},
                {"session_id": session_id, "ts": 2, "event_name": "click", "path": f"/product/{number % 3}"},
            ))
    (data_dir / "sessions.json").write_text(json.dumps({"sessions": sessions}), encoding="utf-8")
    (data_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")

    observed_shapes = []

    def run_pipeline(train, validation, test, seed):
        observed_shapes.append((train.shape, validation.shape, test.shape, seed))
        return np.zeros(len(train), dtype=int), np.zeros(len(validation), dtype=int), np.zeros(len(test), dtype=int), {}

    for pipeline_name in ("A1", "A2", "A3"):
        monkeypatch.setitem(cli.PIPELINES, pipeline_name, run_pipeline)
    output_dir = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["experiment.cli", "--data-dir", str(data_dir), "--feature-subsets", "F11", "F13", "--pipelines", "A1", "A2", "A3", "--seeds", "7", "42", "--split-seed", "2026", "--output-dir", str(output_dir)])

    assert cli.main() == 0
    assert len(observed_shapes) == 12
    assert {shape[:3] for shape in observed_shapes} == {((70, 14), (15, 14), (15, 14))}
    assert all((output_dir / "runs" / f"{subset}_{pipeline}_seed{seed}" / "status.json").exists() for subset in ("F11", "F13") for pipeline in ("A1", "A2", "A3") for seed in (7, 42))


def test_split_manifest_rejects_overlap() -> None:
    manifest = SplitManifest(("a",), ("a",), ("b",), "persona_id", 1)
    try:
        validate_split_manifest(manifest, 2)
    except RuntimeError as error:
        assert "overlapping" in str(error)
    else:
        raise AssertionError("overlapping splits must fail")
