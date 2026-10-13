from __future__ import annotations

import math

import pandas as pd
import pytest

from src.api import (
    compare_experiments,
    get_experiment_details,
    get_experiment_history,
    inspect_features,
    run_baseline,
    run_experiment,
)
from src.experiments import ENGINE


@pytest.fixture
def loaded(miniature_bundle):
    ENGINE.set_dataset(miniature_bundle)
    return miniature_bundle


def assert_result(result):
    for split in ("validation", "test"):
        assert 0 <= result[split]["ap"] <= 1
        assert 0 <= result[split]["roc_auc"] <= 1
        assert math.isfinite(result[split]["ap"])
        assert math.isfinite(result[split]["roc_auc"])


def test_invalid_feature_is_clear(loaded):
    with pytest.raises(ValueError, match="Unknown feature: source_activity_magic"):
        run_experiment("bad", ["source_activity_magic"])


def test_experiment_requires_comparable_baseline(loaded):
    with pytest.raises(RuntimeError, match="run_baseline"):
        run_experiment("temporal", ["time_since_source_event"])


def test_baseline_temporal_graph_mixed_and_registry(loaded):
    baseline = run_baseline(mode="fast")
    temporal = run_experiment("temporal", ["time_since_source_event"], mode="fast")
    graph = run_experiment("graph", ["source_degree"], mode="fast")
    mixed = run_experiment(
        "mixed", ["source_events_recent", "prior_edge_exists"], mode="fast"
    )
    for result in (baseline, temporal, graph, mixed):
        assert_result(result)
        assert result["configuration"]["model"] == baseline["configuration"]["model"]
        assert result["seed"] == baseline["seed"]
    history = get_experiment_history()
    comparison = compare_experiments()
    assert len(history) == 4
    assert len(comparison) == 4
    assert get_experiment_details(mixed["experiment_id"])["experiment_name"] == "mixed"
    assert math.isclose(
        temporal["delta_vs_baseline"]["ap"],
        temporal["test"]["ap"] - baseline["test"]["ap"],
    )


def test_feature_inspection_is_readable(loaded):
    result = inspect_features(["time_since_source_event", "source_degree"], 5)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "source", "destination", "timestamp", "time_since_source_event", "source_degree"
    ]
    assert len(result) == 5


def test_same_seed_and_configuration_are_deterministic(loaded):
    first = run_baseline(mode="fast", seed=11)
    second = run_baseline(mode="fast", seed=11)
    assert first["configuration"] == second["configuration"]
    assert first["validation"] == second["validation"]
    assert first["test"] == second["test"]
