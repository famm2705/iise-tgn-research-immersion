from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import load_wikipedia, make_splits, preprocess_raw_csv


def test_preprocessing_matches_original_bipartite_layout(tmp_path):
    raw = tmp_path / "wikipedia.csv"
    pd.DataFrame(
        {
            "user_id": [0, 1, 0],
            "item_id": [0, 1, 1],
            "timestamp": [1.0, 2.0, 3.0],
            "state_label": [0, 1, 0],
            "f0": [0.1, 0.2, 0.3],
            "f1": [1.1, 1.2, 1.3],
        }
    ).to_csv(raw, index=False)
    bundle = preprocess_raw_csv(raw, tmp_path)
    assert bundle.interactions["u"].tolist() == [1, 2, 1]
    assert bundle.interactions["i"].tolist() == [3, 4, 4]
    assert bundle.interactions["idx"].tolist() == [1, 2, 3]
    assert bundle.edge_features.shape == (4, 2)
    assert np.all(bundle.edge_features[0] == 0)
    assert bundle.node_features.shape == (5, 172)
    cached = load_wikipedia(tmp_path, download=False)
    assert cached.interactions.equals(bundle.interactions)
    assert np.array_equal(cached.edge_features, bundle.edge_features)


def test_splits_are_chronological_and_deterministic(miniature_bundle):
    first = make_splits(miniature_bundle.interactions)
    second = make_splits(miniature_bundle.interactions)
    assert np.array_equal(first.train.edge_idxs, second.train.edge_idxs)
    assert first.train.timestamps.max() <= first.validation_time
    assert first.validation.timestamps.min() > first.validation_time
    assert first.validation.timestamps.max() <= first.test_time
    assert first.test.timestamps.min() > first.test_time

