from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import DatasetBundle
from src.experiments import ENGINE


@pytest.fixture
def miniature_bundle(tmp_path):
    n = 60
    # IDs follow upstream's one-based, disjoint bipartite indexing.
    frame = pd.DataFrame(
        {
            "u": np.array([1 + (k % 4) for k in range(n)], dtype=np.int64),
            "i": np.array([5 + ((k * 3) % 5) for k in range(n)], dtype=np.int64),
            "ts": np.arange(1, n + 1, dtype=np.float64),
            "label": np.ones(n, dtype=np.float64),
            "idx": np.arange(1, n + 1, dtype=np.int64),
        }
    )
    rng = np.random.default_rng(7)
    edges = np.vstack([np.zeros((1, 4), dtype=np.float32), rng.normal(size=(n, 4)).astype(np.float32)])
    nodes = np.zeros((10, 172), dtype=np.float32)
    bundle = DatasetBundle(frame, edges, nodes, tmp_path)
    yield bundle
    ENGINE.dataset = None
    ENGINE.history.clear()
    ENGINE._feature_cache.clear()

