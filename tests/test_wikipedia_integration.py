from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.api import get_dataset_summary, load_dataset, run_baseline


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_TGN_INTEGRATION") != "1",
    reason="set RUN_TGN_INTEGRATION=1 to download and train on Wikipedia",
)
def test_public_wikipedia_download_and_fast_baseline(tmp_path):
    data_dir = Path(os.environ.get("TGN_DATA_DIR", tmp_path / "data"))
    summary = load_dataset(data_dir, download=True)
    assert summary["number_of_interactions"] == 157474
    assert summary["original_edge_feature_dimensionality"] == 172
    result = run_baseline(mode="fast")
    assert 0 <= result["validation"]["ap"] <= 1
    assert 0 <= result["validation"]["roc_auc"] <= 1
    assert 0 <= result["test"]["ap"] <= 1
    assert 0 <= result["test"]["roc_auc"] <= 1
