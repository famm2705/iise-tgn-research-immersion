"""Stable notebook-facing facade for controlled TGN experiments."""

from __future__ import annotations

import copy
import platform
from pathlib import Path

import pandas as pd

from .catalog import FEATURE_CATALOG, validate_features
from .data import load_wikipedia, make_splits
from .experiments import ENGINE, load_config
from .feature_engineering import compute_causal_features
from .tgn_adapter import seed_everything


def setup_environment(seed: int | None = None) -> dict:
    """Create local runtime directories, set seeds, and report the active stack."""
    config = load_config()
    actual_seed = int(config["model"]["seed"] if seed is None else seed)
    seed_everything(actual_seed)
    for directory in ("data", ".cache/tgn"):
        Path(directory).mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch

    return {
        "status": "ready",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "seed": actual_seed,
    }


def load_dataset(
    data_dir: str | Path = "data",
    download: bool = True,
    force_preprocess: bool = False,
) -> dict:
    dataset = load_wikipedia(data_dir, download=download, force_preprocess=force_preprocess)
    ENGINE.set_dataset(dataset)
    return get_dataset_summary()


def _require_dataset():
    if ENGINE.dataset is None:
        raise RuntimeError("Dataset is not loaded. Call load_dataset() first.")
    return ENGINE.dataset


def get_dataset_summary() -> dict:
    dataset = _require_dataset()
    frame = dataset.interactions
    splits = make_splits(frame)
    return {
        "dataset": "JODIE Wikipedia",
        "number_of_interactions": len(frame),
        "number_of_source_nodes": int(frame["u"].nunique()),
        "number_of_destination_nodes": int(frame["i"].nunique()),
        "time_span": {
            "start": float(frame["ts"].min()),
            "end": float(frame["ts"].max()),
            "duration_seconds": float(frame["ts"].max() - frame["ts"].min()),
        },
        "original_edge_feature_dimensionality": int(dataset.edge_features.shape[1]),
        "chronological_split_sizes": {
            "train": splits.train.n_interactions,
            "validation": splits.validation.n_interactions,
            "test": splits.test.n_interactions,
        },
        "split_boundaries": {
            "validation_time": splits.validation_time,
            "test_time": splits.test_time,
        },
    }


def get_feature_catalog() -> dict:
    return copy.deepcopy(FEATURE_CATALOG)


def get_baseline_config() -> dict:
    config = load_config()
    return copy.deepcopy(config)


def run_baseline(mode: str = "fast", seed: int | None = None) -> dict:
    return ENGINE.run("TGN baseline", [], mode, seed, None, is_baseline=True)


def run_experiment(
    experiment_name: str,
    selected_features: list[str],
    seed: int | None = None,
    hypothesis: str | None = None,
    mode: str = "fast",
) -> dict:
    return ENGINE.run(
        experiment_name, selected_features, mode, seed, hypothesis, is_baseline=False
    )


def get_experiment_history() -> pd.DataFrame:
    return ENGINE.history_frame()


def compare_experiments() -> pd.DataFrame:
    history = ENGINE.history_frame()
    if history.empty:
        return history
    return history.sort_values(
        ["test_ap", "test_roc_auc"], ascending=False, kind="stable"
    ).reset_index(drop=True)


def get_best_experiment(metric: str = "ap") -> dict:
    column = {"ap": "test.ap", "roc_auc": "test.roc_auc"}.get(metric)
    if column is None:
        raise ValueError("metric must be 'ap' or 'roc_auc'")
    if not ENGINE.history:
        raise RuntimeError("No experiments have been run")
    section, key = column.split(".")
    best = max(ENGINE.history, key=lambda item: item[section][key])
    return copy.deepcopy(best)


def get_experiment_details(experiment_id: str) -> dict:
    return ENGINE.details(experiment_id)


def inspect_features(feature_names: list[str], n_examples: int = 10) -> pd.DataFrame:
    names = validate_features(feature_names)
    if n_examples <= 0:
        raise ValueError("n_examples must be positive")
    dataset = _require_dataset()
    config = load_config()
    values = compute_causal_features(
        dataset.interactions,
        float(config["dataset"]["recent_window_seconds"]),
        names,
    ).values
    columns = dataset.interactions[["u", "i", "ts"]].rename(
        columns={"u": "source", "i": "destination", "ts": "timestamp"}
    )
    return pd.concat([columns, values], axis=1).head(n_examples).reset_index(drop=True)


def plot_metric_comparison(metric: str = "ap"):
    if metric not in {"ap", "roc_auc"}:
        raise ValueError("metric must be 'ap' or 'roc_auc'")
    history = ENGINE.history_frame()
    if history.empty:
        raise RuntimeError("No experiments have been run")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(history) * 1.4), 4))
    x = range(len(history))
    ax.bar(x, history[f"test_{metric}"], label=f"Test {metric.upper()}")
    ax.set_xticks(list(x), history["experiment_name"], rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"TGN experiment comparison — test {metric.upper()}")
    ax.legend()
    fig.tight_layout()
    return fig
