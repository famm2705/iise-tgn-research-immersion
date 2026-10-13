"""Experiment orchestration and in-session registry."""

from __future__ import annotations

import copy
import datetime as dt
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .catalog import feature_category, validate_features
from .data import DatasetBundle, chronological_prefix, make_splits
from .feature_engineering import append_scaled_features, compute_causal_features
from .tgn_adapter import train_tgn


CONFIG_PATH = Path(__file__).resolve().parents[1] / "experiment_config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class ExperimentEngine:
    def __init__(self) -> None:
        self.dataset: DatasetBundle | None = None
        self.history: list[dict] = []
        self._feature_cache: dict[tuple[str, int], pd.DataFrame] = {}

    def set_dataset(self, dataset: DatasetBundle) -> None:
        self.dataset = dataset
        self.history.clear()
        self._feature_cache.clear()

    def _prepared(self, mode: str) -> tuple[DatasetBundle, dict, object]:
        if self.dataset is None:
            raise RuntimeError("Dataset is not loaded. Call load_dataset() first.")
        config = load_config()
        if mode not in config["modes"]:
            raise ValueError(f"Unknown mode: {mode}. Choose 'fast' or 'reference'.")
        working = chronological_prefix(
            self.dataset, config["modes"][mode]["max_interactions"]
        )
        splits = make_splits(working.interactions)
        return working, config, splits

    def _features(self, working: DatasetBundle, config: dict) -> pd.DataFrame:
        key = (str(working.data_dir), len(working.interactions))
        if key not in self._feature_cache:
            self._feature_cache[key] = compute_causal_features(
                working.interactions,
                float(config["dataset"]["recent_window_seconds"]),
            ).values
        return self._feature_cache[key]

    def run(
        self,
        experiment_name: str,
        selected_features: list[str],
        mode: str,
        seed: int | None,
        hypothesis: str | None,
        is_baseline: bool = False,
    ) -> dict:
        if not experiment_name or not experiment_name.strip():
            raise ValueError("experiment_name must be a non-empty string")
        names = validate_features(selected_features)
        if is_baseline and names:
            raise ValueError("A baseline cannot contain engineered features")
        if not is_baseline and self._baseline_for(mode) is None:
            raise RuntimeError(
                f"No {mode!r} baseline is registered. Run run_baseline(mode={mode!r}) first."
            )
        working, full_config, splits = self._prepared(mode)
        actual_seed = int(full_config["model"]["seed"] if seed is None else seed)
        model_config = copy.deepcopy(full_config["model"])
        mode_config = copy.deepcopy(full_config["modes"][mode])
        if names:
            engineered = self._features(working, full_config)
            edge_features, scaling = append_scaled_features(
                working.edge_features,
                engineered,
                names,
                working.interactions["idx"].to_numpy(),
                splits.train_row_mask,
            )
        else:
            edge_features = working.edge_features.astype(np.float32, copy=True)
            scaling = {}
        started = time.perf_counter()
        metrics = train_tgn(
            splits=splits,
            node_features=working.node_features,
            edge_features=edge_features,
            config=model_config,
            epochs=int(mode_config["epochs"]),
            seed=actual_seed,
        )
        runtime = time.perf_counter() - started
        baseline = self._baseline_for(mode)
        if is_baseline:
            delta = {"ap": 0.0, "roc_auc": 0.0}
        else:
            delta = {
                "ap": metrics.test_ap - baseline["test"]["ap"],
                "roc_auc": metrics.test_roc_auc - baseline["test"]["roc_auc"],
            }
        temporal = [name for name in names if feature_category(name) == "temporal"]
        graph = [name for name in names if feature_category(name) == "graph"]
        result = {
            "experiment_id": uuid.uuid4().hex,
            "experiment_name": experiment_name.strip(),
            "hypothesis": hypothesis,
            "is_baseline": is_baseline,
            "selected_features": names,
            "temporal_features": temporal,
            "graph_features": graph,
            "feature_categories": sorted({feature_category(name) for name in names}),
            "seed": actual_seed,
            "mode": mode,
            "train_time_seconds": float(runtime),
            "validation": {
                "ap": metrics.validation_ap,
                "roc_auc": metrics.validation_roc_auc,
            },
            "test": {"ap": metrics.test_ap, "roc_auc": metrics.test_roc_auc},
            "delta_vs_baseline": delta,
            "configuration": {
                "model": model_config,
                "mode": mode_config,
                "dataset": copy.deepcopy(full_config["dataset"]),
                "effective_interactions": len(working.interactions),
                "split_seed": 2020,
                "engineered_feature_scaling": scaling,
            },
            "epochs_trained": metrics.epochs_trained,
            "device": metrics.device,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "contract_version": full_config["contract_version"],
        }
        self.history.append(result)
        return copy.deepcopy(result)

    def _baseline_for(self, mode: str) -> dict | None:
        return next(
            (item for item in reversed(self.history) if item["is_baseline"] and item["mode"] == mode),
            None,
        )

    def history_frame(self) -> pd.DataFrame:
        rows = []
        for item in self.history:
            rows.append(
                {
                    "experiment_id": item["experiment_id"],
                    "experiment_name": item["experiment_name"],
                    "mode": item["mode"],
                    "seed": item["seed"],
                    "selected_features": list(item["selected_features"]),
                    "feature_categories": list(item["feature_categories"]),
                    "train_time_seconds": item["train_time_seconds"],
                    "validation_ap": item["validation"]["ap"],
                    "validation_roc_auc": item["validation"]["roc_auc"],
                    "test_ap": item["test"]["ap"],
                    "test_roc_auc": item["test"]["roc_auc"],
                    "delta_ap": item["delta_vs_baseline"]["ap"],
                    "delta_roc_auc": item["delta_vs_baseline"]["roc_auc"],
                }
            )
        return pd.DataFrame(rows)

    def details(self, experiment_id: str) -> dict:
        for item in self.history:
            if item["experiment_id"] == experiment_id:
                return copy.deepcopy(item)
        raise KeyError(f"Unknown experiment_id: {experiment_id}")


ENGINE = ExperimentEngine()
