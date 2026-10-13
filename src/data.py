"""JODIE Wikipedia acquisition, original-compatible preprocessing, and splits."""

from __future__ import annotations

import os
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


WIKIPEDIA_URL = "https://snap.stanford.edu/jodie/wikipedia.csv"


@dataclass(frozen=True)
class DatasetBundle:
    interactions: pd.DataFrame
    edge_features: np.ndarray
    node_features: np.ndarray
    data_dir: Path


@dataclass(frozen=True)
class InteractionData:
    sources: np.ndarray
    destinations: np.ndarray
    timestamps: np.ndarray
    edge_idxs: np.ndarray
    labels: np.ndarray

    @property
    def n_interactions(self) -> int:
        return len(self.sources)


@dataclass(frozen=True)
class DatasetSplits:
    full: InteractionData
    train: InteractionData
    validation: InteractionData
    test: InteractionData
    train_row_mask: np.ndarray
    validation_row_mask: np.ndarray
    test_row_mask: np.ndarray
    validation_time: float
    test_time: float


def _to_data(frame: pd.DataFrame, mask: np.ndarray | None = None) -> InteractionData:
    selected = frame if mask is None else frame.loc[mask]
    return InteractionData(
        selected["u"].to_numpy(dtype=np.int64, copy=True),
        selected["i"].to_numpy(dtype=np.int64, copy=True),
        selected["ts"].to_numpy(dtype=np.float64, copy=True),
        selected["idx"].to_numpy(dtype=np.int64, copy=True),
        selected["label"].to_numpy(dtype=np.float64, copy=True),
    )


def download_wikipedia(destination: Path, url: str = WIKIPEDIA_URL) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()
    return destination


def preprocess_raw_csv(raw_path: Path, data_dir: Path) -> DatasetBundle:
    """Match upstream `utils/preprocess_data.py --bipartite`."""
    # SNAP's header has five conceptual names, while each feature vector is
    # expanded into 172 comma-separated fields. The upstream parser skips the
    # header and reads every data row positionally.
    raw = pd.read_csv(raw_path, header=None, skiprows=1)
    if raw.shape[1] < 5:
        raise ValueError("Wikipedia CSV must contain u, i, timestamp, label, and features")
    base = raw.iloc[:, :4].copy()
    base.columns = ["u", "i", "ts", "label"]
    features = raw.iloc[:, 4:].to_numpy(dtype=np.float32)
    base["u"] = base["u"].astype(np.int64)
    base["i"] = base["i"].astype(np.int64)
    if base["u"].nunique() != base["u"].max() - base["u"].min() + 1:
        raise ValueError("Source IDs must be contiguous, as required by upstream TGN")
    if base["i"].nunique() != base["i"].max() - base["i"].min() + 1:
        raise ValueError("Destination IDs must be contiguous, as required by upstream TGN")
    upper_u = int(base["u"].max()) + 1
    base["i"] = base["i"] + upper_u + 1
    base["u"] = base["u"] + 1
    base["idx"] = np.arange(1, len(base) + 1, dtype=np.int64)
    edge_features = np.vstack(
        [np.zeros((1, features.shape[1]), dtype=np.float32), features]
    )
    max_node = int(max(base["u"].max(), base["i"].max()))
    # Upstream fixes node input width at 172 for Wikipedia/JODIE.
    node_features = np.zeros((max_node + 1, 172), dtype=np.float32)
    data_dir.mkdir(parents=True, exist_ok=True)
    base.to_csv(data_dir / "ml_wikipedia.csv", index=False)
    np.save(data_dir / "ml_wikipedia.npy", edge_features)
    np.save(data_dir / "ml_wikipedia_node.npy", node_features)
    return DatasetBundle(base, edge_features, node_features, data_dir)


def load_wikipedia(
    data_dir: str | Path = "data",
    download: bool = True,
    force_preprocess: bool = False,
) -> DatasetBundle:
    directory = Path(data_dir).expanduser().resolve()
    processed_csv = directory / "ml_wikipedia.csv"
    edge_path = directory / "ml_wikipedia.npy"
    node_path = directory / "ml_wikipedia_node.npy"
    if not force_preprocess and all(path.exists() for path in (processed_csv, edge_path, node_path)):
        return DatasetBundle(
            pd.read_csv(processed_csv),
            np.load(edge_path),
            np.load(node_path),
            directory,
        )
    raw_path = directory / "wikipedia.csv"
    if not raw_path.exists():
        if not download:
            raise FileNotFoundError(
                f"Dataset not found at {raw_path}. Call load_dataset(download=True)."
            )
        download_wikipedia(raw_path)
    return preprocess_raw_csv(raw_path, directory)


def make_splits(interactions: pd.DataFrame, split_seed: int = 2020) -> DatasetSplits:
    if interactions.empty:
        raise ValueError("Cannot split an empty dataset")
    timestamps = interactions["ts"].to_numpy(dtype=np.float64)
    val_time, test_time = np.quantile(timestamps, [0.70, 0.85])
    sources = interactions["u"].to_numpy(dtype=np.int64)
    destinations = interactions["i"].to_numpy(dtype=np.int64)
    node_set = set(sources) | set(destinations)
    future_nodes = sorted(
        set(sources[timestamps > val_time]) | set(destinations[timestamps > val_time])
    )
    sample_size = min(len(future_nodes), int(0.1 * len(node_set)))
    held_out = set(random.Random(split_seed).sample(future_nodes, sample_size))
    observed = np.array(
        [u not in held_out and v not in held_out for u, v in zip(sources, destinations)],
        dtype=bool,
    )
    train_mask = (timestamps <= val_time) & observed
    validation_mask = (timestamps > val_time) & (timestamps <= test_time)
    test_mask = timestamps > test_time
    return DatasetSplits(
        full=_to_data(interactions),
        train=_to_data(interactions, train_mask),
        validation=_to_data(interactions, validation_mask),
        test=_to_data(interactions, test_mask),
        train_row_mask=train_mask,
        validation_row_mask=validation_mask,
        test_row_mask=test_mask,
        validation_time=float(val_time),
        test_time=float(test_time),
    )


def chronological_prefix(bundle: DatasetBundle, max_interactions: int | None) -> DatasetBundle:
    if max_interactions is None or len(bundle.interactions) <= max_interactions:
        return bundle
    frame = bundle.interactions.sort_values("ts", kind="stable").iloc[:max_interactions].copy()
    max_edge = int(frame["idx"].max())
    return DatasetBundle(frame, bundle.edge_features[: max_edge + 1], bundle.node_features, bundle.data_dir)
