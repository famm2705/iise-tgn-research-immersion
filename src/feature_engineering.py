"""Strictly causal streaming temporal and bipartite-graph features."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .catalog import ALL_FEATURES, validate_features


@dataclass(frozen=True)
class FeatureResult:
    values: pd.DataFrame
    recent_window_seconds: float


def compute_causal_features(
    interactions: pd.DataFrame,
    recent_window_seconds: float,
    feature_names: list[str] | None = None,
) -> FeatureResult:
    """Compute features in timestamp groups so equal-time events cannot see each other."""
    names = validate_features(feature_names or list(ALL_FEATURES))
    required = {"u", "i", "ts"}
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"Missing interaction columns: {', '.join(sorted(missing))}")
    if recent_window_seconds <= 0:
        raise ValueError("recent_window_seconds must be positive")

    n = len(interactions)
    output = {name: np.zeros(n, dtype=np.float64) for name in names}
    if n == 0:
        return FeatureResult(pd.DataFrame(output, index=interactions.index), recent_window_seconds)

    # Stable sorting preserves event order without making it part of the causal state.
    order = np.argsort(interactions["ts"].to_numpy(), kind="stable")
    sources = interactions["u"].to_numpy()
    destinations = interactions["i"].to_numpy()
    timestamps = interactions["ts"].to_numpy(dtype=np.float64)

    last_source: dict[int, float] = {}
    last_destination: dict[int, float] = {}
    recent_source: defaultdict[int, deque[float]] = defaultdict(deque)
    recent_destination: defaultdict[int, deque[float]] = defaultdict(deque)
    recent_pair: defaultdict[tuple[int, int], deque[float]] = defaultdict(deque)
    source_neighbors: defaultdict[int, set[int]] = defaultdict(set)
    destination_neighbors: defaultdict[int, set[int]] = defaultdict(set)

    def prune(queue: deque[float], cutoff: float) -> None:
        while queue and queue[0] < cutoff:
            queue.popleft()

    start = 0
    while start < n:
        t = timestamps[order[start]]
        end = start + 1
        while end < n and timestamps[order[end]] == t:
            end += 1
        group = order[start:end]
        cutoff = t - recent_window_seconds

        # Read state first. No event at t is visible to another event at t.
        for row in group:
            u, v = int(sources[row]), int(destinations[row])
            pair = (u, v)
            if "time_since_source_event" in output:
                output["time_since_source_event"][row] = (
                    t - last_source[u] if u in last_source else 0.0
                )
            if "time_since_destination_event" in output:
                output["time_since_destination_event"][row] = (
                    t - last_destination[v] if v in last_destination else 0.0
                )
            if "source_events_recent" in output:
                prune(recent_source[u], cutoff)
                output["source_events_recent"][row] = len(recent_source[u])
            if "destination_events_recent" in output:
                prune(recent_destination[v], cutoff)
                output["destination_events_recent"][row] = len(recent_destination[v])
            if "pair_events_recent" in output:
                prune(recent_pair[pair], cutoff)
                output["pair_events_recent"][row] = len(recent_pair[pair])
            if "source_degree" in output:
                output["source_degree"][row] = len(source_neighbors[u])
            if "destination_degree" in output:
                output["destination_degree"][row] = len(destination_neighbors[v])
            if "prior_edge_exists" in output:
                output["prior_edge_exists"][row] = float(v in source_neighbors[u])
            if "source_two_hop_reach" in output:
                reachable: set[int] = set()
                for neighbor_v in source_neighbors[u]:
                    reachable.update(destination_neighbors[neighbor_v])
                reachable.discard(u)
                output["source_two_hop_reach"][row] = len(reachable)
            if "destination_two_hop_reach" in output:
                reachable_destinations: set[int] = set()
                for neighbor_u in destination_neighbors[v]:
                    reachable_destinations.update(source_neighbors[neighbor_u])
                reachable_destinations.discard(v)
                output["destination_two_hop_reach"][row] = len(reachable_destinations)

        # Only after all reads at t are complete do these events become history.
        for row in group:
            u, v = int(sources[row]), int(destinations[row])
            pair = (u, v)
            last_source[u] = t
            last_destination[v] = t
            recent_source[u].append(t)
            recent_destination[v].append(t)
            recent_pair[pair].append(t)
            source_neighbors[u].add(v)
            destination_neighbors[v].add(u)
        start = end

    frame = pd.DataFrame(output, index=interactions.index)
    return FeatureResult(frame, recent_window_seconds)


def append_scaled_features(
    original_edge_features: np.ndarray,
    engineered: pd.DataFrame,
    selected_features: list[str],
    edge_indices: np.ndarray,
    training_row_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Append train-fitted standardized features to the padded original matrix."""
    names = validate_features(selected_features)
    if not names:
        return original_edge_features.astype(np.float32, copy=True), {}
    raw = engineered[names].to_numpy(dtype=np.float64)
    training = raw[np.asarray(training_row_mask, dtype=bool)]
    means = training.mean(axis=0)
    scales = training.std(axis=0)
    scales[scales == 0] = 1.0
    scaled = ((raw - means) / scales).astype(np.float32)
    result = np.zeros((original_edge_features.shape[0], original_edge_features.shape[1] + len(names)), dtype=np.float32)
    result[:, : original_edge_features.shape[1]] = original_edge_features
    result[np.asarray(edge_indices, dtype=np.int64), original_edge_features.shape[1] :] = scaled
    metadata = {
        name: {"mean": float(mean), "scale": float(scale)}
        for name, mean, scale in zip(names, means, scales)
    }
    return result, metadata
