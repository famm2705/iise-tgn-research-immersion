"""Thin modern training adapter around the vendored Twitter Research TGN."""

from __future__ import annotations

import copy
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import DatasetSplits, InteractionData


UPSTREAM_ROOT = Path(__file__).resolve().parents[1] / "tgn" / "original"
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from model.tgn import TGN  # noqa: E402
from utils.data_processing import compute_time_statistics  # noqa: E402
from utils.utils import RandEdgeSampler, get_neighbor_finder  # noqa: E402


@dataclass(frozen=True)
class TrainingMetrics:
    validation_ap: float
    validation_roc_auc: float
    test_ap: float
    test_roc_auc: float
    epochs_trained: int
    device: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # older supported PyTorch releases lack warn_only
        torch.use_deterministic_algorithms(True)


def _evaluate(
    model: TGN,
    sampler: RandEdgeSampler,
    data: InteractionData,
    n_neighbors: int,
    batch_size: int,
) -> tuple[float, float]:
    if data.n_interactions == 0:
        raise ValueError("Evaluation split is empty")
    sampler.reset_random_state()
    aps: list[float] = []
    aucs: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, data.n_interactions, batch_size):
            end = min(data.n_interactions, start + batch_size)
            size = end - start
            _, negatives = sampler.sample(size)
            positive, negative = model.compute_edge_probabilities(
                data.sources[start:end],
                data.destinations[start:end],
                negatives,
                data.timestamps[start:end],
                data.edge_idxs[start:end],
                n_neighbors,
            )
            scores = np.concatenate(
                [positive.detach().cpu().numpy().reshape(-1), negative.detach().cpu().numpy().reshape(-1)]
            )
            labels = np.concatenate([np.ones(size), np.zeros(size)])
            aps.append(float(average_precision_score(labels, scores)))
            aucs.append(float(roc_auc_score(labels, scores)))
    return float(np.mean(aps)), float(np.mean(aucs))


def train_tgn(
    splits: DatasetSplits,
    node_features: np.ndarray,
    edge_features: np.ndarray,
    config: dict,
    epochs: int,
    seed: int,
    device: str | None = None,
) -> TrainingMetrics:
    """Run the original transductive self-supervised workflow with in-memory checkpoints."""
    seed_everything(seed)
    selected_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(selected_device)
    train_finder = get_neighbor_finder(splits.train, config["uniform_neighbors"])
    full_finder = get_neighbor_finder(splits.full, config["uniform_neighbors"])
    train_sampler = RandEdgeSampler(splits.train.sources, splits.train.destinations)
    validation_sampler = RandEdgeSampler(splits.full.sources, splits.full.destinations, seed=0)
    test_sampler = RandEdgeSampler(splits.full.sources, splits.full.destinations, seed=2)
    mean_src, std_src, mean_dst, std_dst = compute_time_statistics(
        splits.full.sources, splits.full.destinations, splits.full.timestamps
    )
    # Avoid divide-by-zero in the upstream time normalization on tiny fixtures.
    std_src = float(std_src) if std_src > 0 else 1.0
    std_dst = float(std_dst) if std_dst > 0 else 1.0
    model = TGN(
        neighbor_finder=train_finder,
        node_features=node_features,
        edge_features=edge_features,
        device=torch_device,
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        dropout=config["dropout"],
        use_memory=True,
        message_dimension=config["message_dimension"],
        memory_dimension=config["memory_dimension"],
        memory_update_at_start=not config["memory_update_at_end"],
        embedding_module_type=config["embedding_module"],
        message_function=config["message_function"],
        aggregator_type=config["aggregator"],
        memory_updater_type=config["memory_updater"],
        n_neighbors=config["n_neighbors"],
        mean_time_shift_src=mean_src,
        std_time_shift_src=std_src,
        mean_time_shift_dst=mean_dst,
        std_time_shift_dst=std_dst,
    ).to(torch_device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    batch_size = config["batch_size"]
    best_ap = -math.inf
    best_epoch = 0
    stale_epochs = 0
    best_state = None
    best_validation = (math.nan, math.nan)

    for epoch in range(epochs):
        model.memory.__init_memory__()
        model.set_neighbor_finder(train_finder)
        model.train()
        for start in range(0, splits.train.n_interactions, batch_size):
            end = min(splits.train.n_interactions, start + batch_size)
            size = end - start
            _, negatives = train_sampler.sample(size)
            positive, negative = model.compute_edge_probabilities(
                splits.train.sources[start:end],
                splits.train.destinations[start:end],
                negatives,
                splits.train.timestamps[start:end],
                splits.train.edge_idxs[start:end],
                config["n_neighbors"],
            )
            positive_labels = torch.ones(size, dtype=torch.float32, device=torch_device)
            negative_labels = torch.zeros(size, dtype=torch.float32, device=torch_device)
            loss = criterion(positive.reshape(-1), positive_labels) + criterion(
                negative.reshape(-1), negative_labels
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.memory.detach_memory()

        model.set_neighbor_finder(full_finder)
        validation = _evaluate(
            model, validation_sampler, splits.validation, config["n_neighbors"], batch_size
        )
        if validation[0] > best_ap + 1e-10:
            best_ap = validation[0]
            best_epoch = epoch
            stale_epochs = 0
            best_validation = validation
            best_state = copy.deepcopy(model.state_dict())
            best_messages = model.memory.backup_memory()[2]
        else:
            stale_epochs += 1
        if stale_epochs >= config["patience"]:
            break

    if best_state is None:
        raise RuntimeError("TGN training did not produce a checkpoint")
    model.load_state_dict(best_state)
    # state_dict contains memory tensors but not queued raw messages.
    memory_backup = model.memory.backup_memory()
    model.memory.restore_memory((memory_backup[0], memory_backup[1], best_messages))
    model.set_neighbor_finder(full_finder)
    test_ap, test_auc = _evaluate(
        model, test_sampler, splits.test, config["n_neighbors"], batch_size
    )
    metrics = TrainingMetrics(
        validation_ap=float(best_validation[0]),
        validation_roc_auc=float(best_validation[1]),
        test_ap=float(test_ap),
        test_roc_auc=float(test_auc),
        epochs_trained=best_epoch + 1,
        device=selected_device,
    )
    for value in (
        metrics.validation_ap,
        metrics.validation_roc_auc,
        metrics.test_ap,
        metrics.test_roc_auc,
    ):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(f"Invalid evaluation metric: {value}")
    return metrics
