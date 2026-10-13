# TGN Research-Immersion Experimental Backend

This repository wraps the original Twitter Research Temporal Graph Network for
controlled feature-engineering experiments on the JODIE Wikipedia dataset. The
original interaction features are always retained; selected causal temporal and
graph features are appended.

## Colab setup

```python
!git clone https://github.com/famm2705/iise-tgn-research-immersion.git
%cd iise-tgn-research-immersion
!pip install -r requirements.txt
```

Then run:

```python
from src.api import (
    setup_environment, load_dataset, get_dataset_summary, get_feature_catalog,
    run_baseline, run_experiment, get_experiment_history,
    compare_experiments, inspect_features,
)

setup_environment()
load_dataset()  # first call downloads the ~560 MB SNAP CSV and caches preprocessing
baseline = run_baseline(mode="fast")
temporal = run_experiment(
    experiment_name="Temporal experiment",
    selected_features=["time_since_source_event", "source_events_recent"],
    mode="fast",
)
```

All notebook code imports only `src.api`; internal TGN classes are not part of
the public interface.

## Runtime modes

- `fast`: the earliest 20,000 interactions, still chronologically ordered and
  split by the same 70%/15%/15% timestamp-quantile procedure, two epochs, and
  early stopping. Every fast experiment uses exactly this same subset.
- `reference`: all interactions and the upstream 50-epoch/patience-5 setup.

Both modes preserve the upstream inductive training holdout, memory, temporal
neighbor logic, one-negative-per-positive strategy, and batch-averaged AP/AUC.
Mode and effective interaction count are included in every result.

## Causality

For event `(u, v, t)`, features use only events with timestamp strictly less
than `t`. Every same-timestamp event is computed from the same pre-`t` snapshot;
the group updates state only after all its features are emitted. Feature scaling
is fitted on training rows only. Tests verify that changing or deleting future
events cannot alter earlier features.

## Configuration and reproducibility

The fixed configuration is in `experiment_config.yaml`. Python, NumPy, and
PyTorch seeds are controlled and recorded. Validation/test samplers use the
upstream fixed seeds. Students may select approved features and may explicitly
override the experiment seed, but cannot mutate model hyperparameters through
the public API.

The backend is tested on CPU with Python 3.12.14 and PyTorch 2.14.1. A GPU is not
required, but is recommended for reference mode. Current Colab's modern Python
and PyTorch stack is supported; see `docs/tgn_compatibility.md`.

## Tests

```bash
python -m pytest -q
```

The default suite is network-free and executes the actual vendored TGN on a
small deterministic temporal graph. The technical notebook at
`notebooks/backend_smoke_test.ipynb` demonstrates the complete public workflow.

To run the download plus real-Wikipedia baseline integration test:

```bash
RUN_TGN_INTEGRATION=1 python -m pytest tests/test_wikipedia_integration.py -q
```

Set `TGN_DATA_DIR` to an existing processed data directory to reuse its cache.

## Source provenance

`tgn/original` vendors `twitter-research/tgn` commit
`d55bbe678acabb9fc3879c408fd1f2e15919667c` under its Apache-2.0 license. Audit
details are in `docs/tgn_architecture_audit.md`.
