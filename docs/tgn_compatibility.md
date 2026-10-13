# TGN Compatibility Record

## Supported and tested stack

Locally tested: Python 3.12.14, PyTorch 2.14.1 (CPU), NumPy 2.5.4, pandas 3.0.6,
and scikit-learn 1.9.1. The requirements intentionally use compatible ranges
instead of the unavailable 2020 binary pins. GPU is optional; reference mode is
substantially faster with a Colab GPU.

## Material changes

### Modern temporal-attention batch shape

- Original behavior: `attn_output.squeeze()` and
  `attn_output_weights.squeeze()` removed every singleton dimension.
- Problem: for a one-event final batch, this also removed the batch dimension,
  causing shape errors in masking or the merge layer on modern PyTorch.
- Change: `tgn/original/model/temporal_attention.py` removes only the known
  singleton query dimensions (`dim=0` and `dim=1`) and retains batch dimension.
- Evidence: automated baseline/temporal/graph/mixed tests use splits smaller
  than batch size, force one-batch evaluation, and complete with finite metrics.

### Modern boolean indexing

- Original behavior: `invalid_neighborhood_mask.squeeze()` indexed the padding
  mask.
- Problem: a one-event batch produces a scalar boolean index.
- Change: squeeze only dimension 1.
- Evidence: the same real-TGN smoke tests exercise nodes with no earlier neighbors.

### Python inductive-node sampling

- Original behavior: `random.sample(test_node_set, k)` sampled a set directly.
- Problem: modern Python requires a sequence population.
- Change: `src/data.py` samples a sorted list using a local RNG seeded with 2020.
- Consistency: size, eligible nodes, train-edge removal, and chronological masks
  are unchanged; sorting makes the split repeatable across processes.

### Paths and in-memory checkpoints

- Original behavior: data paths are process-relative globals and each epoch
  writes model/result files.
- Problem: repeated Colab API calls would collide and leave internal artifacts.
- Change: the adapter accepts arrays and split objects, and retains the best
  state in memory. Model construction, loss, optimizer, memory lifecycle,
  neighbor selection, negative sampling, and metrics remain upstream-equivalent.

### Writable NumPy arrays

- Original behavior: arrays were passed directly to PyTorch.
- Problem: pandas 3 may expose read-only NumPy views, producing undefined-write
  warnings in current PyTorch.
- Change: split arrays are materialized as writable copies.
- Consistency: values and order are unchanged.

### Engineered-feature scaling

This is an experimental-input addition, not a TGN rewrite. Raw engineered values
are standardized using means and standard deviations fitted only on training
rows, then appended after every original Wikipedia feature. Scaling metadata is
recorded in every result. Inspection returns unscaled human-readable values.

## Verification

`pytest` covers original-compatible preprocessing, deterministic chronological
splits, manually checked temporal/graph values, equal-timestamp strictness,
future-event leakage, the original TGN baseline, appended temporal/graph/mixed
features, validation errors, logging, comparisons, and interpretable inspection.
