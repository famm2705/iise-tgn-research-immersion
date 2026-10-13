# Original TGN Architecture Audit

Audited upstream repository: [`twitter-research/tgn`](https://github.com/twitter-research/tgn),
commit `d55bbe678acabb9fc3879c408fd1f2e15919667c` (vendored at
`tgn/original`). The upstream README and source, rather than a reimplementation
from memory, are the reference for this backend.

## Dataset acquisition and preprocessing

The upstream README points to Stanford SNAP's JODIE datasets. Wikipedia is the
CSV at `https://snap.stanford.edu/jodie/wikipedia.csv` (approximately 560 MB).
`utils/preprocess_data.py` reads each row as:

```text
source_id, destination_id, timestamp, label, feature_0, ..., feature_n
```

The first line is a header. With `--bipartite`, source IDs are shifted by one,
destination IDs are shifted above the source-ID range and then by one, and edge
indices are shifted by one. ID/edge index zero is reserved for padding. The
script writes:

- `ml_wikipedia.csv`: `u`, `i`, `ts`, `label`, `idx`;
- `ml_wikipedia.npy`: interaction features with an all-zero padding row 0;
- `ml_wikipedia_node.npy`: all-zero node features, 172 columns wide.

This backend reproduces those transforms in `src/data.py` and retains the raw
CSV plus processed cache.

## Exact model inputs and edge-feature propagation

`utils/data_processing.py:get_data` loads the processed CSV and the two NumPy
arrays. A `Data` object contains parallel arrays for source IDs, destination IDs,
timestamps, one-based edge indices, and labels.

`model/tgn.py:TGN` converts the full edge feature matrix to `float32` and derives
`n_edge_features` from its second dimension. That dimension propagates to:

1. the raw memory message width:
   `2 * memory_dimension + n_edge_features + time_encoder_dimension`;
2. the identity or MLP message function and memory updater input;
3. each graph embedding layer;
4. the temporal attention key/value width
   `neighbor_features + time_features + n_edge_features`.

At an event, `get_raw_messages` selects `edge_raw_features[edge_idxs]`. During
neighborhood embedding, `GraphEmbedding` selects the historical edge features
returned by `NeighborFinder`. Appending columns to the padded edge-feature
matrix therefore reaches both the event-message and temporal-neighborhood paths
without changing TGN architecture code.

## Chronological splitting and inductive holdout

The upstream split boundaries are the 0.70 and 0.85 timestamp quantiles.
Training uses events with `timestamp <= validation_time`, validation uses
`validation_time < timestamp <= test_time`, and test uses
`timestamp > test_time`. For the self-supervised workflow, Python's RNG is fixed
to 2020, 10% of all unique-node count is sampled from nodes appearing after the
validation boundary, and every training edge incident to those held-out nodes
is removed. Validation/test retain all chronological edges. This backend uses
the same boundary and holdout logic, with a sorted population to remain valid on
modern Python.

The training neighbor finder contains training edges only. Validation/test use
a finder built from the full graph, but `NeighborFinder.find_before` uses
`np.searchsorted(..., cut_time)` with the default left side, so only neighbors
with timestamps strictly earlier than the query time are returned.

## Negative sampling

`RandEdgeSampler` stores the unique observed source IDs and destination IDs. For
each positive event it samples one source and one destination independently and
uniformly; link prediction discards the sampled source and uses the sampled
destination with the positive source. It does not reject an accidentally real
edge. Training uses NumPy's global seeded RNG. Validation and test use resettable
samplers with seeds 0 and 2, respectively. The adapter preserves this behavior.

## AP and ROC-AUC

`evaluation/evaluation.py:eval_edge_prediction` concatenates positive and
negative probabilities for each evaluation batch, constructs equal numbers of
one and zero labels, and calls scikit-learn's `average_precision_score` and
`roc_auc_score`. It returns the unweighted mean of per-batch AP and per-batch
ROC-AUC. Evaluation batch size equals training batch size so memory is advanced
between chronological batches. The adapter preserves these details.

## Original baseline configuration

The upstream `TGN-attn` command is `python train_self_supervised.py --use_memory`.
Its effective defaults are batch size 200, 10 temporal neighbors, two heads, one
graph attention layer, dropout 0.1, learning rate 0.0001, GRU memory updater,
identity message function, last-message aggregator, 172-dimensional memory, and
50 epochs with patience 5. Although the CLI declares `node_dim=100` and
`time_dim=100`, it never passes them to `TGN`; both effective dimensions come
from the 172-column node-feature matrix. The central configuration records the
effective value 172 rather than the unused CLI defaults.

## Current Colab incompatibilities

Upstream pins Python >=3.7 with pandas 1.1.0, PyTorch 1.6.0, and scikit-learn
0.23.1. Current Colab (October 2026) uses Python 3.13; recent published runtimes
use PyTorch 2.11 and Google has announced the move toward 2.14. Those 2020 binary
pins do not provide wheels for current Python. In addition:

- `random.sample(set, ...)` in upstream preprocessing is rejected by modern Python;
- unqualified `Tensor.squeeze()` in temporal attention collapses batch dimension
  when a final batch has one event under current PyTorch;
- current PyTorch warns when converting read-only NumPy views;
- upstream checkpoint files and process-global paths are unsuitable for repeated
  in-notebook experiments.

The compatibility approach and evidence are recorded in
`docs/tgn_compatibility.md`. Colab runtime information was checked against
[Google's runtime release notes](https://colab.research.google.com/notebooks/relnotes.ipynb)
and [runtime-version FAQ](https://research.google.com/colaboratory/runtime-version-faq.html).
