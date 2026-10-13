# IISE Research Immersion — TGN Experimental Backend Contract

**Contract Version:** 1.0\
**Primary implementation owner:** Codex\
**Student-facing notebook owner:** Google Colab / Gemini\
**Source model:** `twitter-research/tgn`

---

# 1. Purpose

Build a reliable experimental backend for an undergraduate research-immersion activity based on the original Temporal Graph Network implementation released by Twitter Research.

The backend must allow a Google Colab notebook to:

1. download/load the approved temporal graph dataset;
2. initialize and train the original TGN architecture;
3. run a fixed baseline experiment;
4. add selected interpretable temporal and graph-derived features to the original interaction features;
5. rerun comparable experiments;
6. calculate and return consistent performance metrics;
7. log all experiments;
8. expose interpretable information that a student can use to formulate and evaluate hypotheses;
9. prevent temporal data leakage;
10. remain simple enough that the Colab notebook does not need to implement ML infrastructure.

The notebook will later provide the educational layer. This repository is responsible only for the reliable experimental engine.

---

# 2. Approved Dataset

## Dataset

**JODIE Wikipedia temporal interaction dataset**

The dataset represents interactions between Wikipedia editors and pages.

Each interaction contains:

- source user/editor;
- destination Wikipedia page;
- timestamp;
- label supplied by the dataset;
- original interaction feature vector.

The original interaction features must remain intact.

## Why Wikipedia

Wikipedia is selected instead of Reddit because:

- it is directly supported by the original TGN repository;
- it uses the same temporal interaction structure expected by TGN;
- it is substantially smaller than Reddit and therefore more appropriate for repeated Google Colab experiments;
- it includes meaningful interaction attributes;
- it allows direct reproduction of the original TGN link-prediction workflow.

---

# 3. Primary Experimental Task

The primary experiment is:

**Temporal link prediction.**

Given the interaction history available before time `t`, the model predicts whether an interaction between two nodes is likely to occur.

The backend must preserve the original TGN link-prediction logic as closely as practical.

The educational research question is:

> Can explicitly engineered temporal and structural graph features improve a TGN's prediction of future interactions beyond the original interaction features?

---

# 4. Baseline Definition

The baseline must use:

- the original Wikipedia interaction dataset;
- the original edge feature vector supplied with the dataset;
- TGN with memory enabled;
- a fixed model configuration;
- the same chronological train/validation/test split for all experiments;
- the same random seed unless explicitly changed by the research team.

The baseline configuration must be stored centrally and must not be editable by students.

Example conceptual configuration:

- memory: enabled
- embedding module: graph attention
- message aggregator: default TGN baseline
- temporal neighbors: fixed
- attention heads: fixed
- learning rate: fixed
- number of layers: fixed
- random seed: fixed

Exact values must be documented in `experiment_config.yaml`.

---

# 5. Experimental Feature Architecture

The original Wikipedia edge features form the fixed baseline.

Engineered features are appended to the existing edge feature vector.

For an experiment:

`experiment_edge_features = original_edge_features + selected_engineered_features`

The original edge features must never be silently removed or changed.

---

# 6. Approved Engineered Features

The backend must expose exactly two student-facing feature categories.

## A. Temporal Features

Temporal features describe **when and how frequently interactions occur**.

They may depend on historical timestamps but must not directly describe graph topology.

Initial approved catalog:

### `time_since_source_event`

Elapsed time between the current event and the source node's most recent previous interaction.

### `time_since_destination_event`

Elapsed time between the current event and the destination node's most recent previous interaction.

### `source_events_recent`

Number of source-node interactions within a predefined recent time window.

### `destination_events_recent`

Number of destination-node interactions within the same predefined recent time window.

### `pair_events_recent`

Number of previous interactions between the same source-destination pair within the predefined time window.

The window size must be configurable centrally but fixed within an experimental comparison.

---

## B. Graph Features

Graph features describe the topology of the temporal graph snapshot existing immediately before the current interaction.

They must not use future edges.

Initial approved catalog:

### `source_degree`

Number of unique destination nodes connected to the source before the current interaction.

### `destination_degree`

Number of unique source nodes connected to the destination before the current interaction.

### `prior_edge_exists`

Binary indicator showing whether the source and destination have interacted previously.

### `source_two_hop_reach`

Number of unique source-side nodes reachable through the source's previously connected destination nodes.

### `destination_two_hop_reach`

Number of unique destination-side nodes reachable through the destination's previously connected source nodes.

---

# 7. Temporal Causality Requirement

This requirement is mandatory.

For an interaction occurring at timestamp `t`, every engineered feature must be calculated exclusively from interactions with timestamp:

`timestamp < t`

The current interaction must not be incorporated before its feature values are calculated.

Validation or test interactions must never influence features associated with earlier events.

The implementation must contain automated leakage tests.

---

# 8. Feature Catalog API

The backend must expose a machine-readable feature catalog.

Required public function:

```python
get_feature_catalog()
```

Expected structure:

```python
{
    "temporal": {
        "time_since_source_event": {
            "description": "...",
            "interpretation": "...",
            "uses_future_information": False
        }
    },
    "graph": {
        "source_degree": {
            "description": "...",
            "interpretation": "...",
            "uses_future_information": False
        }
    }
}
```

Descriptions must correspond exactly with `feature_catalog.md`.

---

# 9. Student Feature Selection

Students do not modify feature-generation code.

They provide only a list:

```python
selected_features = [
    "time_since_source_event",
    "source_events_recent",
    "source_degree"
]
```

The backend must validate the request.

Unknown features must raise a clear error.

Example:

```text
Unknown feature: source_activity_magic
Choose a feature returned by get_feature_catalog().
```

There is no hard limit on the number of experiments.

The backend should allow any subset of approved engineered features.

---

# 10. Experiment Workflow

The required programmatic workflow is:

```python
setup_environment()

load_dataset()

get_dataset_summary()

get_feature_catalog()

run_baseline()

run_experiment(
    experiment_name=...,
    selected_features=[...]
)

get_experiment_history()

compare_experiments()
```

The Colab notebook must not need to know how TGN training, negative sampling, temporal neighbor construction, or feature engineering are implemented.

---

# 11. Public API

The backend must expose at least:

```python
setup_environment()

load_dataset()

get_dataset_summary()

get_feature_catalog()

get_baseline_config()

run_baseline()

run_experiment(
    experiment_name: str,
    selected_features: list[str],
    seed: int | None = None
)

get_experiment_history()

compare_experiments()

get_best_experiment(metric="ap")

get_experiment_details(experiment_id)

plot_metric_comparison(metric="ap")
```

Implementation details may be distributed across modules, but these functions must form a stable notebook-facing interface.

---

# 12. Experiment Result Schema

Each experiment must return a structured object containing at least:

```python
{
    "experiment_id": str,
    "experiment_name": str,

    "selected_features": list[str],
    "temporal_features": list[str],
    "graph_features": list[str],

    "seed": int,

    "train_time_seconds": float,

    "validation": {
        "ap": float,
        "roc_auc": float
    },

    "test": {
        "ap": float,
        "roc_auc": float
    },

    "delta_vs_baseline": {
        "ap": float,
        "roc_auc": float
    }
}
```

If additional valid link-prediction metrics are implemented, they may be included, but AP and ROC-AUC must remain available.

---

# 13. Interpretable Experimental Inputs

For every feature, students must be able to inspect sample values.

Required functionality:

```python
inspect_features(
    feature_names=[...],
    n_examples=10
)
```

Expected output should contain human-readable rows showing:

- source node;
- destination node;
- timestamp;
- selected engineered feature values.

The purpose is to allow students to understand what the model receives.

---

# 14. Dataset Exploration API

Required:

```python
get_dataset_summary()
```

Must provide at least:

- number of interactions;
- number of source nodes;
- number of destination nodes;
- time span;
- original edge feature dimensionality;
- chronological train/validation/test sizes.

Optional but useful:

- interaction frequency distribution;
- source degree distribution;
- destination degree distribution.

---

# 15. Experiment Logging

Every call to `run_experiment()` must append to an experiment registry.

The registry must store:

- experiment ID;
- name;
- selected features;
- configuration;
- seed;
- metrics;
- runtime;
- timestamp;
- source code / contract version if practical.

Student hypothesis text should be accepted optionally:

```python
run_experiment(
    experiment_name="Recent activity hypothesis",
    selected_features=[...],
    hypothesis="..."
)
```

The hypothesis is metadata only and must never affect training.

---

# 16. Reproducibility

The backend must:

- control Python random seed;
- control NumPy random seed;
- control PyTorch random seed;
- preserve chronological data splits;
- record configuration with every experiment;
- avoid accidental changes in baseline hyperparameters.

Repeated runs with the same seed/configuration should produce equivalent results within normal GPU nondeterminism.

---

# 17. Runtime Modes

Implement two modes.

## Student / Fast Mode

Purpose:

- repeated Colab experimentation;
- faster iteration;
- reasonable training time.

May use:

- fewer epochs;
- early stopping;
- reduced training subset only if scientifically defensible.

Any reduction must preserve chronology.

## Reference Mode

Purpose:

- final verification;
- closer reproduction of original TGN settings.

The mode used must be included in experiment results.

---

# 18. TGN Source Integrity

Use the original:

`https://github.com/twitter-research/tgn`

as the model reference.

Do not rewrite TGN from scratch unless technically unavoidable.

Preferred strategy:

- vendor or fork the original implementation;
- wrap it with modern compatibility code;
- preserve core TGN behavior.

Any changes required for modern Python/PyTorch must be documented in:

`docs/tgn_compatibility.md`

---

# 19. Dependency Strategy

The original TGN repository uses old package versions.

The implementation must determine whether:

1. the original dependencies can safely run in current Colab;
2. minimal compatibility patches are required;
3. a dedicated compatible environment is necessary.

Do not silently substitute a different TGN implementation.

The final README must state:

- tested Python version;
- tested PyTorch version;
- whether GPU is required;
- expected Colab setup.

---

# 20. Repository Structure

Preferred structure:

```text
research-immersion-tgn/
│
├── README.md
├── activity_spec.md
├── feature_catalog.md
├── api_contract.md
├── acceptance_tests.md
├── experiment_config.yaml
├── requirements.txt
│
├── tgn/
│   └── original/adapted Twitter TGN source
│
├── src/
│   ├── data.py
│   ├── feature_engineering.py
│   ├── tgn_adapter.py
│   ├── experiments.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── api.py
│
├── tests/
│   ├── test_data.py
│   ├── test_temporal_features.py
│   ├── test_graph_features.py
│   ├── test_no_leakage.py
│   ├── test_tgn_adapter.py
│   └── test_experiment_api.py
│
└── notebooks/
    └── backend_smoke_test.ipynb
```

The student-facing notebook is not the primary Codex deliverable.

---

# 21. Required Automated Tests

At minimum:

### Data

- dataset downloads successfully;
- expected columns exist;
- timestamps are sortable;
- TGN preprocessing produces valid arrays.

### Feature correctness

For a small synthetic temporal graph:

- temporal features equal manually verified values;
- graph features equal manually verified values.

### Leakage

For event `i`:

- feature values do not change when events after `i` are modified or removed.

This test is mandatory.

### Baseline

- baseline can train;
- baseline produces finite AP/AUC values.

### Feature experiment

- baseline + one temporal feature trains;
- baseline + one graph feature trains;
- baseline + mixed features trains.

### Experiment history

- experiments are correctly logged;
- baseline comparison is calculated correctly.

---

# 22. Acceptance Criteria

The backend is complete only when all of the following are true:

1. A clean environment can install dependencies.
2. Wikipedia data can be downloaded/preprocessed automatically.
3. Original TGN can be initialized.
4. Baseline training completes successfully.
5. Baseline AP and ROC-AUC are produced.
6. At least one engineered temporal feature can be appended to TGN edge inputs.
7. At least one engineered graph feature can be appended.
8. Mixed temporal + graph features can be used.
9. Feature values are causally computed with no future leakage.
10. Student-selected feature names are validated.
11. Each experiment generates structured results.
12. Each experiment is compared against the same baseline.
13. Experiment history persists during the notebook session.
14. Students can inspect actual engineered feature values.
15. Fast mode works in Google Colab.
16. Automated tests pass.
17. README documents the complete setup.
18. Notebook-facing API is stable and documented.

---

# 23. Non-Goals

Codex must not:

- design the student lesson;
- write final student explanations;
- generate hypotheses for students;
- change the approved feature definitions without updating the contract;
- substitute XGBoost or another model for TGN;
- use random train/test splitting;
- use future events to construct features;
- silently change the experimental task;
- optimize hyperparameters independently for every feature set.

The objective is feature experimentation under a controlled TGN configuration, not automated hyperparameter search.

---

# 24. Final Backend Deliverables

Codex must deliver:

1. working repository;
2. reproducible environment;
3. TGN adapter;
4. Wikipedia dataset loader/preprocessor;
5. temporal feature engine;
6. graph feature engine;
7. baseline experiment;
8. student-selectable experiment runner;
9. metric generation;
10. experiment logger;
11. interpretable feature inspection;
12. automated tests;
13. backend smoke-test notebook;
14. complete README;
15. compatibility/change log for modifications to the original TGN implementation.
