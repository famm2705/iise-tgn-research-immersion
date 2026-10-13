# Notebook-facing API Contract

The stable facade is `src.api`. It exports `setup_environment`, `load_dataset`,
`get_dataset_summary`, `get_feature_catalog`, `get_baseline_config`,
`run_baseline`, `run_experiment`, `get_experiment_history`,
`compare_experiments`, `get_best_experiment`, `get_experiment_details`,
`inspect_features`, and `plot_metric_comparison`.

Experiment calls return dictionaries containing identity and metadata, selected
feature categories, complete configuration, seed, mode, runtime, validation and
test AP/ROC-AUC, and deltas from the baseline. Tabular inspection and comparison
functions return pandas DataFrames.
