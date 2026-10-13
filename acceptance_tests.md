# Acceptance Tests

Acceptance requires automated coverage of dataset preprocessing and loading,
manually verified temporal and graph features, strict no-future leakage,
baseline execution, temporal/graph/mixed feature experiments, invalid feature
validation, registry and metric comparisons, and deterministic configuration.

The full suite is run with `pytest`. Network/full-dataset validation is marked
`integration`; the default suite uses a deterministic miniature dataset while
executing the same TGN adapter.
