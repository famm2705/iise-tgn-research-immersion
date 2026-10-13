# Research Immersion Activity Specification

This backend supports controlled temporal-link-prediction experiments on the
JODIE Wikipedia interaction dataset using the original Twitter Research TGN
implementation. The independent variable is the subset of approved engineered
features appended to the original interaction vector. Dataset ordering, split,
negative sampling, model configuration, training, evaluation, and seed remain
fixed within a runtime mode.

For an event `(u, v, t)`, engineered features use only events whose timestamp is
strictly less than `t`. Events sharing a timestamp are featurized before any of
them update history.

The consolidated authoritative contract is
`IISE Research Immersion — TGN Experimental Backend Contract.md`.
