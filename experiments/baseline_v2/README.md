# Frozen baseline v2

This directory is an immutable snapshot of World Cup Forecaster experiment
`wci-20260620T171813Z`.

## Contract

- Source-results SHA-256:
  `dacd6dcf2ae54a9ada1dcdac4d7279ba483576f77ea5ed5983b76f08f70b5d78`
- Mean World Cup walk-forward log loss: `0.9596465218`
- Historical-prior log loss: `1.0753408958`
- Three-way outcome accuracy: `0.578125`
- Adaptive conformal coverage: `0.88671875`
- Frozen 2026 blind accuracy after 32 results: `0.59375`
- Frozen 2026 blind log loss after 32 results: `0.9551303897`
- Training window: 28 years
- Seed: 2026

The snapshot contains the fitted model, evaluation predictions, benchmark ladder, feature ablation,
Türkiye blind audit, publication figures, and SHA-256 checksums. New model versions must be compared
against this baseline on the same folds and metrics. Files in this directory must not be overwritten.

`results_snapshot.csv` is the exact source snapshot used by the frozen experiment.
