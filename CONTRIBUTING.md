# Contributing

1. Create a focused branch.
2. Keep feature computation chronological and document source availability.
3. Add tests for leakage boundaries and public contracts.
4. Run:

```bash
ruff check .
pytest --cov=worldcup_predictor
python -m build
```

New model or data families must use the existing World Cup cutoffs and report log loss, accuracy,
calibration, coverage and per-edition behavior. A richer input is not considered an upgrade until it
passes the promotion gate.
