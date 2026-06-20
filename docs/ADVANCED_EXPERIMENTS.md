# Advanced Experiment Protocol

This stage asks whether more data or more compute improves the frozen v2 forecast.

## Controlled families

- Environment: travel distance, timezone displacement, elevation, temperature, apparent
  temperature, humidity, precipitation, wind, gusts and pressure.
- Event history: xG balance, shots, shot quality, possession share, passing, pressure,
  counterpressure, set-piece xG, transition xG and final-third entries.
- Announced lineup: XI continuity, changes, prior starts, debut share, goalkeeper change and
  positional shape.
- Current squad: age, height, caps, goals, club dispersion, domestic share, top-five-league share,
  positional depth and concentration.
- Architecture: calibrated hybrid, CUDA XGBoost and a CUDA residual tabular network.

## Leakage boundaries

ERA5 weather is retrospective reanalysis and is labeled `oracle only`. It cannot be promoted to live
forecasting. The production weather adapter must use an archived forecast run timestamped before
kickoff.

Starting-XI features are joined only to the exact fixture. They belong to a late model that runs after
official lineup publication. They are never forward-filled into an earlier forecast.

Event profiles use an as-of join with exact matches disabled. A match can update a team's profile only
for later fixtures.

## Promotion rule

A normal pre-match family needs all five World Cup folds, lower mean log loss, no accuracy regression,
acceptable calibration, adequate source coverage, and demonstrated 2026 availability. One-fold gains
are research signals, not production upgrades.

## Product-wide integration

The advanced stage is propagated across all dashboard workspaces and publication graphics. Each
surface distinguishes immutable blind evidence, promoted production probabilities, and staged
research. Türkiye Focus additionally exposes the complete official squad, player experience,
international scoring, club geography, roster percentiles, and explicit strengths/risks.
