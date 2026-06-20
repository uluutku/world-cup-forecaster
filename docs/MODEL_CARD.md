# Model Card — World Cup Intelligence v2

## Intended use

World Cup Intelligence estimates three-way match outcomes and tournament stage probabilities for
senior men's international football. It is designed for research,
demonstration, scenario analysis, and portfolio evaluation.

It is not intended for automated betting, financial decisions, player selection, or decisions that
affect athletes.

## Model architecture

The production forecast blends:

- regularized multinomial logistic regression;
- histogram gradient boosting;
- two Poisson goal-intensity regressors;
- a fitted Dixon–Coles correction for dependence among low-scoring outcomes.

Blend weights are optimized on the probability simplex using a chronological calibration window.
Temperature scaling is optimized jointly with the blend. Training examples receive tournament
importance and exponential recency weights.

## Feature state

All state is computed sequentially before kickoff. Feature families include dynamic team rating,
rating uncertainty, volatility, form, momentum, attack/defence processes, trend, clean-sheet and
scoring rates, consistency, activity, rest, schedule strength, opponent prestige, major-tournament
form, experience, venue context, and head-to-head history.

## Evaluation protocol

The model is retrained at the start of the 2006, 2010, 2014, 2018, and 2022 World Cups. Test matches
are never used to construct pre-match state, select blend parameters, or calibrate temperature.

Current aggregate results:

| Metric | Result |
|---|---:|
| Log loss | 0.960 |
| Historical-prior log loss | 1.075 |
| Accuracy | 57.8% |
| Brier score | 0.564 |
| Ranked probability score | 0.196 |
| Expected calibration error | 0.101 |

The 2022 fold underperforms the historical-prior baseline. This is reported as a model-risk signal,
not removed as an outlier.

## Uncertainty

- bootstrap intervals quantify evaluation sampling uncertainty;
- adaptive conformal prediction sets quantify outcome-set uncertainty;
- dynamic rating variance captures sparse or inactive teams;
- model-family disagreement highlights epistemic disagreement;
- predictive entropy highlights intrinsically difficult fixtures;
- Population Stability Index reports changing feature distributions.

Adaptive conformal coverage is empirical and can miss its nominal target under temporal shift.

## Staged research findings

- travel and country-centroid timezone proxies regress the five-fold benchmark;
- retrospective ERA5 weather also regresses and is retained only as an oracle experiment;
- StatsBomb event history improves 2022 probability quality but has insufficient temporal coverage;
- announced-lineup continuity and shape features do not improve the 2022 fold;
- CUDA XGBoost and a residual tabular network underperform the calibrated hybrid;
- official 2026 squad intelligence is available for all 48 teams but lacks equivalent historical
  snapshots required for production promotion.

## Known limitations

The production model does not directly observe player injuries, tactical assignments, authenticated
club workload, market prices, or archived pre-kickoff weather forecasts. Starting lineups are handled
by a separate late-information model, not assumed available for early forecasts. Historical
national-team identity and country naming can involve succession assumptions inherited from source
data.

Tournament simulation approximates third-place bracket assignment when the complete FIFA lookup is
not available in the source fixture data.

## Reproducibility

Each run stores a data SHA-256, experiment identifier, software versions, seed, model parameters,
backtest predictions, diagnostic tables, and generated figures.
