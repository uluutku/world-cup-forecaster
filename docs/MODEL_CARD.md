# Model card: World Cup Forecaster v2

## What it's for

It estimates win / draw / loss chances for senior men's international matches, and how far each
team is likely to go in a tournament. I built it for research, demonstration and scenario
analysis.

It is not built for betting, financial decisions, picking players, or anything that affects real
athletes.

## What's inside

The forecast blends four things:

- a regularised multinomial logistic regression,
- a histogram gradient-boosting classifier,
- two Poisson goal-rate models (one per team),
- a Dixon–Coles correction that handles how low-scoring results depend on each other.

The blend weights and the calibration temperature are tuned together on a chronological hold-out
window. Training games are weighted by how recent and how important they were.

## The features

Everything is built in date order, before kickoff. For each team I track a dynamic rating and how
confident that rating is, volatility, form, momentum, attack and defence, trend, clean-sheet and
scoring rates, consistency, how active and rested they are, schedule strength, opponent prestige,
major-tournament form, experience, home or neutral context, and head-to-head record.

## How it's tested

I retrain from scratch before the 2006, 2010, 2014, 2018 and 2022 World Cups and predict each one.
A test match is never used to build pre-match state, choose blend weights, or set the calibration.

| Metric | Result |
|---|---:|
| Log loss | 0.960 |
| Historical-prior log loss | 1.075 |
| Accuracy | 57.8% |
| Brier score | 0.564 |
| Ranked probability score | 0.196 |
| Expected calibration error | 0.101 |

2022 came out worse than simply guessing from historical win/draw/loss rates. I report that as a
sign of model risk rather than dropping the fold.

## How it shows uncertainty

- bootstrap intervals for how much the score could move on a different sample of matches,
- conformal prediction sets for outcome-level uncertainty,
- rating variance for teams that are sparse or inactive,
- disagreement between the three models, which flags genuinely unclear games,
- predictive entropy for matches that are hard by nature,
- a population-stability check for features whose distribution is drifting.

Conformal coverage is measured, not guaranteed, and can fall short of its target when things shift
over time.

## What I tested and parked

- travel and timezone proxies made the backtest worse,
- retrospective ERA5 weather also made it worse and is kept only as an after-the-fact experiment,
- StatsBomb event data helped on 2022, but I only have it for that one tournament,
- announced-lineup features didn't help the 2022 fold,
- a GPU XGBoost and a residual neural net both lost to the calibrated hybrid,
- official 2026 squad data exists for all 48 teams but there's no matching history to test it on,
  so it stays as context rather than going into the forecast.

## Known limits

The production model doesn't see injuries, tactics, real club workload, market prices, or archived
weather forecasts. Starting lineups are handled by a separate late model, not assumed to be
available for early forecasts. Historical team identities involve a few succession calls inherited
from the source data. The tournament simulation approximates third-place bracket assignment when the
full FIFA rule isn't present in the fixture data.

## Reproducing a run

Every run records the data SHA-256, an experiment id, software versions, the random seed, the model
parameters, the backtest predictions, the diagnostic tables, and the generated figures.
