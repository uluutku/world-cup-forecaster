# Data Source Registry and Promotion Rules

Every source is isolated as a feature family. A source is not promoted because it is richer or more
expensive; it must improve the frozen walk-forward benchmark without introducing future information.

## Ingested open sources

| Source | Current signals | Coverage boundary | Status |
|---|---|---|---|
| [International results](https://github.com/martj42/international_results) | scores, tournaments, venue, scorer events, penalties, shootouts | 1872–present, scorer events from 1916 | evaluated |
| [Historical FIFA ranking](https://github.com/Dato-Futbol/fifa-ranking) | external rating points and one-year momentum | 1992–September 2024 | evaluated |
| [Fjelstul World Cup Database](https://github.com/jfjelstul/worldcup) | squads, player ages, positions, prior appearances | men's World Cups through 2022 | evaluated, rejected |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | xG, shots, possession, passing, pressures, set pieces, transitions, final-third entries, announced-XI continuity | 314 matches across six international competitions, 2018–2024 | evaluated, coverage-limited |
| [Open-Meteo historical API](https://open-meteo.com/en/docs/historical-weather-api) | temperature, humidity, wind, precipitation, apparent temperature, pressure and elevation | 898 of 911 World Cup matches from 1950–2022 | evaluated as retrospective oracle, rejected |
| [OpenFootball World Cup data](https://github.com/openfootball/worldcup.json) | official 2026 squad reconciliation, clubs and birth dates | all 48 squads and 1,248 players | current intelligence layer |
| [FIFA official squad lists](https://fdp.fifa.org/assetspublic/ce281/pdf/SquadLists-English.pdf) | 2026 caps, goals and height | 1,238 player-number joins | current intelligence layer |

## Availability classes

| Class | Example | Model eligibility |
|---|---|---|
| normal pre-match | event history, travel distance | production candidate after five-fold validation |
| forecast snapshot | archived weather forecast | eligible only when the forecast run predates kickoff |
| retrospective oracle | ERA5 observed weather | diagnostic ceiling, never directly promotable |
| announced lineup | official starting XI | separate late model after lineup publication |
| current snapshot | official 2026 squad | intelligence UI until historical equivalents exist |

## Optional authenticated providers

These adapters must remain optional and cannot be required to reproduce the open benchmark.

| Provider | Planned signals |
|---|---|
| [Sportradar Soccer API](https://developer.sportradar.com/soccer/reference/soccer-api-overview) | injuries, missing players, rosters, lineups, transfers, player and team season statistics |
| [football-data.org](https://docs.football-data.org/general/v4/index.html) | squads, persons, lineups, bookings, substitutions, club matches and player minutes |
| [API-Football](https://www.api-football.com/news/post/fifa-world-cup-2026-guide-to-using-data-with-api-sports) | 2026 fixtures, lineups, injuries, player statistics, venue and live-event data |

## Feature-family promotion gate

A new family must satisfy all applicable requirements:

1. Same 2006, 2010, 2014, 2018, and 2022 tournament cutoffs.
2. No source timestamp later than kickoff.
3. Mean log-loss improvement with bootstrap uncertainty reported.
4. No material calibration regression.
5. Performance shown by edition, not only as a pooled mean.
6. Coverage and missingness exposed.
7. Current-production availability demonstrated for 2026.
8. Baseline artifact and predictions remain immutable.

## Current decisions

- Shootout history: probability improvement, accuracy tradeoff; research only.
- Historical FIFA rankings: no global improvement; rejected as a standalone family.
- Scorer/player proxies: no global improvement; rejected as a standalone family.
- Historical World Cup squad composition: substantial regression; rejected.
- No enriched configuration currently passes the production promotion gate.
- Travel and ERA5 weather regress the five-fold benchmark.
- CUDA XGBoost and a CUDA residual tabular network train successfully but underperform the hybrid.
- StatsBomb event history improves 2022 log loss by 0.0131, but its paired bootstrap interval crosses zero and accuracy falls by 3.1 points.
- Announced-lineup features do not improve the 2022 fold.
