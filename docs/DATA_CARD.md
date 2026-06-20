# Data Card — International Match Results

## Source

The project uses the maintained
[`martj42/international_results`](https://github.com/martj42/international_results) dataset.
It covers senior men's full internationals and includes date, teams, score, tournament, location,
and neutral-venue status.

The training pipeline downloads the source file directly and records its SHA-256 in
`artifacts/report.json`.

## Current snapshot

- 49,437 completed matches;
- observations beginning in 1872;
- scheduled 2026 World Cup group fixtures;
- full-time scores excluding penalty shootout results.

The exact snapshot changes as the upstream dataset is maintained. Run metadata should be used instead
of assuming these counts are permanent.

## Data quality checks

The ingestion layer verifies required columns, parses dates, orders matches chronologically,
separates completed matches from fixtures, and prevents unplayed matches from updating team state.

Tests explicitly verify that:

- a match cannot influence its own features;
- future fixtures do not update ratings;
- score probabilities remain normalized;
- group inference creates correct connected components.

## Transformations

The raw data is not randomly shuffled. It is passed through a sequential state engine. Training
features therefore represent information available before each kickoff.

Older observations are retained for state continuity, while model fitting uses a 28-year rolling
window and exponential recency weights.

## Limitations and representation

Match availability is not uniform across countries or eras. Stronger federations and modern periods
are generally more densely observed. Tournament labels vary, historical national identities involve
succession choices, and result-only data cannot represent player-level or event-level match quality.

The system exposes rating uncertainty and activity partly to make sparse-data risk visible.

