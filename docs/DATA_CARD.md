# Data card: international match results

## Where the data comes from

The model trains on the
[`martj42/international_results`](https://github.com/martj42/international_results) dataset, which is
kept up to date and lists every senior men's international: date, teams, score, tournament,
location, and whether it was played at a neutral venue.

The training run downloads it directly and saves the file's SHA-256 in `artifacts/report.json`, so
you can always tell which exact snapshot produced a given result.

## What's in it right now

- 49,437 completed matches,
- going back to 1872,
- plus the scheduled 2026 World Cup group fixtures,
- full-time scores, not counting penalty shootouts.

These numbers move as the upstream dataset is maintained, so trust the run metadata rather than
assuming the figures above are fixed.

## Checks on the way in

Before anything is used, the ingestion step confirms the expected columns are there, parses the
dates, sorts matches into date order, splits played games from upcoming ones, and makes sure an
unplayed match can never update a team's state.

There are tests that specifically check that:

- a match cannot influence its own features,
- upcoming fixtures don't move any ratings,
- score probabilities always add up to one,
- group inference recovers the right groups from the fixture list.

## How it's transformed

The data is never shuffled. It runs through the state engine in date order, so every feature for a
match only reflects what was known before kickoff.

Old matches are kept so team state stays continuous, but the model itself is fit on a 28-year
rolling window with more weight on recent games.

## Limits

Coverage isn't even across countries or eras. Bigger federations and recent years are recorded in
far more detail. Tournament names change over time, some national identities involve judgement about
which country succeeded which, and results-only data can't capture how good a performance actually
was. The model exposes its rating uncertainty partly to keep this thin-data risk visible.
