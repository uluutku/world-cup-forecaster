# Advanced experiments

After freezing the v2 model, I tested whether more data or a heavier model would beat it. I tried
each idea on its own, on the same World Cups, so I could see its real effect instead of burying it
in a pile of other changes.

## What I tried

- **Environment:** travel distance, timezone shift, altitude, temperature, humidity, rain, wind and
  air pressure.
- **Event data (StatsBomb):** xG balance, shots and shot quality, possession, passing, pressing,
  set-piece and transition xG, final-third entries.
- **Announced lineups:** how much the starting XI changed, prior starts, debutant share, goalkeeper
  changes and shape.
- **Current squads:** age, height, caps, goals, how spread across clubs and leagues, and positional
  depth.
- **Heavier models:** a GPU XGBoost and a GPU residual neural network, against the calibrated
  hybrid.

## Rules I kept so nothing cheats

ERA5 weather is looked up after the fact, so it is marked "oracle only" and can never go into a live
forecast. A real weather feature would have to use a forecast that was published before kickoff.

Lineup features are attached only to the exact match they belong to. They are part of a separate
late model that runs once the official XI is out, and they are never copied back onto an earlier
forecast.

Event data uses an as-of join: a match can only update a team's profile for later games, never its
own.

## When something gets promoted

A normal pre-match feature has to clear a real bar before it replaces the baseline: all five World
Cup folds, a lower average log loss, no drop in accuracy, calibration that still holds, enough data
coverage, and proof it is actually available for 2026. A gain on a single tournament is an
interesting signal, not a reason to ship it.

## Where this shows up in the app

The findings run through the whole dashboard and the generated graphics. Every view keeps three
things apart: the frozen pre-tournament predictions, the probabilities actually in use, and the
research that didn't make the cut. The Türkiye page goes further, showing the full official squad,
experience, international goals, where the players play, squad percentiles, and the clear strengths
and risks.
