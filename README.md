# GTA VI Twitch CCU forecast — entry for gta6-twitch-ccu-forecast-01

Forecasts how many people watch Grand Theft Auto VI live on Twitch in
each hour of its release day (19 Nov 2026, UTC), as five quantiles per
hour, from everything recorded up to the end of the day's first hour.

## How it works

See the docstring at the top of `main.py`. In short: a launch day has a
shape — viewers dip after hour 1, jump when the Americas unlock, rise
into the evening, fall overnight — and how steep that shape is depends on
the launch. The program learns the shape from the case's own comparables
(past big launches shipped in the case file):

- each comparable's release day becomes log-ratios to its own hour-1 count;
- comparables are weighted by similarity to the case at hand, with a
  Gaussian kernel on two log-scales: the hour-1 count, and the "dip"
  (the highest count in the 24 hours before the release day, over the
  hour-1 count);
- per forecast hour, the weighted median of the log-ratios is the median
  ratio and the weighted spread around it is the log spread;
  `q = p1 * ratio * exp(z * spread)`.

No usable comparable in the case → it falls back to the persistence
baseline (every hour at the hour-1 count, spread widening like a random
walk), so it always writes a valid forecast. Deterministic, standard
library only, no network.

## Validation (practice cases, pinball loss on the log scale)

Answers reconstructed from SullyGnome's hourly "Average viewers" for the
four practice launches (anchor verified: the value at the release day's
00:00 equals each case's hour-1 print; the starter scored on the same
answers reproduces the 0.387 the rules state, so the answers are right).

| case | launch | starter | this entry |
| --- | --- | --- | --- |
| p00 | Marvel's Wolverine | 0.744 | 0.351 |
| p01 | Red Dead Redemption 2 | 0.287 | 0.217 |
| p02 | God of War Ragnarök | 0.215 | 0.154 |
| p03 | Marvel's Spider-Man 2 | 0.303 | 0.202 |
| **mean** | | **0.387** | **0.231** |

90% coverage 0.87–1.00 across the cases (an honest forecast covers ~0.9).
Leave-one-out over the 37 comparables: 0.396 vs the starter's 0.930
(0.261 vs 0.351 on the big launches, hour-1 count ≥ 10,000).

## Files

- `main.py` — the entry (Python 3.13, standard library only)
- `README.md` — this file
- `LICENSE` — MIT
- `.makefun/owner.json` — payout wallet

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
