"""Forecast GTA VI's Twitch viewers, hour by hour, on its release day.

The program reads one case (schema makefun/twitch-ccu-case/v1): everything
recorded up to the end of the release day's first hour, that hour's count
included. It writes, for each of hours 2 to 24, five quantiles of that
hour's concurrent viewers in the Grand Theft Auto VI Twitch category.

How it forecasts
---------------
A launch day has a shape: viewers dip after the first hour (only New
Zealand and Asia are awake), jump when the Americas wake and unlock, rise
into the evening and fall overnight. How steep that shape is depends on the
launch: a game whose first hour is already huge (early access, or a
simultaneous midnight unlock) barely doubles, while one whose first hour is
only the New Zealand trickle grows many times over.

So the program learns the shape from the case's own comparables -- the past
big launches shipped in the case -- instead of assuming one shape for all:

* For every comparable with a complete release day it takes the hourly
  counts, expressed as log-ratios to that launch's own hour-1 count.
* It weights each comparable by how similar it is to the case at hand,
  with a Gaussian kernel on two log-scales: the hour-1 count itself, and
  the "dip" -- the highest count in the 24 hours before the release day
  divided by the hour-1 count. The dip says how much of the launch was
  already live before midnight UTC: a big dip means hour 1 is only a
  small early wave and the day grows steeply.
* For each forecast hour it takes the weighted median of the comparables'
  log-ratios as the median ratio, and the weighted spread around it as the
  log spread. Quantiles follow the usual lognormal form:
  q = p1 * ratio_h * exp(z_tau * spread_h).

If a case ships no usable comparable, the program falls back to the
persistence baseline (every hour at the hour-1 count, spread widening
like a random walk), so it always writes a valid forecast.

Everything is deterministic: no randomness, no clock, no network, standard
library only. Two runs write the same bytes.

Validation (practice cases p00-p03, answers from SullyGnome's hourly
averages, pinball loss on the log scale, lower is better):
  starter (persistence baseline): 0.387
  this program:                   0.231
  per case: p00 0.351, p01 0.217, p02 0.154, p03 0.202

Run it the way the grader does:
    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# The kernel's bandwidths, in natural-log units, on the two similarities.
BANDWIDTH_LOG_P1 = 1.0
BANDWIDTH_LOG_DIP = 1.0
# The spread never goes below this: even the most similar launches differ.
SPREAD_FLOOR = 0.15
# The persistence fallback's spread, in log units per square-root hour.
FALLBACK_SIGMA = 0.15
# Output bounds the contract sets on a quantile (viewers).
Q_MIN = 0.01
Q_MAX = 1e9


def parse_time(text):
    """An ISO-8601 timestamp with a Z suffix, as a naive-UTC datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_of(day_start, text):
    """Which hour of the release day a timestamp falls in (1-based)."""
    return int((parse_time(text) - day_start).total_seconds() // 3600) + 1


def release_day_counts(comp):
    """A comparable's release-day hourly viewers, hours 1..24, or None.

    None when the day is incomplete or hour 1 has no viewers: a partial
    day cannot say what a full launch day looks like.
    """
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    hours = {}
    for point in comp.get("hourly") or []:
        try:
            hour = hour_of(day_start, point["from"])
            viewers = point["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= hour <= 24 and isinstance(viewers, (int, float)):
            hours[hour] = float(viewers)
    if len(hours) != 24 or hours[1] < 1:
        return None
    return hours


def pre_release_peak(comp):
    """The highest hourly count in the 24 hours before the release day.

    None when the comparable has no points there.
    """
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    peak = None
    for point in comp.get("hourly") or []:
        try:
            delta = (parse_time(point["from"]) - day_start).total_seconds() / 3600
            viewers = point["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if -24 <= delta < 0 and isinstance(viewers, (int, float)):
            peak = viewers if peak is None else max(peak, viewers)
    return peak


def case_dip(case, p1):
    """The case's own dip: pre-release peak over its hour-1 count.

    1.0 when the case has no pre-release points: a neutral dip, which
    leaves the kernel to weight on the hour-1 count alone.
    """
    day_start = parse_time(case["day_start"])
    peak = None
    for point in case.get("history_hourly") or []:
        try:
            delta = (parse_time(point["from"]) - day_start).total_seconds() / 3600
            viewers = point["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if -24 <= delta < 0 and isinstance(viewers, (int, float)):
            peak = viewers if peak is None else max(peak, viewers)
    if peak is None or peak <= 0:
        return 1.0
    return peak / p1


def weighted_median(pairs):
    """The median of values weighted by their weights.

    `pairs` is a list of (value, weight) with positive weights.
    """
    ordered = sorted(pairs)
    total = sum(weight for _, weight in ordered)
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= total / 2:
            return value
    return ordered[-1][0]


def weighted_spread(pairs, centre):
    """The weighted standard deviation of values around `centre`."""
    total = sum(weight for _, weight in pairs)
    variance = sum(weight * (value - centre) ** 2 for value, weight in pairs) / total
    return math.sqrt(max(0.0, variance))


def round2(x):
    """x rounded to 2 decimals, half up, as the starter rounds it.

    A whole number comes back as an int, so json.dumps writes 1240
    where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def kernel_forecast(p1, dip, comparables):
    """One case's forecast from its comparables, or None when unusable."""
    # Every comparable's release-day curve and its similarity to the case.
    weighted = []
    log_p1 = math.log(p1)
    log_dip = math.log(dip)
    for comp in comparables:
        counts = release_day_counts(comp)
        if counts is None:
            continue
        peak = pre_release_peak(comp)
        comp_dip = (peak / counts[1]) if peak and peak > 0 else 1.0
        d_p1 = (math.log(counts[1]) - log_p1) / BANDWIDTH_LOG_P1
        d_dip = (math.log(comp_dip) - log_dip) / BANDWIDTH_LOG_DIP
        weight = math.exp(-0.5 * (d_p1 * d_p1 + d_dip * d_dip))
        weighted.append((counts, weight))
    if not weighted:
        return None

    hours = []
    for hour in range(2, 25):
        pairs = [
            (math.log(counts[hour] / counts[1]), weight)
            for counts, weight in weighted
        ]
        median_log_ratio = weighted_median(pairs)
        spread = max(SPREAD_FLOOR, weighted_spread(pairs, median_log_ratio))
        centre = math.log(p1) + median_log_ratio
        row = []
        previous = None
        for z in Z:
            value = round2(min(Q_MAX, max(Q_MIN, math.exp(centre + z * spread))))
            if previous is not None and value < previous:
                value = previous
            row.append(value)
            previous = value
        hours.append({"hour": hour, "q": row})
    return {"hours": hours}


def persistence_forecast(p1, forecast_rows):
    """The fallback: every hour at the hour-1 count, widening spread."""
    hours = []
    for row in forecast_rows:
        hour = row["hour"]
        values = [
            round2(min(Q_MAX, max(Q_MIN, p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(hour - 1)))))
            for z in Z
        ]
        hours.append({"hour": hour, "q": values})
    return {"hours": hours}


def forecast_case(case, name):
    """One case's forecast: the learned shape, or persistence as fallback."""
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        # No hour-1 print: fall back to the last recorded hour instead of
        # failing the case outright. The final case always has hour 1.
        history = case.get("history_hourly") or []
        p1 = max(1, history[-1]["viewers"]) if history else 1
    else:
        p1 = max(1, first["viewers"])
    dip = case_dip(case, p1)
    forecast = kernel_forecast(p1, dip, case.get("comparables") or [])
    if forecast is None:
        forecast = persistence_forecast(p1, case["forecast"])
    return forecast


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
