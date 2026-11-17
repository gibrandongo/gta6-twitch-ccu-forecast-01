#!/usr/bin/env python3
"""Forecast GTA VI launch-day Twitch viewers, hour by hour (make.fun
gta6-twitch-ccu-forecast-01).

Reads every <id>.in case file in $MAKEFUN_INPUT_DIR (else /input/) and writes
$MAKEFUN_OUTPUT_DIR/<id>.out (else /output/) with five quantiles per hour
for hours 2..24.

Method (fitted at run time from the case's own comparables, i.e. past big
game launches' release-day hourly curves, which the case file carries):

* Each comparable launch gives a release-day curve of log-ratios
  log(viewers_h / viewers_hour1).  The amplitude of that curve depends
  strongly on the hour-1 print: launches already hot at 00:00 UTC stay
  near it, launches still quiet at 00:00 UTC ramp many-fold as time zones
  unlock through the day.  So per hour we predict the log-ratio with two
  estimators and average them:
    1. a one-variable linear regression of log-ratio on log(hour-1 print)
       over all usable comparables;
    2. the median log-ratio of the 15 comparables nearest in log(hour-1
       print) (k-nearest neighbours).
* The log spread s_h is the mean of the regression's residual sd and the
  neighbours' robust (IQR) sd, scaled by 0.8 (the raw dispersion
  overstates the pinball-optimal spread on the practice cases), floored
  at 0.15 so the band never collapses.
* Quantiles: q_tau = p1 * exp(median_h + z_tau * s_h) with the standard
  normal quantiles z = (-1.645, -0.674, 0, 0.674, 1.645).

Pure Python 3 stdlib, deterministic (no clock, no randomness, no network),
single pass over the case file.  Falls back to the global median curve
with a fixed spread if too few comparables are usable.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone

LEVELS_Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
HOURS = range(2, 25)
K_NEIGHBOURS = 15
SPREAD_SCALE = 0.8
SPREAD_FLOOR = 0.15
SPREAD_CAP = 1.5
MIN_COMPARABLES = 5


def parse_ts(text):
    """Parse an ISO-8601 UTC timestamp; return a datetime or None."""
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except (TypeError, ValueError, AttributeError):
        return None


def median(values):
    return statistics.median(values)


def iqr_sd(values):
    """Robust sd from the inter-quartile range; floors at SPREAD_FLOOR."""
    if len(values) >= 4:
        ordered = sorted(values)
        q25 = ordered[len(ordered) // 4]
        q75 = ordered[(3 * len(ordered)) // 4]
        return max((q75 - q25) / 1.349, SPREAD_FLOOR)
    if len(values) > 1:
        return max(statistics.pstdev(values), SPREAD_FLOOR)
    return 0.30


def release_day_hours(comp):
    """Hour (1..24) -> viewers on the comparable's release day, or None."""
    day_start = parse_ts(comp.get("day_start"))
    if day_start is None:
        return None
    hours = {}
    for point in comp.get("hourly") or []:
        when = parse_ts(point.get("from"))
        viewers = point.get("viewers")
        if when is None or viewers is None:
            continue
        hour = int((when - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            hours[hour] = float(viewers)
    if 1 not in hours or hours[1] <= 0:
        return None
    return hours


def build_curves(case):
    """Return list of (log_hour1, {hour: log-ratio}) for usable comparables."""
    curves = []
    for comp in case.get("comparables") or []:
        hours = release_day_hours(comp)
        if hours is None:
            continue
        p1 = hours[1]
        curves.append((math.log(p1),
                       {h: math.log(max(v, 1.0) / p1)
                        for h, v in hours.items()}))
    return curves


def hour1_print(case):
    """Hour-1 print: known row, else last pre-checkpoint hourly point."""
    known = case.get("known") or []
    if known:
        viewers = known[0].get("viewers")
        if viewers is not None and float(viewers) > 0:
            return float(viewers)
    hourly = case.get("history_hourly") or []
    if hourly:
        viewers = hourly[-1].get("viewers")
        if viewers is not None and float(viewers) > 0:
            return float(viewers)
    history = case.get("history") or []
    if history:
        viewers = history[-1].get("viewers")
        if viewers is not None and float(viewers) > 0:
            return float(viewers)
    return 1.0


def linreg(hour, log_p1, curves):
    """log-ratio ~ a + b * log(hour-1 print); returns (median, resid_sd)."""
    xs, ys = [], []
    for lp1, logr in curves:
        if hour in logr:
            xs.append(lp1)
            ys.append(logr[hour])
    if len(xs) < 3:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    var = sum((x - mx) ** 2 for x in xs)
    if var <= 0:
        return (my, max(statistics.pstdev(ys), SPREAD_FLOOR))
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = cov / var
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = statistics.pstdev(resid) if len(resid) > 1 else 0.30
    return (a + b * log_p1, max(sd, SPREAD_FLOOR))


def knn_median(hour, log_p1, curves):
    """Median + robust sd of log-ratio among the K nearest comparables."""
    near = sorted(curves, key=lambda c: (abs(c[0] - log_p1), c[0]))[:K_NEIGHBOURS]
    vals = [logr[hour] for _, logr in near if hour in logr]
    if not vals:
        return None
    return (median(vals), iqr_sd(vals))


def observed_range(hour, curves):
    """(min, max) log-ratio observed for this hour across comparables."""
    vals = [logr[hour] for _, logr in curves if hour in logr]
    if not vals:
        return None
    return (min(vals), max(vals))


def forecast_case(case):
    p1 = max(hour1_print(case), 1.0)
    log_p1 = math.log(p1)
    curves = build_curves(case)
    rows = []
    for hour in HOURS:
        if len(curves) >= MIN_COMPARABLES:
            reg = linreg(hour, log_p1, curves)
            knn = knn_median(hour, log_p1, curves)
            parts = [p for p in (reg, knn) if p is not None]
            if parts:
                med = sum(p[0] for p in parts) / len(parts)
                spread = sum(p[1] for p in parts) / len(parts)
            else:
                med, spread = 0.0, 0.35
            # Revision 1: never let the regression extrapolate beyond what
            # any comparable launch actually did in this hour (plus a
            # small margin): GTA VI's hour-1 print may sit outside the
            # comparable range, and a linear fit must not run away there.
            rng = observed_range(hour, curves)
            if rng is not None:
                med = min(max(med, rng[0] - 0.25), rng[1] + 0.25)
        else:
            vals = [logr[hour] for _, logr in curves if hour in logr]
            med = median(vals) if vals else 0.0
            spread = iqr_sd(vals) if vals else 0.35
        spread = min(max(spread * SPREAD_SCALE, SPREAD_FLOOR), SPREAD_CAP)
        if len(curves) >= MIN_COMPARABLES:
            # Revision 2: if the target's hour-1 print sits outside the
            # comparables' hour-1 range we are extrapolating, so widen the
            # band in proportion to the distance outside that range.
            lo = min(lp1 for lp1, _ in curves)
            hi = max(lp1 for lp1, _ in curves)
            outside = max(lo - log_p1, log_p1 - hi, 0.0)
            spread = min(spread + 0.25 * outside, SPREAD_CAP)
        quantiles = []
        prev = 0.0
        for z in LEVELS_Z:
            q = p1 * math.exp(med + z * spread)
            q = min(max(q, 1.0), 1e9)
            q = round(q, 2)
            if q < prev:
                q = prev
            quantiles.append(q)
            prev = q
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input/")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output/")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        forecast = forecast_case(case)
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(forecast, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
