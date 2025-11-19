#!/usr/bin/env python3
"""
publish_feed.py (LIVE MODE or BACKTEST MODE)

- For LIVE:
    Uses actual current UTC timestamp and standard half-life.
- For BACKTEST:
    Uses SNAPSHOT_AS_OF from workflow and uses backtest half-life.

- Reads INPUT_PATH (prices.json or snapshot)
- Uses engine.signals.compute_symbol_view() for unified TS/NS/CDS logic
- Writes unified feed to OUTPUT_PATH
- Writes score distribution summary to SCORE_STATS_OUT
"""

from __future__ import annotations
import os, json
from datetime import datetime, timezone
from pathlib import Path

from engine import read_json, write_json
from sigengine.signals import (
    compute_symbol_view,
    percentiles,
)

# ---- ENV ----
INPUT_PATH  = os.getenv("INPUT_PATH",  "data/prices.json")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "data/prices_final.json")
SCORE_STATS_OUT = os.getenv("SCORE_STATS_OUT", "data/score_stats.json")

BACKTEST_MODE = os.getenv("BACKTEST_MODE", "0") == "1"
SNAPSHOT_AS_OF = os.getenv("SNAPSHOT_AS_OF")   # YYYY-MM-DD


def resolve_as_of():
    """
    Determine the timestamp used for score computation:
      - LIVE: use real now()
      - BACKTEST: use SNAPSHOT_AS_OF + "23:59:00Z"
    """
    if BACKTEST_MODE:
        if not SNAPSHOT_AS_OF:
            raise RuntimeError("BACKTEST_MODE=1 but SNAPSHOT_AS_OF is missing!")

        # Interpret snapshot as end-of-day UTC
        dt = datetime.fromisoformat(SNAPSHOT_AS_OF).replace(
            hour=23, minute=59, second=0, tzinfo=timezone.utc
        )
        iso = dt.isoformat()
        return dt, iso

    # Live mode
    dt = datetime.now(timezone.utc)
    iso = dt.isoformat()
    return dt, iso


def main():
    base = read_json(INPUT_PATH)
    syms = base.get("symbols") or {}

    # Determine correct reference timestamp (live or backtest)
    as_of_dt, as_of_iso = resolve_as_of()

    out = {
        "as_of_utc": as_of_iso,
        "timeframe": base.get("timeframe", "1Day"),
        "indicators_window": base.get("indicators_window", 200),
        "intraday": base.get("intraday") or {},
        "symbols": {},
    }

    ts_list = []
    ns_list = []
    cds_list = []

    for sym, node in syms.items():
        res = compute_symbol_view(sym, node, as_of_dt, as_of_iso)
        if not res:
            continue
        sym_out, ts_val, ns_val, cds_val = res
        out["symbols"][sym] = sym_out
        ts_list.append(ts_val)
        ns_list.append(ns_val)
        cds_list.append(cds_val)

    # write feed
    write_json(OUTPUT_PATH, out)

    # score distributions
    stats = {
        "as_of_utc": as_of_iso,
        "counts": {"symbols": len(out["symbols"])},
        "TS": percentiles(ts_list),
        "NS": percentiles(ns_list),
        "CDS": percentiles(cds_list),
    }
    write_json(SCORE_STATS_OUT, stats)

    # console logs
    def fmt(d):
        if not d:
            return "n/a"
        return ", ".join(f"{k}={v:.3f}" for k, v in d.items())

    print("[score-stats] TS:", fmt(stats["TS"]))
    print("[score-stats] NS:", fmt(stats["NS"]))
    print("[score-stats] CDS:", fmt(stats["CDS"]))
    print(f"[OK] wrote {OUTPUT_PATH} and {SCORE_STATS_OUT}")


if __name__ == "__main__":
    main()
