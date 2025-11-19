#!/usr/bin/env python3
"""
publish_feed.py (LIVE MODE)

- Reads base prices (historical bars + metadata) from INPUT_PATH
- Uses engine.signals to compute indicators + TS/NS/CDS
- Writes unified feed with per-symbol `indicators`, `signals`, `guidance`, `news`
- Writes score distribution stats to SCORE_STATS_OUT
"""

from __future__ import annotations
import os, json
from datetime import datetime, timezone
from pathlib import Path

from engine import read_json, write_json   # you already have these helpers
from engine.signals import (
    utcnow_iso,
    compute_symbol_view,
    percentiles,
)

# ---- ENV ----
INPUT_PATH  = os.getenv("INPUT_PATH",  "data/prices.json")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "data/prices_final.json")
SCORE_STATS_OUT = os.getenv("SCORE_STATS_OUT", "data/score_stats.json")


def main():
    base = read_json(INPUT_PATH)
    syms = base.get("symbols") or {}

    now_dt = datetime.now(timezone.utc)
    now_iso = utcnow_iso()

    out = {
        "as_of_utc": now_iso,
        "timeframe": base.get("timeframe", "1Day"),
        "indicators_window": base.get("indicators_window", 200),
        "intraday": base.get("intraday") or {},
        "symbols": {},
    }

    ts_list = []
    ns_list = []
    cds_list = []

    for sym, node in syms.items():
        res = compute_symbol_view(sym, node, now_dt, now_iso)
        if not res:
            continue
        sym_out, ts_val, ns_val, cds_val = res
        out["symbols"][sym] = sym_out
        ts_list.append(ts_val)
        ns_list.append(ns_val)
        cds_list.append(cds_val)

    # write feed
    write_json(OUTPUT_PATH, out)

    # stats
    stats = {
        "as_of_utc": now_iso,
        "counts": {"symbols": len(out["symbols"])},
        "TS": percentiles(ts_list),
        "NS": percentiles(ns_list),
        "CDS": percentiles(cds_list),
    }
    write_json(SCORE_STATS_OUT, stats)

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
