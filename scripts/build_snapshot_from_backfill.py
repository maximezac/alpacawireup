#!/usr/bin/env python3
"""
build_snapshot_from_backfill.py

Reads a multi-year backfill file (prices + news) and produces a
"snapshot" feed as of a given date, suitable as INPUT_PATH to publish_feed.py.

Env:
- BACKTEST_PRICES_PATH    : default "data/prices_backfill.json"
- SNAPSHOT_OUTPUT_PATH    : default "data/prices_snapshot.json"
- SNAPSHOT_AS_OF          : required; YYYY-MM-DD or ISO (we use its date part)
- SNAPSHOT_WINDOW_DAYS    : default "200"  (how many most-recent daily bars to keep)
- SNAPSHOT_MIN_BARS       : default "60"   (skip symbols with too little history)
"""

from __future__ import annotations
import os, sys, json
from datetime import datetime, timezone
from typing import Any, Dict, List

from fetch_prices import compute_indicators_from_bars  # reuse your indicator math


BACKTEST_PRICES_PATH = os.environ.get("BACKTEST_PRICES_PATH", "data/prices_backfill.json")
SNAPSHOT_OUTPUT_PATH = os.environ.get("SNAPSHOT_OUTPUT_PATH", "data/prices_snapshot.json")
SNAPSHOT_AS_OF       = os.environ.get("SNAPSHOT_AS_OF")
SNAPSHOT_WINDOW_DAYS = int(os.environ.get("SNAPSHOT_WINDOW_DAYS", "200"))
SNAPSHOT_MIN_BARS    = int(os.environ.get("SNAPSHOT_MIN_BARS", "60"))


def parse_ts(ts: str) -> datetime:
    # Bars are already ISO with Z offset
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_as_of_date(as_of: str) -> datetime:
    # Accept "YYYY-MM-DD" or full ISO, but only keep the date
    try:
        if "T" in as_of:
            dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(as_of + "T00:00:00+00:00")
    except Exception as e:
        raise SystemExit(f"SNAPSHOT_AS_OF invalid: {as_of!r} ({e})")
    # End of that day UTC
    return datetime(dt.year, dt.month, dt.day, 23, 59, 59, tzinfo=timezone.utc)


def main():
    if not SNAPSHOT_AS_OF:
        raise SystemExit("SNAPSHOT_AS_OF is required (YYYY-MM-DD)")

    as_of_dt = parse_as_of_date(SNAPSHOT_AS_OF)
    print(f"[INFO] Building snapshot as of {as_of_dt.isoformat()}")

    if not os.path.exists(BACKTEST_PRICES_PATH):
        raise SystemExit(f"Backfill file not found: {BACKTEST_PRICES_PATH}")

    with open(BACKTEST_PRICES_PATH, "r", encoding="utf-8") as f:
        backfill = json.load(f)

    symbols = backfill.get("symbols", {}) or {}
    out_root: Dict[str, Any] = {
        "as_of_utc": as_of_dt.isoformat(),
        "timeframe": "1Day",
        "indicators_window": SNAPSHOT_WINDOW_DAYS,
        "intraday": {
            "timeframe": "",   # we skip intraday in this first backtest pass
            "days_back": 0,
        },
        "symbols": {},
    }

    kept = 0
    skipped = 0

    for sym, node in symbols.items():
        bars = node.get("bars", []) or []
        if not bars:
            skipped += 1
            continue

        # Keep only bars with t <= as_of_dt
        trimmed_bars: List[Dict[str, Any]] = []
        for b in bars:
            try:
                bt = parse_ts(b.get("t"))
            except Exception:
                continue
            if bt <= as_of_dt:
                trimmed_bars.append(b)

        if len(trimmed_bars) < SNAPSHOT_MIN_BARS:
            skipped += 1
            continue

        # Limit to window days for indicators + storage
        if SNAPSHOT_WINDOW_DAYS > 0 and len(trimmed_bars) > SNAPSHOT_WINDOW_DAYS:
            trimmed_bars = trimmed_bars[-SNAPSHOT_WINDOW_DAYS:]

        # Compute indicators for this as-of slice
        indicators = compute_indicators_from_bars(trimmed_bars)
        last_bar = trimmed_bars[-1]
        price = last_bar.get("c")

        # Filter news up to as_of
        news_items = node.get("news", []) or []
        trimmed_news = []
        for n in news_items:
            ts = n.get("ts")
            if not ts:
                continue
            try:
                dt = parse_ts(ts)
            except Exception:
                continue
            if dt <= as_of_dt:
                trimmed_news.append(n)

        out_root["symbols"][sym] = {
            "symbol": sym,
            "price": price,
            "ts": last_bar.get("t"),
            "bars": trimmed_bars,
            "sector": node.get("sector", ""),
            "indicators": indicators,
            "news": trimmed_news,
        }
        kept += 1

    print(f"[INFO] Snapshot symbols: kept={kept}, skipped={skipped}")
    os.makedirs(os.path.dirname(SNAPSHOT_OUTPUT_PATH), exist_ok=True)
    with open(SNAPSHOT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_root, f, indent=2)
    print(f"[DONE] Wrote snapshot to {SNAPSHOT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
