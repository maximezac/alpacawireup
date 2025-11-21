#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Reuse core signal engine pieces so backtest == live logic
from sigengine.signals import (
    compute_indicators_daily_from_bars,
    compute_TS_daily,
    compute_NS,
    dynamic_weights,
    clamp_unit,
)

# Backtest-specific decay half-life (in hours).
# Default kept for compatibility; prefer BACKTEST_HALF_LIFE_HOURS env (set in sigengine/signals.py)
BACKTEST_DECAY_HALF_LIFE_HOURS = float(
    os.getenv("BACKTEST_DECAY_HALF_LIFE_HOURS", "1440.0")
)
# Preferred canonical env name for backtests (hours)
BACKTEST_HALF_LIFE_HOURS = float(os.getenv("BACKTEST_HALF_LIFE_HOURS", os.getenv("BACKTEST_DECAY_HALF_LIFE_HOURS", "1440")))
# Default maximum age of news to include in snapshots (days)
BACKTEST_NS_MAX_AGE_DAYS = int(os.getenv("BACKTEST_NS_MAX_AGE_DAYS", "90"))

SECTOR_NUDGE = float(os.getenv("SECTOR_NUDGE", "0.05"))




# ---------- basic JSON helpers ----------

def read_json(p: str) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(p: str, obj: dict) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ---------- main snapshot builder ----------

def build_snapshot(as_of: str, input_path: str, output_path: str) -> None:
    """
    Build a daily-only snapshot feed from a multi-year backfill.

    - Trims `bars` to <= as_of date
    - Trims `news` to <= as_of end-of-day
    - Recomputes daily indicators from trimmed bars
    - Recomputes TS_daily, NS (with backtest half-life), and CDS
    - Writes a prices_final-style JSON suitable for postprocess_v3/update_performance_v3
    """
    base = read_json(input_path)
    syms = base.get("symbols") or {}

    as_of_date = datetime.fromisoformat(as_of).date()
    # Treat snapshot as “end of that day” UTC
    as_of_dt = datetime(
        as_of_date.year, as_of_date.month, as_of_date.day,
        23, 59, 59, tzinfo=timezone.utc
    )
    as_of_iso = as_of_dt.isoformat().replace("+00:00", "Z")

    out = {
        "as_of_utc": as_of_iso,
        "timeframe": base.get("timeframe", "1Day"),
        "indicators_window": base.get("indicators_window", 200),
        "symbols": {},
    }

    ts_list: List[float] = []
    ns_list: List[float] = []
    cds_list: List[float] = []

    for sym, node in syms.items():
        bars = node.get("bars") or []
        if not bars:
            continue

        # 1) Trim daily bars to <= as_of_date
        bars_trim: List[Dict[str, Any]] = []
        for b in bars:
            t = b.get("t")
            if not t:
                continue
            try:
                d = datetime.fromisoformat(t.replace("Z", "+00:00")).date()
            except Exception:
                continue
            if d <= as_of_date:
                bars_trim.append(b)

        if not bars_trim:
            continue

        closes = [
            float(b.get("c", 0.0) or 0.0)
            for b in bars_trim
            if b.get("c") is not None
        ]
        if not closes:
            continue

        last_bar = bars_trim[-1]
        px = float(last_bar.get("c", 0.0) or 0.0)
        ts = last_bar.get("t") or as_of_iso
        if px <= 0:
            continue

        # 2) Recompute DAILY indicators using shared engine helper
        ind_daily = compute_indicators_daily_from_bars(bars_trim, px)

                # 3) Trim news to <= as_of_dt
        news_all = node.get("news") or []
        news_trim: List[Dict[str, Any]] = []
        for n in news_all:
            t_iso = n.get("ts")
            if not t_iso:
                continue
            try:
                t = datetime.fromisoformat(t_iso.replace("Z", "+00:00"))
            except Exception:
                continue
            if t <= as_of_dt:
                news_trim.append(n)

                # Optional: truncate very old news to avoid long-tail dilution
        try:
            max_age_days = float(os.getenv("BACKTEST_NS_MAX_AGE_DAYS", str(BACKTEST_NS_MAX_AGE_DAYS)))
            cutoff_dt = as_of_dt - timedelta(days=max_age_days)
            news_trim = [n for n in news_trim if n.get("ts") and datetime.fromisoformat(n.get("ts").replace("Z", "+00:00")) >= cutoff_dt]
        except Exception:
            # if parse fails, keep original news_trim
            pass



                # 4) Recompute NS / TS / CDS with backtest half-life
        # Use BACKTEST_HALF_LIFE_HOURS if provided to align with sigengine.signals
        half_life = BACKTEST_HALF_LIFE_HOURS
        ns = compute_NS(news_trim, as_of_dt, half_life)

        ts_val = compute_TS_daily(ind_daily)
        wT, wN = dynamic_weights(ns)
        cds = clamp_unit(
            wT * ts_val
            + wN * ns
            + (SECTOR_NUDGE if node.get("sector_bias") else 0.0)
        )

        # 5) Guidance from SMA20 (same scheme as live)
        guidance: Dict[str, Any] = {}
        sma20_last = ind_daily.get("sma20")
        if sma20_last is not None:
            guidance["sma20"] = round(sma20_last, 4)
            guidance["buy_on_dip_below"] = round(sma20_last * 0.99, 4)
            guidance["trim_above"] = round(sma20_last * 1.08, 4)

            out["symbols"][sym] = {
            "price": px,
            "ts": ts,
            "bars": bars_trim,
            "bars_5m": [],
            "indicators": {
                "daily": ind_daily,
                # Backtest uses daily-only; intraday left empty.
                "intraday": {},
            },
            "signals": {
                "TS": round(ts_val, 6),
                "TS_daily": round(ts_val, 6),
                "TS_intraday": None,
                "NS": round(ns, 6),
                "CDS": round(cds, 6),
                "wT": round(wT, 6),
                "wN": round(wN, 6),
            },
            "guidance": guidance,
            "news": news_trim,
        }


        ts_list.append(ts_val)
        ns_list.append(ns)
        cds_list.append(cds)

    write_json(output_path, out)
    print(
        f"[OK] snapshot built for {as_of} → {output_path} "
        f"with {len(out['symbols'])} symbols."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", required=True, help="Snapshot date YYYY-MM-DD")
    ap.add_argument(
        "--input",
        required=True,
        help="Backfill JSON (e.g. data/prices_backfill.json)",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output snapshot JSON (e.g. data/prices_final_snapshot.json)",
    )
    args = ap.parse_args()
    build_snapshot(args.as_of, args.input, args.output)


if __name__ == "__main__":
    main()
