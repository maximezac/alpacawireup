#!/usr/bin/env python3
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path

from engine import read_json, write_json
from engine.signals import compute_symbol_view

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", required=True, help="Snapshot date (YYYY-MM-DD or ISO)")
    ap.add_argument("--input",  default="data/prices_backfill.json")
    ap.add_argument("--output", default="data/prices_final_snapshot.json")
    return ap.parse_args()

def parse_asof(s: str) -> datetime:
    if len(s) == 10:
        d = datetime.fromisoformat(s).date()
        return datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)

def _ts(obj) -> datetime | None:
    t = obj.get("t")
    if not t:
        return None
    try:
        return datetime.fromisoformat(t.replace("Z","+00:00")).astimezone(timezone.utc)
    except Exception:
        return None

def _nts(obj) -> datetime | None:
    t = obj.get("ts")
    if not t:
        return None
    try:
        return datetime.fromisoformat(t.replace("Z","+00:00")).astimezone(timezone.utc)
    except Exception:
        return None

def main():
    args = parse_args()
    snap_end = parse_asof(args.as_of)

    base = read_json(args.input)
    syms = base.get("symbols") or {}

    out = {
        "as_of_utc": snap_end.isoformat().replace("+00:00", "Z"),
        "timeframe": base.get("timeframe", "1Day"),
        "indicators_window": base.get("indicators_window", 200),
        "intraday": base.get("intraday") or {},
        "symbols": {},
    }

    for sym, node in syms.items():
        # Slice DAILY bars up to snapshot date
        bars = node.get("bars") or []
        bars = [b for b in bars if (_ts(b) or snap_end) <= snap_end]
        node["bars"] = bars

        # Slice INTRADAY bars if you store them (optional)
        bars_5m = node.get("bars_5m") or []
        bars_5m = [b for b in bars_5m if (_ts(b) or snap_end) <= snap_end]
        node["bars_5m"] = bars_5m

        # Slice news up to snapshot date
        news = node.get("news") or []
        news = [n for n in news if (_nts(n) or snap_end) <= snap_end]
        node["news"] = news

        # NOTE: for fully accurate intraday TS per date, you'd ideally
        # recompute indicators_5m from bars_5m here. For now we assume
        # indicators_5m is either absent (so TS falls back to daily),
        # or already prepared for this snapshot time.
        # node["indicators_5m"] = recompute_indicators_5m(bars_5m)

        res = compute_symbol_view(sym, node, snap_end, out["as_of_utc"])
        if not res:
            continue
        sym_out, ts_val, ns_val, cds_val = res
        out["symbols"][sym] = sym_out

    write_json(args.output, out)
    print(f"[OK] wrote snapshot feed {args.output} as of {out['as_of_utc']}")

if __name__ == "__main__":
    main()
