#!/usr/bin/env python3
"""
verify_backtest_snapshot.py

Quick verification: compares feed signals vs ledger signal_snapshot
for a given feed and ledger and prints mismatches.

Usage:
  python verify_backtest_snapshot.py --feed data/prices_final.json --ledger data/backtest_trades_ledger.csv --date 2025-11-03T23:59:59Z
"""
import argparse, json, csv, sys
from pathlib import Path

def load_feed(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_ledger(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--date", required=True, help="as_of_utc to filter ledger rows (ISO string)")
    ap.add_argument("--portfolio", default=None, help="optional portfolio id to filter ledger rows")
    args = ap.parse_args()

    feed = load_feed(args.feed)
    ledger = load_ledger(args.ledger)
    asof = args.date

    symbols = (feed.get("symbols") or {})
    # collect feed symbols with non-zero NS (abs>1e-9)
    feed_ns = {}
    for s, n in symbols.items():
        sig = (n.get("signals") or {})
        ns = sig.get("NS")
        if ns is None:
            continue
        try:
            v = float(ns)
        except Exception:
            continue
        if abs(v) > 1e-9:
            feed_ns[s.upper()] = v

    print(f"Feed as_of_utc: {feed.get('as_of_utc')}")
    print(f"Found {len(feed_ns)} symbols with non-zero NS in feed (showing up to 30):")
    for i,(s,v) in enumerate(sorted(feed_ns.items(), key=lambda kv: -abs(kv[1]))):
        if i>=30: break
        print(f"  {s}: NS={v}")

    # filter ledger rows for the given date (+ optional portfolio)
    matched = [r for r in ledger if r.get("datetime_utc")==asof and (args.portfolio is None or r.get("portfolio_id")==args.portfolio)]
    print(f"\nLedger rows matching datetime_utc={asof}: {len(matched)}")

    mismatches = []
    for r in matched:
        sym = r.get("symbol","").upper()
        try:
            snap = json.loads(r.get("signal_snapshot") or "{}")
        except Exception:
            snap = {}
        ns_ledger = snap.get("NS")
        if ns_ledger is None:
            ns_ledger_val = None
        else:
            try:
                ns_ledger_val = float(ns_ledger)
            except Exception:
                ns_ledger_val = None

        ns_feed_val = feed_ns.get(sym)
        # If feed has non-zero NS but ledger doesn't match, report
        if ns_feed_val is not None:
            if ns_ledger_val is None or abs(ns_feed_val - ns_ledger_val) > 1e-6:
                mismatches.append((sym, ns_feed_val, ns_ledger_val, r))
    if mismatches:
        print("\nMISMATCHES (feed NS != ledger NS):")
        for sym, f_ns, l_ns, row in mismatches:
            print(f"  {sym}: feed NS={f_ns} ledger NS={l_ns}  row={row}")
    else:
        print("\nNo mismatches found for feed symbols with non-zero NS. Ledger timestamps and signals appear consistent.")

if __name__ == '__main__':
    main()