#!/usr/bin/env python3
"""
fetch_prices.py
Fetch ~9 months of 1D bars per symbol from Alpaca and write data/prices.json.

Env:
- ALPACA_KEY_ID, ALPACA_SECRET_KEY (required)
- SYMBOLS_PATH (default: data/symbols.txt)
- OUTPUT_PATH (default: data/prices.json)
- DAYS_BACK (default: 270)

Output schema:
{
  "as_of_utc": "...",
  "timeframe": "1Day",
  "indicators_window": null,
  "symbols": {
    "NVDA": {
      "symbol": "NVDA",
      "price": 201.03,
      "ts": "2025-10-28T04:00:00Z",
      "history": [{"t": "...", "o":..., "h":..., "l":..., "c":..., "v":...}, ...],
      "sector": "",
      "news": []
    }, ...
  }
}
"""
import os, sys, json, time, math
from datetime import datetime, timedelta, timezone
import requests

API_BASE = "https://data.alpaca.markets/v2/stocks"

ALPACA_KEY_ID = os.environ.get("ALPACA_KEY_ID")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
SYMBOLS_PATH = os.environ.get("SYMBOLS_PATH", "data/symbols.txt")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "data/prices.json")
DAYS_BACK = int(os.environ.get("DAYS_BACK", "270"))

if not ALPACA_KEY_ID or not ALPACA_SECRET_KEY:
    print("Missing ALPACA_KEY_ID / ALPACA_SECRET_KEY.", file=sys.stderr)
    sys.exit(1)

headers = {
    "APCA-API-KEY-ID": ALPACA_KEY_ID,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

def read_symbols(path):
    syms = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                syms.append(s)
    return syms

def fetch_bars(symbol, start_iso, end_iso):
    """
    Loop through pages for 1D bars from start to end.
    """
    url = f"{API_BASE}/{symbol}/bars"
    params = {
        "timeframe": "1Day",
        "start": start_iso,
        "end": end_iso,
        "limit": 10000,
        "adjustment": "raw",
        "sort": "asc",
    }
    out = []
    next_token = None
    while True:
        p = dict(params)
        if next_token:
            p["page_token"] = next_token
        r = requests.get(url, headers=headers, params=p, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Bars fetch failed for {symbol}: {r.status_code} {r.text}")
        data = r.json()
        bars = data.get("bars") or []
        for b in bars:
            out.append({
                "t": b.get("t"),
                "o": b.get("o"),
                "h": b.get("h"),
                "l": b.get("l"),
                "c": b.get("c"),
                "v": b.get("v"),
            })
        next_token = data.get("next_page_token")
        if not next_token:
            break
        # Be gentle
        time.sleep(0.15)
    return out

def main():
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=DAYS_BACK)
    start_iso = start.isoformat()
    end_iso = now.isoformat()

    symbols = read_symbols(SYMBOLS_PATH)
    res = {
        "as_of_utc": now.isoformat(),
        "timeframe": "1Day",
        "indicators_window": None,
        "symbols": {}
    }

    for sym in symbols:
        try:
            hist = fetch_bars(sym, start_iso, end_iso)
            if not hist:
                print(f"[WARN] No bars for {sym}", file=sys.stderr)
                continue
            last = hist[-1]
            res["symbols"][sym] = {
                "symbol": sym,
                "price": float(last["c"]),
                "ts": last["t"],
                "history": hist,
                "sector": "",
                "news": []  # optional; NS stays neutral if empty
            }
            print(f"[OK] {sym}: {len(hist)} bars through {last['t']}")
        except Exception as e:
            print(f"[ERR] {sym}: {e}", file=sys.stderr)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {OUTPUT_PATH} with {len(res['symbols'])} symbols.")

if __name__ == "__main__":
    main()
