#!/usr/bin/env python3
# scripts/fetch_prices.py
# Fetches recent daily bars from Alpaca and writes data/prices.json.

import os, sys, json
from datetime import datetime, timezone
import requests

ALPACA_KEY_ID = os.environ.get("ALPACA_KEY_ID") or os.environ.get("ALPACA_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_SECRET")
TIMEFRAME = os.environ.get("TIMEFRAME", "1Day")
BARS_LIMIT = int(os.environ.get("BARS_LIMIT", "120"))
SYMBOLS_PATH = os.environ.get("SYMBOLS_PATH", "data/symbols.txt")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "data/prices.json")

BASE = "https://data.alpaca.markets/v2"

if not (ALPACA_KEY_ID and ALPACA_SECRET_KEY):
    print("Missing ALPACA_KEY_ID / ALPACA_SECRET_KEY", file=sys.stderr)
    sys.exit(1)

def read_symbols(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]

def fetch_bars(symbol):
    url = f"{BASE}/stocks/{symbol}/bars"
    params = {"timeframe": TIMEFRAME, "limit": BARS_LIMIT}
    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY_ID,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }
    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code != 200:
        print(f"{symbol} -> {r.status_code} {r.text[:200]}", file=sys.stderr)
        return []
    try:
        return r.json().get("bars", [])
    except Exception as e:
        print(f"{symbol} JSON parse error: {e}", file=sys.stderr)
        return []

def main():
    symbols = read_symbols(SYMBOLS_PATH)
    out = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "timeframe": TIMEFRAME,
        "indicators_window": BARS_LIMIT,
        "symbols": {}
    }
    for sym in symbols:
        bars = fetch_bars(sym)
        if not bars:
            continue
        last = bars[-1]
        out["symbols"][sym] = {
            "symbol": sym,
            "price": last.get("c"),
            "ts": last.get("t"),
            "history": [
                {"t": b.get("t"), "o": b.get("o"), "h": b.get("h"),
                 "l": b.get("l"), "c": b.get("c"), "v": b.get("v")}
                for b in bars
            ],
            "sector": "",
            "news": []
        }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUTPUT_PATH} with {len(out['symbols'])} symbols.")

if __name__ == "__main__":
    main()
