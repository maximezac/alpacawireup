#!/usr/bin/env python3
"""
scripts/publish_feed.py
-----------------------
Fetches market data from Alpaca (latest + recent bars), computes baseline indicators,
collects recent news per symbol, and writes a compact feed JSON at data/prices.json.

Env Vars (required):
- ALPACA_KEY
- ALPACA_SECRET

Optional:
- SYMBOLS (comma-separated list; defaults to a safe set if not provided)
- DATA_BASE (default: https://data.alpaca.markets)  # for market data
- NEWS_BASE (default: https://data.alpaca.markets)  # for news
- MAX_BARS (default: 200)
- NEWS_LOOKBACK_DAYS (default: 7)
- OUTPUT_PATH (default: data/prices.json)

Notes:
- This script *only* fetches and prepares the base feed. It does *not* do trading logic.
- Use scripts/analyze_feed.py to run Algorithm v2.2 on the generated feed.
"""

import os, sys, json, math, time, datetime as dt
from typing import List, Dict, Any
import requests
from pathlib import Path

def getenv(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    if not v:
        return default
    return v

ALPACA_KEY    = getenv("ALPACA_KEY")
ALPACA_SECRET = getenv("ALPACA_SECRET")
if not ALPACA_KEY or not ALPACA_SECRET:
    print("ERROR: Missing ALPACA_KEY or ALPACA_SECRET", file=sys.stderr)
    sys.exit(1)

DATA_BASE  = getenv("DATA_BASE", "https://data.alpaca.markets")
NEWS_BASE  = getenv("NEWS_BASE", DATA_BASE)
MAX_BARS   = int(getenv("MAX_BARS", "200"))
NEWS_LOOKBACK_DAYS = int(getenv("NEWS_LOOKBACK_DAYS", "7"))
OUTPUT_PATH = getenv("OUTPUT_PATH", "data/prices.json")

DEFAULT_SYMBOLS = [
    "QBTS","VTI","ASTS","RKLB","VOO","NVDA","ARCC","TTWO",
    "AMZN","RGTI","QUBT","SPY","QQQ","DIA","IWM","AAPL",
    "MSFT","GOOGL","META","TSLA","AMD","AVGO","INTC","SMCI","ASML"
]
SYMBOLS = [s.strip().upper() for s in getenv("SYMBOLS", ",".join(DEFAULT_SYMBOLS)).split(",") if s.strip()]

H = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET
}

def fetch_recent_bars(symbol: str, limit: int = 200) -> List[Dict[str, Any]]:
    # Alpaca v2 bars endpoint: GET /v2/stocks/{symbol}/bars
    # params: timeframe=1Day, limit, adjustment=raw
    url = f"{DATA_BASE}/v2/stocks/{symbol}/bars"
    params = {"timeframe":"1Day", "limit": limit, "adjustment": "raw"}
    r = requests.get(url, headers=H, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("bars", [])

def ema(series, span):
    if not series:
        return []
    k = 2/(span+1)
    out = []
    ema_prev = series[0]
    out.append(ema_prev)
    for x in series[1:]:
        ema_prev = x*k + ema_prev*(1-k)
        out.append(ema_prev)
    return out

def rsi(series, period=14):
    if len(series) < period+1:
        return [None]*len(series)
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(series)):
        ch = series[i] - series[i-1]
        gains.append(max(ch,0.0))
        losses.append(max(-ch,0.0))
    avg_gain = sum(gains[1:period+1]) / period
    avg_loss = sum(losses[1:period+1]) / period
    rs_list = [None]*(period)
    rs_list.append(avg_gain/avg_loss if avg_loss != 0 else float('inf'))
    rsi_vals = [None]*(period+1)
    rsi_vals.append(100 - (100/(1 + rs_list[-1])) if rs_list[-1] is not None else None)
    for i in range(period+1, len(series)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        rs = (avg_gain/avg_loss) if avg_loss != 0 else float('inf')
        rs_list.append(rs)
        rsi_vals.append(100 - (100/(1+rs)) if rs is not None else None)
    return rsi_vals

def sma(series, n):
    out = []
    s = 0.0
    q = []
    for x in series:
        q.append(x)
        s += x
        if len(q) > n:
            s -= q.pop(0)
        out.append(s/len(q))
    return out

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = [ (a - b) if (a is not None and b is not None) else None
                  for a,b in zip(ema_fast, ema_slow) ]
    # Replace None with first non-None
    for i in range(len(macd_line)):
        if macd_line[i] is None:
            macd_line[i] = 0.0
            break
    signal_line = ema(macd_line, signal)
    hist = [ (m - s) for m,s in zip(macd_line, signal_line) ]
    return macd_line, signal_line, hist, ema_fast, ema_slow

def fetch_news(symbol: str, start_iso: str) -> List[Dict[str, Any]]:
    # Alpaca news endpoint: GET /v1beta1/news?symbols=...&start=...
    url = f"{NEWS_BASE}/v1beta1/news"
    params = {"symbols": symbol, "start": start_iso, "limit": 50}
    r = requests.get(url, headers=H, params=params, timeout=30)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    data = r.json()
    return data.get("news", [])

def main():
    start = dt.datetime.utcnow() - dt.timedelta(days=NEWS_LOOKBACK_DAYS)
    start_iso = start.replace(microsecond=0).isoformat() + "Z"

    feed = {
        "metadata": {
            "source": "alpaca",
            "timeframe": "1Day",
            "indicators_window": MAX_BARS,
            "generated_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        },
        "symbols": {}  # symbol -> latest snapshot with indicators + small news cache
    }

    for sym in SYMBOLS:
        try:
            bars = fetch_recent_bars(sym, MAX_BARS)
            if not bars:
                print(f"warn: no bars for {sym}", file=sys.stderr)
                continue
            closes = [b["c"] for b in bars]
            highs  = [b["h"] for b in bars]
            lows   = [b["l"] for b in bars]
            vols   = [b["v"] for b in bars]

            macd_line, signal_line, hist, ema12, ema26 = macd(closes, 12, 26, 9)
            rsi14 = rsi(closes, 14)
            sma20 = sma(closes, 20)

            latest = bars[-1]
            news = fetch_news(sym, start_iso)

            feed["symbols"][sym] = {
                "price": latest["c"],
                "volume": latest["v"],
                "ts": latest["t"],
                "bars_tail":  min(len(bars), 30),  # diagnostic only
                "indicators": {
                    "ema12":  ema12[-1],
                    "ema26":  ema26[-1],
                    "macd":   macd_line[-1],
                    "macd_signal": signal_line[-1],
                    "macd_hist":   hist[-1],
                    "rsi14": rsi14[-1],
                    "sma20": sma20[-1]
                },
                "series": {
                    "close": closes[-30:],          # last 30 for slope calcs
                    "macd_hist": hist[-30:],
                    "ema20": sma(closes, 20)[-30:]  # proxy for slope if desired
                },
                "news": [
                    {
                        "id": n.get("id"),
                        "updated_at": n.get("updated_at"),
                        "headline": n.get("headline"),
                        "summary": n.get("summary"),
                        "url": n.get("url"),
                        "author": n.get("author"),
                        "source": n.get("source")
                    } for n in news[:15]
                ]
            }
        except Exception as e:
            print(f"error: symbol {sym}: {e}", file=sys.stderr)

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(feed, f, indent=2, ensure_ascii=False)

    print(f"Wrote feed to {out_path} with {len(feed['symbols'])} symbols.")

if __name__ == "__main__":
    main()
