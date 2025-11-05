
#!/usr/bin/env python3
"""
fetch_prices.py — robust daily price + indicators fetcher for Alpaca
--------------------------------------------------------------------
- Reads tickers from data/symbols.txt (one per line, '#' for comments).
- Fetches up to N days of 1Day bars from Alpaca (v2).
- Computes indicators (EMA12/26, MACD, MACD signal/hist, RSI14, SMA20).
- Optionally loads sectors from data/sectors.json if present.
- Writes a feed file at data/prices.json in the format expected by publish_feed.py.

Environment variables:
- ALPACA_KEY_ID           : required (your Alpaca key id)
- ALPACA_SECRET_KEY       : required (your Alpaca secret)
- ALPACA_DATA_FEED        : optional; default "iex" (use "sip" only if your sub supports it)
- SYMBOLS_PATH            : optional; default "data/symbols.txt"
- OUTPUT_PATH             : optional; default "data/prices.json"
- DAYS_BACK               : optional; default "200" (days of 1Day bars to request, max used for indicators)
- MAX_SYMBOLS             : optional; default "100"  (cap to avoid rate limiting)
- REQUEST_TIMEOUT_SEC     : optional; default "15"
- REQUEST_SLEEP_SEC       : optional; default "0.25" (sleep between symbols to ease rate limits)

Notes on feed fallback:
- If ALPACA_DATA_FEED="sip" causes a 403 (subscription), we automatically retry with "iex".
"""

import os
import json
import time
import math
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import requests

# =========================
# CONFIG
# =========================
ALPACA_KEY_ID     = os.getenv("ALPACA_KEY_ID", "").strip()
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "").strip()
DATA_FEED         = os.getenv("ALPACA_DATA_FEED", "iex").strip().lower()
SYMBOLS_PATH      = os.getenv("SYMBOLS_PATH", "data/symbols.txt")
SECTORS_PATH      = os.getenv("SECTORS_PATH", "data/sectors.json")
OUTPUT_PATH       = os.getenv("OUTPUT_PATH", "data/prices.json")
DAYS_BACK         = int(os.getenv("DAYS_BACK", "200"))
MAX_SYMBOLS       = int(os.getenv("MAX_SYMBOLS", "100"))
REQUEST_TIMEOUT   = int(os.getenv("REQUEST_TIMEOUT_SEC", "15"))
REQUEST_SLEEP     = float(os.getenv("REQUEST_SLEEP_SEC", "0.25"))

API_URL_BASE = "https://data.alpaca.markets/v2/stocks"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY_ID,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

# =========================
# Utilities
# =========================

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def read_symbols(path: str) -> List[str]:
    syms = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().upper()
                if s and not s.startswith("#"):
                    syms.append(s)
    except FileNotFoundError:
        print(f"[WARN] {path} not found — falling back to a minimal core list.")
        syms = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "VTI", "VOO"]
    # Dedup & limit
    uniq = sorted(set(syms))
    if len(uniq) > MAX_SYMBOLS:
        print(f"[INFO] Limiting symbols to first {MAX_SYMBOLS} of {len(uniq)}")
    return uniq[:MAX_SYMBOLS]

def read_sectors(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Expecting {"AAPL": "Tech", ...} or {"symbols": {"AAPL":"Tech", ...}}
            if isinstance(data, dict) and "symbols" in data and isinstance(data["symbols"], dict):
                return {k.upper(): v for k, v in data["symbols"].items()}
            elif isinstance(data, dict):
                return {k.upper(): v for k, v in data.items()}
    except Exception as e:
        print(f"[WARN] Failed to read sectors from {path}: {e}")
    return {}

# =========================
# Indicator math
# =========================

def sma(values: List[float], window: int) -> List[float]:
    out = []
    q = []
    s = 0.0
    for v in values:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q) if q else math.nan)
    return out

def ema(values: List[float], span: int) -> List[float]:
    out = []
    if not values:
        return out
    alpha = 2.0 / (span + 1.0)
    ema_prev = values[0]
    for i, v in enumerate(values):
        if i == 0:
            ema_prev = v
        else:
            ema_prev = (v - ema_prev) * alpha + ema_prev
        out.append(ema_prev)
    return out

def rsi(values: List[float], period: int = 14) -> List[float]:
    if not values:
        return []
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(values)):
        delta = values[i] - values[i-1]
        gains.append(max(0.0, delta))
        losses.append(max(0.0, -delta))
    # Wilder's smoothing
    avg_gain = sum(gains[1:period+1]) / period if len(gains) > period else 0.0
    avg_loss = sum(losses[1:period+1]) / period if len(losses) > period else 0.0
    out = [math.nan] * len(values)
    for i in range(period+1, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out

def macd(values: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
    if not values:
        return [], [], []
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line = [ (f - s) for f, s in zip(ema_fast, ema_slow) ]
    signal_line = ema(macd_line, signal)
    hist = [ (m - s) for m, s in zip(macd_line, signal_line) ]
    return macd_line, signal_line, hist

# =========================
# Fetching
# =========================

def fetch_bars(symbol: str, feed: str) -> Dict[str, Any]:
    """Fetch full 1Day history for one symbol with explicit start date and generous limit."""
    from datetime import datetime, timedelta, timezone

    LOOKBACK_DAYS = int(os.environ.get("DAYS_BACK", "300"))
    PAD_DAYS = max(LOOKBACK_DAYS + 60, 120)  # pad for weekends/holidays

    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=PAD_DAYS)
    start_iso = start_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    url = f"{API_URL_BASE}/{symbol}/bars"
    params = {
        "timeframe": "1Day",
        "start": start_iso,
        "limit": 5000,          # plenty of bars for long history
        "adjustment": "raw",
        "feed": feed
    }

    r = requests.get(url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT)
    if r.status_code == 403 and feed == "sip":
        print(f"[WARN] {symbol}: 403 on SIP — retrying with IEX")
        return fetch_bars(symbol, "iex")
    r.raise_for_status()
    return r.json()

def compute_indicators_from_bars(bars: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute key indicators from a list of Alpaca bar dicts (expects 'c' field for close)."""
    closes = [b.get("c") for b in bars if b.get("c") is not None]
    if not closes:
        return {}
    # SMA20, EMA12/26, MACD, RSI14
    sma20_series = sma(closes, 20)
    ema12_series = ema(closes, 12)
    ema26_series = ema(closes, 26)
    macd_line, macd_signal, macd_hist = macd(closes, 12, 26, 9)
    rsi14_series = rsi(closes, 14)

    # Use last available values (may be NaN early; guard with fallback None)
    def last_valid(arr):
        for v in reversed(arr):
            if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                return float(v)
        return None

    return {
        "sma20": last_valid(sma20_series),
        "ema12": last_valid(ema12_series),
        "ema26": last_valid(ema26_series),
        "macd": last_valid(macd_line),
        "macd_signal": last_valid(macd_signal),
        "macd_hist": last_valid(macd_hist),
        "rsi14": last_valid(rsi14_series),
    }

def fetch_latest_quote(symbol: str) -> dict:
    """Fetch latest quote/trade for a symbol from Alpaca (1 quote)."""
    url = f"{API_URL_BASE}/{symbol}/quotes/latest"
    r = requests.get(url, headers=HEADERS, timeout=10)
    if r.status_code != 200:
        raise Exception(f"{symbol}: latest quote fetch failed ({r.status_code})")
    j = r.json().get("quote", {})
    return {
        "symbol": symbol,
        "price": j.get("ap") or j.get("bp") or j.get("p"),
        "ts": j.get("t"),
    }

def get_now_data(symbols: list[str]) -> dict:
    """Return latest quotes for a list of symbols."""
    out = {}
    for sym in symbols:
        try:
            out[sym] = fetch_latest_quote(sym)
        except Exception as e:
            print(f"[WARN] now-data fetch failed for {sym}: {e}")
    return out

def main():
    if not ALPACA_KEY_ID or not ALPACA_SECRET_KEY:
        raise SystemExit("ALPACA_KEY_ID / ALPACA_SECRET_KEY not set. Please add them to repo secrets and/or env.")

    symbols = read_symbols(SYMBOLS_PATH)
    sectors = read_sectors(SECTORS_PATH)

    feed_root: Dict[str, Any] = {
        "as_of_utc": now_utc_iso(),
        "timeframe": "1Day",
        "indicators_window": DAYS_BACK,
        "symbols": {}
    }

    for i, sym in enumerate(symbols, 1):
        try:
            resp = fetch_bars(sym, DATA_FEED)
            bars = resp.get("bars", [])
            if not bars:
                print(f"[WARN] {sym}: no bars returned")
                continue

            # Normalize bars to expected schema (use only fields we need)
            normalized_bars = []
            for b in bars[-DAYS_BACK:]:
                normalized_bars.append({
                    "t": b.get("t"),
                    "o": b.get("o"),
                    "h": b.get("h"),
                    "l": b.get("l"),
                    "c": b.get("c"),
                    "v": b.get("v"),
                })

            last = normalized_bars[-1]
            indicators = compute_indicators_from_bars(normalized_bars)

            feed_root["symbols"][sym] = {
                "symbol": sym,
                "price": last.get("c"),
                "ts": last.get("t"),
                "bars": normalized_bars[-200:],  # <-- publish_feed.py expects this key
                "sector": sectors.get(sym, ""),      # optional; enrich via sectors.json
                "indicators": indicators,
                "news": []                           # news added by fetch_news.py later
            }

            print(f"[OK] {i:>3}/{len(symbols)} {sym}: {len(normalized_bars)} bars, indicators: "
                  f"RSI={indicators.get('rsi14')}, MACD_HIST={indicators.get('macd_hist')}")

        except requests.HTTPError as http_err:
            # Special case: 403 SIP handled in fetch_bars; others report
            try:
                text = http_err.response.text
            except Exception:
                text = ""
            print(f"[ERR] {sym}: HTTPError {http_err} {text[:200]}")
        except Exception as e:
            print(f"[ERR] {sym}: {e}")
        finally:
            time.sleep(REQUEST_SLEEP)

    # Ensure output directory
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(feed_root, f, indent=2)
    print(f"[DONE] Wrote {len(feed_root['symbols'])} symbols to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
