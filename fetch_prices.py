#!/usr/bin/env python3
"""
fetch_prices.py — robust daily + intraday price + indicators fetcher for Alpaca
-------------------------------------------------------------------------------
- Reads tickers from data/symbols.txt (one per line, '#' for comments).
- Fetches up to N days of 1Day bars from Alpaca (v2) for daily indicators.
- Fetches intraday bars (default 5Min) for higher-resolution indicators.
- Computes indicators:
    Daily:    EMA12/26, MACD, MACD signal/hist, RSI14, SMA20
    Intraday: EMA50, EMA200, MACD, MACD signal/hist, MACD hist prev, RSI14
- Optionally loads sectors from data/sectors.json if present.
- Writes a feed file at data/prices.json in the format expected by publish_feed.py.

Env (required):
- ALPACA_KEY_ID
- ALPACA_SECRET_KEY

Env (optional):
- ALPACA_DATA_FEED        : default "iex"  (use "sip" only if your sub supports it)
- SYMBOLS_PATH            : default "data/symbols.txt"
- SECTORS_PATH            : default "data/sectors.json"
- OUTPUT_PATH             : default "data/prices.json"
- DAYS_BACK               : default "200"     (daily indicators window)
- MAX_SYMBOLS             : default "0"       (0 = unlimited; >0 = soft cap before fetch)
- SYMBOLS_BATCH           : default "75"      (placeholder for future multi-symbol endpoints)
- REQUEST_TIMEOUT_SEC     : default "15"
- REQUEST_SLEEP_SEC       : default "0.25"
- COVERAGE_STRICT         : default "1"       (fail if any requested symbols are missing)

Intraday (optional):
- INTRADAY_TIMEFRAME      : default "5Min"    (set empty or "none" to disable)
- INTRADAY_DAYS_BACK      : default "10"      (how many calendar days of intraday to request)
  * alias: INTRADAY_DAYS_BACK_5M is also accepted, if INTRADAY_DAYS_BACK is not set
- INTRADAY_BARS_MAX       : default "500"     (cap bars_5m list length per symbol in output)

Notes:
- If ALPACA_DATA_FEED="sip" causes a 403, we automatically retry with "iex".
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
MAX_SYMBOLS       = int(os.getenv("MAX_SYMBOLS", "0"))            # 0 = unlimited
SYMBOLS_BATCH     = int(os.getenv("SYMBOLS_BATCH", "75"))         # future use (multi-symbol endpoints)
REQUEST_TIMEOUT   = int(os.getenv("REQUEST_TIMEOUT_SEC", "15"))
REQUEST_SLEEP     = float(os.getenv("REQUEST_SLEEP_SEC", "0.25"))
COVERAGE_STRICT   = (os.getenv("COVERAGE_STRICT", "1") == "1")
DAILY_BARS_MAX   = int(os.getenv("DAILY_BARS_MAX", "200"))  # 0 = no cap


# Intraday config
INTRADAY_TIMEFRAME_RAW = os.getenv("INTRADAY_TIMEFRAME", "5Min").strip()
# Allow either INTRADAY_DAYS_BACK or INTRADAY_DAYS_BACK_5M
_intraday_back_env = (
    os.getenv("INTRADAY_DAYS_BACK")
    or os.getenv("INTRADAY_DAYS_BACK_5M")
    or "10"
)
INTRADAY_DAYS_BACK = int(_intraday_back_env)
INTRADAY_BARS_MAX  = int(os.getenv("INTRADAY_BARS_MAX", "500"))

# Normalize timeframe and decide if intraday is enabled
INTRADAY_TIMEFRAME = INTRADAY_TIMEFRAME_RAW
if INTRADAY_TIMEFRAME.lower() in ("", "none", "off", "disable"):
    INTRADAY_TIMEFRAME = ""
INTRADAY_ENABLED = bool(INTRADAY_TIMEFRAME) and INTRADAY_DAYS_BACK > 0

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
    uniq = sorted(set(syms))
    if MAX_SYMBOLS and len(uniq) > MAX_SYMBOLS:
        print(f"[INFO] Truncating symbols to MAX_SYMBOLS={MAX_SYMBOLS} of {len(uniq)}")
        uniq = uniq[:MAX_SYMBOLS]
    return uniq

def read_sectors(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
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
    macd_line = [(f - s) for f, s in zip(ema_fast, ema_slow)]
    signal_line = ema(macd_line, signal)
    hist = [(m - s) for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, hist

def last_valid(arr: List[Any]) -> float | None:
    for v in reversed(arr):
        if v is None:
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        return float(v)
    return None

# =========================
# Fetching
# =========================

def fetch_bars(symbol: str, feed: str) -> Dict[str, Any]:
    """Fetch full 1Day history for one symbol with explicit start date and generous limit."""
    LOOKBACK_DAYS_LOCAL = int(os.environ.get("DAYS_BACK", str(DAYS_BACK)))
    PAD_DAYS = max(LOOKBACK_DAYS_LOCAL + 60, 120)

    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=PAD_DAYS)
    start_iso = start_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    url = f"{API_URL_BASE}/{symbol}/bars"
    params = {
        "timeframe": "1Day",
        "start": start_iso,
        "limit": 5000,
        "adjustment": "raw",
        "feed": feed
    }

    r = requests.get(url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT)
    if r.status_code == 403 and feed == "sip":
        print(f"[WARN] {symbol}: 403 on SIP — retrying with IEX")
        return fetch_bars(symbol, "iex")
    r.raise_for_status()
    return r.json()

def fetch_intraday_bars(symbol: str, feed: str) -> Dict[str, Any]:
    """Fetch intraday history (e.g., 5Min) for one symbol."""
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=INTRADAY_DAYS_BACK)
    start_iso = start_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    url = f"{API_URL_BASE}/{symbol}/bars"
    params = {
        "timeframe": INTRADAY_TIMEFRAME,
        "start": start_iso,
        "limit": 5000,  # safe upper limit; Alpaca will cap if needed
        "adjustment": "raw",
        "feed": feed,
    }

    r = requests.get(url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT)
    if r.status_code == 403 and feed == "sip":
        print(f"[WARN] {symbol} (intraday): 403 on SIP — retrying with IEX")
        return fetch_intraday_bars(symbol, "iex")
    r.raise_for_status()
    return r.json()

def compute_indicators_from_bars(bars: List[Dict[str, Any]]) -> Dict[str, float]:
    """Daily indicators"""
    closes = [b.get("c") for b in bars if b.get("c") is not None]
    if not closes:
        return {}
    sma20_series = sma(closes, 20)
    ema12_series = ema(closes, 12)
    ema26_series = ema(closes, 26)
    macd_line, macd_signal, macd_hist = macd(closes, 12, 26, 9)
    rsi14_series = rsi(closes, 14)

    return {
        "sma20": last_valid(sma20_series),
        "ema12": last_valid(ema12_series),
        "ema26": last_valid(ema26_series),
        "macd": last_valid(macd_line),
        "macd_signal": last_valid(macd_signal),
        "macd_hist": last_valid(macd_hist),
        "rsi14": last_valid(rsi14_series),
    }

def compute_intraday_indicators_from_bars(bars: List[Dict[str, Any]]) -> Dict[str, float]:
    """Intraday indicators (e.g., 5Min EMA50/200, MACD, RSI, MACD hist prev)."""
    closes = [b.get("c") for b in bars if b.get("c") is not None]
    if not closes:
        return {}

    ema50_series = ema(closes, 50)
    ema200_series = ema(closes, 200)
    macd_line, macd_signal, macd_hist = macd(closes, 12, 26, 9)
    rsi14_series = rsi(closes, 14)

    ema50_last = last_valid(ema50_series)
    ema200_last = last_valid(ema200_series)
    macd_last = last_valid(macd_line)
    macd_signal_last = last_valid(macd_signal)
    macd_hist_last = last_valid(macd_hist)

    # Find previous valid MACD hist value for slope/phase
    macd_hist_prev = None
    seen_last = False
    for v in reversed(macd_hist):
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            continue
        if not seen_last:
            # first valid from the end is "last"
            seen_last = True
            continue
        macd_hist_prev = float(v)
        break

    rsi14_last = last_valid(rsi14_series)

    return {
        "ema50_5m": ema50_last,
        "ema200_5m": ema200_last,
        "macd_5m": macd_last,
        "macd_signal_5m": macd_signal_last,
        "macd_hist_5m": macd_hist_last,
        "macd_hist_prev_5m": macd_hist_prev,
        "rsi14_5m": rsi14_last,
    }

def fetch_latest_quote(symbol: str) -> dict:
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
    """Return latest quotes for a list of symbols (sequential; keep rate gentle)."""
    out = {}
    for sym in symbols:
        try:
            out[sym] = fetch_latest_quote(sym)
        except Exception as e:
            print(f"[WARN] now-data fetch failed for {sym}: {e}")
        time.sleep(REQUEST_SLEEP)
    return out

# =========================
# Main
# =========================

def main():
    if not ALPACA_KEY_ID or not ALPACA_SECRET_KEY:
        raise SystemExit("ALPACA_KEY_ID / ALPACA_SECRET_KEY not set. Please add them to repo secrets and/or env.")

    symbols = read_symbols(SYMBOLS_PATH)
    sectors = read_sectors(SECTORS_PATH)

    feed_root: Dict[str, Any] = {
        "as_of_utc": now_utc_iso(),
        "timeframe": "1Day",
        "indicators_window": DAYS_BACK,
        "intraday": {
            "timeframe": INTRADAY_TIMEFRAME if INTRADAY_ENABLED else "",
            "days_back": INTRADAY_DAYS_BACK if INTRADAY_ENABLED else 0,
        },
        "symbols": {}
    }

    requested_set = list(symbols)  # preserve order for logging
    missing_no_bars = []
    zero_priced = []

    # (Per-symbol fetch; SYMBOLS_BATCH kept for future multi-symbol endpoints)
    for i, sym in enumerate(symbols, 1):
        try:
            # -----------------------
            # Daily bars + indicators
            # -----------------------
            resp = fetch_bars(sym, DATA_FEED)
            bars = resp.get("bars", [])
            if not bars:
                print(f"[WARN] {sym}: no daily bars returned")
                missing_no_bars.append(sym)
                time.sleep(REQUEST_SLEEP)
                continue

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
            indicators_daily = compute_indicators_from_bars(normalized_bars)

            price = last.get("c")
            if price in (None, 0, 0.0):
                zero_priced.append(sym)

            # Decide how many daily bars to store (we already limited indicator window to DAYS_BACK)
            if DAILY_BARS_MAX > 0:
                bars_to_store = normalized_bars[-DAILY_BARS_MAX:]
            else:
                bars_to_store = normalized_bars  # no cap
            
            symbol_payload: Dict[str, Any] = {
                "symbol": sym,
                "price": price,
                "ts": last.get("t"),
                "bars": bars_to_store,
                "sector": sectors.get(sym, ""),
                "indicators": indicators_daily,
                "news": [],
}


            # -----------------------
            # Intraday bars + indicators (5Min by default)
            # -----------------------
            if INTRADAY_ENABLED:
                try:
                    intraday_resp = fetch_intraday_bars(sym, DATA_FEED)
                    intraday_bars = intraday_resp.get("bars", []) or []
                    if intraday_bars:
                        normalized_5m = []
                        for b in intraday_bars:
                            normalized_5m.append({
                                "t": b.get("t"),
                                "o": b.get("o"),
                                "h": b.get("h"),
                                "l": b.get("l"),
                                "c": b.get("c"),
                                "v": b.get("v"),
                            })
                        # Cap length so feed doesn't explode
                        if INTRADAY_BARS_MAX > 0:
                            normalized_5m = normalized_5m[-INTRADAY_BARS_MAX:]

                        intraday_indicators = compute_intraday_indicators_from_bars(normalized_5m)

                        symbol_payload["bars_5m"] = normalized_5m
                        symbol_payload["indicators_5m"] = intraday_indicators
                    else:
                        print(f"[WARN] {sym}: no intraday bars returned ({INTRADAY_TIMEFRAME})")
                except requests.HTTPError as http_err:
                    try:
                        text = http_err.response.text
                    except Exception:
                        text = ""
                    print(f"[WARN] {sym}: intraday HTTPError {http_err} {text[:200]}")
                except Exception as e:
                    print(f"[WARN] {sym}: intraday fetch failed: {e}")

            feed_root["symbols"][sym] = symbol_payload

            rsi_dbg = indicators_daily.get("rsi14")
            macd_dbg = indicators_daily.get("macd_hist")
            print(f"[OK] {i:>3}/{len(symbols)} {sym}: {len(normalized_bars)} daily bars, "
                  f"RSI={rsi_dbg}, MACD_HIST={macd_dbg}")

        except requests.HTTPError as http_err:
            try:
                text = http_err.response.text
            except Exception:
                text = ""
            print(f"[ERR] {sym}: HTTPError {http_err} {text[:200]}")
        except Exception as e:
            print(f"[ERR] {sym}: {e}")
        finally:
            time.sleep(REQUEST_SLEEP)

    # --- Coverage checks & diagnostics
    returned_set = set((feed_root.get("symbols") or {}).keys())
    requested_set_unique = list(dict.fromkeys(requested_set))  # preserve order, dedup
    missing = [s for s in requested_set_unique if s not in returned_set]

    if missing:
        print(f"[ERROR] Missing symbols after fetch: {missing[:25]}{' …' if len(missing) > 25 else ''} "
              f"(total {len(missing)})")

    if zero_priced:
        print(f"[WARN] Zero/None prices for: {zero_priced[:25]}{' …' if len(zero_priced) > 25 else ''} "
              f"(total {len(zero_priced)})")

    if missing and COVERAGE_STRICT:
        # Fail fast to avoid downstream surprises
        raise SystemExit(2)

    # Ensure output directory & write
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(feed_root, f, indent=2)
    print(f"[DONE] Wrote {len(feed_root['symbols'])} symbols to {OUTPUT_PATH}")
    print(
        f"[INFO] Intraday enabled={INTRADAY_ENABLED}, "
        f"timeframe='{INTRADAY_TIMEFRAME}', days_back={INTRADAY_DAYS_BACK}, "
        f"bars_max={INTRADAY_BARS_MAX}"
    )

if __name__ == "__main__":
    main()
