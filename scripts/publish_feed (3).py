#!/usr/bin/env python3
"""
publish_feed.py
- Fetches OHLCV bars (Alpaca if credentials are set; otherwise uses data/prices.json as a local fallback).
- Computes indicators (EMA12, EMA26, EMA20, MACD, MACD signal, MACD histogram, RSI14, SMA20).
- Runs Strategy Engine v2.2 to produce TS, NS, weights, CDS, and a decision per symbol with robust fallbacks so TS doesn't collapse to all zeros.
- Writes a single enriched JSON: data/prices_final.json

Environment (optional):
  ALPACA_KEY, ALPACA_SECRET  -> if set, script will fetch from Alpaca market data v2 (US stocks, 1D bars, ~200 days history)
  WATCHLIST                  -> comma-separated symbols; otherwise reads ./data/watchlist.txt (one symbol per line)
  DAYS_BACK                  -> history window (default 220)
  OUT_PATH                   -> output JSON path (default ./data/prices_final.json)

Input fallback (if no Alpaca or request fails):
  ./data/prices.json (same rough schema as your existing feed; indicators will be recomputed)

Notes:
- The algorithm uses regression slope for MACD histogram, percent-slope for EMA20 with fallbacks, and RSI mapped to [-1, +1].
- Includes debug fields to diagnose zeros: series lengths, lookback L, and which fallbacks were used.
"""

import os
import sys
import json
import math
import time
import datetime as dt
from collections import defaultdict

# --------- Small, dependency-free helpers ---------
def ema(series, length):
    if not series or length <= 0:
        return []
    k = 2 / (length + 1)
    out = []
    ema_val = None
    for price in series:
        if ema_val is None:
            ema_val = price
        else:
            ema_val = price * k + ema_val * (1 - k)
        out.append(ema_val)
    return out

def sma(series, length):
    if length <= 0 or len(series) < 1:
        return []
    out = []
    window_sum = 0.0
    q = []
    for x in series:
        q.append(x)
        window_sum += x
        if len(q) > length:
            window_sum -= q.pop(0)
        out.append(window_sum / len(q))
    return out

def rsi(series, length=14):
    # Wilder's RSI
    if len(series) < length + 1:
        return [None] * len(series)
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(series)):
        change = series[i] - series[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    # Wilder smoothing
    avg_gain = sum(gains[1:length+1]) / length
    avg_loss = sum(losses[1:length+1]) / length
    rsis = [None] * (length)  # first 'length' entries undefined
    for i in range(length+1, len(series)+1):
        idx = i - 1
        gain = gains[idx]
        loss = losses[idx]
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length
        if avg_loss == 0:
            rs_val = float('inf')
            rsi_val = 100.0
        else:
            rs_val = avg_gain / avg_loss
            rsi_val = 100.0 - (100.0 / (1.0 + rs_val))
        rsis.append(rsi_val)
    # Align length to series
    pad = len(series) - len(rsis)
    if pad > 0:
        rsis = [None]*pad + rsis
    return rsis

def linreg_slope(y):
    """Return slope of linear regression (x = 0..n-1) normalized by range and length, robust to flat/short series.
       If insufficient or constant series, returns 0.0."""
    n = len(y)
    if n < 3:
        return 0.0
    x_sum = (n - 1) * n / 2
    x2_sum = (n - 1) * n * (2*n - 1) / 6
    y_sum = sum(y)
    xy_sum = sum(i * v for i, v in enumerate(y))
    denom = n * x2_sum - x_sum * x_sum
    if denom == 0:
        return 0.0
    slope = (n * xy_sum - x_sum * y_sum) / denom
    # Normalize to be usable across symbols with different absolute scales
    y_max = max(y)
    y_min = min(y)
    yrange = (y_max - y_min) if (y_max is not None and y_min is not None) else 0.0
    if yrange <= 0:
        # fallback: scale by mean magnitude to avoid divide-by-zero
        mean_mag = (abs(y_sum) / n) if n else 1.0
        if mean_mag == 0:
            return 0.0
        norm = slope / mean_mag
    else:
        norm = slope / yrange
    # keep within [-1,1] gently (slope should be small after normalization)
    return max(-1.0, min(1.0, norm))

def pct_slope(series, L=10):
    """Percent slope over last L bars vs the starting value; robust to flat or short series."""
    if not series or len(series) < 2:
        return 0.0, {"used": "none", "reason": "too_short"}
    window = series[-L:] if len(series) >= L else series
    start = window[0]
    end = window[-1]
    if start is None or end is None or start == 0:
        return 0.0, {"used": "none", "reason": "invalid_start"}
    pct = (end - start) / abs(start)
    # Clip to reasonable bounds to avoid outliers
    pct = max(-1.5, min(1.5, pct))
    return pct, {"used": "series", "L": len(window)}

def map_rsi_to_unit(r):
    """Map RSI to [-1,+1] centered at 50, clamp and handle None."""
    if r is None:
        return 0.0
    # 50 -> 0, 70 -> +0.67, 30 -> -0.67, extremes -> +/-1
    val = (r - 50.0) / 50.0
    return max(-1.0, min(1.0, val))

def compute_indicators_from_prices(prices):
    ema12 = ema(prices, 12)
    ema26 = ema(prices, 26)
    ema20 = ema(prices, 20)
    macd_line = [ (a - b) if (a is not None and b is not None) else None for a, b in zip(ema12, ema26) ]
    # simple EMA signal for MACD; if macd_line has None at start, treat as zeros progressively
    macd_clean = [0.0 if v is None else v for v in macd_line]
    macd_signal = ema(macd_clean, 9)
    macd_hist = [ (m - s) if (m is not None and s is not None) else 0.0 for m, s in zip(macd_clean, macd_signal) ]
    rsi14 = rsi(prices, 14)
    sma20 = sma(prices, 20)
    return {
        "ema12": ema12,
        "ema26": ema26,
        "ema20": ema20,
        "macd": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "rsi14": rsi14,
        "sma20": sma20
    }

def strategy_engine_v22(indicators, price_series, news_sentiment=None, L=10):
    """Return signals dict with TS, NS, weights, CDS, components, decision, and debug.
       Robust fallbacks so TS rarely collapses to zeros."""
    hist = indicators.get("macd_hist") or []
    ema20_series = indicators.get("ema20") or []
    rsi14_series = indicators.get("rsi14") or []
    price_series = price_series or []
    # --- Component 1: MACD histogram slope via regression ---
    macd_len = min(len(hist), L)
    macd_window = hist[-macd_len:] if macd_len > 0 else []
    macd_hist_slope = linreg_slope(macd_window) if macd_window else 0.0

    # --- Component 2: EMA trend percent slope with fallbacks ---
    ema_slope_used = "ema20"
    ema20_pct, ema_debug = pct_slope(ema20_series, L=L)
    ema20_slope = ema20_pct
    if ema20_slope == 0.0:
        # fallback to EMA12
        ema12_series = indicators.get("ema12") or []
        ema12_pct, ema_debug = pct_slope(ema12_series, L=L)
        ema20_slope = ema12_pct
        ema_slope_used = "ema12"
        if ema20_slope == 0.0:
            # final fallback to price
            price_pct, ema_debug = pct_slope(price_series, L=L)
            ema20_slope = price_pct
            ema_slope_used = "price"

    # --- Component 3: RSI position mapped to [-1,+1] ---
    rsi_val = rsi14_series[-1] if rsi14_series else None
    rsi_pos = map_rsi_to_unit(rsi_val)

    # --- Technical Score TS (0.4 MACD slope + 0.4 RSI + 0.2 EMA slope) ---
    TS = 0.4 * macd_hist_slope + 0.4 * rsi_pos + 0.2 * ema20_slope
    TS = max(-1.0, min(1.0, TS))

    # --- News Sentiment NS ---
    # If not provided, assume neutral; if provided, clamp to [-1,+1]
    if news_sentiment is None:
        NS = 0.0
        ns_debug = {"source": "default_neutral"}
    else:
        NS = max(-1.0, min(1.0, float(news_sentiment)))
        ns_debug = {"source": "feed_value"}

    # --- Weight flipping rule ---
    wT, wN = 0.6, 0.4
    if abs(NS) >= 0.7:
        wT, wN = 0.4, 0.6

    CDS = wT * TS + wN * NS

    # --- Decision thresholds ---
    if CDS > 0.35:
        decision = "Buy/Add"
    elif CDS < -0.25:
        decision = "Sell/Trim"
    else:
        decision = "Hold"

    debug = {
        "ema20_series_len": len(ema20_series),
        "macd_hist_series_len": len(hist),
        "rsi14_series_len": len(rsi14_series),
        "price_series_len": len(price_series),
        "lookback_L": L,
        "ema_slope_used": ema_slope_used,
        "ema_slope_meta": ema_debug,
        "ns_debug": ns_debug,
    }

    return {
        "TS": round(TS, 6),
        "NS": round(NS, 6),
        "wT": wT,
        "wN": wN,
        "CDS": round(CDS, 6),
        "components": {
            "rsi_pos": round(rsi_pos, 6),
            "macd_hist_slope": round(macd_hist_slope, 6),
            "ema20_slope": round(ema20_slope, 6),
        },
        "decision": decision,
        "debug": debug,
    }

def load_watchlist():
    env_list = os.getenv("WATCHLIST", "").strip()
    if env_list:
        return [s.strip().upper() for s in env_list.split(",") if s.strip()]
    # Otherwise read file
    wl_path = os.path.join("data", "watchlist.txt")
    if os.path.exists(wl_path):
        with open(wl_path, "r", encoding="utf-8") as f:
            return [line.strip().upper() for line in f if line.strip() and not line.strip().startswith("#")]
    # Fallback demo list
    return ["SPY", "QQQ", "AAPL", "NVDA"]

def fetch_from_alpaca(symbols, days_back=220):
    """Attempt to fetch 1D bars for symbols from Alpaca. Returns dict {symbol: {'t': [iso], 'c': [close], 'v': [volume]}}.
       If creds missing or fetch fails, return {} to trigger local fallback.
       NOTE: Uses simple urllib to avoid external deps."""
    key = os.getenv("ALPACA_KEY")
    sec = os.getenv("ALPACA_SECRET")
    if not key or not sec:
        return {}
    try:
        import urllib.request, urllib.error
        import urllib.parse
        end = dt.datetime.utcnow().date()
        start = end - dt.timedelta(days=days_back*2)  # *2 to account for weekends/holidays
        base_url = "https://data.alpaca.markets/v2/stocks/bars"
        headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": sec,
        }
        out = {}
        for sym in symbols:
            params = urllib.parse.urlencode({
                "symbols": sym,
                "timeframe": "1Day",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": days_back+50,
                "adjustment": "raw",
                "feed": "iex",
                "sort": "asc",
            })
            url = f"{base_url}?{params}"
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            # Parse
            bars = payload.get("bars", {}).get(sym, [])
            t = [b.get("t") for b in bars]
            c = [float(b.get("c")) for b in bars]
            v = [int(b.get("v", 0)) for b in bars]
            out[sym] = {"t": t, "c": c, "v": v}
        return out
    except Exception as e:
        # print to stderr but don't crash; caller will fallback
        print(f"[WARN] Alpaca fetch failed: {e}", file=sys.stderr)
        return {}

def load_local_prices():
    """Load local fallback feed at ./data/prices.json, but normalize into {symbol: {'t': [], 'c': [], 'v': []}}.
       Accepts both your previous schema and a minimal one."""
    path = os.path.join("data", "prices.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Try flexible parsing
    out = {}
    if isinstance(raw, dict) and "data" in raw:
        # Assume structure: {"data": {"SYMBOL": {"history": [{"ts", "close", "volume"}, ...], ...}, ...}}
        data = raw["data"]
        for sym, entry in data.items():
            hist = entry.get("history") or entry.get("bars") or []
            t = [h.get("ts") or h.get("t") for h in hist]
            c = [float(h.get("close") or h.get("c")) for h in hist if (h.get("close") or h.get("c")) is not None]
            v = [int(h.get("volume") or h.get("v") or 0) for h in hist]
            out[sym] = {"t": t, "c": c, "v": v}
    elif isinstance(raw, dict):
        # Possibly { "SYMBOL": {"t":[...], "c":[...], "v":[...]}, ...}
        ok = True
        for sym, entry in raw.items():
            if not isinstance(entry, dict) or "c" not in entry:
                ok = False
                break
        if ok:
            return raw
    return out

def enrich_and_publish(series_by_symbol, out_path):
    """Compute indicators, strategy signals, and write final JSON."""
    enriched = {"as_of_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z", "data": {}}
    for sym, series in series_by_symbol.items():
        closes = series.get("c") or []
        t = series.get("t") or []
        v = series.get("v") or []
        if not closes:
            continue
        ind = compute_indicators_from_prices(closes)
        # news sentiment placeholder: None (neutral) unless provided in future.
        signals = strategy_engine_v22(ind, closes, news_sentiment=None, L=10)
        latest_price = closes[-1]
        latest_ts = t[-1] if t else None
        sector = None  # optionally fill from your own mapping
        enriched["data"][sym] = {
            "symbol": sym,
            "price": round(float(latest_price), 4),
            "ts": latest_ts,
            "sector": sector,
            "indicators": {
                "ema12": ind["ema12"][-1] if ind["ema12"] else None,
                "ema26": ind["ema26"][-1] if ind["ema26"] else None,
                "macd": ind["macd"][-1] if ind["macd"] else None,
                "macd_signal": ind["macd_signal"][-1] if ind["macd_signal"] else None,
                "macd_hist": ind["macd_hist"][-1] if ind["macd_hist"] else None,
                "rsi14": ind["rsi14"][-1] if ind["rsi14"] else None,
                "sma20": ind["sma20"][-1] if ind["sma20"] else None,
            },
            "signals": signals,
            "decision": signals["decision"],
        }
    # Write
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)
    print(f"Wrote {out_path} with {len(enriched['data'])} symbols. as_of_utc={enriched['as_of_utc']}")

def main():
    symbols = load_watchlist()
    days_back = int(os.getenv("DAYS_BACK", "220"))
    out_path = os.getenv("OUT_PATH", os.path.join("data", "prices_final.json"))

    # Try Alpaca first
    series_by_symbol = fetch_from_alpaca(symbols, days_back=days_back)
    if not series_by_symbol:
        print("[INFO] Using local fallback at ./data/prices.json")
        series_by_symbol = load_local_prices()
        # If local is empty, exit gracefully
        if not series_by_symbol:
            print("[ERROR] No data available from Alpaca or local fallback. Provide credentials or data/prices.json.")
            sys.exit(1)

    enrich_and_publish(series_by_symbol, out_path)

if __name__ == "__main__":
    main()
