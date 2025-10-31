#!/usr/bin/env python3
"""
publish_feed.py — v2.2.1
Reads raw prices (URL or local), computes indicators (SMA20/EMA20/EMA26/MACD/RSI14),
then TS/NS/CDS per Strategy Engine v2.2, and writes an enriched JSON.

Inputs (env):
- INPUT_PATH: local path to raw prices JSON. Default: data/prices.json
- OUTPUT_PATH: where to write enriched JSON. Default: data/prices_final.json
- FEED_URL: optional; if set, fetched via HTTP instead of local file.

Optional:
- MAX_ARTICLES (default 10)
- DECAY_HALF_LIFE_HOURS (default 24)
- SECTOR_NUDGE (default 0.05)
"""
import os, sys, json, math
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

import requests
from dateutil import parser as dtparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DEFAULT_INPUT_PATH = os.environ.get("INPUT_PATH", "data/prices.json")
DEFAULT_OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "data/prices_final.json")
FEED_URL = os.environ.get("FEED_URL")

MAX_ARTICLES = int(os.environ.get("MAX_ARTICLES", "10"))
HALF_LIFE_HOURS = float(os.environ.get("DECAY_HALF_LIFE_HOURS", "24"))
SECTOR_NUDGE = float(os.environ.get("SECTOR_NUDGE", "0.05"))

analyzer = SentimentIntensityAnalyzer()

# Optional sector defaults if symbol.sector == "" or missing
SECTOR_FALLBACKS = {
    "NVDA": "Tech", "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech",
    "AMZN": "Consumer Discretionary", "META": "Tech", "VTI": "ETF",
    "VOO": "ETF", "SPY": "ETF", "QQQ": "ETF",
}

NEWS_SOURCE_WEIGHTS = {
    "finnhub": 1.00,
    "benzinga": 0.90,
    "mtnewswires": 0.95,
    "google_rss": 0.50,
    "reddit": 0.35,
    "newsapi": 0.30
}

def safe_get(d: Dict, path: List[str], default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def load_feed() -> Dict[str, Any]:
    if FEED_URL:
        try:
            r = requests.get(FEED_URL, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[WARN] Failed to fetch FEED_URL={FEED_URL}: {e}. Falling back to local file.", file=sys.stderr)
    with open(DEFAULT_INPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def norm_clip(x: float, lo: float=-1.0, hi: float=1.0) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return max(lo, min(hi, x))

# ------------------------
# Indicator computations
# ------------------------
def sma(values: List[float], n: int) -> List[float]:
    out = []
    s = 0.0
    q = []
    for v in values:
        q.append(v)
        s += v
        if len(q) > n:
            s -= q.pop(0)
        out.append(s / len(q))
    return out

# ---- Indicator helpers (pure Python, no pandas required) ----
def _ema(values, span):
    if not values or span <= 1:
        return None
    k = 2 / (span + 1.0)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema

def _sma(values, window):
    if not values or len(values) < window:
        return None
    return sum(values[-window:]) / float(window)

def _rsi(values, period=14):
    # Wilder's RSI on closes
    if not values or len(values) <= period:
        return None
    gains, losses = 0.0, 0.0
    # seed average gains/losses
    for i in range(1, period + 1):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0 and avg_gain == 0:
        return 50.0
    # smooth
    for i in range(period + 1, len(values)):
        diff = values[i] - values[i - 1]
        gain = max(diff, 0.0)
        loss = max(-diff, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def _macd_full(values, fast=12, slow=26, signal=9):
    # returns (macd, signal, hist)
    if not values:
        return (None, None, None)
    ema_fast = _ema(values, fast)
    ema_slow = _ema(values, slow)
    if ema_fast is None or ema_slow is None:
        return (None, None, None)
    macd_line = ema_fast - ema_slow
    # Build a crude signal by simulating EMA on MACD over last `signal` steps.
    # If we don't have enough values to simulate, fall back to None.
    if len(values) < slow + signal:
        return (macd_line, None, None)
    # Approximate MACD series for last `signal` points by recomputing over trailing window
    macd_series = []
    # choose a trailing window long enough
    tail_needed = slow + signal
    tail = values[-tail_needed:]
    for i in range(1, len(tail) + 1):
        sub = tail[:i]
        ef = _ema(sub, fast)
        es = _ema(sub, slow)
        if ef is None or es is None:
            macd_series.append(None)
        else:
            macd_series.append(ef - es)
    macd_series = [m for m in macd_series if m is not None]
    if len(macd_series) < signal:
        return (macd_line, None, None)
    signal_line = _ema(macd_series, signal)
    if signal_line is None:
        return (macd_line, None, None)
    hist = macd_line - signal_line
    return (macd_line, signal_line, hist)

def ensure_indicators(sym_data):
    """
    Return an indicators dict. Prefer existing indicators; otherwise compute from history.
    History items expected shape: {"t": "...", "o":..., "h":..., "l":..., "c": <close>, "v": ...}
    """
    # If indicators already present, pass-through
    indicators = (sym_data.get("indicators") or {}).copy()

    need = {"sma20", "ema12", "ema26", "macd", "macd_signal", "macd_hist", "rsi14"}
    have = set(k for k, v in indicators.items()) if indicators else set()
    missing = need - have

    if not missing:
        # normalize keys/order and return
        return {
            "sma20": indicators.get("sma20"),
            "ema12": indicators.get("ema12"),
            "ema26": indicators.get("ema26"),
            "macd": indicators.get("macd"),
            "macd_signal": indicators.get("macd_signal"),
            "macd_hist": indicators.get("macd_hist"),
            "rsi14": indicators.get("rsi14"),
        }

    # Try to compute from history if available
    hist = sym_data.get("history") or []
    closes = [h.get("c") for h in hist if isinstance(h, dict) and "c" in h]
    closes = [float(c) for c in closes if c is not None]

    sma20 = indicators.get("sma20")
    if sma20 is None:
        sma20 = _sma(closes, 20)

    ema12 = indicators.get("ema12")
    if ema12 is None:
        ema12 = _ema(closes, 12)

    ema26 = indicators.get("ema26")
    if ema26 is None:
        ema26 = _ema(closes, 26)

    macd, macd_signal, macd_hist = (
        indicators.get("macd"),
        indicators.get("macd_signal"),
        indicators.get("macd_hist"),
    )
    if macd is None or macd_signal is None or macd_hist is None:
        m, s, h = _macd_full(closes, 12, 26, 9)
        macd = m if macd is None else macd
        macd_signal = s if macd_signal is None else macd_signal
        macd_hist = h if macd_hist is None else macd_hist

    rsi14 = indicators.get("rsi14")
    if rsi14 is None:
        rsi14 = _rsi(closes, 14)

    return {
        "sma20": sma20,
        "ema12": ema12,
        "ema26": ema26,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "rsi14": rsi14,
    }


def ema(values: List[float], n: int) -> List[float]:
    out = []
    alpha = 2.0 / (n + 1.0)
    ema_val = None
    for v in values:
        if ema_val is None:
            ema_val = v
        else:
            ema_val = alpha * v + (1 - alpha) * ema_val
        out.append(ema_val)
    return out

def rsi(values: List[float], n: int = 14) -> List[float]:
    gains, losses = 0.0, 0.0
    rsis = []
    prev = None
    avg_gain = None
    avg_loss = None
    for i, v in enumerate(values):
        if prev is None:
            rsis.append(None)
            prev = v
            continue
        change = v - prev
        prev = v
        gain = max(0.0, change)
        loss = max(0.0, -change)
        if i < n:
            gains += gain
            losses += loss
            rsis.append(None)
            if i == n-1:
                avg_gain = gains / n
                avg_loss = losses / n
        else:
            avg_gain = (avg_gain * (n-1) + gain) / n
            avg_loss = (avg_loss * (n-1) + loss) / n
            rs = float('inf') if avg_loss == 0 else (avg_gain / avg_loss)
            rsi_val = 100.0 - (100.0 / (1.0 + rs))
            rsis.append(rsi_val)
    return rsis

def macd(values: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line = [ (f - s) if (f is not None and s is not None) else None
                  for f, s in zip(ema_fast, ema_slow) ]
    # replace None with 0 at beginning for stability
    macd_line = [0.0 if v is None else v for v in macd_line]
    signal_line = ema(macd_line, signal)
    hist = [m - s for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, hist

def attach_indicators(sym: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure sym['indicators'] has rsi14, ema20, ema26, macd, macd_signal, macd_hist, sma20.
    Uses sym['history'] if provided.
    """
    out = dict(sym)
    ind = dict(out.get("indicators") or {})
    hist = out.get("history") or []
    closes = [float(h.get("c")) for h in hist if h.get("c") is not None]

    if closes:
        sma20 = sma(closes, 20)[-1] if len(closes) >= 1 else None
        ema20 = ema(closes, 20)[-1] if len(closes) >= 1 else None
        ema26 = ema(closes, 26)[-1] if len(closes) >= 1 else None
        macd_line, signal_line, hist_line = macd(closes, 12, 26, 9)
        macd_val = macd_line[-1] if macd_line else None
        macd_sig = signal_line[-1] if signal_line else None
        macd_hist_val = hist_line[-1] if hist_line else None
        rsi_list = rsi(closes, 14)
        rsi14 = rsi_list[-1] if rsi_list and rsi_list[-1] is not None else None

        ind.setdefault("sma20", sma20)
        ind.setdefault("ema20", ema20)
        ind.setdefault("ema26", ema26)
        ind.setdefault("macd", macd_val)
        ind.setdefault("macd_signal", macd_sig)
        ind.setdefault("macd_hist", macd_hist_val)
        ind.setdefault("rsi14", rsi14)

    out["indicators"] = ind
    return out

# ------------------------
# Scoring
# ------------------------
def compute_ts(t: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    price = t.get("price")
    ind = t.get("indicators", {}) or {}

    # RSI position [-1, 1]
    rsi_val = ind.get("rsi14")
    if rsi_val is None:
        rsi_pos = 0.0
        rsi_src = "missing_rsi"
    else:
        rsi_pos = norm_clip((float(rsi_val) - 50.0) / 50.0, -1, 1)
        rsi_src = "rsi14"

    # MACD histogram (or proxy)
    macd_hist = ind.get("macd_hist")
    if macd_hist is None and ("macd" in ind and "macd_signal" in ind):
        macd_hist = float(ind.get("macd", 0.0)) - float(ind.get("macd_signal", 0.0))
        mh_src = "macd_minus_signal_proxy"
    else:
        mh_src = "macd_hist"
    macd_component = math.tanh(float(macd_hist or 0.0))

    # EMA slope proxy using ema20 (or ema26) else sma20
    ema_base = ind.get("ema20", ind.get("ema26", ind.get("sma20")))
    if price is not None and ema_base is not None and float(ema_base) != 0.0:
        ema_slope = norm_clip((float(price) - float(ema_base)) / float(ema_base), -1, 1)
        ema_src = "ema20/26_or_sma20_proxy"
    else:
        ema_slope = 0.0
        ema_src = "missing_ema_and_price"

    ts = 0.4 * macd_component + 0.4 * rsi_pos + 0.2 * ema_slope
    debug = {
        "components": {
            "rsi_pos": rsi_pos,
            "macd_hist_slope": macd_component,
            "ema20_slope": ema_slope
        },
        "sources": {
            "rsi": rsi_src,
            "macd": mh_src,
            "ema": ema_src
        }
    }
    return ts, debug

def _freshness_weight(article_ts: str, now_dt: datetime) -> float:
    try:
        art_dt = dtparser.isoparse(article_ts)
        if art_dt.tzinfo is None:
            art_dt = art_dt.replace(tzinfo=timezone.utc)
        age_hours = (now_dt - art_dt).total_seconds() / 3600.0
        if age_hours < 0:
            age_hours = 0.0
    except Exception:
        age_hours = 9999
    return 0.5 ** (age_hours / HALF_LIFE_HOURS)

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def compute_ns_for_ticker(sym_data: Dict[str, Any], now_dt: datetime) -> Tuple[float, Dict[str, Any]]:
    articles = sym_data.get("news") or []
    if not articles:
        return 0.0, {"source":"default_neutral", "articles_used": 0}
    selected = []
    scores = []
    for a in articles[:MAX_ARTICLES]:
        text_parts = [a.get("headline") or "", a.get("summary") or ""]
        if not any(tp.strip() for tp in text_parts):
            continue

        text = ". ".join(tp for tp in text_parts if tp)
        vs = analyzer.polarity_scores(text)["compound"]

        rel = float(a.get("relevance", 1.0) or 1.0)
        dec = _freshness_weight(a.get("ts") or a.get("time") or "", now_dt)
        src = (a.get("source") or "").lower()
        src_w = NEWS_SOURCE_WEIGHTS.get(src, 0.5)  # default if not found

        # combine relevance × decay × source weight
        w = max(0.0, rel) * max(0.0, dec) * max(0.0, src_w)
        if w == 0.0:
            continue

        scores.append((vs, w))
        selected.append({
            "headline": a.get("headline"),
            "source": a.get("source"),
            "ts": a.get("ts") or a.get("time"),
            "vader": vs,
            "relevance": rel,
            "decay_w": dec,
            "source_w": src_w,
            "weight": w
        })
    if not scores:
        return 0.0, {"source":"default_neutral", "articles_used": 0}
    num = sum(v * w for v, w in scores)
    den = sum(w for _, w in scores)
    base_ns = num / den if den else 0.0
    return norm_clip(base_ns, -1, 1), {"source":"vader_weighted", "articles_used": len(selected), "samples": selected}

def finalize_ns_with_sector(ns_map: Dict[str, float], sector_map: Dict[str, str]) -> Tuple[Dict[str,float], Dict[str,Any]]:
    sector_groups = {}
    for sym, sector in sector_map.items():
        sector_groups.setdefault(sector or "Unknown", []).append(ns_map.get(sym, 0.0))
    sector_avg = {sec: (sum(vals)/len(vals) if vals else 0.0) for sec, vals in sector_groups.items()}
    adjusted = {}
    debug = {}
    for sym, ns in ns_map.items():
        sec = sector_map.get(sym, "Unknown")
        avg = sector_avg.get(sec, 0.0)
        if ns * avg < 0:
            nudged = ns + (SECTOR_NUDGE if avg > 0 else -SECTOR_NUDGE)
            adjusted[sym] = norm_clip(nudged, -1, 1)
            debug[sym] = {"sector_avg": avg, "nudge": (SECTOR_NUDGE if avg>0 else -SECTOR_NUDGE)}
        else:
            adjusted[sym] = ns
            debug[sym] = {"sector_avg": avg, "nudge": 0.0}
    return adjusted, {"sector_avg": sector_avg, "notes":"nudge applied when sign disagrees"}

def decide(ts: float, ns: float, ns_meta: dict | None, regime_bias: float | None = None) -> tuple[float, float, float, str, dict]:
    """
    Adaptive weighting per v2.2:
      - Base: wT=0.6, wN=0.4 (your default)
      - If |ns| >= 0.7: start from risk-on news rule (wN=0.6, wT=0.4)
      - Confidence scaling using news evidence & freshness:
          conf = sqrt(min(articles_used,10)/10) * avg_decay_w
          wN = clip(0.35 * (0.7 + 0.6 * conf), 0.2, 0.6)
      - Regime nudge based on portfolio-average NS (if provided):
          if regime_bias <= -0.4: wN -= 0.05
          if regime_bias >= +0.4: wN += 0.05
          (still clipped 0.2..0.6)
    Returns: (wT, wN, cds, decision, dbg)
    """
    # Base weights
    if abs(ns) >= 0.7:
        wN, wT = 0.6, 0.4
    else:
        wT, wN = 0.6, 0.4

    # Pull evidence meta
    articles_used = 0
    avg_decay = 0.0
    if ns_meta and ns_meta.get("source") != "default_neutral":
        samples = ns_meta.get("samples") or []
        articles_used = len(samples)
        if samples:
            avg_decay = sum(s.get("decay_w", 0.0) or 0.0 for s in samples) / len(samples)

    # Confidence scaling (0..~1); if no/old news, conf ~ 0 → wN ~ 0.245; with fresh plentiful news, wN → 0.6
    import math
    conf = math.sqrt(min(articles_used, 10) / 10.0) * avg_decay
    wN = 0.35 * (0.7 + 0.6 * conf)
    wN = _clip(wN, 0.2, 0.6)
    wT = 1.0 - wN

    # Regime nudge from portfolio-average NS
    if regime_bias is not None:
        if regime_bias <= -0.4:
            wN -= 0.05
        elif regime_bias >= +0.4:
            wN += 0.05
        wN = _clip(wN, 0.2, 0.6)
        wT = 1.0 - wN

    # Compose CDS & decision
    cds = wT * ts + wN * ns
    if cds > 0.35:
        decision = "Buy/Add"
    elif cds < -0.25:
        decision = "Sell/Trim"
    else:
        decision = "Hold"

    dbg = {
        "articles_used": articles_used,
        "avg_decay_w": round(avg_decay, 6),
        "conf": round(conf, 6),
        "regime_bias": None if regime_bias is None else round(regime_bias, 6)
    }
    return wT, wN, cds, decision, dbg

def main():
    raw = load_feed()
    now_str = raw.get("as_of_utc") or raw.get("as_of")
    now_dt = dtparser.isoparse(now_str).astimezone(timezone.utc) if now_str else datetime.now(timezone.utc)

    symbols_in = raw.get("symbols", {}) or {}

    # 1) Enrich each symbol with computed indicators (from history if needed)
    enriched = {}
    for sym, t in symbols_in.items():
        t2 = attach_indicators(t)
        if not t2.get("sector"):
            t2["sector"] = SECTOR_FALLBACKS.get(sym, t2.get("sector", "Unknown"))
        enriched[sym] = t2

    # 2) First pass TS & base NS
    base_ns_map, sector_map = {}, {}
    ts_debug_map, ns_debug_map = {}, {}
    ts_val_map = {}

    for sym, t in enriched.items():
        t["indicators"] = ensure_indicators(t)
        
        ts_val, ts_dbg = compute_ts(t)
        ts_val_map[sym] = ts_val
        ts_debug_map[sym] = ts_dbg

        base_ns, ns_dbg = compute_ns_for_ticker(t, now_dt)
        base_ns_map[sym] = base_ns
        ns_debug_map[sym] = ns_dbg

        sector_map[sym] = t.get("sector") or "Unknown"

    # 3) Sector adjustment on NS
    ns_adj, sector_dbg = finalize_ns_with_sector(base_ns_map, sector_map)

    regime_bias = 0.0
    if ns_adj:
        regime_bias = sum(ns_adj.values()) / len(ns_adj)

    # 4) Build output
    out = {
        "as_of_utc": now_dt.isoformat(),
        "timeframe": raw.get("timeframe", "1Day"),
        "indicators_window": raw.get("indicators_window"),
        "symbols": {}
    }

    for sym, t in enriched.items():
        ts_val = ts_val_map[sym]
        ns_val = ns_adj.get(sym, 0.0)

        wT, wN, cds, decision, decide_dbg = decide(
            ts_val, ns_val,
            ns_meta=ns_debug_map.get(sym),     # may be "default_neutral"
            regime_bias=regime_bias
        )
        
        out["symbols"][sym] = {
            "symbol": sym,
            "price": t.get("price"),
            "ts": t.get("ts"),
            "sector": t.get("sector"),
            "indicators": t.get("indicators") or None,
            "signals": {
                "TS": round(ts_val, 6),
                "NS": round(ns_val, 6),
                "wT": wT,
                "wN": wN,
                "CDS": round(cds, 6),
                "components": ts_debug_map[sym]["components"],
                "ns_debug": ns_debug_map[sym],
                "decide_debug": decide_dbg,
                "sector_debug": sector_dbg
            },
            "decision": decision
        }

    # ----------------------------------------------------------
    # ADDITIONAL SECTION: live snapshot ("now") data
    # ----------------------------------------------------------
    # ----------------------------------------------------------
    # 🕒 Real-time “now” snapshot (separate from daily bars)
    # ----------------------------------------------------------
    try:
        import fetch_prices  # uses your existing get_now_data()
        symbols_list = list(out["symbols"].keys())

        # Optional toggle (set GENERATE_NOW=0 in CI to skip)
        if os.getenv("GENERATE_NOW", "1") != "1":
            raise RuntimeError("GENERATE_NOW disabled")

        live_quotes = fetch_prices.get_now_data(symbols_list)

        now_block = {}
        for sym in symbols_list:
            # Skip if no live quote returned
            node = live_quotes.get(sym)
            if not node:
                continue

            latest_price = node.get("price")
            if latest_price is None:
                continue
    
            # Reuse the daily indicators so TS can be computed at the latest price
            # Construct a minimal ticker payload for compute_ts()
            ref = out["symbols"][sym]
            ticker_for_ts = {
                "price": float(latest_price),
                "indicators": ref.get("indicators", {}),
            }

            # Compute TS using your existing helper; keep NS from the daily signals
            ts_val, ts_dbg = compute_ts(ticker_for_ts)
            ns_val = ref.get("signals", {}).get("NS", 0.0)

            # Combine into CDS/decision using your existing decide()
            wT, wN, cds, decision, decide_dbg = decide(ts_val, ns_val, None, None)

            now_block[sym] = {
                "price": float(latest_price),
                "ts": node.get("ts") or datetime.now(timezone.utc).isoformat(),
                "signals": {"TS": ts_val, "NS": ns_val, "CDS": cds},
                "decision": decision,
            }

        if now_block:
            out["now"] = now_block
            print(f"[INFO] Added {len(now_block)} symbols to 'now' snapshot.")
            print(f"[ASSERT] now key present? {'now' in out}, symbols={len(out['now'])}")
        else:
            print("[WARN] No live quotes available for 'now' snapshot.")
    except Exception as e:
        print(f"[WARN] Skipped 'now' snapshot: {e}")


    with open(DEFAULT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # ✅ Reopen and verify what was written
    try:
        with open(DEFAULT_OUTPUT_PATH, "r", encoding="utf-8") as f:
            verify = json.load(f)
        print(
            "[VERIFY]",
            f"path={DEFAULT_OUTPUT_PATH}",
            f"has_now={'now' in verify}",
            f"now_count={len(verify.get('now', {})) if 'now' in verify else 0}",
            f"symbols={len(verify.get('symbols', {}))}",
        )
    except Exception as e:
        print(f"[VERIFY] Failed to re-open output: {e}")

if __name__ == "__main__":
    main()
