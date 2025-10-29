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
        w = max(0.0, rel) * max(0.0, dec)
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

def decide(ts: float, ns: float) -> Tuple[float, float, float, str]:
    if abs(ns) >= 0.7:
        wN, wT = 0.6, 0.4
    else:
        wT, wN = 0.6, 0.4
    cds = wT * ts + wN * ns
    if cds > 0.35:
        decision = "Buy/Add"
    elif cds < -0.25:
        decision = "Sell/Trim"
    else:
        decision = "Hold"
    return wT, wN, cds, decision

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
        ts_val, ts_dbg = compute_ts(t)
        ts_val_map[sym] = ts_val
        ts_debug_map[sym] = ts_dbg

        base_ns, ns_dbg = compute_ns_for_ticker(t, now_dt)
        base_ns_map[sym] = base_ns
        ns_debug_map[sym] = ns_dbg

        sector_map[sym] = t.get("sector") or "Unknown"

    # 3) Sector adjustment on NS
    ns_adj, sector_dbg = finalize_ns_with_sector(base_ns_map, sector_map)

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
        wT, wN, cds, decision = decide(ts_val, ns_val)
        out["symbols"][sym] = {
            "symbol": sym,
            "price": t.get("price"),
            "ts": t.get("ts"),
            "sector": t.get("sector"),
            "signals": {
                "TS": round(ts_val, 6),
                "NS": round(ns_val, 6),
                "wT": wT,
                "wN": wN,
                "CDS": round(cds, 6),
                "components": ts_debug_map[sym]["components"],
                "sector_debug": sector_dbg
            },
            "decision": decision
        }

    with open(DEFAULT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote enriched feed to {DEFAULT_OUTPUT_PATH} with {len(out['symbols'])} symbols.")

if __name__ == "__main__":
    main()
