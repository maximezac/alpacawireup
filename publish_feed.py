#!/usr/bin/env python3
"""
publish_feed.py — v2.2
Reads a raw prices feed (URL or local), computes Technical Score (TS), News Score (NS),
and Composite Decision Score (CDS) per your Strategy Engine v2.2, then writes an enriched JSON.

Inputs (env or CLI):
- FEED_URL: URL to raw prices JSON (e.g., your gist). If provided, overrides INPUT_PATH.
- INPUT_PATH: local path to raw prices JSON. Default: data/prices.json
- OUTPUT_PATH: where to write enriched JSON. Default: data/prices_final.json

Optional env:
- MAX_ARTICLES: cap news items per ticker (default 10)
- DECAY_HALF_LIFE_HOURS: NS half-life hours (default 24)
- SECTOR_NUDGE: small nudge when sector avg sentiment disagrees (default 0.05)

Schema expectations (flexible/defensive):
{
  "as_of_utc": "2025-10-27T15:32:09Z",
  "timeframe": "1Day",
  "indicators_window": 200,
  "symbols": {
     "QBTS": {
        "symbol": "QBTS",
        "price": 35.53,
        "volume": 2410,
        "ts": "2025-10-27T15:30:00Z",
        "sector": "Tech",
        "indicators": {
            "ema12": 33.9897, "ema26": 31.7434,
            "macd": 2.2463, "macd_signal": 3.3309,
            "macd_hist": -1.0845, "rsi14": 49.73, "sma20":34.285
        },
        "news": [
           {"headline":"...", "summary":"...", "ts":"2025-10-27T14:00:00Z", "relevance": 0.9, "source":"..."}
        ]
     }, ...
  }
}
"""
import os, sys, json, math, time
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
    if x is None or math.isnan(x):
        return 0.0
    return max(lo, min(hi, x))

def compute_ts(t: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Technical score with robust fallbacks. Returns (TS, debug)"""
    price = t.get("price")
    ind = t.get("indicators", {}) or {}
    # RSI position in [-1,1]
    rsi = ind.get("rsi14")
    if rsi is None:
        rsi_pos = 0.0
        rsi_src = "missing_rsi"
    else:
        rsi_pos = norm_clip((float(rsi) - 50.0) / 50.0, -1, 1)
        rsi_src = "rsi14"
    # MACD histogram slope proxy
    macd_hist = ind.get("macd_hist")
    if macd_hist is None and ("macd" in ind and "macd_signal" in ind):
        macd_hist = float(ind.get("macd", 0.0)) - float(ind.get("macd_signal", 0.0))
        mh_src = "macd_minus_signal_proxy"
    else:
        mh_src = "macd_hist"
    # Normalize by price scale to avoid exploding small caps; use tanh to squash
    macd_component = math.tanh(float(macd_hist or 0.0))
    # EMA slope proxy using ema20 (or ema26) if available; else SMA20; else price baseline -> 0
    ema = ind.get("ema20", ind.get("ema26", ind.get("sma20")))
    if price is not None and ema is not None and ema != 0:
        ema_slope = norm_clip((float(price) - float(ema)) / float(ema), -1, 1)
        ema_src = "ema20/26_or_sma20_proxy"
    else:
        ema_slope = 0.0
        ema_src = "missing_ema_and_price"
    # Weighted TS per v2.1 spec (unchanged weights): 0.4 MACD, 0.4 RSI, 0.2 EMA
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
        # Unknown timestamp → treat as old
        age_hours = 9999
    # Exponential decay by half-life
    return 0.5 ** (age_hours / HALF_LIFE_HOURS)

def compute_ns_for_ticker(sym_data: Dict[str, Any], now_dt: datetime) -> Tuple[float, Dict[str, Any]]:
    """News sentiment score in [-1,1] using VADER, half-life decay, relevance weight, and sector nudge."""
    articles = sym_data.get("news") or []
    symbol = sym_data.get("symbol") or ""
    sector = sym_data.get("sector") or ""
    selected = []
    scores = []
    for a in articles[:MAX_ARTICLES]:
        text_parts = [a.get("headline") or "", a.get("summary") or ""]
        # Skip empty
        if not any(tp.strip() for tp in text_parts):
            continue
        text = ". ".join(tp for tp in text_parts if tp)
        vs = analyzer.polarity_scores(text)["compound"]  # [-1,1]
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
    # Weighted average
    num = sum(v * w for v, w in scores)
    den = sum(w for _, w in scores)
    base_ns = num / den if den else 0.0
    base_ns = norm_clip(base_ns, -1, 1)
    # Sector nudge: compute running sector avg later; return base here and let caller adjust
    return base_ns, {"source":"vader_weighted", "articles_used": len(selected), "samples": selected}

def finalize_ns_with_sector(ns_map: Dict[str, float], sector_map: Dict[str, str]) -> Tuple[Dict[str,float], Dict[str,Any]]:
    """Apply small sector-bias nudge when a ticker's NS disagrees with its sector average."""
    # Build sector averages
    sector_groups = {}
    for sym, sector in sector_map.items():
        sector_groups.setdefault(sector or "Unknown", []).append(ns_map.get(sym, 0.0))
    sector_avg = {sec: (sum(vals)/len(vals) if vals else 0.0) for sec, vals in sector_groups.items()}
    adjusted = {}
    debug = {}
    for sym, ns in ns_map.items():
        sec = sector_map.get(sym, "Unknown")
        avg = sector_avg.get(sec, 0.0)
        if ns * avg < 0:  # disagreement in sign
            # nudge toward sector by small amount
            nudged = ns + (SECTOR_NUDGE if avg > 0 else -SECTOR_NUDGE)
            adjusted[sym] = norm_clip(nudged, -1, 1)
            debug[sym] = {"sector_avg": avg, "nudge": (SECTOR_NUDGE if avg>0 else -SECTOR_NUDGE)}
        else:
            adjusted[sym] = ns
            debug[sym] = {"sector_avg": avg, "nudge": 0.0}
    return adjusted, {"sector_avg": sector_avg, "notes":"nudge applied when sign disagrees"}

def decide(ts: float, ns: float) -> Tuple[float, float, float, str]:
    """Compose CDS per v2.2 rules and return (wT, wN, CDS, decision)."""
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
    now_str = raw.get("as_of_utc") or raw.get("as_of") or None
    now_dt = dtparser.isoparse(now_str).astimezone(timezone.utc) if now_str else datetime.now(timezone.utc)
    symbols: Dict[str, Any] = raw.get("symbols", {})
    out = {
        "as_of_utc": now_dt.isoformat(),
        "timeframe": raw.get("timeframe", "1Day"),
        "indicators_window": raw.get("indicators_window"),
        "symbols": {}
    }
    # First pass: TS and base NS
    base_ns_map = {}
    sector_map = {}
    ns_debug_map = {}
    ts_debug_map = {}
    for sym, t in symbols.items():
        ts_val, ts_dbg = compute_ts(t)
        ts_debug_map[sym] = ts_dbg
        base_ns, ns_dbg = compute_ns_for_ticker(t, now_dt)
        base_ns_map[sym] = base_ns
        ns_debug_map[sym] = ns_dbg
        sector_map[sym] = t.get("sector") or "Unknown"
    # Sector adjustment
    ns_adj, sector_dbg = finalize_ns_with_sector(base_ns_map, sector_map)
    # Assemble output
    for sym, t in symbols.items():
        ts_val, _ = compute_ts(t)
        ns_val = ns_adj.get(sym, 0.0)
        wT, wN, cds, decision = decide(ts_val, ns_val)
        # write back
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
                "ts_debug": ts_debug_map[sym]["sources"],
                "ns_debug": ns_debug_map[sym],
                "sector_debug": sector_dbg
            },
            "decision": decision
        }
    # Write file
    out_path = DEFAULT_OUTPUT_PATH
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote enriched feed to {out_path} with {len(out['symbols'])} symbols.")

if __name__ == "__main__":
    main()
