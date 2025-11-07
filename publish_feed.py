#!/usr/bin/env python3
"""
publish_feed.py  (LIVE-ONLY MODE)

- Reads base prices (historical bars + metadata) from INPUT_PATH
- Ensures each symbol has a fresh `price` and `ts`
- Recomputes indicators and TS/NS/CDS on every run
- Writes unified feed with NO top-level `now` and NO per-symbol `now`
- Logs TS/NS/CDS distribution percentiles and writes to data/score_stats.json
"""

import os, json, math, statistics
from datetime import datetime, timezone
from pathlib import Path

# ---- ENV ----
INPUT_PATH  = os.getenv("INPUT_PATH",  "data/prices.json")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "data/prices_final.json")

# indicator windows (match what you used before)
EMA_FAST = int(os.getenv("EMA_FAST", "12"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "26"))
RSI_LEN  = int(os.getenv("RSI_LEN",  "14"))
SMA_LEN  = int(os.getenv("SMA_LEN",  "20"))

# News decay/weighting knobs (used to compute NS every run)
DECAY_HALF_LIFE_HOURS = float(os.getenv("DECAY_HALF_LIFE_HOURS", "12"))
SECTOR_NUDGE = float(os.getenv("SECTOR_NUDGE", "0.05"))  # optional tiny nudge
NS_DEFAULT_WT = float(os.getenv("NS_DEFAULT_WT", "0.4"))
TS_DEFAULT_WT = float(os.getenv("TS_DEFAULT_WT", "0.6"))
NS_STRONG_ABS = float(os.getenv("NS_STRONG_ABS", "0.7"))  # flip weights if |NS| >= this
SCORE_STATS_OUT = os.getenv("SCORE_STATS_OUT", "data/score_stats.json")

# ---- helpers ----
def utcnow_iso():
    return datetime.now(timezone.utc).isoformat()

def read_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def last_price_and_ts(node):
    """
    Pull best-available latest price & timestamp from the base file.
    Expected shapes seen historically:
      node["bars"] = [{"t": "...", "c": number}, ...]  # daily or intraday
      node["price"] and node["ts"] may exist already
    Preference order:
      - most-recent bar close
      - existing "price" with "ts"
    """
    bars = node.get("bars") or []
    if bars:
        b = bars[-1]
        px = float(b.get("c", 0.0) or 0.0)
        ts = b.get("t")
        if px > 0 and ts:
            return px, ts
    # fallback to existing
    px = float(node.get("price") or 0.0)
    ts = node.get("ts")
    return px, ts

# --- Indicators (simple, deterministic) ---
def ema(values, n):
    if not values: return []
    k = 2.0/(n+1.0)
    out = []
    s = values[0]
    out.append(s)
    for v in values[1:]:
        s = v*k + s*(1-k)
        out.append(s)
    return out

def sma(values, n):
    out = []
    s = 0.0
    q = []
    for v in values:
        q.append(v); s += v
        if len(q) > n:
            s -= q.pop(0)
        out.append(s / len(q))
    return out

def rsi(values, n):
    if len(values) < 2: return [50.0]*len(values)
    gains = [0.0]; losses=[0.0]
    for i in range(1,len(values)):
        d = values[i]-values[i-1]
        gains.append(max(d,0.0))
        losses.append(max(-d,0.0))
    ema_g = ema(gains, n)
    ema_l = ema(losses, n)
    out = []
    for g,l in zip(ema_g, ema_l):
        if l == 0: out.append(100.0)
        else:
            rs = g/l
            out.append(100.0 - (100.0/(1.0+rs)))
    return out

def macd(values, fast=12, slow=26, sig=9):
    if not values: return [], [], []
    e_fast = ema(values, fast)
    e_slow = ema(values, slow)
    line = [a-b for a,b in zip(e_fast, e_slow)]
    signal = ema(line, sig)
    hist = [a-b for a,b in zip(line, signal)]
    return line, signal, hist

def clamp_unit(x):
    # clamp to [-1,1]
    return max(-1.0, min(1.0, x))

def compute_TS(ind):
    """Scale indicators into [-1,1] then combine."""
    # Normalize MACD hist slope and RSI pos & EMA slope lightly
    rsi_val = ind.get("rsi")
    ema_fast = ind.get("ema_fast")
    ema_slow = ind.get("ema_slow")
    macd_hist = ind.get("macd_hist")
    sma20 = ind.get("sma20")

    # simple slopes (last - prev)
    def slope(arr):
        return 0.0 if (arr is None or len(arr)<2) else (arr[-1] - arr[-2])

    ts_rsi = 0.0 if rsi_val is None else clamp_unit((rsi_val-50.0)/50.0)
    ts_macd = 0.0 if macd_hist is None else clamp_unit(macd_hist/ max(1e-6, abs(macd_hist)+1e-6))  # relative
    ts_ema = 0.0
    if ema_fast is not None and ema_slow is not None:
        ts_ema = clamp_unit((ema_fast - ema_slow)/max(1e-6, abs(ema_slow)))

    # 0.4 * MACD + 0.4 * RSI + 0.2 * EMA diff
    out = 0.4*ts_macd + 0.4*ts_rsi + 0.2*ts_ema
    # tiny bonus for price> SMA20 (trend confirmation)
    if sma20 is not None and ind.get("price") is not None:
        if ind["price"] > sma20:
            out += 0.03
        else:
            out -= 0.03
    return clamp_unit(out)

def age_hours(dt_iso: str|None, now: datetime) -> float:
    if not dt_iso: return 9999.0
    try:
        t = datetime.fromisoformat(dt_iso.replace("Z","+00:00"))
        return max(0.0, (now - t).total_seconds()/3600.0)
    except:
        return 9999.0

def compute_NS(news_list, now_utc):
    """
    news_list entries expected to include:
      {"headline":..., "source":..., "ts": "... ISO ...", "tone": float in [-1,1]}
    Apply half-life decay on tone by age in hours. Average the remaining.
    """
    if not news_list: return 0.0
    hl = DECAY_HALF_LIFE_HOURS
    lam = math.log(2.0)/max(1e-6, hl)
    num=0.0; den=0.0
    for n in news_list:
        tone = n.get("tone")
        if tone is None: continue
        t_iso = n.get("ts")
        h = age_hours(t_iso, now_utc)
        w = math.exp(-lam*h)
        num += float(tone)*w
        den += w
    if den <= 0: return 0.0
    val = num/den
    # optional tiny sector nudge (if you store sector_tone on node)
    return clamp_unit(val)

def dynamic_weights(ns):
    if abs(ns) >= NS_STRONG_ABS:
        return 0.4, 0.6  # wT, wN
    return TS_DEFAULT_WT, NS_DEFAULT_WT

def percentiles(vals):
    if not vals:
        return {}
    xs = sorted(vals)
    def p(q):
        if len(xs)==1: return xs[0]
        k = (len(xs)-1)*q
        i = int(k); f=k-i
        if i+1 < len(xs):
            return xs[i]*(1-f) + xs[i+1]*f
        return xs[-1]
    return {
        "min": xs[0],
        "p25": p(0.25),
        "p50": p(0.50),
        "p75": p(0.75),
        "p90": p(0.90),
        "max": xs[-1],
        "mean": statistics.fmean(xs)
    }

def main():
    base = read_json(INPUT_PATH)
    syms = base.get("symbols") or {}
    now_iso = utcnow_iso()
    now_dt = datetime.now(timezone.utc)

    out = {
        "as_of_utc": now_iso,
        "timeframe": base.get("timeframe", "1Day"),
        "indicators_window": base.get("indicators_window", 200),
        "symbols": {}
    }

    ts_list = []; ns_list = []; cds_list = []

    for sym, node in syms.items():
        px, ts = last_price_and_ts(node)
        if px <= 0:  # skip unpriced
            continue

        # pull price history (close array) for indicators
        closes = [float(b.get("c", 0.0) or 0.0) for b in (node.get("bars") or []) if b.get("c") is not None]
        if not closes or closes[-1] != px:
            closes = (closes or []) + [px]

        # indicators
        e_fast = ema(closes, EMA_FAST)
        e_slow = ema(closes, EMA_SLOW)
        m_line, m_sig, m_hist = macd(closes, EMA_FAST, EMA_SLOW, 9)
        rsi_arr = rsi(closes, RSI_LEN)
        sma_arr = sma(closes, SMA_LEN)

        ind = {
            "ema_fast": e_fast[-1] if e_fast else None,
            "ema_slow": e_slow[-1] if e_slow else None,
            "macd": m_line[-1] if m_line else None,
            "macd_signal": m_sig[-1] if m_sig else None,
            "macd_hist": m_hist[-1] if m_hist else None,
            "macd_hist_prev": (m_hist[-2] if m_hist and len(m_hist) > 1 else (m_hist[-1] if m_hist else None)),
            "rsi": rsi_arr[-1] if rsi_arr else None,
            "sma20": sma_arr[-1] if sma_arr else None,
            "price": px
        }

        # news-based score
        news = node.get("news") or []   # fetch_news.py attaches raw items with `tone` and `ts`
        ns = compute_NS(news, now_dt)

        # technical score
        ts_val = compute_TS(ind)

        # composite
        wT, wN = dynamic_weights(ns)
        cds = clamp_unit(wT*ts_val + wN*ns + (SECTOR_NUDGE if node.get("sector_bias") else 0.0))

        # decision (v2.2 thresholds kept external if needed; we store only scores)
        out["symbols"][sym] = {
            "price": px,
            "ts": ts or now_iso,
            "indicators": ind,
            "signals": {
                "TS": round(ts_val, 6),
                "NS": round(ns, 6),
                "CDS": round(cds, 6),
                "wT": round(wT, 6),
                "wN": round(wN, 6)
            },
            "news": news  # unchanged list; consumer can slice MAX_ARTICLES later
        }

        ts_list.append(ts_val); ns_list.append(ns); cds_list.append(cds)

    # write feed
    write_json(OUTPUT_PATH, out)

    # stats
    stats = {
        "as_of_utc": now_iso,
        "counts": {"symbols": len(out["symbols"])},
        "TS": percentiles(ts_list),
        "NS": percentiles(ns_list),
        "CDS": percentiles(cds_list)
    }
    write_json(SCORE_STATS_OUT, stats)

    # console log
    def fmt(d):
        return ", ".join(f"{k}={v:.3f}" for k,v in d.items()) if d else "n/a"
    print("[score-stats] TS:", fmt(stats["TS"]))
    print("[score-stats] NS:", fmt(stats["NS"]))
    print("[score-stats] CDS:", fmt(stats["CDS"]))
    print(f"[OK] wrote {OUTPUT_PATH} and {SCORE_STATS_OUT}")

if __name__ == "__main__":
    main()
