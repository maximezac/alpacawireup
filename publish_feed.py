#!/usr/bin/env python3
"""
publish_feed.py  (LIVE-ONLY MODE, v2.3 TS upgrade)

- Reads base prices (historical bars + metadata) from INPUT_PATH
- Ensures each symbol has a fresh `price` and `ts`
- Recomputes indicators and TS/NS/CDS on every run
- Prefers intraday (5Min) indicators for TS when available
- Writes unified feed with NO top-level `now` and NO per-symbol `now`
- Logs TS/NS/CDS distribution percentiles and writes to data/score_stats.json
"""

import os, json, math, statistics
from datetime import datetime, timezone
from pathlib import Path

# ---- ENV ----
INPUT_PATH  = os.getenv("INPUT_PATH",  "data/prices.json")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "data/prices_final.json")

# indicator windows (daily context; TS itself uses its own intraday inputs)
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

# extra MAs for trend context (daily, not used directly in TS anymore)
MA_TREND_FAST = int(os.getenv("MA_TREND_FAST", "50"))
MA_TREND_SLOW = int(os.getenv("MA_TREND_SLOW", "200"))


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

    Preference order:
      - most-recent intraday bar close (bars_5m)
      - most-recent daily bar close (bars)
      - existing "price" with "ts"
    """
    # Prefer intraday (5m) if present
    bars_5m = node.get("bars_5m") or []
    if bars_5m:
        b = bars_5m[-1]
        px = float(b.get("c", 0.0) or 0.0)
        ts = b.get("t")
        if px > 0 and ts:
            return px, ts

    # Fall back to daily bars
    bars = node.get("bars") or []
    if bars:
        b = bars[-1]
        px = float(b.get("c", 0.0) or 0.0)
        ts = b.get("t")
        if px > 0 and ts:
            return px, ts

    # Fallback to existing
    px = float(node.get("price") or 0.0)
    ts = node.get("ts")
    return px, ts

# --- Indicators (simple, deterministic) --- (daily context)
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

# --- TS logic ---

def compute_TS_daily(ind):
    """
    Original daily TS logic (kept as a fallback if intraday data is missing).

    Uses:
      - ema_fast, ema_slow
      - macd_hist, macd_hist_prev
      - rsi, rsi_prev
      - sma20, ma50, ma200
      - price, prev_close
    """
    rsi_val        = ind.get("rsi")
    rsi_prev       = ind.get("rsi_prev")
    ema_fast       = ind.get("ema_fast")
    ema_slow       = ind.get("ema_slow")
    macd_hist      = ind.get("macd_hist")
    macd_hist_prev = ind.get("macd_hist_prev")
    sma20          = ind.get("sma20")
    ma50           = ind.get("ma50")
    ma200          = ind.get("ma200")
    price          = ind.get("price")
    prev_close     = ind.get("prev_close")

    def diff(curr, prev):
        if curr is None or prev is None:
            return 0.0
        return curr - prev

    ts_rsi = 0.0 if rsi_val is None else clamp_unit((rsi_val - 50.0) / 50.0)

    ts_macd = 0.0
    if macd_hist is not None:
        ts_macd = clamp_unit(macd_hist / (abs(macd_hist) + 1e-6))

    ts_ema = 0.0
    if ema_fast is not None and ema_slow is not None:
        ts_ema = clamp_unit((ema_fast - ema_slow) / max(1e-6, abs(ema_slow)))

    out = 0.4 * ts_macd + 0.4 * ts_rsi + 0.2 * ts_ema

    # tiny bonus/penalty vs SMA20
    if sma20 is not None and price is not None:
        if price > sma20:
            out += 0.03
        else:
            out -= 0.03

    # Trend classification (up / down / chop)
    trend = "chop"
    if price is not None and ma50 is not None and ma200 is not None:
        if price > ma50 > ma200:
            trend = "up"
        elif price < ma50 < ma200:
            trend = "down"

    if trend == "up":
        out += 0.08
    elif trend == "down":
        out -= 0.08

    # Momentum phase via RSI + MACD hist slope
    hist_slope = diff(macd_hist, macd_hist_prev)
    if rsi_val is not None:
        if rsi_val < 70 and hist_slope > 0:
            out += 0.04
        elif rsi_val >= 75 and hist_slope <= 0:
            out -= 0.06

    # Don't chase extension: way above SMA20 + hot RSI
    if price is not None and sma20 is not None and rsi_val is not None:
        dist = (price - sma20) / max(1e-6, sma20)
        if dist > 0.07 and rsi_val > 70:
            out -= 0.08

    # One-bar spike dampening
    if price is not None and prev_close is not None and rsi_val is not None:
        day_change = (price - prev_close) / max(1e-6, prev_close)
        if day_change > 0.08 and rsi_val > 65:
            out -= 0.05

    return clamp_unit(out)

def compute_TS_intraday(ind):
    """
    Intraday TS (v2.3):

      TS = 0.40 * MACD_hist_slope
         + 0.35 * RSI_position
         + 0.25 * EMA_trend_strength

    Expected `ind` keys:
      - rsi         (intraday RSI)
      - macd_hist   (last intraday MACD histogram)
      - macd_hist_prev (previous intraday MACD histogram)
      - ema50       (intraday EMA50)
      - ema200      (intraday EMA200)
      - price       (latest price, ideally 5m close)
    """
    rsi_val        = ind.get("rsi")
    macd_hist      = ind.get("macd_hist")
    macd_hist_prev = ind.get("macd_hist_prev")
    ema50          = ind.get("ema50")
    ema200         = ind.get("ema200")
    price          = ind.get("price")

    # RSI component: scaled so 50 is neutral, 0/100 ~ +/-1
    if rsi_val is None:
        ts_rsi = 0.0
    else:
        ts_rsi = clamp_unit((rsi_val - 50.0) / 50.0)

    # MACD slope component
    ts_macd_slope = 0.0
    if macd_hist is not None and macd_hist_prev is not None:
        slope = macd_hist - macd_hist_prev
        denom = abs(macd_hist_prev) + 1e-3
        ts_macd_slope = clamp_unit(slope / denom)

    # EMA trend strength: distance from EMA50 and EMA200
    ts_ema_trend = 0.0
    if price is not None and ema50 is not None and ema200 is not None and price != 0:
        dist50 = (price - ema50) / price  # fraction above/below EMA50
        dist200 = (price - ema200) / price
        # normalize: ~10% distance maps near +/-1
        n50 = clamp_unit(dist50 / 0.10)
        n200 = clamp_unit(dist200 / 0.10)
        ts_ema_trend = clamp_unit(0.5 * n50 + 0.5 * n200)

    ts = (
        0.40 * ts_macd_slope +
        0.35 * ts_rsi +
        0.25 * ts_ema_trend
    )
    return clamp_unit(ts)

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
        # pass through intraday metadata if present (from fetch_prices.py)
        "intraday": base.get("intraday") or {},
        "symbols": {}
    }

    ts_list = []; ns_list = []; cds_list = []

    for sym, node in syms.items():
        px, ts = last_price_and_ts(node)
        if px <= 0:  # skip unpriced
            continue

        # ----- DAILY CONTEXT (for guidance / context, not the main TS anymore) -----
        daily_bars = node.get("bars") or []
        closes_daily = [float(b.get("c", 0.0) or 0.0) for b in daily_bars if b.get("c") is not None]

        e_fast = ema(closes_daily, EMA_FAST) if closes_daily else []
        e_slow = ema(closes_daily, EMA_SLOW) if closes_daily else []
        m_line, m_sig, m_hist = macd(closes_daily, EMA_FAST, EMA_SLOW, 9) if closes_daily else ([], [], [])
        rsi_arr = rsi(closes_daily, RSI_LEN) if closes_daily else []
        sma_arr = sma(closes_daily, SMA_LEN) if closes_daily else []

        sma50_arr = sma(closes_daily, MA_TREND_FAST) if closes_daily else []
        sma200_arr = sma(closes_daily, MA_TREND_SLOW) if closes_daily else []

        prev_close = closes_daily[-1] if closes_daily else None  # previous daily close (context)
        macd_hist_prev_daily = m_hist[-2] if m_hist and len(m_hist) > 1 else (m_hist[-1] if m_hist else None)
        rsi_prev_daily = rsi_arr[-2] if rsi_arr and len(rsi_arr) > 1 else (rsi_arr[-1] if rsi_arr else None)

        ind_daily = {
            "ema_fast": e_fast[-1] if e_fast else None,
            "ema_slow": e_slow[-1] if e_slow else None,
            "macd": m_line[-1] if m_line else None,
            "macd_signal": m_sig[-1] if m_sig else None,
            "macd_hist": m_hist[-1] if m_hist else None,
            "macd_hist_prev": macd_hist_prev_daily,
            "rsi": rsi_arr[-1] if rsi_arr else None,
            "rsi_prev": rsi_prev_daily,
            "sma20": sma_arr[-1] if sma_arr else None,
            "ma50": sma50_arr[-1] if sma50_arr else None,
            "ma200": sma200_arr[-1] if sma200_arr else None,
            "prev_close": prev_close,
            "price": px
        }

        # ----- INTRADAY INPUTS (preferred for TS) -----
        intr = node.get("indicators_5m") or {}
        rsi_5m        = intr.get("rsi14_5m")
        macd_hist_5m  = intr.get("macd_hist_5m")
        macd_hist_prev_5m = intr.get("macd_hist_prev_5m")
        ema50_5m      = intr.get("ema50_5m")
        ema200_5m     = intr.get("ema200_5m")

        ind_intraday_for_ts = {
            "rsi": rsi_5m,
            "macd_hist": macd_hist_5m,
            "macd_hist_prev": macd_hist_prev_5m,
            "ema50": ema50_5m,
            "ema200": ema200_5m,
            "price": px,
        }

        has_intraday_ts = all([
            rsi_5m is not None,
            macd_hist_5m is not None,
            macd_hist_prev_5m is not None,
            ema50_5m is not None,
            ema200_5m is not None,
        ])

        # news-based score
        news = node.get("news") or []   # fetch_news.py attaches raw items with `tone` and `ts`
        ns = compute_NS(news, now_dt)

        # technical score (TS)
        ts_daily = compute_TS_daily(ind_daily)
        ts_intraday = compute_TS_intraday(ind_intraday_for_ts) if has_intraday_ts else None

        # Primary TS we expose to downstream logic:
        ts_val = ts_intraday if ts_intraday is not None else ts_daily

        # composite
        wT, wN = dynamic_weights(ns)
        cds = clamp_unit(wT*ts_val + wN*ns + (SECTOR_NUDGE if node.get("sector_bias") else 0.0))

        # simple guideline levels anchored on daily SMA20 if available
        guidance = {}
        sma20_last = ind_daily.get("sma20")
        if sma20_last is not None:
            guidance["sma20"] = round(sma20_last, 4)
            guidance["buy_on_dip_below"] = round(sma20_last * 0.99, 4)  # ~1% under SMA20
            guidance["trim_above"] = round(sma20_last * 1.08, 4)        # ~8% above SMA20

        # We include both daily + intraday TS in indicators for transparency
        indicators_out = {
            "daily": ind_daily,
            "intraday": {
                "rsi": rsi_5m,
                "macd_hist": macd_hist_5m,
                "macd_hist_prev": macd_hist_prev_5m,
                "ema50": ema50_5m,
                "ema200": ema200_5m,
                "price": px,
            }
        }

        out["symbols"][sym] = {
            "price": px,
            "ts": ts or now_iso,
            "indicators": indicators_out,
            "signals": {
                "TS": round(ts_val, 6),
                "TS_daily": round(ts_daily, 6),
                "TS_intraday": round(ts_intraday, 6) if ts_intraday is not None else None,
                "NS": round(ns, 6),
                "CDS": round(cds, 6),
                "wT": round(wT, 6),
                "wN": round(wN, 6)
            },
            "guidance": guidance,
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
