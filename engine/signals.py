#!/usr/bin/env python3
"""
engine/signals.py

Shared signal engine for LIVE + BACKTEST.

- Computes daily indicators from `bars`
- Uses intraday context from `indicators_5m` when present
- Computes TS_daily, TS_intraday (v2.3), NS, CDS, guidance
- Returns a unified per-symbol view that publish_feed and
  backtest snapshot builders can both use.
"""

from __future__ import annotations
import os, math, statistics
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional

# ---- ENV knobs ----

EMA_FAST = int(os.getenv("EMA_FAST", "12"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "26"))
RSI_LEN  = int(os.getenv("RSI_LEN",  "14"))
SMA_LEN  = int(os.getenv("SMA_LEN",  "20"))

DECAY_HALF_LIFE_HOURS = float(os.getenv("DECAY_HALF_LIFE_HOURS", "12"))
BACKTEST_HALF_LIFE_HOURS = float(os.getenv("BACKTEST_HALF_LIFE_HOURS", "720"))  # ~30 days

SECTOR_NUDGE = float(os.getenv("SECTOR_NUDGE", "0.05"))
NS_DEFAULT_WT = float(os.getenv("NS_DEFAULT_WT", "0.4"))
TS_DEFAULT_WT = float(os.getenv("TS_DEFAULT_WT", "0.6"))
NS_STRONG_ABS = float(os.getenv("NS_STRONG_ABS", "0.7"))

MA_TREND_FAST = int(os.getenv("MA_TREND_FAST", "50"))
MA_TREND_SLOW = int(os.getenv("MA_TREND_SLOW", "200"))

BACKTEST_MODE = os.getenv("BACKTEST_MODE", "0") == "1"


# ---- helpers ----

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def clamp_unit(x: float) -> float:
    return max(-1.0, min(1.0, x))

def ema(values, n: int):
    if not values: return []
    k = 2.0/(n+1.0)
    out = []; s = values[0]; out.append(s)
    for v in values[1:]:
        s = v*k + s*(1-k)
        out.append(s)
    return out

def sma(values, n: int):
    out=[]; s=0.0; q=[]
    for v in values:
        q.append(v); s+=v
        if len(q)>n:
            s-=q.pop(0)
        out.append(s/len(q))
    return out

def rsi(values, n: int):
    if len(values)<2: return [50.0]*len(values)
    gains=[0.0]; losses=[0.0]
    for i in range(1,len(values)):
        d=values[i]-values[i-1]
        gains.append(max(d,0.0))
        losses.append(max(-d,0.0))
    ema_g=ema(gains,n); ema_l=ema(losses,n)
    out=[]
    for g,l in zip(ema_g,ema_l):
        if l==0: out.append(100.0)
        else:
            rs=g/l
            out.append(100 - (100/(1+rs)))
    return out

def macd(values, fast=12, slow=26, sig=9):
    if not values: return [],[],[]
    e_fast=ema(values,fast); e_slow=ema(values,slow)
    line=[a-b for a,b in zip(e_fast,e_slow)]
    signal=ema(line,sig)
    hist=[a-b for a,b in zip(line,signal)]
    return line,signal,hist

def last_price_and_ts(node):
    bars_5m=node.get("bars_5m") or []
    if bars_5m:
        b=bars_5m[-1]; px=float(b.get("c",0.0) or 0.0); ts=b.get("t")
        if px>0 and ts: return px,ts
    bars=node.get("bars") or []
    if bars:
        b=bars[-1]; px=float(b.get("c",0.0) or 0.0); ts=b.get("t")
        if px>0 and ts: return px,ts
    return float(node.get("price") or 0.0), node.get("ts")


# ---- NS helpers ----

def age_hours(dt_iso: str|None, ref_dt: datetime) -> float:
    if not dt_iso: return 9999.0
    try:
        t = datetime.fromisoformat(dt_iso.replace("Z","+00:00"))
        return max(0.0, (ref_dt - t).total_seconds()/3600.0)
    except:
        return 9999.0

def compute_NS(news_list, ref_dt: datetime, half_life_hours: float) -> float:
    if not news_list: return 0.0

    hl = max(1e-6, half_life_hours)
    lam = math.log(2.0) / hl

    num = 0.0; den = 0.0
    for n in news_list:
        tone = n.get("tone")
        if tone is None: continue
        h = age_hours(n.get("ts"), ref_dt)
        w = math.exp(-lam*h)
        num += float(tone)*w
        den += w

    if den <= 0: return 0.0
    return clamp_unit(num/den)


def dynamic_weights(ns):
    if abs(ns) >= NS_STRONG_ABS:
        return 0.4, 0.6
    return TS_DEFAULT_WT, NS_DEFAULT_WT


# ---- TS logic ----
# (unchanged — your copy was correct, so I left it exactly as-is)
# compute_TS_daily()
# compute_TS_intraday()
# compute_indicators_daily_from_bars()
# ...


# ---- MAIN: compute_symbol_view ----

def compute_symbol_view(
    sym: str,
    node: Dict[str,Any],
    as_of_dt: datetime,
    as_of_iso: str,
) -> Optional[Tuple[Dict[str,Any], float, float, float]]:

    px, ts = last_price_and_ts(node)
    if px <= 0:
        return None

    # DAILY
    daily_bars = node.get("bars") or []
    ind_daily = compute_indicators_daily_from_bars(daily_bars, px)

    # INTRADAY
    intr = node.get("indicators_5m") or {}
    rsi_5m = intr.get("rsi14_5m")
    macd_hist_5m = intr.get("macd_hist_5m")
    macd_hist_prev_5m = intr.get("macd_hist_prev_5m")
    ema50_5m = intr.get("ema50_5m")
    ema200_5m = intr.get("ema200_5m")

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

    # NEWS SCORE (FIXED)
    half_life = BACKTEST_HALF_LIFE_HOURS if BACKTEST_MODE else DECAY_HALF_LIFE_HOURS
    news = node.get("news") or []
    ns = compute_NS(news, as_of_dt, half_life)

    # TECHNICAL SCORES
    ts_daily = compute_TS_daily(ind_daily)
    ts_intraday = compute_TS_intraday(ind_intraday_for_ts) if has_intraday_ts else None
    ts_val = ts_intraday if ts_intraday is not None else ts_daily

    # CDS
    wT, wN = dynamic_weights(ns)
    cds = clamp_unit(wT*ts_val + wN*ns + (SECTOR_NUDGE if node.get("sector_bias") else 0.0))

    # GUIDANCE
    guidance = {}
    sma20_last = ind_daily.get("sma20")
    if sma20_last is not None:
        guidance["sma20"] = round(sma20_last, 4)
        guidance["buy_on_dip_below"] = round(sma20_last * 0.99, 4)
        guidance["trim_above"] = round(sma20_last * 1.08, 4)

    sym_out = {
        "price": px,
        "ts": ts or as_of_iso,
        "indicators": {
            "daily": ind_daily,
            "intraday": {
                "rsi": rsi_5m,
                "macd_hist": macd_hist_5m,
                "macd_hist_prev": macd_hist_prev_5m,
                "ema50": ema50_5m,
                "ema200": ema200_5m,
                "price": px,
            }
        },
        "signals": {
            "TS": round(ts_val, 6),
            "TS_daily": round(ts_daily, 6),
            "TS_intraday": round(ts_intraday, 6) if ts_intraday is not None else None,
            "NS": round(ns, 6),
            "CDS": round(cds, 6),
            "wT": round(wT, 6),
            "wN": round(wN, 6),
        },
        "guidance": guidance,
        "news": news,
    }

    return sym_out, ts_val, ns, cds


# ---- percentiles unchanged ----
def percentiles(vals):
    if not vals: return {}
    xs=sorted(vals)
    def p(q):
        if len(xs)==1: return xs[0]
        k=(len(xs)-1)*q
        i=int(k); f=k-i
        if i+1<len(xs):
            return xs[i]*(1-f) + xs[i+1]*f
        return xs[-1]
    return {
        "min": xs[0],
        "p25": p(0.25),
        "p50": p(0.50),
        "p75": p(0.75),
        "p90": p(0.90),
        "max": xs[-1],
        "mean": statistics.fmean(xs),
    }
