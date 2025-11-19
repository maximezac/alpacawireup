#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple

# --- Indicator + scoring knobs (match publish_feed) ---
EMA_FAST = 12
EMA_SLOW = 26
RSI_LEN  = 14
SMA_LEN  = 20
MA_TREND_FAST = 50
MA_TREND_SLOW = 200

DECAY_HALF_LIFE_HOURS = 12.0
SECTOR_NUDGE = 0.05
NS_DEFAULT_WT = 0.4
TS_DEFAULT_WT = 0.6
NS_STRONG_ABS = 0.7


# ---------- basic JSON helpers ----------

def read_json(p: str) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(p: str, obj: dict) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ---------- indicator math (same as publish_feed) ----------

def ema(values, n):
    if not values:
        return []
    k = 2.0 / (n + 1.0)
    out = []
    s = values[0]
    out.append(s)
    for v in values[1:]:
        s = v * k + s * (1 - k)
        out.append(s)
    return out

def sma(values, n):
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

def rsi(values, n):
    if len(values) < 2:
        return [50.0] * len(values)
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(values)):
        d = values[i] - values[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ema_g = ema(gains, n)
    ema_l = ema(losses, n)
    out = []
    for g, l in zip(ema_g, ema_l):
        if l == 0:
            out.append(100.0)
        else:
            rs = g / l
            out.append(100.0 - (100.0 / (1.0 + rs)))
    return out

def macd(values, fast=12, slow=26, sig=9):
    if not values:
        return [], [], []
    e_fast = ema(values, fast)
    e_slow = ema(values, slow)
    line = [a - b for a, b in zip(e_fast, e_slow)]
    signal = ema(line, sig)
    hist = [a - b for a, b in zip(line, signal)]
    return line, signal, hist

def clamp_unit(x: float) -> float:
    return max(-1.0, min(1.0, x))


# ---------- TS / NS logic (daily-only) ----------

def compute_TS_daily(ind: dict) -> float:
    rsi_val        = ind.get("rsi")
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

    # vs SMA20
    if sma20 is not None and price is not None:
        if price > sma20:
            out += 0.03
        else:
            out -= 0.03

    # trend classification vs 50/200
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

    # momentum phase via RSI + MACD slope
    hist_slope = diff(macd_hist, macd_hist_prev)
    if rsi_val is not None:
        if rsi_val < 70 and hist_slope > 0:
            out += 0.04
        elif rsi_val >= 75 and hist_slope <= 0:
            out -= 0.06

    # don't chase extended moves
    if price is not None and sma20 is not None and rsi_val is not None:
        dist = (price - sma20) / max(1e-6, sma20)
        if dist > 0.07 and rsi_val > 70:
            out -= 0.08

    # one-bar spike dampening
    if price is not None and prev_close is not None and rsi_val is not None:
        day_change = (price - prev_close) / max(1e-6, prev_close)
        if day_change > 0.08 and rsi_val > 65:
            out -= 0.05

    return clamp_unit(out)

def age_hours(dt_iso: str | None, now: datetime) -> float:
    if not dt_iso:
        return 9999.0
    try:
        t = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
        return max(0.0, (now - t).total_seconds() / 3600.0)
    except Exception:
        return 9999.0

def compute_NS(news_list, now_utc: datetime) -> float:
    if not news_list:
        return 0.0
    hl = DECAY_HALF_LIFE_HOURS
    lam = math.log(2.0) / max(1e-6, hl)
    num = 0.0
    den = 0.0
    for n in news_list:
        tone = n.get("tone")
        if tone is None:
            continue
        t_iso = n.get("ts")
        h = age_hours(t_iso, now_utc)
        w = math.exp(-lam * h)
        num += float(tone) * w
        den += w
    if den <= 0:
        return 0.0
    val = num / den
    return clamp_unit(val)

def dynamic_weights(ns: float) -> Tuple[float, float]:
    if abs(ns) >= NS_STRONG_ABS:
        return 0.4, 0.6  # wT, wN
    return TS_DEFAULT_WT, NS_DEFAULT_WT


# ---------- main snapshot builder ----------

def build_snapshot(as_of: str, input_path: str, output_path: str) -> None:
    base = read_json(input_path)
    syms = base.get("symbols") or {}

    as_of_date = datetime.fromisoformat(as_of).date()
    # treat snapshot as “end of that day” UTC
    as_of_dt = datetime(as_of_date.year, as_of_date.month, as_of_date.day, 23, 59, 59, tzinfo=timezone.utc)
    as_of_iso = as_of_dt.isoformat().replace("+00:00", "Z")

    out = {
        "as_of_utc": as_of_iso,
        "timeframe": base.get("timeframe", "1Day"),
        "indicators_window": base.get("indicators_window", 200),
        "symbols": {}
    }

    ts_list: List[float] = []
    ns_list: List[float] = []
    cds_list: List[float] = []

    for sym, node in syms.items():
        bars = node.get("bars") or []

        # 1) trim daily bars to <= as_of_date
        bars_trim = []
        for b in bars:
            t = b.get("t")
            if not t:
                continue
            try:
                d = datetime.fromisoformat(t.replace("Z", "+00:00")).date()
            except Exception:
                continue
            if d <= as_of_date:
                bars_trim.append(b)

        if not bars_trim:
            continue

        closes = [float(b.get("c", 0.0) or 0.0) for b in bars_trim if b.get("c") is not None]
        if not closes:
            continue

        last_bar = bars_trim[-1]
        px = float(last_bar.get("c", 0.0) or 0.0)
        ts = last_bar.get("t") or as_of_iso
        if px <= 0:
            continue

        # 2) recompute daily indicators on trimmed closes
        e_fast = ema(closes, EMA_FAST)
        e_slow = ema(closes, EMA_SLOW)
        m_line, m_sig, m_hist = macd(closes, EMA_FAST, EMA_SLOW, 9)
        rsi_arr = rsi(closes, RSI_LEN)
        sma_arr = sma(closes, SMA_LEN)
        sma50_arr = sma(closes, MA_TREND_FAST)
        sma200_arr = sma(closes, MA_TREND_SLOW)

        prev_close = closes[-2] if len(closes) > 1 else closes[-1]
        macd_hist_prev_daily = m_hist[-2] if len(m_hist) > 1 else (m_hist[-1] if m_hist else None)
        rsi_prev_daily = rsi_arr[-2] if len(rsi_arr) > 1 else (rsi_arr[-1] if rsi_arr else None)

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
            "price": px,
        }

        # 3) trim news to <= as_of_dt
        news_all = node.get("news") or []
        news_trim = []
        for n in news_all:
            t_iso = n.get("ts")
            if not t_iso:
                continue
            try:
                t = datetime.fromisoformat(t_iso.replace("Z", "+00:00"))
            except Exception:
                continue
            if t <= as_of_dt:
                news_trim.append(n)

        # 4) recompute NS / TS / CDS
        ns = compute_NS(news_trim, as_of_dt)
        ts_val = compute_TS_daily(ind_daily)
        wT, wN = dynamic_weights(ns)
        cds = clamp_unit(wT * ts_val + wN * ns + (SECTOR_NUDGE if node.get("sector_bias") else 0.0))

        # 5) simple guidance from SMA20
        guidance = {}
        sma20_last = ind_daily.get("sma20")
        if sma20_last is not None:
            guidance["sma20"] = round(sma20_last, 4)
            guidance["buy_on_dip_below"] = round(sma20_last * 0.99, 4)
            guidance["trim_above"] = round(sma20_last * 1.08, 4)

        out["symbols"][sym] = {
            "price": px,
            "ts": ts,
            "indicators": {
                "daily": ind_daily,
                "intraday": {}  # we’re daily-only in backtest
            },
            "signals": {
                "TS": round(ts_val, 6),
                "TS_daily": round(ts_val, 6),
                "TS_intraday": None,
                "NS": round(ns, 6),
                "CDS": round(cds, 6),
                "wT": round(wT, 6),
                "wN": round(wN, 6),
            },
            "guidance": guidance,
            "news": news_trim,
        }

        ts_list.append(ts_val)
        ns_list.append(ns)
        cds_list.append(cds)

    write_json(output_path, out)
    print(f"[OK] snapshot built for {as_of} → {output_path} with {len(out['symbols'])} symbols.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", required=True, help="Snapshot date YYYY-MM-DD")
    ap.add_argument("--input", required=True, help="Backfill JSON (prices_backfill.json)")
    ap.add_argument("--output", required=True, help="Output snapshot JSON (prices_final_snapshot.json)")
    args = ap.parse_args()
    build_snapshot(args.as_of, args.input, args.output)


if __name__ == "__main__":
    main()
