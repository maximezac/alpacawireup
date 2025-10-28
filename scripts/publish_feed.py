#!/usr/bin/env python3
# (see docstring for details)
"""
publish_feed.py
Builds an enriched market feed with technical + news sentiment signals and writes a final JSON. 

Highlights:
- Robust News Sentiment (NS) using VADER with 1-day half-life decay, ticker relevance weighting,
  optional sector cluster bias, and transparent ns_debug.
- Technical Score (TS) with fallbacks when full indicator history isn't present.
- Composite v2.2 logic: dynamic weighting (wT/wN), CDS, and decision thresholds.
- Accepts input from a URL or local path (default: data/prices.json). Writes to data/prices_final.json.

Environment:
- FEED_URL (optional): load JSON from URL if set.
- INPUT_PATH (optional): local input path; default "data/prices.json" when FEED_URL not set.
- OUTPUT_PATH (optional): output path; default "data/prices_final.json".
"""

import os
import re
import json
import math
import typing as _t
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_VADER = SentimentIntensityAnalyzer()
_TZUTC = timezone.utc
_HALF_LIFE_DAYS = 1.0
_LN2 = math.log(2.0)

DEFAULT_INPUT_PATH = "data/prices.json"
DEFAULT_OUTPUT_PATH = "data/prices_final.json"


def parse_dt(ts: str) -> datetime:
    if not ts:
        return datetime.now(tz=_TZUTC)
    try:
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.now(tz=_TZUTC)


def recency_weight(article_time: datetime, now: datetime) -> float:
    dt_days = max(0.0, (now - article_time).total_seconds() / 86400.0)
    return math.exp(-_LN2 * (dt_days / _HALF_LIFE_DAYS))


def relevance_weight(text: str, ticker: str) -> float:
    if not text or not ticker:
        return 1.0
    t = ticker.upper()
    s = text.upper()
    if re.search(rf"\\b{re.escape(t)}\\b", s):
        return 1.3
    if f"{t}:" in s or f"{t}/" in s or f"/{t}" in s:
        return 1.15
    return 1.0


def clamp_unit(x: float) -> float:
    return max(-1.0, min(1.0, x))


def safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


@dataclass
class TSComponents:
    rsi_pos: float = 0.0
    macd_hist_signal: float = 0.0
    ema_slope: float = 0.0
    debug_source: str = "full"


def compute_ts_from_indicators(symbol_obj: dict):
    price = safe_get(symbol_obj, "price", default=None)
    ind   = safe_get(symbol_obj, "indicators", default={}) or {}
    ema20 = ind.get("ema20")
    ema26 = ind.get("ema26")
    macd  = ind.get("macd")
    macd_signal = ind.get("macd_signal")
    macd_hist = ind.get("macd_hist")
    rsi14 = ind.get("rsi14")

    comp = TSComponents()
    dbg = {
        "inputs_present": {
            "price": price is not None,
            "ema20": ema20 is not None,
            "ema26": ema26 is not None,
            "macd": macd is not None,
            "macd_signal": macd_signal is not None,
            "macd_hist": macd_hist is not None,
            "rsi14": rsi14 is not None,
        }
    }

    if isinstance(rsi14, (int, float)):
        comp.rsi_pos = clamp_unit((rsi14 - 50.0) / 50.0)
    else:
        comp.rsi_pos = 0.0

    if isinstance(macd_hist, (int, float)):
        comp.macd_hist_signal = clamp_unit(macd_hist / 2.0)
    elif macd is not None and macd_signal is not None:
        comp.macd_hist_signal = clamp_unit((macd - macd_signal) / 2.0)
        comp.debug_source = "fallback_macd_minus_signal"
    else:
        comp.macd_hist_signal = 0.0
        comp.debug_source = "fallback_no_macd_hist"

    ema_for_slope = ema20 if isinstance(ema20, (int, float)) else ema26 if isinstance(ema26, (int, float)) else None
    if isinstance(price, (int, float)) and isinstance(ema_for_slope, (int, float)) and ema_for_slope:
        comp.ema_slope = clamp_unit((price - ema_for_slope) / ema_for_slope)
    else:
        comp.ema_slope = 0.0
        if comp.debug_source == "full":
            comp.debug_source = "fallback_no_ema_slope"

    ts = 0.4 * comp.macd_hist_signal + 0.4 * comp.rsi_pos + 0.2 * comp.ema_slope
    ts = clamp_unit(ts)

    ts_debug = {
        "rsi_pos": round(comp.rsi_pos, 4),
        "macd_hist_slope": round(comp.macd_hist_signal, 4),
        "ema20_slope": round(comp.ema_slope, 4),
        "debug_source": comp.debug_source,
        "TS": round(ts, 4),
    }
    return ts, ts_debug


def compute_news_sentiment_for_symbol(symbol: str, news_items, now, sector_avg_hint=None):
    used_scores = []
    used_weights = []
    article_ids = []

    for idx, it in enumerate(news_items or []):
        headline = (it.get("headline") or it.get("title") or "").strip()
        summary  = (it.get("summary") or it.get("description") or "").strip()
        text = headline if headline else summary
        if not text:
            continue

        ts_raw = it.get("ts") or it.get("time") or it.get("published_at") or it.get("date") or ""
        at = parse_dt(ts_raw) if ts_raw else now

        base = _VADER.polarity_scores(text)["compound"]
        if base == 0.0 and summary and summary != headline:
            base = _VADER.polarity_scores(summary)["compound"]

        w_recency = recency_weight(at, now)
        w_rel     = relevance_weight(f"{headline} {summary}", symbol)

        w = w_recency * w_rel
        if w <= 1e-9:
            continue

        used_scores.append(base)
        used_weights.append(w)
        article_ids.append({
            "i": idx,
            "ts": at.isoformat(),
            "headline": headline[:140],
            "compound": round(base, 4),
            "w_recency": round(w_recency, 4),
            "w_rel": round(w_rel, 2),
            "w_final": round(w, 4)
        })

    debug = {"source": "vader", "articles_used": len(used_scores), "articles_detail": article_ids}

    if not used_scores:
        debug["source"] = "default_neutral"
        return 0.0, debug

    num = sum(s * w for s, w in zip(used_scores, used_weights))
    den = sum(used_weights)
    ns = num / den if den > 0 else 0.0

    if sector_avg_hint is not None:
        if (sector_avg_hint < -0.25 and ns > 0):
            ns = ns - 0.10
            debug["sector_bias"] = "downward_0.10"
        elif (sector_avg_hint > 0.25 and ns < 0):
            ns = ns + 0.10
            debug["sector_bias"] = "upward_0.10"

    ns = clamp_unit(ns)
    debug["ns_final"] = round(ns, 4)
    return ns, debug


def enrich_with_news_sentiment(symbols: dict) -> dict:
    now = datetime.now(tz=_TZUTC)

    tmp_ns, tmp_debug = {}, {}
    for sym, obj in symbols.items():
        news = obj.get("news") or []
        ns, dbg = compute_news_sentiment_for_symbol(sym, news, now, sector_avg_hint=None)
        tmp_ns[sym] = ns
        tmp_debug[sym] = dbg

    sector_buckets = {}
    for sym, obj in symbols.items():
        sector = (obj.get("sector") or "Other").strip()
        sector_buckets.setdefault(sector, []).append(tmp_ns[sym])
    sector_avg = {sec: (sum(v)/len(v) if v else 0.0) for sec, v in sector_buckets.items()}

    final_ns, final_debug = {}, {}
    for sym, obj in symbols.items():
        sector = (obj.get("sector") or "Other").strip()
        news = obj.get("news") or []
        ns, dbg = compute_news_sentiment_for_symbol(sym, news, now, sector_avg_hint=sector_avg.get(sector))
        final_ns[sym] = ns
        final_debug[sym] = dbg

    for sym, obj in symbols.items():
        sig = obj.setdefault("signals", {})
        sig["NS"] = round(final_ns[sym], 4)
        sig["ns_debug"] = final_debug[sym]

    return symbols


def apply_v22_weights_and_decisions(symbols: dict) -> dict:
    for sym, obj in symbols.items():
        sig = obj.setdefault("signals", {})
        ts_val = float(sig.get("TS", 0.0))
        ns_val = float(sig.get("NS", 0.0))

        if abs(ns_val) >= 0.7:
            wN, wT = 0.6, 0.4
        else:
            wT, wN = 0.6, 0.4

        sig["wT"] = wT
        sig["wN"] = wN
        cds = wT * ts_val + wN * ns_val
        sig["CDS"] = round(cds, 4)

        if cds > 0.35:
            decision = "Buy/Add"
        elif cds < -0.25:
            decision = "Sell/Trim"
        else:
            decision = "Hold"
        obj["decision"] = decision
    return symbols


def load_json_from_url(url: str) -> dict:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def load_input_feed() -> dict:
    feed_url = os.getenv("FEED_URL", "").strip()
    input_path = os.getenv("INPUT_PATH", DEFAULT_INPUT_PATH).strip()

    if feed_url:
        print(f"[publish_feed] Loading from FEED_URL={feed_url}")
        return load_json_from_url(feed_url)
    else:
        print(f"[publish_feed] Loading from local path: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)


def write_output_feed(symbols: dict):
    output_path = os.getenv("OUTPUT_PATH", DEFAULT_OUTPUT_PATH).strip()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(symbols, f, indent=2, ensure_ascii=False)
    print(f"[publish_feed] Wrote enriched feed → {output_path}")


def main():
    symbols = load_input_feed()

    for sym, obj in symbols.items():
        if not isinstance(obj, dict):
            continue
        ts_val, ts_dbg = compute_ts_from_indicators(obj)
        sig = obj.setdefault("signals", {})
        sig["TS"] = round(ts_val, 4)
        components = sig.setdefault("components", {})
        components["rsi_pos"] = ts_dbg["rsi_pos"]
        components["macd_hist_slope"] = ts_dbg["macd_hist_slope"]
        components["ema20_slope"] = ts_dbg["ema20_slope"]
        sig["ts_debug"] = ts_dbg

    symbols = enrich_with_news_sentiment(symbols)
    symbols = apply_v22_weights_and_decisions(symbols)
    write_output_feed(symbols)


if __name__ == "__main__":
    main()
