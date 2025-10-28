#!/usr/bin/env python3
"""
scripts/analyze_feed.py
-----------------------
Consumes a base feed JSON (from scripts/publish_feed.py) and applies
Algorithm v2.2 Adaptive Hybrid to produce enriched decisions per symbol
and portfolio-level recommendations.

Inputs:
  --in  path to prices.json (default: data/prices.json)
  --out path to enriched json (default: data/prices_final.json)

Optional env:
  PORTFOLIOS_PATH (default: in-script defaults from user's context)
  NEWS_DECAY (default: 0.5 per day)
  SECTOR_CORRELATION_PENALTY (default: 0.1 when clustering negatives)

Outputs:
  - Enriched JSON with TS, NS, CDS, weights, decisions
  - Per-portfolio suggested actions (no order execution here)
"""

import os, sys, json, math, argparse, datetime as dt
from typing import Dict, Any, List

NEWS_DECAY = float(os.getenv("NEWS_DECAY", "0.5"))
SECTOR_CORRELATION_PENALTY = float(os.getenv("SECTOR_CORRELATION_PENALTY", "0.1"))

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def rsi_position(rsi: float) -> float:
    if rsi is None:
        return 0.0
    # map 0..100 -> -1..+1 centered at 50
    return max(-1.0, min(1.0, (rsi - 50.0)/50.0))

def slope(series: List[float], lookback: int = 5) -> float:
    if not series or len(series) < lookback+1:
        return 0.0
    a = series[-1]
    b = series[-1 - lookback]
    # normalize by average absolute value to reduce scale effects
    denom = max(1e-9, (abs(a) + abs(b))/2.0)
    return (a - b) / denom

def simple_news_sentiment(items: List[Dict[str, Any]]) -> float:
    """
    Simple lexicon-based sentiment for headlines + summary.
    Returns -1..+1.
    """
    if not items:
        return 0.0
    pos = set(["beat","beats","surge","surges","rally","record","upgrade","upgrades","raises","raise","tops","exceed","strong","bullish","growth","profit","profits","outperform"])
    neg = set(["miss","misses","plunge","plunges","falls","fall","downgrade","downgrades","cuts","cut","warning","bearish","fraud","lawsuit","shortfall","weak","loss","losses","guidance cut","recall"])
    score = 0
    total = 0
    for n in items:
        txt = " ".join([str(n.get("headline","")), str(n.get("summary",""))]).lower()
        s = 0
        for w in pos:
            if w in txt: s += 1
        for w in neg:
            if w in txt: s -= 1
        if s != 0:
            total += 1
            score += s
    if total == 0:
        return 0.0
    # squash roughly into -1..+1
    return max(-1.0, min(1.0, score/10.0))

def decide(cds: float) -> str:
    if cds > 0.35:
        return "Buy/Add"
    if cds < -0.25:
        return "Sell/Trim"
    return "Hold"

# Default portfolios (from user's context, 2025-10-27 snapshot)
DEFAULT_PORTFOLIOS = {
    "Portfolio A": {
        "cash": 53950.0,
        "positions": {
            "AMD": {"qty": 14, "avg": 254.15},
            "PLTR": {"qty": 5,  "avg": None},
            "SNOW": {"qty": 4,  "avg": 265.35},
            "SOFI": {"qty": 25, "avg": None},
            "DE":   {"qty": 1,  "avg": None},
        },
        "slippage": 0.0010,
        "liquidity_impact_smallcap": 0.0005,
        "exposure_limit_step": 0.15
    },
    "Portfolio B": {
        "cash": 250.0,
        "positions": {
            "QBTS": {"qty": 234.08, "avg": 35.25},
            "VTI":  {"qty": 11.48,  "avg": 336.8},
            "QUBT": {"qty": 83.39,  "avg": 16.18},
            "TTWO": {"qty": 4,      "avg": 254.0},
            "ASTS": {"qty": 10.49,  "avg": 76.9},
            "VOO":  {"qty": 1.14,   "avg": 628.0},
            "AMD":  {"qty": 2,      "avg": 254.15},
            "ARCC": {"qty": 24.36,  "avg": 20.25},
            "RKLB": {"qty": 7.27,   "avg": 64.9},
            "RGTI": {"qty": 9.67,   "avg": 40.9},
            "NVDA": {"qty": 2,      "avg": 190.9},
            "SNOW": {"qty": 1,      "avg": 265.3},
            "AMZN": {"qty": 0.1013, "avg": 226.9}
        },
        "slippage": 0.0005,
        "liquidity_impact_smallcap": 0.0005,
        "exposure_limit_step": 0.15
    }
}

def attach_sector(symbol: str) -> str:
    # ultra-simple sector mapping for demo; extend as needed
    tech = {"AAPL","MSFT","GOOGL","META","TSLA","AMD","NVDA","SMCI","ASML","SNOW","PLTR","QUBT","QBTS","RGTI"}
    etf  = {"SPY","QQQ","DIA","IWM","VTI","VOO"}
    space= {"RKLB","ASTS"}
    fin  = {"SOFI","ARCC"}
    ind  = {"DE"}
    if symbol in tech: return "Tech"
    if symbol in etf:  return "ETF"
    if symbol in space:return "Space"
    if symbol in fin:  return "Finance"
    if symbol in ind:  return "Industrial"
    return "Other"

def analyze_symbol(sym: str, node: Dict[str, Any]) -> Dict[str, Any]:
    ind = node.get("indicators", {})
    series = node.get("series", {})

    # Technical components
    rsi14 = ind.get("rsi14")
    rsi_pos = rsi_position(rsi14)

    macd_hist_series = series.get("macd_hist", [])
    macd_slope = slope(macd_hist_series, lookback=3)

    ema20_series = series.get("ema20", [])
    ema_slope = slope(ema20_series, lookback=5)

    TS = 0.4*macd_slope + 0.4*rsi_pos + 0.2*ema_slope
    TS = max(-1.0, min(1.0, TS))

    # News/Sentiment
    ns_raw = simple_news_sentiment(node.get("news", []))
    # decay is applied at portfolio aggregation step if historical NS exists; for now we expose raw
    NS = ns_raw

    # Weight switching
    wT, wN = (0.6, 0.4)
    if abs(NS) >= 0.7:
        wT, wN = (0.4, 0.6)

    CDS = wT*TS + wN*NS
    CDS = max(-1.0, min(1.0, CDS))

    decision = decide(CDS)

    return {
        "symbol": sym,
        "price": node.get("price"),
        "ts": node.get("ts"),
        "sector": attach_sector(sym),
        "signals": {
            "TS": TS,
            "NS": NS,
            "wT": wT,
            "wN": wN,
            "CDS": CDS,
            "components": {
                "rsi_pos": rsi_pos,
                "macd_hist_slope": macd_slope,
                "ema20_slope": ema_slope
            }
        },
        "decision": decision
    }

def portfolio_summary(name: str, portfolio: Dict[str, Any], symbols_map: Dict[str, Any]) -> Dict[str, Any]:
    items = []
    sector_bucket = {}
    for sym, lot in portfolio["positions"].items():
        node = symbols_map.get(sym)
        if not node:
            continue
        sym_analysis = analyze_symbol(sym, node)
        items.append(sym_analysis)
        sector_bucket.setdefault(sym_analysis["sector"], []).append(sym_analysis["signals"]["CDS"])

    # sector correlation penalty if cluster of negatives
    sector_adjustments = {}
    for sector, cds_list in sector_bucket.items():
        if len(cds_list) >= 2:
            avg = sum(cds_list)/len(cds_list)
            if avg < -0.25:
                sector_adjustments[sector] = -SECTOR_CORRELATION_PENALTY

    # apply sector penalty
    for x in items:
        sector = x["sector"]
        if sector in sector_adjustments:
            x["signals"]["CDS"] = max(-1.0, min(1.0, x["signals"]["CDS"] + sector_adjustments[sector]))
            x["decision"] = decide(x["signals"]["CDS"])

    # portfolio-level NS bias
    avg_NS = 0.0
    if items:
        avg_NS = sum([it["signals"]["NS"] for it in items]) / len(items)
    exposure_bias = 0.0
    if avg_NS < -0.4:
        exposure_bias = -0.2  # reduce exposure by 20%

    # sizing modifier from NS extremes per symbol
    for x in items:
        ns = x["signals"]["NS"]
        size_mult = 1.0
        if ns < -0.5: size_mult = 0.5
        if ns >  0.5: size_mult = 1.5
        x["sizing_multiplier_from_NS"] = size_mult

    return {
        "name": name,
        "avg_NS": avg_NS,
        "exposure_bias": exposure_bias,
        "items": items
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/prices.json")
    ap.add_argument("--out", dest="out_path", default="data/prices_final.json")
    args = ap.parse_args()

    feed = load_json(args.in_path)
    symbols_map = feed.get("symbols", {})

    # Load portfolios (or use defaults)
    portfolios_path = os.getenv("PORTFOLIOS_PATH")
    if portfolios_path and os.path.exists(portfolios_path):
        portfolios = load_json(portfolios_path)
    else:
        portfolios = DEFAULT_PORTFOLIOS

    results = {
        "metadata": {
            "algo_version": "v2.2 Adaptive Hybrid",
            "generated_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "source_feed_generated_at_utc": feed.get("metadata",{}).get("generated_at_utc"),
            "notes": "TS=0.4*MACD_hist_slope + 0.4*RSI_position + 0.2*EMA20_slope; dynamic wT/wN; sector penalty; portfolio NS bias."
        },
        "per_symbol": {},
        "portfolios": []
    }

    # per symbol analysis (for all symbols in the feed)
    for sym, node in symbols_map.items():
        results["per_symbol"][sym] = analyze_symbol(sym, node)

    # per portfolio summaries
    for name, pf in portfolios.items():
        results["portfolios"].append(portfolio_summary(name, pf, symbols_map))

    save_json(results, args.out_path)
    print(f"Wrote enriched analysis to {args.out_path}")

if __name__ == "__main__":
    main()
