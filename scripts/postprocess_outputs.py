#!/usr/bin/env python3
import os, json, math, sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

# ---------- Config via env ----------
INPUT_FINAL = os.environ.get("INPUT_FINAL", "data/prices_final.json")

# thresholds (align with v2.2)
BUY_TH  = float(os.environ.get("BUY_THRESHOLD", "0.35"))
SELL_TH = float(os.environ.get("SELL_THRESHOLD", "-0.25"))

# watchlist thresholds
TS_WATCH  = float(os.environ.get("TS_WATCH", "0.50"))
NS_WATCH  = float(os.environ.get("NS_WATCH", "0.50"))
WATCH_TOP = int(os.environ.get("WATCH_TOP", "20"))

# sizing (notional caps)
A_MAX_DEPLOY_FRAC  = float(os.environ.get("A_MAX_DEPLOY_FRAC", "0.15"))   # total deploy ≤ 15% of A
A_PER_LINE_FRAC    = float(os.environ.get("A_PER_LINE_FRAC", "0.04"))     # per-line cap 4% of A
B_PER_LINE_FRAC    = float(os.environ.get("B_PER_LINE_FRAC", "0.02"))     # per-line cap 2% of B

# files
PORT_A_CSV = os.environ.get("PORT_A_CSV", "data/portfolio_a.csv")
PORT_B_CSV = os.environ.get("PORT_B_CSV", "data/portfolio_b.csv")

OUT_SLIM        = os.environ.get("OUT_SLIM", "data/prices_final_slim.json")
OUT_PORTA_ONLY  = os.environ.get("OUT_PORTA_ONLY", "data/prices_final_porta.json")
OUT_PORTB_ONLY  = os.environ.get("OUT_PORTB_ONLY", "data/prices_final_portb.json")
OUT_WATCHLIST   = os.environ.get("OUT_WATCHLIST", "data/watchlist_summary.json")
OUT_TRADES      = os.environ.get("OUT_TRADES", "data/recommended_trades.json")

# ---------- Helpers ----------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_portfolio_csv(path):
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["symbol","quantity"]).to_csv(p, index=False)
    df = pd.read_csv(p)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
    return df

def total_value(port, feed):
    total = 0.0
    for _, row in port.iterrows():
        sym = row["symbol"]
        qty = float(row["quantity"])
        node = feed["symbols"].get(sym, {})
        px = node.get("price")
        if px is not None:
            total += qty * float(px)
    return total

def cds_for(sym, feed):
    node = feed["symbols"].get(sym, {})
    sig = node.get("signals", {}) if node else {}
    return (node.get("price"), sig.get("TS"), sig.get("NS"), sig.get("CDS"), node.get("decision"))

def propose_trades(port, feed, is_A):
    """Return a list of trade rows for the given portfolio using only file decisions."""
    rows = []
    # Existing positions → follow file decision
    for _, r in port.iterrows():
        sym = r["symbol"]; qty = float(r["quantity"])
        price, TS, NS, CDS, decision = cds_for(sym, feed)
        if price is None or CDS is None:
            continue
        if CDS > BUY_TH:
            rows.append({"side":"BUY_ADD","symbol":sym,"qty_hint":None,"price":price,"CDS":CDS,"why":"CDS>BUY_TH"})
        elif CDS < SELL_TH:
            rows.append({"side":"SELL_TRIM","symbol":sym,"qty_hint":None,"price":price,"CDS":CDS,"why":"CDS<SELL_TH"})
        else:
            rows.append({"side":"HOLD","symbol":sym,"qty_hint":None,"price":price,"CDS":CDS,"why":"hold_band"})

    # New buys → top CDS not already held
    held = set(port["symbol"].tolist())
    extra = []
    for sym, node in feed["symbols"].items():
        if sym in held: 
            continue
        sig = node.get("signals", {}) or {}
        CDS = sig.get("CDS")
        if CDS is not None and CDS > BUY_TH:
            extra.append((sym, float(node.get("price") or 0), float(CDS)))
    # Top extras
    extra = sorted(extra, key=lambda x: x[2], reverse=True)
    if is_A:
        extra = extra[:10]   # allow more breadth in A
    else:
        extra = extra[:5]    # keep B conservative
    for sym, px, cds in extra:
        if px:
            rows.append({"side":"BUY_NEW","symbol":sym,"qty_hint":None,"price":px,"CDS":cds,"why":"High CDS"})
    return rows

def size_trades_A(trades, port_val):
    """Use 15% total cap; 4% per line cap; sells 10–20% by CDS severity."""
    budget = port_val * A_MAX_DEPLOY_FRAC
    out = []
    # First append sells (increase budget)
    for tr in trades:
        if tr["side"] != "SELL_TRIM": 
            continue
        cds = tr["CDS"]; px = tr["price"]
        # Sell fraction: 10% → 20% as CDS worsens
        frac = 0.10 if cds > -0.50 else (0.15 if cds > -0.75 else 0.20)
        out.append({**tr, "qty": "0.10–0.20 of position", "notional": f"~{int(frac*100)}% * position * {px:.2f}"})
    # Then buys within caps
    per_line_cap = port_val * A_PER_LINE_FRAC
    for tr in sorted([t for t in trades if t["side"] in ("BUY_ADD","BUY_NEW")], key=lambda x: x["CDS"], reverse=True):
        if budget <= 0: 
            break
        alloc = min(per_line_cap, budget)
        qty = int(alloc // tr["price"])
        if qty <= 0: 
            continue
        notional = qty * tr["price"]
        out.append({**tr, "qty": qty, "notional": round(notional,2)})
        budget -= notional
    return out, budget

def size_trades_B(trades, port_val, cash=0.0):
    """Trims fund buys; 2% per-line cap; conservative."""
    budget = float(cash)
    out = []
    # Sells first to raise budget
    for tr in trades:
        if tr["side"] != "SELL_TRIM": 
            continue
        cds = tr["CDS"]; px = tr["price"]
        frac = 0.10 if cds > -0.50 else (0.15 if cds > -0.75 else 0.20)
        out.append({**tr, "qty": "0.10–0.20 of position", "notional": f"~{int(frac*100)}% * position * {px:.2f}"})
        # We can’t know exact position size here; sizing remains a hint.
        # Budget update will happen when execution applies; keep as annotation.
    per_line_cap = port_val * B_PER_LINE_FRAC
    for tr in sorted([t for t in trades if t["side"] in ("BUY_ADD","BUY_NEW")], key=lambda x: x["CDS"], reverse=True):
        if budget <= 0: 
            break
        alloc = min(per_line_cap, budget)
        qty = int(alloc // tr["price"])
        if qty <= 0: 
            continue
        notional = qty * tr["price"]
        out.append({**tr, "qty": qty, "notional": round(notional,2)})
        budget -= notional
    return out, budget

def main():
    feed = load_json(INPUT_FINAL)

    # ---- (1) Slim file with just TS/NS per symbol ----
    slim = {
        "as_of_utc": feed.get("as_of_utc"),
        "symbols": {
            s: {"TS": (node.get("signals") or {}).get("TS"),
                "NS": (node.get("signals") or {}).get("NS")}
            for s, node in (feed.get("symbols") or {}).items()
        }
    }
    Path(OUT_SLIM).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SLIM, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2)

    # Load portfolios
    portA = ensure_portfolio_csv(PORT_A_CSV)
    portB = ensure_portfolio_csv(PORT_B_CSV)

    # ---- (2) prices_final but filtered to each portfolio ----
    def filter_feed(port_df):
        keep = set(port_df["symbol"].tolist())
        return {
            "as_of_utc": feed.get("as_of_utc"),
            "timeframe": feed.get("timeframe"),
            "symbols": {s: feed["symbols"][s] for s in keep if s in feed.get("symbols", {})}
        }
    with open(OUT_PORTA_ONLY, "w", encoding="utf-8") as f:
        json.dump(filter_feed(portA), f, indent=2)
    with open(OUT_PORTB_ONLY, "w", encoding="utf-8") as f:
        json.dump(filter_feed(portB), f, indent=2)

    # ---- (3) Summary of NEW tickers to watch (high TS or high NS), excluding holdings ----
    held_all = set(portA["symbol"].tolist()) | set(portB["symbol"].tolist())
    watch = []
    for s, node in (feed.get("symbols") or {}).items():
        if s in held_all: 
            continue
        sig = node.get("signals") or {}
        TS = sig.get("TS"); NS = sig.get("NS"); CDS = sig.get("CDS")
        if (TS is not None and TS >= TS_WATCH) or (NS is not None and NS >= NS_WATCH):
            watch.append({
                "symbol": s,
                "price": node.get("price"),
                "TS": TS, "NS": NS, "CDS": CDS,
                "reason": "High TS" if (TS or -1) >= TS_WATCH else ("High NS" if (NS or -1) >= NS_WATCH else "")
            })
    watch = sorted(watch, key=lambda r: (r["CDS"] if r["CDS"] is not None else -999), reverse=True)[:WATCH_TOP]
    with open(OUT_WATCHLIST, "w", encoding="utf-8") as f:
        json.dump({"as_of_utc": feed.get("as_of_utc"), "watch": watch}, f, indent=2)

    # ---- (4) Summary of recommended trades for A & B ----
    # Compute portfolio notional values (no cash known here; trade sizing uses caps; B is conservative)
    valA = total_value(portA, feed)
    valB = total_value(portB, feed)

    recA = propose_trades(portA, feed, is_A=True)
    sizedA, leftoverA = size_trades_A(recA, valA)

    recB = propose_trades(portB, feed, is_A=False)
    # If you want to include a cash file for B, set PORTFOLIO_B_CASH env and pass here
    cashB = float(os.environ.get("PORTFOLIO_B_CASH", "0"))
    sizedB, leftoverB = size_trades_B(recB, valB, cash=cashB)

    with open(OUT_TRADES, "w", encoding="utf-8") as f:
        json.dump({
            "as_of_utc": feed.get("as_of_utc"),
            "rules": {
                "BUY_THRESHOLD": BUY_TH,
                "SELL_THRESHOLD": SELL_TH,
                "A_MAX_DEPLOY_FRAC": A_MAX_DEPLOY_FRAC,
                "A_PER_LINE_FRAC": A_PER_LINE_FRAC,
                "B_PER_LINE_FRAC": B_PER_LINE_FRAC
            },
            "portfolio_A": {"value_est": round(valA,2), "trades": sizedA, "residual_budget_hint": round(leftoverA,2)},
            "portfolio_B": {"value_est": round(valB,2), "trades": sizedB, "residual_budget_hint": round(leftoverB,2)}
        }, f, indent=2)

    print(f"[OK] Wrote:\n - {OUT_SLIM}\n - {OUT_PORTA_ONLY}\n - {OUT_PORTB_ONLY}\n - {OUT_WATCHLIST}\n - {OUT_TRADES}")

if __name__ == "__main__":
    main()
