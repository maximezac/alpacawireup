#!/usr/bin/env python3
"""
scripts/postprocess_outputs.py

Reads an enriched feed (prices_final.json) and produces:
1) OUT_SLIM:    Slimmed feed with {symbol, price, TS, NS, CDS, decision}
2) OUT_PAPER_ONLY / OUT_PERSONAL_ONLY: feed filtered to the symbols in each portfolio
3) OUT_WATCHLIST: summary of new tickers to watch (high TS or high NS)
4) OUT_TRADES:  recommended trades for portfolio_paper and portfolio_personal
5) Uses/maintains portfolio CSVs under /data (symbol,quantity)

Environment variables (with defaults):
- INPUT_FINAL: data/prices_final.json
- OUT_SLIM: data/prices_final_slim.json
- OUT_PAPER_ONLY: data/prices_final_paper.json
- OUT_PERSONAL_ONLY: data/prices_final_personal.json
- OUT_WATCHLIST: data/watchlist_summary.json
- OUT_TRADES: data/recommended_trades.json

Portfolio files:
- PORT_PAPER_CSV: data/portfolio_paper.csv
- PORT_PERSONAL_CSV: data/portfolio_personal.csv
- PORTFOLIO_PERSONAL_CASH: "0"  (available cash for the personal portfolio)

Thresholds (string numbers, parsed to float):
- BUY_THRESHOLD: "0.35"         # v2.2 Buy/Add
- SELL_THRESHOLD: "-0.25"       # v2.2 Sell/Trim
- TS_WATCH: "0.50"              # watch if TS >= TS_WATCH
- NS_WATCH: "0.50"              # watch if NS >= NS_WATCH
- WATCH_TOP: "20"               # max tickers in watchlist

Sizing knobs (fractions of portfolio equity):
- A_MAX_DEPLOY_FRAC: "0.15"     # max total BUY cash to deploy today (paper)
- A_PER_LINE_FRAC:  "0.04"      # max per-line size (paper)
- B_PER_LINE_FRAC:  "0.02"      # target per-line size (personal)
- SELL_TRIM_FRACTION: "0.25"    # % of current shares to trim on Sell/Trim

Notes:
- We DO NOT recompute indicators; we only read values as-is from INPUT_FINAL.
- Share counts are floored to whole shares; any zero-sized result is omitted.
- New tickers are allowed in proposes for both portfolios (subject to sizing rules).
"""

import os, csv, json, math
from typing import Dict, Any, List, Tuple

# =============== Env & paths ===============
INPUT_FINAL          = os.environ.get("INPUT_FINAL", "data/prices_final.json")

OUT_SLIM             = os.environ.get("OUT_SLIM", "data/prices_final_slim.json")
OUT_PAPER_ONLY       = os.environ.get("OUT_PAPER_ONLY", "data/prices_final_paper.json")
OUT_PERSONAL_ONLY    = os.environ.get("OUT_PERSONAL_ONLY", "data/prices_final_personal.json")
OUT_WATCHLIST        = os.environ.get("OUT_WATCHLIST", "data/watchlist_summary.json")
OUT_TRADES           = os.environ.get("OUT_TRADES", "data/recommended_trades.json")

PORT_PAPER_CSV       = os.environ.get("PORT_PAPER_CSV", "data/portfolio_paper.csv")
PORT_PERSONAL_CSV    = os.environ.get("PORT_PERSONAL_CSV", "data/portfolio_personal.csv")
PERSONAL_CASH        = float(os.environ.get("PORTFOLIO_PERSONAL_CASH", "0") or "0")

BUY_THRESHOLD        = float(os.environ.get("BUY_THRESHOLD", "0.35"))
SELL_THRESHOLD       = float(os.environ.get("SELL_THRESHOLD", "-0.25"))

TS_WATCH             = float(os.environ.get("TS_WATCH", "0.50"))
NS_WATCH             = float(os.environ.get("NS_WATCH", "0.50"))
WATCH_TOP            = int(os.environ.get("WATCH_TOP", "20"))

A_MAX_DEPLOY_FRAC    = float(os.environ.get("A_MAX_DEPLOY_FRAC", "0.15"))
A_PER_LINE_FRAC      = float(os.environ.get("A_PER_LINE_FRAC", "0.04"))
B_PER_LINE_FRAC      = float(os.environ.get("B_PER_LINE_FRAC", "0.02"))
SELL_TRIM_FRACTION   = float(os.environ.get("SELL_TRIM_FRACTION", "0.25"))  # 25% trim by default


# =============== Helpers ===============
def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def ensure_portfolio_csv(path: str) -> Dict[str, float]:
    """
    Read CSV: symbol,quantity
    Returns dict {symbol: float(quantity)}
    Creates file if missing (empty).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        # create an empty CSV
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "quantity"])
        return {}
    out = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sym = (row.get("symbol") or "").strip().upper()
            qty = row.get("quantity")
            if not sym:
                continue
            try:
                out[sym] = float(qty)
            except Exception:
                continue
    return out

def save_portfolio_csv(path: str, pos: Dict[str, float]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "quantity"])
        for sym, qty in sorted(pos.items()):
            w.writerow([sym, qty])

def feed_symbols(feed: Dict[str, Any]) -> Dict[str, Any]:
    return feed.get("symbols", {}) or {}

def get_price(node: Dict[str, Any]) -> float:
    # prefer "now" price if present; else daily price
    if "now" in node and isinstance(node["now"], dict) and "price" in node["now"]:
        return float(node["now"]["price"])
    if "price" in node:
        return float(node["price"])
    return 0.0

def get_signals(node: Dict[str, Any]) -> Dict[str, Any]:
    return node.get("signals") or {}

def get_cds(node: Dict[str, Any]) -> float:
    return float(get_signals(node).get("CDS", 0.0) or 0.0)

def get_ts(node: Dict[str, Any]) -> float:
    return float(get_signals(node).get("TS", 0.0) or 0.0)

def get_ns(node: Dict[str, Any]) -> float:
    return float(get_signals(node).get("NS", 0.0) or 0.0)

def total_value(port: Dict[str, float], feed: Dict[str, Any]) -> float:
    syms = feed_symbols(feed)
    val = 0.0
    for sym, qty in port.items():
        node = syms.get(sym)
        if not node:
            continue
        px = get_price(node)
        val += px * float(qty)
    return val

def filter_feed_by_symbols(feed: Dict[str, Any], symbols: List[str]) -> Dict[str, Any]:
    slim = {"as_of_utc": feed.get("as_of_utc"),
            "timeframe": feed.get("timeframe"),
            "indicators_window": feed.get("indicators_window"),
            "symbols": {}}
    src = feed_symbols(feed)
    for s in symbols:
        if s in src:
            slim["symbols"][s] = src[s]
    return slim

def build_slim(feed: Dict[str, Any]) -> Dict[str, Any]:
    out = {"as_of_utc": feed.get("as_of_utc"),
           "symbols": {}}
    for sym, node in feed_symbols(feed).items():
        out["symbols"][sym] = {
            "price": node.get("price"),
            "ts": node.get("ts"),
            "TS": get_ts(node),
            "NS": get_ns(node),
            "CDS": get_cds(node),
            "decision": node.get("decision")
        }
    # include live "now" if present at root
    if "now" in feed:
        out["now"] = feed["now"]
    return out

def watchlist(feed: Dict[str, Any], ts_thr: float, ns_thr: float, top_n: int) -> List[Dict[str, Any]]:
    rows = []
    for sym, node in feed_symbols(feed).items():
        ts = get_ts(node)
        ns = get_ns(node)
        hit = (ts is not None and ts >= ts_thr) or (ns is not None and ns >= ns_thr)
        if hit:
            rows.append({
                "symbol": sym,
                "price": get_price(node) or node.get("price"),
                "TS": ts, "NS": ns, "CDS": get_cds(node),
                "reason": "TS_high" if (ts is not None and ts >= ts_thr) else "NS_high"
            })
    # sort by |CDS| desc, then TS desc
    rows.sort(key=lambda r: (abs(r["CDS"] or 0.0), r["TS"] or 0.0), reverse=True)
    return rows[:top_n]

def propose_trades(port: Dict[str, float], feed: Dict[str, Any], allow_new: bool = True) -> List[Dict[str, Any]]:
    """
    Returns a flat list of proposed actions:
      {"symbol":..., "action": "BUY"/"SELL", "reason": "...", "px": price, "qty_hint": float, "cds": float}
    qty_hint is un-sized; sizing is handled later per portfolio rules.
    """
    src = feed_symbols(feed)
    actions = []

    # 1) Sells / Trims for owned symbols
    for sym, qty in port.items():
        node = src.get(sym)
        if not node:
            continue
        cds = get_cds(node)
        px  = get_price(node)
        if cds < SELL_THRESHOLD:
            # propose a trim; actual shares decided later
            actions.append({
                "symbol": sym, "action": "SELL", "reason": "CDS<SellThreshold",
                "px": px, "qty_hint": float(qty) * SELL_TRIM_FRACTION, "cds": cds
            })

    # 2) Buys/Adds for owned and (optionally) new symbols
    for sym, node in src.items():
        cds = get_cds(node)
        if cds > BUY_THRESHOLD:
            already_own = sym in port and port[sym] > 0
            if already_own or allow_new:
                actions.append({
                    "symbol": sym, "action": "BUY", "reason": "CDS>BuyThreshold",
                    "px": get_price(node), "qty_hint": 0.0, "cds": cds
                })

    # de-dup: if both BUY and SELL exist for a symbol (shouldn’t), keep stronger |cds|
    by_sym = {}
    for a in actions:
        s = a["symbol"]
        if s not in by_sym:
            by_sym[s] = a
        else:
            # prefer higher |cds|
            if abs(a["cds"]) > abs(by_sym[s]["cds"]):
                by_sym[s] = a
    return list(by_sym.values())

def size_trades_A(actions: List[Dict[str, Any]], port_value: float) -> Tuple[List[Dict[str, Any]], float]:
    """
    Portfolio PAPER sizing:
      - Total BUY budget <= A_MAX_DEPLOY_FRAC * port_value
      - Per-line BUY target <= A_PER_LINE_FRAC * port_value
      - SELL trims use SELL_TRIM_FRACTION of current qty (already set), floored to whole shares
    """
    buy_budget_total = port_value * A_MAX_DEPLOY_FRAC
    per_line_cap_val = port_value * A_PER_LINE_FRAC

    sized = []
    spent = 0.0

    # SELLs first (don’t affect budget directly here)
    for a in actions:
        if a["action"] != "SELL":
            continue
        px = a.get("px") or 0.0
        qty = math.floor(max(0.0, a.get("qty_hint", 0.0)))  # whole shares
        if qty <= 0:
            continue
        sized.append({**a, "qty": qty, "notional": round(qty * px, 2)})

    # BUYs with caps
    for a in sorted([x for x in actions if x["action"] == "BUY"], key=lambda x: x["cds"], reverse=True):
        px = a.get("px") or 0.0
        if px <= 0:
            continue
        # target value per line
        target_val = min(per_line_cap_val, buy_budget_total - spent)
        if target_val <= 0:
            break
        qty = math.floor(target_val / px)
        if qty <= 0:
            continue
        notional = qty * px
        spent += notional
        sized.append({**a, "qty": qty, "notional": round(notional, 2)})

    leftover = round(buy_budget_total - spent, 2)
    return sized, max(0.0, leftover)

def size_trades_B(actions: List[Dict[str, Any]], port_value: float, cash: float) -> Tuple[List[Dict[str, Any]], float]:
    """
    Portfolio PERSONAL sizing:
      - BUYs limited by available 'cash'
      - Per-line BUY target ~ B_PER_LINE_FRAC * (port_value + cash)
      - SELL trims like A (fractional -> whole shares)
    """
    equity_plus_cash = port_value + cash
    per_line_target = equity_plus_cash * B_PER_LINE_FRAC

    sized = []
    cash_left = cash

    # SELLs first (no cash change accounted here; you can optionally add proceeds)
    for a in actions:
        if a["action"] != "SELL":
            continue
        px = a.get("px") or 0.0
        qty = math.floor(max(0.0, a.get("qty_hint", 0.0)))
        if qty <= 0:
            continue
        sized.append({**a, "qty": qty, "notional": round(qty * px, 2)})

    # BUYs with cash limit
    for a in sorted([x for x in actions if x["action"] == "BUY"], key=lambda x: x["cds"], reverse=True):
        px = a.get("px") or 0.0
        if px <= 0 or cash_left <= 0:
            continue
        target_val = min(per_line_target, cash_left)
        qty = math.floor(target_val / px)
        if qty <= 0:
            continue
        notional = qty * px
        cash_left -= notional
        sized.append({**a, "qty": qty, "notional": round(notional, 2)})

    return sized, round(cash_left, 2)


# =============== Main ===============
def main():
    feed = read_json(INPUT_FINAL)

    # 1) Slim file
    slim = build_slim(feed)
    write_json(OUT_SLIM, slim)

    # 2) Load portfolios
    portPaper    = ensure_portfolio_csv(PORT_PAPER_CSV)
    portPersonal = ensure_portfolio_csv(PORT_PERSONAL_CSV)

    # 3) Portfolio-only feeds
    paper_syms = sorted(portPaper.keys())
    pers_syms  = sorted(portPersonal.keys())
    write_json(OUT_PAPER_ONLY,    filter_feed_by_symbols(feed, paper_syms))
    write_json(OUT_PERSONAL_ONLY, filter_feed_by_symbols(feed, pers_syms))

    # 4) Watchlist
    wl = watchlist(feed, TS_WATCH, NS_WATCH, WATCH_TOP)
    write_json(OUT_WATCHLIST, {
        "as_of_utc": feed.get("as_of_utc"),
        "criteria": {"TS_WATCH": TS_WATCH, "NS_WATCH": NS_WATCH, "TOP": WATCH_TOP},
        "items": wl
    })

    # 5) Trades for each portfolio
    paper_val = total_value(portPaper, feed)
    pers_val  = total_value(portPersonal, feed)

    recPaper = propose_trades(portPaper, feed, allow_new=True)
    recPers  = propose_trades(portPersonal, feed, allow_new=True)

    sizedPaper, paper_leftover = size_trades_A(recPaper, paper_val)
    sizedPers,  pers_cash_left = size_trades_B(recPers, pers_val, cash=PERSONAL_CASH)

    # Final trades output
    out_trades = {
        "as_of_utc": feed.get("as_of_utc"),
        "thresholds": {
            "BUY_THRESHOLD": BUY_THRESHOLD,
            "SELL_THRESHOLD": SELL_THRESHOLD
        },
        "portfolio_paper": {
            "portfolio_value_est": round(paper_val, 2),
            "deploy_rules": {"max_deploy_frac": A_MAX_DEPLOY_FRAC, "per_line_frac": A_PER_LINE_FRAC},
            "trades": sizedPaper,
            "residual_budget_hint": paper_leftover
        },
        "portfolio_personal": {
            "portfolio_value_est": round(pers_val, 2),
            "cash_start": PERSONAL_CASH,
            "per_line_frac": B_PER_LINE_FRAC,
            "trades": sizedPers,
            "cash_left": pers_cash_left
        }
    }
    write_json(OUT_TRADES, out_trades)

    # 6) (Optional) persist unchanged CSVs back (no edits here, but function kept for future autoupdates)
    # save_portfolio_csv(PORT_PAPER_CSV, portPaper)
    # save_portfolio_csv(PORT_PERSONAL_CSV, portPersonal)

    print(f"[OK] Wrote:\n"
          f"  - {OUT_SLIM}\n"
          f"  - {OUT_PAPER_ONLY}\n"
          f"  - {OUT_PERSONAL_ONLY}\n"
          f"  - {OUT_WATCHLIST}\n"
          f"  - {OUT_TRADES}")

if __name__ == "__main__":
    main()
