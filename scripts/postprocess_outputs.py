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
PAPER_CASH_MODE       = (os.environ.get("PAPER_CASH_MODE", "1") == "1")  # default ON
PORTFOLIO_PAPER_CASH  = float(os.environ.get("PORTFOLIO_PAPER_CASH", "0") or "0")

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
            w.writerow(["symbol", "quantity"])  # legacy default
        return {}
    out = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sym = (row.get("symbol") or "").strip().upper()
            qty = row.get("qty")
            if qty is None:
                qty = row.get("quantity")
            if not sym:
                continue
            if sym == "__CASH__":
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

def read_cash_from_csv(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sym = (row.get("symbol") or "").strip().upper()
            if sym == "__CASH__":
                try:
                    return float(row.get("qty") or row.get("quantity") or 0)  # supports both schemas
                except:
                    return 0.0
    return 999.0

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

def size_trades_cash(actions: List[Dict[str, Any]], port_value: float, cash: float, per_line_frac: float,
                     max_deploy_frac: float | None = None) -> Tuple[List[Dict[str, Any]], float]:
    """
    Unified cash-constrained sizing used by BOTH portfolios.
      - BUYs limited by available 'cash'
      - Optional max_deploy_frac (cap total BUY spend to that fraction of equity+cash)
      - Per-line BUY target ~= per_line_frac * (equity+cash)
      - SELL trims floor to whole shares
    """
    equity = port_value + cash
    per_line_target = equity * per_line_frac
    total_budget = cash
    if max_deploy_frac is not None:
        total_budget = min(total_budget, equity * max_deploy_frac)

    sized, spent = [], 0.0

    # SELLs first
    for a in actions:
        if a["action"] != "SELL":
            continue
        px = a.get("px") or 0.0
        qty = math.floor(max(0.0, a.get("qty_hint", 0.0)))
        if qty <= 0: 
            continue
        sized.append({**a, "qty": qty, "notional": round(qty * px, 2)})

    # BUYs with caps (respect total_budget and per-line target)
    for a in sorted([x for x in actions if x["action"] == "BUY"], key=lambda x: x["cds"], reverse=True):
        px = a.get("px") or 0.0
        if px <= 0:
            continue
        remain = total_budget - spent
        if remain <= 0:
            break
        target_val = min(per_line_target, remain)
        qty = round(target_val / px, 4)  # fractional ok
        if qty <= 0:
            continue
        notional = qty * px
        spent += notional
        sized.append({**a, "qty": qty, "notional": round(notional, 2)})

    cash_left = round(cash - spent, 2)
    return sized, cash_left

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
        qty = round(target_val / px, 4)  # fractional
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

    personal_cash_csv = read_cash_from_csv(PORT_PERSONAL_CSV)
    paper_cash_csv = read_cash_from_csv(PORT_PAPER_CSV)
    personal_cash = personal_cash_csv if personal_cash_csv > 0 else PERSONAL_CASH
    paper_cash = paper_cash_csv

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


    # new (one engine, different knobs):
    sizedPaper, paper_cash_left = size_trades_cash(
        recPaper, paper_val, cash=paper_cash,
        per_line_frac=A_PER_LINE_FRAC,       # paper’s per-line % (keep separate knob)
        max_deploy_frac=A_MAX_DEPLOY_FRAC    # paper’s optional daily cap
    )
    sizedPers, pers_cash_left = size_trades_cash(
        recPers, pers_val, cash=personal_cash,
        per_line_frac=B_PER_LINE_FRAC,       # personal’s per-line %
        max_deploy_frac=None                 # usually no daily cap for personal
    )


    # Final trades output
    out_trades = {
        "as_of_utc": feed.get("as_of_utc"),
        "thresholds": {
            "BUY_THRESHOLD": BUY_THRESHOLD,
            "SELL_THRESHOLD": SELL_THRESHOLD
        },
        "portfolio_paper": {
            "portfolio_value_est": round(paper_val, 2),
            "cash_start": paper_cash,
            "per_line_frac": A_PER_LINE_FRAC,
            "max_deploy_frac": A_MAX_DEPLOY_FRAC,   # informational
            "trades": sizedPaper,
            "cash_left": paper_cash_left
        },
        "portfolio_personal": {
            "portfolio_value_est": round(pers_val, 2),
            "cash_start": personal_cash,
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
