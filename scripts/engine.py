#!/usr/bin/env python3
from __future__ import annotations
import os, csv, json, math
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Literal
from pathlib import Path

DecisionSource = Literal["CDS", "TS_ONLY", "NS_ONLY"]
TradeScope = Literal["none", "paper", "real", "both"]

# ---------- Config dataclasses ----------
@dataclass
class StrategyConfig:
    buy_threshold: float
    sell_threshold: float
    decision_source: DecisionSource = "CDS"
    ns_weight_boost: float = 1.0
    allow_new_symbols: bool = True

@dataclass
class SizingConfig:
    per_line_frac: float
    max_deploy_frac: float | None = 0.15
    trim_fraction: float = 0.25
    fractional_buys: bool = True

@dataclass
class PortfolioConfig:
    id: str
    title: str
    path: str
    positions_csv: str = "positions.csv"
    strategy: StrategyConfig = None
    sizing: SizingConfig = None
    trade_scope: TradeScope = "none"
    tags: list[str] | None = None

# ---------- IO helpers ----------
def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# engine.py

def _latest_px(node: dict) -> float:
    if not node:
        return 0.0
    v = node.get("price")
    if v is None:
        v = (node.get("now") or {}).get("price")  # legacy fallback
    try:
        return float(v or 0.0)
    except:
        return 0.0

def total_value(positions: dict, feed: dict) -> float:
    syms = (feed or {}).get("symbols") or {}
    val = 0.0
    for sym, qty in positions.items():
        node = syms.get(sym)
        if not node:
            continue
        px = _latest_px(node)
        val += float(qty) * px
    return val


def read_positions_csv(path: str) -> tuple[float, Dict[str, dict]]:
    """
    CSV schema:
      symbol,qty,avg_cost
      __CASH__,10000,0
      NVDA,2,190.9
    Returns: (cash, {SYM: {qty, avg_cost}})
    """
    if not Path(path).exists():
        return 0.0, {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cash = 0.0
        pos: Dict[str, dict] = {}
        for row in r:
            sym = (row.get("symbol") or "").strip().upper()
            if not sym:
                continue
            qty = float(row.get("qty") or row.get("quantity") or 0)
            avg = float(row.get("avg_cost") or 0)
            if sym == "__CASH__":
                cash = qty
                continue
            if qty != 0:
                pos[sym] = {"qty": qty, "avg_cost": avg}
        return round(cash, 2), pos

def write_positions_csv(path: str, cash: float, pos: Dict[str, dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "qty", "avg_cost"])
        w.writerow(["__CASH__", round(cash, 2), 0])
        for sym in sorted(pos.keys()):
            p = pos[sym]
            w.writerow([sym, round(p["qty"], 6), round(p["avg_cost"], 6)])

# ---------- Prices/Signals helpers ----------
def feed_symbols(feed: Dict[str, Any]) -> Dict[str, Any]:
    return feed.get("symbols", {}) or {}

def get_price(node: Dict[str, Any]) -> float:
    if isinstance(node.get("now"), dict) and node["now"].get("price") is not None:
        return float(node["now"]["price"])
    if node.get("price") is not None:
        return float(node["price"])
    return 0.0

def score_for_decision(node: Dict[str, Any], strat: StrategyConfig) -> float:
    sig = node.get("signals") or {}
    if strat.decision_source == "CDS":
        # Trust CDS as provided by your v2.2 pipeline; optional NS boost when you later choose to recombine
        return float(sig.get("CDS") or 0.0)
    elif strat.decision_source == "TS_ONLY":
        return float(sig.get("TS") or 0.0)
    else:  # NS_ONLY
        return float(sig.get("NS") or 0.0) * float(strat.ns_weight_boost or 1.0)

# ---------- Proposals ----------
def propose_actions(port: Dict[str, float], feed: Dict[str, Any], strat: StrategyConfig) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    symbols = feed_symbols(feed)

    # Sells/Trims for existing holdings
    for sym, qty in port.items():
        node = symbols.get(sym)
        if not node:
            continue
        score = score_for_decision(node, strat)
        if score < strat.sell_threshold:
            px = get_price(node)
            actions.append({
                "symbol": sym, "action": "SELL", "px": px,
                "qty_hint": qty,          # holder qty; final trim by sizer
                "reason": "score<sell", "score": score
            })

    # Buys/Adds (existing or new)
    for sym, node in symbols.items():
        score = score_for_decision(node, strat)
        if score > strat.buy_threshold and (strat.allow_new_symbols or sym in port):
            px = get_price(node)
            actions.append({
                "symbol": sym, "action": "BUY", "px": px,
                "qty_hint": 0.0, "reason": "score>buy", "score": score
            })

    # Dedup by strongest |score|
    by_sym: Dict[str, Dict[str, Any]] = {}
    for a in actions:
        if a["symbol"] not in by_sym or abs(a["score"]) > abs(by_sym[a["symbol"]]["score"]):
            by_sym[a["symbol"]] = a
    return list(by_sym.values())

# ---------- Sizing ----------
def total_value(pos: Dict[str, dict], feed: Dict[str, Any]) -> float:
    syms = feed_symbols(feed)
    v = 0.0
    for sym, p in pos.items():
        node = syms.get(sym)
        if not node:
            continue
        v += p["qty"] * get_price(node)
    return v

def size_with_cash(actions: List[Dict[str, Any]], equity: float, cash: float, sizing: SizingConfig) -> Tuple[List[Dict[str, Any]], float]:
    # Max total buy budget = min(cash, equity * max_deploy_frac) if cap set
    if sizing.max_deploy_frac is None:
        buy_budget_total = cash
    else:
        buy_budget_total = min(cash, (equity + cash) * sizing.max_deploy_frac)

    per_line_target = (equity + cash) * sizing.per_line_frac
    buys = [a for a in actions if a["action"] == "BUY"]
    sells = [a for a in actions if a["action"] == "SELL"]

    sized: List[Dict[str, Any]] = []

    # SELL trims
    for a in sells:
        px = a.get("px") or 0.0
        qty_trim = a.get("qty_hint", 0.0) * sizing.trim_fraction
        qty = math.floor(qty_trim)  # trims are whole shares
        if qty > 0:
            sized.append({**a, "qty": qty, "notional": round(qty * px, 2)})

    # BUYs
    buys.sort(key=lambda x: x["score"], reverse=True)
    spent = 0.0
    for a in buys:
        px = a.get("px") or 0.0
        if px <= 0:
            continue
        remaining = buy_budget_total - spent
        if remaining <= 0:
            break
        target_val = min(per_line_target, remaining)
        qty = target_val / px
        qty = qty if sizing.fractional_buys else math.floor(qty)
        qty = round(qty, 4)
        if qty <= 0:
            continue
        spent += qty * px
        sized.append({**a, "qty": qty, "notional": round(qty * px, 2)})

    cash_left = round(cash - spent, 2)
    return sized, cash_left
