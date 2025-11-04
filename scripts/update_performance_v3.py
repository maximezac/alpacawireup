#!/usr/bin/env python3
from __future__ import annotations
import os, csv, json, math
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime, timezone
from engine import (read_json, write_json, read_positions_csv, write_positions_csv)

PRICES_FINAL   = os.getenv("PRICES_FINAL", "data/prices_final.json")
RECS_PATH      = os.getenv("TRADES_RECS", "data/recommended_trades.json")
LEDGER_PATH    = os.getenv("TRADES_LEDGER", "data/trades_ledger.csv")

# Selection filters
EXECUTE_SCOPE  = os.getenv("EXECUTE_SCOPE", "paper")  # paper | real | both
EXECUTE_TAGS   = [t.strip() for t in os.getenv("EXECUTE_TAGS", "").split(",") if t.strip()]  # optional filter

# Friction
SLIP_PAPER_BPS = float(os.getenv("SLIPPAGE_BPS_PAPER", "10"))
SLIP_REAL_BPS  = float(os.getenv("SLIPPAGE_BPS_PERSONAL", "5"))
LIQ_SC_BPS     = float(os.getenv("LIQ_IMPACT_BPS_SMALLCAP", "5"))
SMALLCAPS      = set(s.strip().upper() for s in os.getenv("SMALLCAPS", "").split(",") if s.strip())

# Utility
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def choose_slippage_bps(scope: str, sym: str) -> float:
    base = SLIP_PAPER_BPS if scope == "paper" else SLIP_REAL_BPS
    return base + (LIQ_SC_BPS if sym in SMALLCAPS else 0.0)

def exec_price(mark_px: float, bps: float, side: str) -> float:
    mult = 1.0 + (bps / 10000.0) if side.upper()=="BUY" else 1.0 - (bps / 10000.0)
    return mark_px * mult

def ensure_ledger(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("datetime_utc,portfolio_id,symbol,action,qty,price_exec,slippage_bps,liquidity_impact_bps,fees,gross_amount,post_trade_cash,post_trade_position,signal_snapshot,rationale,run_id\n")

def append_ledger(path: str, row: dict):
    header = ["datetime_utc","portfolio_id","symbol","action","qty","price_exec","slippage_bps","liquidity_impact_bps","fees","gross_amount","post_trade_cash","post_trade_position","signal_snapshot","rationale","run_id"]
    ensure_ledger(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([row.get(k,"") for k in header])

def load_last_applied(path: str) -> str | None:
    if Path(path).exists():
        try:
            return (json.loads(Path(path).read_text()) or {}).get("last_as_of_utc")
        except:
            return None
    return None

def write_last_applied(path: str, asof: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"last_as_of_utc": asof}, indent=2))

def mark_to_market(cash: float, pos: Dict[str, dict], prices: Dict[str, Any]) -> Tuple[float, float]:
    mv = 0.0
    syms = prices.get("symbols") or {}
    for sym, p in pos.items():
        node = syms.get(sym)
        if not node:
            continue
        px = (node.get("now") or {}).get("price", node.get("price", 0.0)) or 0.0
        mv += p["qty"] * float(px)
    equity = cash + mv
    return equity, mv

def append_history_row(folder: Path, portfolio_id: str, cash: float, pos: Dict[str, dict], prices: Dict[str, Any]) -> None:
    hist_path = folder / "history.csv"
    header = ["date","run_id","portfolio","cash","market_value","equity","daily_pnl","daily_return","cumulative_return","holdings_json","signals_json","notes"]
    # read
    rows: List[List[str]] = []
    if hist_path.exists():
        with open(hist_path, "r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            rows = list(r)
    if not rows:
        rows = [header]

    # previous equity
    col = {h:i for i,h in enumerate(rows[0])}
    prev_equity = None
    for r in rows[1:]:
        try:
            prev_equity = float(r[col["equity"]])
        except: pass
    equity, mv = mark_to_market(cash, pos, prices)
    daily_pnl = (equity - prev_equity) if prev_equity is not None else 0.0
    daily_ret = (equity/prev_equity - 1.0) if prev_equity and prev_equity != 0 else 0.0
    # first equity is baseline for cumulative
    first_equity = None
    for r in rows[1:]:
        try:
            first_equity = float(r[col["equity"]]); break
        except: pass
    cum_ret = (equity/first_equity - 1.0) if first_equity and first_equity != 0 else 0.0

    syms = prices.get("symbols") or {}
    holdings_snap = {
        s: {"qty": round(p["qty"], 6), "avg_cost": round(p["avg_cost"], 6), "mark": (syms.get(s, {}).get("now") or {}).get("price", syms.get(s, {}).get("price"))}
        for s, p in sorted(pos.items())
    }
    signals_snap = {
        s: (syms.get(s) or {}).get("signals", {})
        for s in sorted(pos.keys())
    }

    row = [
        datetime.now(timezone.utc).date().isoformat(),
        os.getenv("GITHUB_RUN_ID", utc_now_iso()),
        portfolio_id,
        round(cash, 2),
        round(mv, 2),
        round(equity, 2),
        round(daily_pnl, 2),
        round(daily_ret, 6),
        round(cum_ret, 6),
        json.dumps(holdings_snap, separators=(",",":")),
        json.dumps(signals_snap, separators=(",",":")),
        ""
    ]

    # Upsert by (date, portfolio)
    if len(rows) == 1:
        rows.append(row)
    else:
        # find existing today row
        today = row[0]
        replaced = False
        for i in range(1, len(rows)):
            if rows[i][0] == today and rows[i][2] == portfolio_id:
                rows[i] = row
                replaced = True
                break
        if not replaced:
            rows.append(row)

    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

def apply_trades_to_portfolio(portfolio_id: str, scope: str, folder: Path, positions_csv: Path,
                              prices: Dict[str, Any], trades: List[Dict[str, Any]], ledger_path: str):
    cash, pos = read_positions_csv(str(positions_csv))

    for t in trades:
        sym = str(t.get("symbol","")).upper()
        if not sym: continue
        side = str(t.get("action","")).upper()
        qty  = float(t.get("qty", 0))
        if qty <= 0: continue

        node = (prices.get("symbols") or {}).get(sym) or {}
        mark = (node.get("now") or {}).get("price", node.get("price", 0.0)) or 0.0
        if t.get("px"):
            mark = float(t["px"])  # prefer rec’s px if present

        bps = choose_slippage_bps(scope, sym)
        px_exec = exec_price(float(mark), bps, side)
        gross = qty * px_exec

        if side == "BUY":
            if gross > cash:  # insufficient cash
                continue
            cash -= gross
            cur = pos.get(sym, {"qty":0.0, "avg_cost":0.0})
            new_qty = cur["qty"] + qty
            new_avg = ((cur["qty"] * cur["avg_cost"]) + gross) / new_qty if new_qty > 0 else cur["avg_cost"]
            pos[sym] = {"qty": new_qty, "avg_cost": new_avg}
        elif side == "SELL":
            cur = pos.get(sym, {"qty":0.0, "avg_cost":0.0})
            sell_qty = min(qty, cur["qty"])
            if sell_qty <= 0:  # nothing to sell
                continue
            proceeds = sell_qty * px_exec
            cash += proceeds
            remain = cur["qty"] - sell_qty
            if remain > 0:
                pos[sym] = {"qty": remain, "avg_cost": cur["avg_cost"]}
            else:
                pos.pop(sym, None)
        else:
            continue

        append_ledger(ledger_path, {
            "datetime_utc": utc_now_iso(),
            "portfolio_id": portfolio_id,
            "symbol": sym,
            "action": side,
            "qty": qty,
            "price_exec": round(px_exec, 6),
            "slippage_bps": bps,
            "liquidity_impact_bps": LIQ_SC_BPS if sym in SMALLCAPS else 0.0,
            "fees": 0.0,
            "gross_amount": round((qty * px_exec) * (1 if side=="SELL" else -1), 2),
            "post_trade_cash": round(cash, 2),
            "post_trade_position": round(pos.get(sym, {}).get("qty", 0.0), 6),
            "signal_snapshot": json.dumps((node.get("signals") or {}), separators=(",",":")),
            "rationale": t.get("reason",""),
            "run_id": os.getenv("GITHUB_RUN_ID",""),
        })

    # Persist positions and history
    write_positions_csv(str(positions_csv), cash, pos)
    append_history_row(folder, portfolio_id, cash, pos, prices)

def main():
    prices = read_json(PRICES_FINAL)
    recs = read_json(RECS_PATH)
    asof = recs.get("as_of_utc")
    ports = recs.get("portfolios") or {}

    executed = []
    for pid, block in ports.items():
        meta = block.get("meta") or {}
        scope = meta.get("trade_scope", "none")
        tags  = meta.get("tags") or []
        if scope not in ("paper","real","both"):
            continue
        if EXECUTE_SCOPE not in (scope, "both") and not (EXECUTE_SCOPE=="paper" and scope=="both") and not (EXECUTE_SCOPE=="real" and scope=="both"):
            continue
        if EXECUTE_TAGS and not any(t in tags for t in EXECUTE_TAGS):
            continue

        folder = Path(meta["positions_path"]).resolve().parent
        positions_csv = Path(meta["positions_path"])
        last_applied = load_last_applied(folder / "last_applied.json")
        # idempotency: apply once per as_of_utc
        if last_applied == asof:
            continue

        trades = block.get("trades") or []
        if not trades:
            # still stamp history for the day
            cash, pos = read_positions_csv(str(positions_csv))
            append_history_row(folder, pid, cash, pos, prices)
            write_last_applied(folder / "last_applied.json", asof)
            continue

        apply_trades_to_portfolio(pid, scope, folder, positions_csv, prices, trades, LEDGER_PATH)
        write_last_applied(folder / "last_applied.json", asof)
        executed.append(pid)

    print(f"[OK] update_performance_v3 complete. executed: {executed}")

if __name__ == "__main__":
    main()
