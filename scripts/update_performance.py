#!/usr/bin/env python3
"""
Appends portfolio history, logs trades, and writes performance summaries.

Inputs (env/defaults):
  PRICES_FINAL              = data/prices_final.json
  PORTFOLIO_PAPER_CSV       = data/portfolio_paper.csv
  PORTFOLIO_PERSONAL_CSV    = data/portfolio_personal.csv
  HISTORY_PAPER             = data/portfolio_paper_history.csv
  HISTORY_PERSONAL          = data/portfolio_personal_history.csv
  TRADES_RECS               = data/recommended_trades.json
  TRADES_LEDGER             = data/trades_ledger.csv

  # Execution / friction
  APPLY_TRADES              = "0"            # "1" to apply recs into positions; "0" = dry run
  SLIPPAGE_BPS_PAPER        = "10"           # 0.10%
  SLIPPAGE_BPS_PERSONAL     = "5"            # 0.05%
  LIQ_IMPACT_BPS_SMALLCAP   = "5"            # +0.05%
  SMALLCAPS                 = "QBTS,QUBT,RGTI,ASTS,RKLB,SOFI"  # comma list (optional)

Outputs (appends or overwrites):
  - Append a row to HISTORY_* CSVs (or update for same date)
  - Append rows to TRADES_LEDGER (if APPLY_TRADES=1 and recs present)
  - Write perf_summary_paper.json / perf_summary_personal.json
"""

import os, sys, json, csv, math
from pathlib import Path
from datetime import datetime, timezone, date
from typing import Dict, Any, Tuple

# ---------- env & paths ----------
PRICES_FINAL = os.getenv("PRICES_FINAL", "data/prices_final.json")
PAPER_CSV    = os.getenv("PORTFOLIO_PAPER_CSV", "data/portfolio_paper.csv")
PERS_CSV     = os.getenv("PORTFOLIO_PERSONAL_CSV", "data/portfolio_personal.csv")
HIST_PAPER   = os.getenv("HISTORY_PAPER", "data/portfolio_paper_history.csv")
HIST_PERS    = os.getenv("HISTORY_PERSONAL", "data/portfolio_personal_history.csv")
TRADES_RECS  = os.getenv("TRADES_RECS", "data/recommended_trades.json")
TRADES_LEDGER= os.getenv("TRADES_LEDGER", "data/trades_ledger.csv")
ledger_path = Path(os.environ.get("TRADES_LEDGER", "data/trades_ledger.csv"))
ledger_path.parent.mkdir(parents=True, exist_ok=True)
if not ledger_path.exists():
    ledger_header = (
        "datetime_utc,portfolio,symbol,action,qty,price_exec,slippage_bps,"
        "liquidity_impact_bps,fees,gross_amount,post_trade_cash,post_trade_position,"
        "signal_snapshot,rationale,run_id\n"
    )
    ledger_path.write_text(ledger_header)


APPLY_TRADES = os.getenv("APPLY_TRADES", "0") == "1"

SLIP_PAPER_BPS = float(os.getenv("SLIPPAGE_BPS_PAPER", "10"))
SLIP_PERS_BPS  = float(os.getenv("SLIPPAGE_BPS_PERSONAL", "5"))
LIQ_SC_BPS     = float(os.getenv("LIQ_IMPACT_BPS_SMALLCAP", "5"))
SMALLCAPS      = set(s.strip().upper() for s in os.getenv("SMALLCAPS", "").split(",") if s.strip())

# ---------- utils ----------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def iso_date_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def read_csv_optional(path: str) -> Tuple[list, list]:
    if not Path(path).exists():
        return [], []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        rows = list(r)
    if not rows:
        return [], []
    return rows[0], rows[1:]

def write_csv(path: str, header: list, rows: list):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def write_positions(csv_path: str, cash: float, pos: Dict[str, dict]):
    header = ["symbol","qty","avg_cost"]
    rows = []
    rows.append(["__CASH__", str(round(cash,2)), "0"])
    for sym in sorted(pos.keys()):
        p = pos[sym]
        rows.append([sym, str(p["qty"]), str(p["avg_cost"])])
    write_csv(csv_path, header, rows)

def append_or_upsert_history(path: str, key_cols: list, row: dict):
    header, rows = read_csv_optional(path)
    if not header:
        header = list(row.keys())
        write_csv(path, header, [[row[k] for k in header]])
        return
    # align header (add any new columns if needed)
    for k in row.keys():
        if k not in header:
            header.append(k)
    # upsert by key
    key_idx = [header.index(k) for k in key_cols]
    updated = False
    for i, r in enumerate(rows):
        # pad row to header length
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
            rows[i] = r
        if all((r[idx] == str(row[header[idx]])) for idx in key_idx):
            # replace entire row with aligned values
            rows[i] = [str(row.get(h, "")) for h in header]
            updated = True
            break
    if not updated:
        rows.append([str(row.get(h, "")) for h in header])
    write_csv(path, header, rows)

def append_csv(path: str, row: dict):
    header, rows = read_csv_optional(path)
    if not header:
        header = list(row.keys())
    else:
        for k in row.keys():
            if k not in header:
                header.append(k)
        # pad existing rows
        rows = [r + [""] * (len(header) - len(r)) for r in rows]
    rows.append([str(row.get(h, "")) for h in header])
    write_csv(path, header, rows)

# ---------- core calc ----------
def load_prices_and_signals(prices_path: str) -> Tuple[dict, dict]:
    """
    Returns:
      prices:  {SYM: price_float}
      signals: {SYM: {TS, NS, CDS, decision, ...}}
    """
    data = read_json(prices_path)
    syms = data.get("symbols", {})
    prices, signals = {}, {}
    for sym, node in syms.items():
        prices[sym] = float(node.get("price")) if node.get("price") is not None else None
        sigs = node.get("signals") or {}
        # keep as-is from file (no recalculation)
        signals[sym] = {
            "TS": sigs.get("TS"),
            "NS": sigs.get("NS"),
            "CDS": sigs.get("CDS"),
            "decision": node.get("decision"),
            "wT": sigs.get("wT"),
            "wN": sigs.get("wN"),
        }
    return prices, signals

def read_positions(csv_path: str) -> Tuple[float, Dict[str, dict]]:
    """
    Expected CSV schema (minimal):
      symbol,qty,avg_cost   (+ optional price columns; we only need symbol/qty/avg_cost)
      Special row with symbol="__CASH__" and qty=cash, avg_cost ignored
    Returns: (cash, {SYM: {qty, avg_cost}})
    """
    if not Path(csv_path).exists():
        return 0.0, {}
    header, rows = read_csv_optional(csv_path)
    if not header: return 0.0, {}
    col = {h: i for i, h in enumerate(header)}
    def get(row, k, default=""):
        return row[col[k]] if k in col and col[k] < len(row) else default

    cash = 0.0
    pos = {}
    for r in rows:
        sym = get(r, "symbol", "").upper()
        if not sym: continue
        if sym == "__CASH__":
            try: cash = float(get(r, "qty", "0"))
            except: cash = 0.0
            continue
        try:
            qty = float(get(r, "qty", "0"))
            avg = float(get(r, "avg_cost", "0"))
        except:
            qty, avg = 0.0, 0.0
        if qty != 0.0:
            pos[sym] = {"qty": qty, "avg_cost": avg}
    return cash, pos

def mark_to_market(cash: float, pos: Dict[str, dict], prices: dict) -> Tuple[float, float]:
    mv = 0.0
    for sym, p in pos.items():
        px = prices.get(sym)
        if px is None: continue
        mv += p["qty"] * px
    equity = cash + mv
    return equity, mv

def choose_slippage_bps(portfolio: str, sym: str) -> float:
    base = SLIP_PAPER_BPS if portfolio == "paper" else SLIP_PERS_BPS
    liq = LIQ_SC_BPS if (sym in SMALLCAPS) else 0.0
    return base + liq

def exec_price(mark_px: float, bps: float, side: str) -> float:
    # BUY = pay up; SELL = receive less
    mult = 1.0 + (bps / 10000.0) if side.upper()=="BUY" else 1.0 - (bps / 10000.0)
    return mark_px * mult

def apply_trades_to_positions(
    portfolio: str,
    cash: float,
    pos: Dict[str, dict],
    prices: dict,
    recs: list,
) -> Tuple[float, Dict[str, dict], list]:
    """
    Apply trades (if enabled) and emit ledger rows.
    rec item shape (from your postprocess): 
      {"portfolio":"paper|personal","symbol":"AMD","action":"BUY","qty":5,"price":229.3,"reason":"CDS>0.5"...}
    We *do not* recalc price; we use the provided 'price' if present else current mark.
    """
    ledger_rows = []
    for t in recs:
        if t.get("portfolio") not in (portfolio,):
            continue
        sym = str(t.get("symbol", "")).upper()
        if not sym: continue
        side = t.get("action", "").upper()
        qty  = float(t.get("qty", 0))
        if qty == 0: continue

        mark = float(t.get("price")) if t.get("price") is not None else float(prices.get(sym) or 0.0)
        if mark == 0.0:
            # cannot price -> skip
            continue

        bps = choose_slippage_bps(portfolio, sym)
        px_exec = exec_price(mark, bps, side)
        gross = qty * px_exec

        if side == "BUY":
            cost = gross
            if cost > cash:  # guard: insufficient cash
                continue
            cash -= cost
            cur = pos.get(sym, {"qty":0.0, "avg_cost":0.0})
            new_qty = cur["qty"] + qty
            new_avg = ((cur["qty"] * cur["avg_cost"]) + cost) / new_qty if new_qty != 0 else cur["avg_cost"]
            pos[sym] = {"qty": new_qty, "avg_cost": new_avg}
        elif side == "SELL":
            cur = pos.get(sym, {"qty":0.0, "avg_cost":0.0})
            sell_qty = min(qty, cur["qty"])
            if sell_qty <= 0:  # nothing to sell
                continue
            proceeds = sell_qty * px_exec
            cash += proceeds
            remaining = cur["qty"] - sell_qty
            if remaining > 0:
                pos[sym] = {"qty": remaining, "avg_cost": cur["avg_cost"]}
            else:
                pos.pop(sym, None)
        else:
            continue

        ledger_rows.append({
            "datetime_utc": utc_now_iso(),
            "portfolio": portfolio,
            "symbol": sym,
            "action": side,
            "qty": qty,
            "price_exec": round(px_exec, 6),
            "slippage_bps": bps,
            "liquidity_impact_bps": LIQ_SC_BPS if (sym in SMALLCAPS) else 0.0,
            "fees": 0.0,
            "gross_amount": round((qty * px_exec) * (1 if side=="SELL" else -1), 2),
            "post_trade_cash": round(cash, 2),
            "post_trade_position": round(pos.get(sym, {}).get("qty", 0.0), 6),
            "signal_snapshot": "",  # optional: filled below if available
            "rationale": t.get("reason", ""),
            "run_id": os.getenv("GITHUB_RUN_ID", ""),
        })
    return cash, pos, ledger_rows

def load_previous_equity(history_csv: str) -> Tuple[float, float]:
    """Returns (equity_prev, equity_start) or (None, None) if no history."""
    header, rows = read_csv_optional(history_csv)
    if not rows: return (None, None)
    col = {h:i for i,h in enumerate(header)}
    def gv(r,k): 
        try: return float(r[col[k]])
        except: return None
    equity_prev = gv(rows[-1], "equity")
    # find first equity to compute cumulative
    equity_start = None
    for r in rows:
        v = gv(r, "equity")
        if v is not None:
            equity_start = v
            break
    return (equity_prev, equity_start)

def build_history_row(
    portfolio: str,
    cash: float,
    pos: Dict[str, dict],
    prices: dict,
    signals: dict,
    equity_prev: float | None,
    equity_start: float | None,
) -> dict:
    equity, mv = mark_to_market(cash, pos, prices)
    daily_pnl = (equity - equity_prev) if (equity_prev is not None) else 0.0
    daily_ret = (equity / equity_prev - 1.0) if (equity_prev and equity_prev != 0) else 0.0
    cum_ret   = (equity / equity_start - 1.0) if (equity_start and equity_start != 0) else 0.0

    # holdings & signals snapshot
    holdings_json = {
        sym: {"qty": round(p["qty"], 6), "avg_cost": p["avg_cost"], "mark": prices.get(sym)}
        for sym, p in sorted(pos.items())
    }
    sig_snap = {
        sym: signals.get(sym, {})
        for sym in sorted(pos.keys())
    }

    row = {
        "date": iso_date_utc(),
        "run_id": os.getenv("GITHUB_RUN_ID", utc_now_iso()),
        "portfolio": portfolio,
        "cash": round(cash, 2),
        "market_value": round(mv, 2),
        "equity": round(equity, 2),
        "daily_pnl": round(daily_pnl, 2),
        "daily_return": round(daily_ret, 6),
        "cumulative_return": round(cum_ret, 6),
        "holdings_json": json.dumps(holdings_json, separators=(",",":")),
        "signals_json":  json.dumps(sig_snap, separators=(",",":")),
        "notes": ""
    }
    return row

def make_perf_summary(history_csv: str, out_json: str):
    header, rows = read_csv_optional(history_csv)
    if not rows:
        write_json(out_json, {"as_of_utc": utc_now_iso(), "note": "no history yet"})
        return
    col = {h:i for i,h in enumerate(header)}
    def gf(r, k, d=0.0):
        try: return float(r[col[k]])
        except: return d

    equities = [gf(r, "equity") for r in rows if "equity" in col]
    rets     = [gf(r, "daily_return") for r in rows if "daily_return" in col]

    equity_today = equities[-1]
    equity_start = equities[0] if equities else 0.0

    # max drawdown (equity series)
    peak = -1e18
    max_dd = 0.0
    for e in equities:
        if e > peak: peak = e
        dd = (e/peak - 1.0) if peak > 0 else 0.0
        if dd < max_dd: max_dd = dd

    # rolling metrics (20d)
    n = len(rets)
    if n >= 20:
        last20 = rets[-20:]
        mean20 = sum(last20)/20.0
        var20  = sum((x-mean20)**2 for x in last20)/20.0
        vol20  = math.sqrt(var20) * math.sqrt(252)
        sharpe = (mean20/vol20) if vol20 > 0 else 0.0
    else:
        mean20 = 0.0; vol20 = 0.0; sharpe = 0.0

    write_json(out_json, {
        "as_of_utc": utc_now_iso(),
        "equity_start": round(equity_start, 2),
        "equity_today": round(equity_today, 2),
        "cumulative_return": round((equity_today/equity_start - 1.0) if equity_start>0 else 0.0, 6),
        "vol_20d": round(vol20, 6),
        "sharpe_20d": round(sharpe, 6),
        "max_drawdown_alltime": round(max_dd, 6),
        "days": n
    })

def main():
    prices, signals = load_prices_and_signals(PRICES_FINAL)

    # --- load positions & prior equity ---
    paper_cash, paper_pos = read_positions(PAPER_CSV)
    pers_cash,  pers_pos  = read_positions(PERS_CSV)

    equity_prev_p, equity_start_p = load_previous_equity(HIST_PAPER)
    equity_prev_u, equity_start_u = load_previous_equity(HIST_PERS)

    # --- load trade recommendations (once) ---
    recs_list = []
    if Path(TRADES_RECS).exists():
        recs = read_json(TRADES_RECS)

        def _normalize_trade(t: dict, portfolio: str) -> dict:
        # prefer explicit exec price from postprocess ("px")
            px = t.get("price", t.get("px", None))
            out = dict(t)
            out["portfolio"] = portfolio
            if px is not None:
                out["price"] = px
            return out

        if isinstance(recs, dict):
            # v1 schema: {"paper":[...], "personal":[...]}
            if "paper" in recs or "personal" in recs:
                recs_list += [_normalize_trade(t, "paper")    for t in (recs.get("paper") or [])]
                recs_list += [_normalize_trade(t, "personal") for t in (recs.get("personal") or [])]

            # v2 schema: {"portfolio_paper":{"trades":[...]}, "portfolio_personal":{"trades":[...]}}
            if "portfolio_paper" in recs or "portfolio_personal" in recs:
                recs_list += [
                    _normalize_trade(t, "paper")
                    for t in (recs.get("portfolio_paper", {}).get("trades") or [])
                ]
                recs_list += [
                    _normalize_trade(t, "personal")
                    for t in (recs.get("portfolio_personal", {}).get("trades") or [])
                ]

        elif isinstance(recs, list):
            # flat list with explicit "portfolio" in each item
            for t in recs:
                p = (t.get("portfolio") or "").lower()
                if p in ("paper", "personal"):
                    recs_list.append(_normalize_trade(t, p))

        print(f"[debug] loaded {len(recs_list)} trade recs from {TRADES_RECS}")


    # --- optionally apply trades ONCE ---
    ledger_to_append = []
    if APPLY_TRADES and recs_list:
        # PAPER
        paper_cash, paper_pos, logs1 = apply_trades_to_positions("paper", paper_cash, paper_pos, prices, recs_list)
        ledger_to_append += logs1
        write_positions(PAPER_CSV, paper_cash, paper_pos)

        # PERSONAL
        pers_cash, pers_pos, logs2 = apply_trades_to_positions("personal", pers_cash, pers_pos, prices, recs_list)
        ledger_to_append += logs2
        write_positions(PERS_CSV, pers_cash, pers_pos)

    # --- build history rows AFTER any trade application ---
    row_paper = build_history_row("paper", paper_cash, paper_pos, prices, signals, equity_prev_p, equity_start_p)
    append_or_upsert_history(HIST_PAPER, ["date","portfolio"], row_paper)

    row_pers  = build_history_row("personal", pers_cash, pers_pos, prices, signals, equity_prev_u, equity_start_u)
    append_or_upsert_history(HIST_PERS, ["date","portfolio"], row_pers)

    # --- attach signal snapshots to ledger rows and persist ---
    for r in ledger_to_append:
        sym = r["symbol"]
        r["signal_snapshot"] = json.dumps(signals.get(sym, {}), separators=(",",":"))
    for log in ledger_to_append:
        append_csv(TRADES_LEDGER, log)

    # --- summaries ---
    make_perf_summary(HIST_PAPER, "data/perf_summary_paper.json")
    make_perf_summary(HIST_PERS,  "data/perf_summary_personal.json")

    print("[OK] update_performance complete.")
    print(f"  paper:    equity={row_paper['equity']}  cash={row_paper['cash']}")
    print(f"  personal: equity={row_pers['equity']}   cash={row_pers['cash']}")
    if APPLY_TRADES:
        print(f"  trades logged: {len(ledger_to_append)}")

if __name__ == "__main__":
    main()
