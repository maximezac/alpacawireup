#!/usr/bin/env python3
from __future__ import annotations

import os, csv, json, math
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime, timezone

# local helpers
from engine import (
    read_json, write_json,
    read_positions_csv, write_positions_csv
)

# -------- Global mode switches --------
BACKTEST_MODE = os.getenv("BACKTEST_MODE", "0") == "1"
EXPERIMENT_ID = os.getenv("EXPERIMENT_ID", "baseline_v1")
SNAPSHOT_AS_OF = os.getenv("SNAPSHOT_AS_OF")  # e.g. "2023-06-15" for backtest step date

# -------- Base config / paths (v3 defaults) --------
PRICES_FINAL   = os.getenv("PRICES_FINAL", "data/prices_final.json")
RECS_PATH      = os.getenv("TRADES_RECS", "artifacts/recommended_trades_v3.json")  # v3 trade plan
LEDGER_PATH    = os.getenv("TRADES_LEDGER", "data/trades_ledger.csv")

# Backtest overrides (optional)
if BACKTEST_MODE:
    PRICES_FINAL = os.getenv("BACKTEST_PRICES_FINAL", PRICES_FINAL)
    RECS_PATH    = os.getenv("BACKTEST_TRADES_PATH", RECS_PATH)
    LEDGER_PATH  = os.getenv("BACKTEST_TRADES_LEDGER", LEDGER_PATH)

# Selection filters
EXECUTE_SCOPE  = (os.getenv("EXECUTE_SCOPE", "paper") or "paper").lower()  # paper | real | both
EXECUTE_TAGS   = [t.strip().lower() for t in os.getenv("EXECUTE_TAGS", "").split(",") if t.strip()]

# Friction
SLIP_PAPER_BPS = float(os.getenv("SLIPPAGE_BPS_PAPER", "10"))
SLIP_REAL_BPS  = float(os.getenv("SLIPPAGE_BPS_PERSONAL", "5"))
LIQ_SC_BPS     = float(os.getenv("LIQ_IMPACT_BPS_SMALLCAP", "5"))
SMALLCAPS      = set(s.strip().upper() for s in os.getenv("SMALLCAPS", "").split(",") if s.strip())

# -------- Artifact controls (declutter by default) --------
APPLY_TRADES = os.getenv("APPLY_TRADES", "1").strip() == "1"
DISABLE_VERSIONED = os.getenv("V3_DISABLE_VERSIONED_ARTIFACTS", "0").strip() == "1"

ARTIFACT_VERSIONED   = (os.getenv("ARTIFACT_VERSIONED", "false").lower() == "true")
DISABLE_VERSIONED    = (os.getenv("V3_DISABLE_VERSIONED_ARTIFACTS", "0").strip() == "1")
VERSIONING_ENABLED   = ARTIFACT_VERSIONED and not DISABLE_VERSIONED

# -------- Time helpers --------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _utc_stamp() -> str:
    # e.g., 20251107_203755.325986Z0000
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S.%fZ0000")

# -------- Artifact helpers --------
def write_artifact(base_dir: Path, stem: str, obj: dict) -> None:
    """
    Always write rolling file <stem>.json into base_dir.
    If ARTIFACT_VERSIONED is true, also write a timestamped snapshot into base_dir/.history/
    and prune to ARTIFACT_RETENTION most recent.
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    # Rolling (no timestamp)
    rolling = base_dir / f"{stem}.json"
    write_json(str(rolling), obj)

    if not VERSIONING_ENABLED:
        return

    ARTIFACT_RETENTION = int(os.getenv("ARTIFACT_RETENTION", "3"))
    hist_dir = base_dir / ".history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    snap = hist_dir / f"{stem}_{_utc_stamp()}.json"
    write_json(str(snap), obj)

    snaps = sorted(hist_dir.glob(f"{stem}_*.json"))
    if ARTIFACT_RETENTION >= 0 and len(snaps) > ARTIFACT_RETENTION:
        for old in snaps[:-ARTIFACT_RETENTION]:
            try:
                old.unlink(missing_ok=True)
            except Exception:
                pass

# -------- Utils --------
def utc_now_iso() -> str:
    return _utc_now_iso()

def choose_slippage_bps(scope: str, sym: str) -> float:
    base = SLIP_PAPER_BPS if scope == "paper" else SLIP_REAL_BPS
    return base + (LIQ_SC_BPS if sym in SMALLCAPS else 0.0)

def latest_px(node: dict) -> float:
    """Single source of truth for price (live-only format; keeps legacy fallback)."""
    if not node:
        return 0.0
    v = node.get("price")
    if v is None:
        v = (node.get("now") or {}).get("price")  # legacy fallback
    try:
        return float(v or 0.0)
    except Exception:
        return 0.0

def exec_price(mark_px: float, bps: float, side: str) -> float:
    # BUY -> price up by slippage; SELL -> price down
    mult = 1.0 + (bps / 10000.0) if side.upper() == "BUY" else 1.0 - (bps / 10000.0)
    return mark_px * mult

def ensure_ledger(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(
            "datetime_utc,portfolio_id,symbol,action,qty,price_exec,slippage_bps,liquidity_impact_bps,fees,"
            "gross_amount,post_trade_cash,post_trade_position,signal_snapshot,rationale,run_id\n"
        )

def append_ledger(path: str, row: dict):
    header = [
        "datetime_utc","portfolio_id","symbol","action","qty","price_exec","slippage_bps",
        "liquidity_impact_bps","fees","gross_amount","post_trade_cash","post_trade_position",
        "signal_snapshot","rationale","run_id"
    ]
    ensure_ledger(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([row.get(k, "") for k in header])

def load_last_applied(path: str | Path) -> str | None:
    p = Path(path)
    if p.exists():
        try:
            return (json.loads(p.read_text()) or {}).get("last_as_of_utc")
        except Exception:
            return None
    return None

def portfolio_snapshot(portfolio_id: str, cash: float, pos: Dict[str, dict], prices: Dict[str, Any]) -> dict:
    syms = (prices or {}).get("symbols") or {}
    market_value = 0.0
    holdings = {}
    for sym, p in sorted(pos.items()):
        node = syms.get(sym) or {}
        px = latest_px(node)
        mv = float(p["qty"]) * float(px)
        market_value += mv
        holdings[sym] = {
            "qty": round(p["qty"], 6),
            "avg_cost": round(p["avg_cost"], 6),
            "mark": round(px, 6),
            "market_value": round(mv, 2),
        }
    equity = cash + market_value
    return {
        "as_of_utc": (prices or {}).get("as_of_utc"),
        "portfolio_id": portfolio_id,
        "cash": round(cash, 2),
        "market_value": round(market_value, 2),
        "equity": round(equity, 2),
        "holdings": holdings,
    }

def write_portfolio_value(folder: Path, snap: dict) -> None:
    write_json(str(folder / "portfolio_value.json"), snap)
    if not VERSIONING_ENABLED:
        return
    asof = (snap.get("as_of_utc") or "")
    safe_asof = asof.replace(":", "").replace("-", "").replace("T","_").replace("+","Z")
    if safe_asof:
        write_json(str(folder / f"portfolio_value_{safe_asof}.json"), snap)

def write_applied_trades(folder: Path, portfolio_id: str, prices: dict, payload: dict) -> None:
    """
    payload should contain pre/post snapshots + trades list.
    """
    asof = (prices or {}).get("as_of_utc", "")
    obj = {"as_of_utc": asof, "portfolio_id": portfolio_id}
    obj.update(payload)
    write_json(str(folder / "trades_applied.json"), obj)
    if not VERSIONING_ENABLED:
        return
    # You could add versioned trades_applied here in future if desired.

def ensure_portfolio_ledger(folder: Path) -> Path:
    """Create per-portfolio ledger with header if missing."""
    p = folder / "trades_ledger.csv"
    if not p.exists():
        p.write_text(
            "datetime_utc,portfolio_id,symbol,action,qty,price_exec,slippage_bps,liquidity_impact_bps,fees,"
            "gross_amount,post_trade_cash,post_trade_position,signal_snapshot,rationale,run_id\n"
        )
    return p

def write_execution_summary(folder: Path, portfolio_id: str, positions_csv: Path, cash: float, snap: dict | None = None) -> None:
    base = {
        "datetime_utc": utc_now_iso(),
        "portfolio_id": portfolio_id,
        "positions_path": str(positions_csv),
        "cash": round(cash, 2),
        "run_id": os.getenv("GITHUB_RUN_ID", ""),
        "backtest_mode": BACKTEST_MODE,
        "experiment_id": EXPERIMENT_ID if BACKTEST_MODE else "",
    }
    if snap:
        base.update({
            "market_value": snap.get("market_value"),
            "equity": snap.get("equity"),
            "as_of_utc": snap.get("as_of_utc"),
        })
    write_artifact(folder, "last_execution_summary", base)

def write_last_applied(path: str | Path, asof: str):
    """
    Only used in live mode to avoid re-applying the same snapshot.
    In BACKTEST_MODE we deliberately do NOT write this.
    """
    if BACKTEST_MODE:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"last_as_of_utc": asof}, indent=2))

def mark_to_market(cash: float, pos: Dict[str, dict], prices: Dict[str, Any]) -> Tuple[float, float]:
    mv = 0.0
    syms = prices.get("symbols") or {}
    for sym, p in pos.items():
        node = syms.get(sym)
        if not node:
            continue
        px = latest_px(node)
        mv += p["qty"] * float(px)
    equity = cash + mv
    return equity, mv

def append_history_row(
    folder: Path,
    portfolio_id: str,
    cash: float,
    pos: Dict[str, dict],
    prices: Dict[str, Any],
    date_override: str | None = None,
) -> None:
    """
    Writes/updates a history row for (date, portfolio).

    In live mode: date defaults to today's date.
    In BACKTEST_MODE: pass SNAPSHOT_AS_OF so history date = simulated trade date.
    """
    folder.mkdir(parents=True, exist_ok=True)
    hist_path = folder / "history.csv"
    header = [
        "date","run_id","portfolio","cash","market_value","equity","daily_pnl","daily_return",
        "cumulative_return","holdings_json","signals_json","notes"
    ]

    # read existing
    rows: List[List[str]] = []
    if hist_path.exists():
        with open(hist_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    if not rows:
        rows = [header]

    col = {h: i for i, h in enumerate(rows[0])}

    equity, mv = mark_to_market(cash, pos, prices)

    # previous equity for this portfolio
    prev_equity = None
    for r in rows[1:]:
        try:
            if r[col["portfolio"]] == portfolio_id:
                prev_equity = float(r[col["equity"]])
        except Exception:
            pass

    daily_pnl = (equity - prev_equity) if prev_equity is not None else 0.0
    daily_ret = (equity / prev_equity - 1.0) if (prev_equity is not None and prev_equity != 0) else 0.0

    # baseline equity for cumulative return
    first_equity = None
    for r in rows[1:]:
        try:
            if r[col["portfolio"]] == portfolio_id:
                first_equity = float(r[col["equity"]]); break
        except Exception:
            pass
    cum_ret = (equity / first_equity - 1.0) if (first_equity is not None and first_equity != 0) else 0.0

    syms = prices.get("symbols") or {}
    holdings_snap = {
        s: {
            "qty": round(p["qty"], 6),
            "avg_cost": round(p["avg_cost"], 6),
            "mark": latest_px(syms.get(s, {})),
        }
        for s, p in sorted(pos.items())
    }
    signals_snap = {s: (syms.get(s) or {}).get("signals", {}) for s in sorted(pos.keys())}

    # Use override date (backtest) or "today" (live)
    row_date = date_override or datetime.now(timezone.utc).date().isoformat()

    row = [
        row_date,
        os.getenv("GITHUB_RUN_ID", utc_now_iso()),
        portfolio_id,
        round(cash, 2),
        round(mv, 2),
        round(equity, 2),
        round(daily_pnl, 2),
        round(daily_ret, 6),
        round(cum_ret, 6),
        json.dumps(holdings_snap, separators=(",", ":")),
        json.dumps(signals_snap, separators=(",", ":")),
        ""
    ]

    # upsert (date, portfolio)
    if len(rows) == 1:
        rows.append(row)
    else:
        replaced = False
        for i in range(1, len(rows)):
            if rows[i][0] == row_date and rows[i][2] == portfolio_id:
                rows[i] = row
                replaced = True
                break
        if not replaced:
            rows.append(row)

    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

def apply_trades_to_portfolio(
    portfolio_id: str,
    scope: str,
    folder: Path,
    positions_csv: Path,
    prices: Dict[str, Any],
    trades: List[Dict[str, Any]],
    ledger_path: str
):
    folder.mkdir(parents=True, exist_ok=True)
    cash, pos = read_positions_csv(str(positions_csv))
    pre_snap = portfolio_snapshot(portfolio_id, cash, pos, prices)
    write_portfolio_value(folder, pre_snap)
    portfolio_ledger_path = ensure_portfolio_ledger(folder)
    applied_trades: List[dict] = []  # capture what we actually execute (after sizing)

    for t in trades:
        sym = str(t.get("symbol", "")).upper()
        if not sym:
            continue
        side = str(t.get("action", "")).upper()
        qty  = float(t.get("qty", 0))
        if qty <= 0:
            continue

        node = (prices.get("symbols") or {}).get(sym) or {}
        mark = latest_px(node)
        if t.get("px") is not None:
            try:
                mark = float(t["px"])
            except Exception:
                pass

        bps = choose_slippage_bps(scope, sym)
        px_exec = exec_price(float(mark), bps, side)
        gross = qty * px_exec

        if side == "BUY":
            if gross > cash:
                # insufficient cash; skip
                continue
            cash -= gross
            cur = pos.get(sym, {"qty": 0.0, "avg_cost": 0.0})
            new_qty = cur["qty"] + qty
            new_avg = ((cur["qty"] * cur["avg_cost"]) + gross) / new_qty if new_qty > 0 else cur["avg_cost"]
            pos[sym] = {"qty": new_qty, "avg_cost": new_avg}

        elif side == "SELL":
            cur = pos.get(sym, {"qty": 0.0, "avg_cost": 0.0})
            sell_qty = min(qty, cur["qty"])
            if sell_qty <= 0:
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

        row = {
            "datetime_utc": utc_now_iso(),
            "portfolio_id": portfolio_id,
            "symbol": sym,
            "action": side,
            "qty": qty,
            "price_exec": round(px_exec, 6),
            "slippage_bps": bps,
            "liquidity_impact_bps": LIQ_SC_BPS if sym in SMALLCAPS else 0.0,
            "fees": 0.0,
            "gross_amount": round((qty * px_exec) * (1 if side == "SELL" else -1), 2),
            "post_trade_cash": round(cash, 2),
            "post_trade_position": round(pos.get(sym, {}).get("qty", 0.0), 6),
            "signal_snapshot": json.dumps((node.get("signals") or {}), separators=(",", ":")),
            "rationale": t.get("reason", ""),
            "run_id": os.getenv("GITHUB_RUN_ID", ""),
        }
        
        # Global ledger (existing)
        append_ledger(ledger_path, row)
        # Per-portfolio ledger (new)
        append_ledger(str(portfolio_ledger_path), row)
        
        # Track what we executed this run (for trades_applied*.json)
        applied_trades.append({
            "symbol": sym,
            "action": side,
            "qty": qty,
            "px_exec": row["price_exec"],
            "slippage_bps": row["slippage_bps"],
            "reason": row["rationale"],
            "signals": json.loads(row["signal_snapshot"] or "{}"),
        })

    # --- Snapshot AFTER trades (post state)
    post_snap = portfolio_snapshot(portfolio_id, cash, pos, prices)
    
    # Persist applied trades w/ pre/post equity
    write_applied_trades(folder, portfolio_id, prices, {
        "pre": pre_snap,
        "post": post_snap,
        "trades": applied_trades,
    })
    
    # Update summary with equity/mv
    write_execution_summary(folder, portfolio_id, positions_csv, cash, snap=post_snap)

    # Persist positions & history
    write_positions_csv(str(positions_csv), cash, pos)
    append_history_row(
        folder,
        portfolio_id,
        cash,
        pos,
        prices,
        date_override=SNAPSHOT_AS_OF if BACKTEST_MODE else None,
    )

def _scope_matches(exec_scope: str, port_scope: str) -> bool:
    exec_scope = exec_scope.lower()
    port_scope = port_scope.lower()
    if exec_scope == "both":
        return port_scope in ("paper", "real", "both")
    if port_scope == "both":
        return exec_scope in ("paper", "real")
    return exec_scope == port_scope

# -------- Main --------
def main():
    prices = read_json(PRICES_FINAL)
    recs = read_json(RECS_PATH)
    asof = recs.get("as_of_utc")
    ports = recs.get("portfolios") or {}

    executed: List[str] = []

    for pid, block in ports.items():
        meta = block.get("meta") or {}
        scope = (meta.get("trade_scope") or "none").lower()
        tags  = [t.lower() for t in (meta.get("tags") or [])]

        if scope not in ("paper", "real", "both"):
            continue
        if not _scope_matches(EXECUTE_SCOPE, scope):
            continue
        if EXECUTE_TAGS and not any(t in tags for t in EXECUTE_TAGS):
            continue

        positions_path = meta.get("positions_path") or f"data/portfolios/{pid}/positions.csv"
        positions_csv = Path(positions_path)
        folder = positions_csv.parent

        # idempotency per as_of_utc (LIVE ONLY)
        last_applied = None
        if not BACKTEST_MODE:
            last_applied = load_last_applied(folder / "last_applied.json")

        # Only skip if we're actually applying trades; gather runs should still proceed.
        if not BACKTEST_MODE and APPLY_TRADES and last_applied == asof:
            # already applied this snapshot to this portfolio in live mode
            continue

        # If positions file missing, don't block execution for this snapshot later.
        if not positions_csv.exists():
            folder.mkdir(parents=True, exist_ok=True)
            print(f"[skip] {pid}: positions file missing at {positions_csv}")
            continue

        trades = block.get("trades") or []

        if not trades:
            # No trades: still maintain history/snapshots.
            cash, pos = read_positions_csv(str(positions_csv))
            append_history_row(
                folder,
                pid,
                cash,
                pos,
                prices,
                date_override=SNAPSHOT_AS_OF if BACKTEST_MODE else None,
            )
            if APPLY_TRADES and not BACKTEST_MODE:
                write_last_applied(folder / "last_applied.json", asof)
            snap = portfolio_snapshot(pid, cash, pos, prices)
            write_portfolio_value(folder, snap)
            write_execution_summary(folder, pid, positions_csv, cash, snap=snap)
            continue

        if APPLY_TRADES:
            # REAL/PAPER execution path
            apply_trades_to_portfolio(
                pid,
                scope if scope in ("paper", "real") else EXECUTE_SCOPE,
                folder,
                positions_csv,
                prices,
                trades,
                LEDGER_PATH
            )
            if not BACKTEST_MODE:
                write_last_applied(folder / "last_applied.json", asof)
            executed.append(pid)
        else:
            # DRY-RUN / GATHER: do NOT apply trades (no ledger/positions changes).
            cash, pos = read_positions_csv(str(positions_csv))
            snap = portfolio_snapshot(pid, cash, pos, prices)
            write_portfolio_value(folder, snap)
            write_execution_summary(folder, pid, positions_csv, cash, snap=snap)
            append_history_row(
                folder,
                pid,
                cash,
                pos,
                prices,
                date_override=SNAPSHOT_AS_OF if BACKTEST_MODE else None,
            )
            # Intentionally NOT writing last_applied here.

    print(f"[OK] update_performance_v3 complete. executed: {executed}, backtest_mode={BACKTEST_MODE}")

if __name__ == "__main__":
    main()
