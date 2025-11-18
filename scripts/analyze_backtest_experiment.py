#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, csv, math
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import yaml  # pyyaml is already in your workflows

# ----------------------------------------------------------------------
# Inputs (via CLI + env)
# ----------------------------------------------------------------------
# CLI:
#   python scripts/analyze_backtest_experiment.py \
#       data/prices_backfill.json \
#       data/portfolios/port_backfill_5each/history.csv \
#       data/portfolios/port_backfill_5each/trades_ledger.csv
#
# Env:
#   EXPERIMENT_ID       (e.g. exp_v1)
#   BENCHMARK_SYMBOL    (e.g. SPY)
#   CONFIG_PORTFOLIOS   (e.g. config/portfolios.yml)
#   CONFIG_STRATEGIES   (e.g. config/strategies.yml)
#   PORTFOLIO_ID        (e.g. port_backfill_5each)
# ----------------------------------------------------------------------

if len(sys.argv) < 4:
    print(
        "Usage: analyze_backtest_experiment.py PRICES_BACKFILL HISTORY_CSV LEDGER_CSV",
        file=sys.stderr,
    )
    sys.exit(1)

prices_path = Path(sys.argv[1])
history_path = Path(sys.argv[2])
ledger_path = Path(sys.argv[3])

EXPERIMENT_ID = os.getenv("EXPERIMENT_ID", "exp_v1")
BENCHMARK = os.getenv("BENCHMARK_SYMBOL", "SPY").upper()
PORTFOLIO_ID = os.getenv("PORTFOLIO_ID", "port_backfill_5each")

CONFIG_PORTFOLIOS = os.getenv("CONFIG_PORTFOLIOS", "config/portfolios.yml")
CONFIG_STRATEGIES = os.getenv("CONFIG_STRATEGIES", "config/strategies.yml")

# Where to drop summary artifacts
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 1) Load prices + build per-symbol date->close map
# ----------------------------------------------------------------------
prices = json.loads(prices_path.read_text(encoding="utf-8"))
syms = prices.get("symbols", {})

if BENCHMARK not in syms:
    print(f"[WARN] Benchmark {BENCHMARK} not found in prices_backfill.json", file=sys.stderr)

def build_date_close_map(sym: str):
    node = syms.get(sym, {})
    bars = node.get("bars", []) or []
    m = {}
    for b in bars:
        ts = b.get("t")
        if not ts:
            continue
        d = ts.split("T", 1)[0]
        m[d] = float(b.get("c") or 0.0)
    return m

bench_by_date = build_date_close_map(BENCHMARK) if BENCHMARK in syms else {}

sector_by_symbol = {s: (node.get("sector") or "") for s, node in syms.items()}

# ----------------------------------------------------------------------
# 2) Load portfolio history and wire in benchmark series
# ----------------------------------------------------------------------
rows = []
with history_path.open("r", newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        # If you ever put multiple portfolios in one history, filter:
        if row.get("portfolio") and row["portfolio"] != PORTFOLIO_ID:
            continue
        rows.append(row)

if not rows:
    print("[WARN] No history rows found for portfolio", PORTFOLIO_ID, file=sys.stderr)
    sys.exit(0)

rows.sort(key=lambda r: r["date"])

dates = [r["date"] for r in rows]
equity_series = [float(r["equity"]) for r in rows]
first_equity = equity_series[0]
last_equity = equity_series[-1]
total_return = (last_equity / first_equity - 1.0) if first_equity else 0.0

# Portfolio max drawdown
peak = -1.0
max_drawdown = 0.0  # as negative fraction
for eq in equity_series:
    if eq > peak:
        peak = eq
    if peak > 0:
        dd = (eq / peak) - 1.0
        if dd < max_drawdown:
            max_drawdown = dd

# Build benchmark cumulative returns aligned to the same dates
bench_first_close = None
bench_first_date = None
bench_returns = []
for d in dates:
    px = bench_by_date.get(d)
    if px is None:
        bench_returns.append(None)
        continue
    if bench_first_close is None:
        bench_first_close = px
        bench_first_date = d
    bench_returns.append((px / bench_first_close) - 1.0 if bench_first_close else 0.0)

final_bench_return = None
for v in reversed(bench_returns):
    if v is not None:
        final_bench_return = v
        break

# Write history_with_<BENCHMARK>.csv alongside the original history
history_with_bench_path = history_path.with_name(
    history_path.stem + f"_with_{BENCHMARK}.csv"
)

fieldnames = list(rows[0].keys())
extra_cols = ["bench_symbol", "bench_close", "bench_cum_return"]
with history_with_bench_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames + extra_cols)
    w.writeheader()
    for r, bench_cum in zip(rows, bench_returns):
        d = r["date"]
        bench_close = bench_by_date.get(d)
        out = dict(r)
        out["bench_symbol"] = BENCHMARK
        out["bench_close"] = bench_close if bench_close is not None else ""
        out["bench_cum_return"] = round(bench_cum, 6) if bench_cum is not None else ""
        w.writerow(out)

print(f"[INFO] Wrote {history_with_bench_path}")

# ----------------------------------------------------------------------
# 3) Per-symbol PnL from ledger (realized + unrealized)
# ----------------------------------------------------------------------
if not ledger_path.exists():
    print(f"[WARN] Ledger not found at {ledger_path}, skipping per-symbol stats", file=sys.stderr)
    per_symbol = {}
else:
    ledger_rows = []
    with ledger_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("portfolio_id") != PORTFOLIO_ID:
                continue
            ledger_rows.append(row)

    # Sort by datetime_utc for replay correctness
    def _parse_dt(s: str):
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return datetime.min

    ledger_rows.sort(key=lambda r: _parse_dt(r["datetime_utc"]))

    per_symbol = {}
    for row in ledger_rows:
        sym = row["symbol"].upper()
        action = row["action"].upper()
        qty = float(row["qty"])
        px_exec = float(row["price_exec"])

        state = per_symbol.setdefault(sym, {
            "qty": 0.0,
            "avg_cost": 0.0,
            "realized_pnl": 0.0,
            "buy_volume": 0.0,
            "sell_volume": 0.0,
            "trades": 0,
            "first_trade": None,
            "last_trade": None,
        })

        dt = _parse_dt(row["datetime_utc"])
        if state["first_trade"] is None or dt < state["first_trade"]:
            state["first_trade"] = dt
        if state["last_trade"] is None or dt > state["last_trade"]:
            state["last_trade"] = dt

        state["trades"] += 1

        if action == "BUY":
            cost = qty * px_exec
            state["buy_volume"] += cost
            total_cost = state["qty"] * state["avg_cost"] + cost
            new_qty = state["qty"] + qty
            if new_qty > 0:
                state["avg_cost"] = total_cost / new_qty
            state["qty"] = new_qty

        elif action == "SELL":
            proceeds = qty * px_exec
            state["sell_volume"] += proceeds
            # realized PnL on sold portion
            state["realized_pnl"] += (px_exec - state["avg_cost"]) * qty
            new_qty = state["qty"] - qty
            state["qty"] = new_qty
            if new_qty <= 0:
                state["avg_cost"] = 0.0

    # Attach mark-to-market at last backtest date
    last_date = dates[-1]
    # Build date->close per symbol just once if needed
    # (small universe so on-demand is fine)
    for sym, s in per_symbol.items():
        node = syms.get(sym, {})
        bars = node.get("bars", []) or []
        date_to_close = {}
        for b in bars:
            ts = b.get("t")
            if not ts:
                continue
            d = ts.split("T", 1)[0]
            date_to_close[d] = float(b.get("c") or 0.0)

        # Find price on last_date or nearest prior
        px = None
        if last_date in date_to_close:
            px = date_to_close[last_date]
        else:
            # search backwards
            for d in sorted(date_to_close.keys(), reverse=True):
                if d <= last_date:
                    px = date_to_close[d]
                    break

        s["mark"] = px if px is not None else 0.0
        qty = s["qty"]
        avg_cost = s["avg_cost"]
        s["unrealized_pnl"] = (s["mark"] - avg_cost) * qty if qty > 0 and s["mark"] else 0.0
        s["total_pnl"] = s["realized_pnl"] + s["unrealized_pnl"]
        s["market_value"] = qty * (s["mark"] or 0.0)
        if s["buy_volume"] > 0:
            s["exposure_return"] = s["total_pnl"] / s["buy_volume"]
        else:
            s["exposure_return"] = None

# ----------------------------------------------------------------------
# 4) Per-sector aggregation
# ----------------------------------------------------------------------
per_sector = defaultdict(lambda: {
    "sector": "",
    "market_value": 0.0,
    "realized_pnl": 0.0,
    "unrealized_pnl": 0.0,
    "total_pnl": 0.0,
    "buy_volume": 0.0,
    "sell_volume": 0.0,
    "symbols": set(),
})

for sym, s in per_symbol.items():
    sector = sector_by_symbol.get(sym, "") or "UNSPECIFIED"
    agg = per_sector[sector]
    agg["sector"] = sector
    agg["market_value"] += s["market_value"]
    agg["realized_pnl"] += s["realized_pnl"]
    agg["unrealized_pnl"] += s["unrealized_pnl"]
    agg["total_pnl"] += s["total_pnl"]
    agg["buy_volume"] += s["buy_volume"]
    agg["sell_volume"] += s["sell_volume"]
    agg["symbols"].add(sym)

for sector, agg in per_sector.items():
    if agg["buy_volume"] > 0:
        agg["exposure_return"] = agg["total_pnl"] / agg["buy_volume"]
    else:
        agg["exposure_return"] = None
    agg["symbols"] = sorted(agg["symbols"])

# ----------------------------------------------------------------------
# 5) Load experiment config from YAMLs
# ----------------------------------------------------------------------
def safe_load_yaml(path_str: str):
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[WARN] Failed to load YAML {path_str}: {e}", file=sys.stderr)
        return None

portfolios_cfg = safe_load_yaml(CONFIG_PORTFOLIOS)
strategies_cfg = safe_load_yaml(CONFIG_STRATEGIES)

defaults_cfg = (portfolios_cfg or {}).get("defaults", {}) if isinstance(portfolios_cfg, dict) else {}

# ----------------------------------------------------------------------
# 6) Build experiment summary JSON
# ----------------------------------------------------------------------
timeline = []
for r, bench_cum in zip(rows, bench_returns):
    d = r["date"]
    equity = float(r["equity"])
    cum_ret = float(r.get("cumulative_return", 0.0))
    entry = {
        "date": d,
        "equity": equity,
        "cumulative_return": cum_ret,
    }
    if bench_cum is not None:
        entry["benchmark_cum_return"] = bench_cum
    timeline.append(entry)

summary = {
    "experiment_id": EXPERIMENT_ID,
    "portfolio_id": PORTFOLIO_ID,
    "benchmark_symbol": BENCHMARK,
    "date_range": {
        "start": dates[0],
        "end": dates[-1],
        "steps": len(dates),
    },
    "topline": {
        "initial_equity": first_equity,
        "final_equity": last_equity,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "benchmark_total_return": final_bench_return,
        "excess_return_vs_benchmark":
            (total_return - final_bench_return) if (final_bench_return is not None) else None,
    },
    "experiment_config": {
        "portfolios_defaults": defaults_cfg,
        "strategies": strategies_cfg,
        "env": {
            # a few interesting knobs; you can add more here over time
            "BACKTEST_MODE": os.getenv("BACKTEST_MODE"),
            "WINDOW_DAYS": os.getenv("WINDOW_DAYS"),
            "NEWS_LOOKBACK_DAYS": os.getenv("NEWS_LOOKBACK_DAYS"),
            "NEWS_MAX_ARTICLES_TOTAL": os.getenv("NEWS_MAX_ARTICLES_TOTAL"),
            "ALPACA_DATA_FEED": os.getenv("ALPACA_DATA_FEED"),
        },
    },
    "timeline": timeline,
}

summary_path = ARTIFACT_DIR / f"experiment_summary_{EXPERIMENT_ID}.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"[INFO] Wrote {summary_path}")

# ----------------------------------------------------------------------
# 7) Write per-symbol + per-sector CSVs
# ----------------------------------------------------------------------
if per_symbol:
    per_symbol_path = ARTIFACT_DIR / f"experiment_summary_{EXPERIMENT_ID}_per_symbol.csv"
    fieldnames = [
        "symbol", "sector", "qty_end", "avg_cost_end", "mark",
        "market_value", "realized_pnl", "unrealized_pnl", "total_pnl",
        "buy_volume", "sell_volume", "exposure_return",
        "trades", "first_trade", "last_trade",
    ]
    with per_symbol_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for sym, s in sorted(per_symbol.items()):
            w.writerow({
                "symbol": sym,
                "sector": sector_by_symbol.get(sym, "") or "UNSPECIFIED",
                "qty_end": round(s["qty"], 6),
                "avg_cost_end": round(s["avg_cost"], 6),
                "mark": round(s["mark"], 6),
                "market_value": round(s["market_value"], 2),
                "realized_pnl": round(s["realized_pnl"], 2),
                "unrealized_pnl": round(s["unrealized_pnl"], 2),
                "total_pnl": round(s["total_pnl"], 2),
                "buy_volume": round(s["buy_volume"], 2),
                "sell_volume": round(s["sell_volume"], 2),
                "exposure_return": round(s["exposure_return"], 6) if s["exposure_return"] is not None else "",
                "trades": s["trades"],
                "first_trade": s["first_trade"].isoformat() if s["first_trade"] else "",
                "last_trade": s["last_trade"].isoformat() if s["last_trade"] else "",
            })
    print(f"[INFO] Wrote {per_symbol_path}")

if per_sector:
    per_sector_path = ARTIFACT_DIR / f"experiment_summary_{EXPERIMENT_ID}_per_sector.csv"
    fieldnames = [
        "sector", "market_value", "realized_pnl", "unrealized_pnl",
        "total_pnl", "buy_volume", "sell_volume", "exposure_return", "symbols",
    ]
    with per_sector_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for sector, agg in sorted(per_sector.items()):
            w.writerow({
                "sector": sector,
                "market_value": round(agg["market_value"], 2),
                "realized_pnl": round(agg["realized_pnl"], 2),
                "unrealized_pnl": round(agg["unrealized_pnl"], 2),
                "total_pnl": round(agg["total_pnl"], 2),
                "buy_volume": round(agg["buy_volume"], 2),
                "sell_volume": round(agg["sell_volume"], 2),
                "exposure_return": round(agg["exposure_return"], 6) if agg["exposure_return"] is not None else "",
                "symbols": ",".join(agg["symbols"]),
            })
    print(f"[INFO] Wrote {per_sector_path}")
