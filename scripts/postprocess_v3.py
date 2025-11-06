#!/usr/bin/env python3
from __future__ import annotations
import os, json, math, yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple

from engine import (
    read_json, write_json, read_positions_csv, total_value,
    StrategyConfig, SizingConfig, PortfolioConfig,
    propose_actions, size_with_cash
)

# -------- IO paths --------
INPUT_FINAL    = os.environ.get("INPUT_FINAL", "data/prices_final.json")
PORTFOLIOS_YML = os.environ.get("PORTFOLIOS_YML", "config/portfolios.yml")
# Default OUT_TRADES aligns with v3 executor (artifacts/*)
OUT_TRADES     = os.environ.get("OUT_TRADES", "artifacts/recommended_trades_v3.json")
OUT_WATCHLIST  = os.environ.get("OUT_WATCHLIST", "data/watchlist_summary.json")

# -------- Watchlist knobs (optional) --------
TS_WATCH  = float(os.environ.get("TS_WATCH", "0.50"))
NS_WATCH  = float(os.environ.get("NS_WATCH", "0.50"))
WATCH_TOP = int(os.environ.get("WATCH_TOP", "20"))

# -------- helpers --------
def _latest_px(node: dict) -> float:
    """Single source of truth for price (live-only shape; legacy-safe)."""
    if not node:
        return 0.0
    v = node.get("price")
    if v is None:
        v = (node.get("now") or {}).get("price")
    try:
        return float(v or 0.0)
    except Exception:
        return 0.0

def _news_preview(news_list, max_items=3):
    prev = []
    for n in (news_list or [])[:max_items]:
        prev.append({
            "ts": n.get("ts"),
            "source": n.get("source"),
            "tone": n.get("tone"),
            "headline": n.get("headline"),
        })
    return prev

def build_watchlist(feed: Dict[str, Any], ts_thr: float, ns_thr: float, top_n: int):
    out = []
    for sym, node in (feed.get("symbols") or {}).items():
        sig = node.get("signals") or {}
        ts  = float(sig.get("TS")  or 0.0)
        ns  = float(sig.get("NS")  or 0.0)
        cds = float(sig.get("CDS") or 0.0)
        if ts >= ts_thr or ns >= ns_thr:
            out.append({
                "symbol": sym,
                "price": _latest_px(node),
                "TS": ts, "NS": ns, "CDS": cds
            })
    out.sort(key=lambda r: (abs(r["CDS"]), r["TS"]), reverse=True)
    return out[:top_n]

def load_portfolios_config(path: str) -> Tuple[dict, dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"portfolios config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("defaults") or {}, cfg.get("portfolios") or {}

def mk_portfolio_config(pid: str, defaults: dict, node: dict) -> PortfolioConfig:
    # Strategy
    strat = StrategyConfig(
        buy_threshold=float(node.get("buy_threshold", defaults.get("buy_threshold", 0.35))),
        sell_threshold=float(node.get("sell_threshold", defaults.get("sell_threshold", -0.25))),
        decision_source=(node.get("decision_source", defaults.get("decision_source", "CDS"))),
        ns_weight_boost=float(node.get("ns_weight_boost", defaults.get("ns_weight_boost", 1.0))),
        allow_new_symbols=bool(node.get("allow_new_symbols", defaults.get("allow_new_symbols", True))),
    )
    # Sizing
    max_deploy = node.get("max_deploy_frac", defaults.get("max_deploy_frac", 0.15))
    max_deploy = None if max_deploy is None else float(max_deploy)
    sizing = SizingConfig(
        per_line_frac=float(node.get("per_line_frac", defaults.get("per_line_frac", 0.02))),
        max_deploy_frac=max_deploy,
        trim_fraction=float(node.get("trim_fraction", defaults.get("trim_fraction", 0.25))),
        fractional_buys=bool(node.get("fractional_buys", defaults.get("fractional_buys", True))),
    )
    # Portfolio
    base_path = node.get("path", defaults.get("path"))
    return PortfolioConfig(
        id=pid,
        title=node.get("title", pid),
        path=base_path,
        positions_csv=node.get("positions_csv", "positions.csv"),
        strategy=strat,
        sizing=sizing,
        trade_scope=node.get("trade_scope", defaults.get("trade_scope", "none")),
        tags=node.get("tags", defaults.get("tags", [])),
    )

def _safe_read_positions(pos_path: str) -> Tuple[float, Dict[str, dict]]:
    """Don't explode if positions.csv isn't there yet."""
    p = Path(pos_path)
    if not p.exists():
        # Allow portfolio to exist with zero cash/positions; executor will skip trading until it exists.
        return 0.0, {}
    return read_positions_csv(pos_path)

def main():
    feed = read_json(INPUT_FINAL)
    defaults, portfolios = load_portfolios_config(PORTFOLIOS_YML)

    out: Dict[str, Any] = {
        "as_of_utc": feed.get("as_of_utc"),
        "portfolios": {}
    }

    # Optional global watchlist
    wl = build_watchlist(feed, TS_WATCH, NS_WATCH, WATCH_TOP)
    write_json(OUT_WATCHLIST, {
        "as_of_utc": feed.get("as_of_utc"),
        "criteria": {"TS_WATCH": TS_WATCH, "NS_WATCH": NS_WATCH, "TOP": WATCH_TOP},
        "items": wl
    })

    # Portfolios
    total_trades = 0
    for pid, node in portfolios.items():
        cfg = mk_portfolio_config(pid, defaults, node)

        # Resolve positions path (config path + positions_csv)
        if not cfg.path:
            # If omitted, default to data/portfolios/<pid>
            base = Path("data/portfolios") / pid
        else:
            base = Path(cfg.path)
        pos_path = str(base / cfg.positions_csv)

        cash, pos = _safe_read_positions(pos_path)
        port_val  = total_value(pos, feed)  # should use live price under the hood

        # actions → sizing
        qtys = {k: v["qty"] for k, v in pos.items()}  # propose_actions expects {sym: qty}
        actions = propose_actions(qtys, feed, cfg.strategy)
        sized, cash_left = size_with_cash(actions, port_val, cash, cfg.sizing)

        # --- Enrich each trade with price, signals, indicators, and a few news items
        symbols_map = (feed or {}).get("symbols", {})
        for t in sized:
            sym = t.get("symbol")
            n = symbols_map.get(sym, {})  # node for the symbol in feed

            # price
            if "px" not in t or t["px"] is None:
                t["px"] = n.get("price")

            # signals (TS/NS/CDS and weights)
            sig = n.get("signals") or {}
            t["signals"] = {
                "TS":  sig.get("TS"),
                "NS":  sig.get("NS"),
                "CDS": sig.get("CDS"),
                "wT":  sig.get("wT"),
                "wN":  sig.get("wN"),
            }

            # indicators (accept either publisher or fetcher keys)
            ind = n.get("indicators") or {}
            t["indicators"] = {
                "ema_fast":    ind.get("ema_fast", ind.get("ema12")),
                "ema_slow":    ind.get("ema_slow", ind.get("ema26")),
                "macd":        ind.get("macd"),
                "macd_signal": ind.get("macd_signal"),
                "macd_hist":   ind.get("macd_hist"),
                "rsi14":       ind.get("rsi", ind.get("rsi14")),
                "sma20":       ind.get("sma20"),
            }

            # short news preview
            t["news_preview"] = _news_preview(n.get("news"), max_items=3)

        out["portfolios"][pid] = {
            "meta": {
                "title": cfg.title,
                "trade_scope": cfg.trade_scope,
                "decision_source": cfg.strategy.decision_source,
                "ns_weight_boost": cfg.strategy.ns_weight_boost,
                "per_line_frac": cfg.sizing.per_line_frac,
                "max_deploy_frac": cfg.sizing.max_deploy_frac,
                "trim_fraction": cfg.sizing.trim_fraction,
                "fractional_buys": cfg.sizing.fractional_buys,
                "tags": cfg.tags,
                "positions_path": pos_path,
            },
            "portfolio_value_est": round(port_val, 2),
            "cash_start": round(cash, 2),
            "trades": sized,
            "cash_left": round(float(cash_left or 0.0), 2),
        }

        # --- NEW: also write a per-portfolio trade plan into its folder
        from pathlib import Path
        plan = {
            "as_of_utc": out["as_of_utc"],
            "portfolio_id": pid,
            "meta": out["portfolios"][pid]["meta"],
            "trades": sized,
        }
        base = Path(out["portfolios"][pid]["meta"]["positions_path"]).parent
        base.mkdir(parents=True, exist_ok=True)
        # rolling pointer
        write_json(str(base / "trades_plan.json"), plan)
        # versioned by as_of_utc (safe for history / diffing)
        safe_asof = (out["as_of_utc"] or "").replace(":", "").replace("-", "").replace("T","_").replace("+","Z")
        if safe_asof:
            write_json(str(base / f"trades_plan_{safe_asof}.json"), plan)

        total_trades += len(sized)

    Path(OUT_TRADES).parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT_TRADES, out)
    print(f"[OK] wrote {OUT_TRADES} with {len(out['portfolios'])} portfolios, {total_trades} total trades.")

if __name__ == "__main__":
    main()
