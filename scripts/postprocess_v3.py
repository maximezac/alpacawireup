#!/usr/bin/env python3
from __future__ import annotations
import os, json, math, yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
from engine import (read_json, write_json, read_positions_csv, total_value,
                    StrategyConfig, SizingConfig, PortfolioConfig,
                    propose_actions, size_with_cash)

INPUT_FINAL = os.environ.get("INPUT_FINAL", "data/prices_final.json")
PORTFOLIOS_YML = os.environ.get("PORTFOLIOS_YML", "config/portfolios.yml")
OUT_TRADES = os.environ.get("OUT_TRADES", "data/recommended_trades.json")
OUT_WATCHLIST = os.environ.get("OUT_WATCHLIST", "data/watchlist_summary.json")

# Optional: global watchlist knobs (still supported)
TS_WATCH = float(os.environ.get("TS_WATCH", "0.50"))
NS_WATCH = float(os.environ.get("NS_WATCH", "0.50"))
WATCH_TOP = int(os.environ.get("WATCH_TOP", "20"))

def build_watchlist(feed: Dict[str, Any], ts_thr: float, ns_thr: float, top_n: int):
    out = []
    for sym, node in (feed.get("symbols") or {}).items():
        sig = node.get("signals") or {}
        ts = float(sig.get("TS") or 0.0)
        ns = float(sig.get("NS") or 0.0)
        cds = float(sig.get("CDS") or 0.0)
        if ts >= ts_thr or ns >= ns_thr:
            # live-only shape: price is on the symbol root
            price = node.get("price")
            try:
                price = float(price) if price is not None else 0.0
            except:
                price = 0.0
            out.append({"symbol": sym, "price": price, "TS": ts, "NS": ns, "CDS": cds})
    out.sort(key=lambda r: (abs(r["CDS"]), r["TS"]), reverse=True)
    return out[:top_n]

def load_portfolios_config(path: str) -> Tuple[dict, dict]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
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
    return PortfolioConfig(
        id=pid,
        title=node.get("title", pid),
        path=node.get("path"),
        positions_csv=node.get("positions_csv", "positions.csv"),
        strategy=strat,
        sizing=sizing,
        trade_scope=node.get("trade_scope", defaults.get("trade_scope", "none")),
        tags=node.get("tags", defaults.get("tags", [])),
    )

def main():
    feed = read_json(INPUT_FINAL)
    defaults, portfolios = load_portfolios_config(PORTFOLIOS_YML)

    out: Dict[str, Any] = {"as_of_utc": feed.get("as_of_utc"), "portfolios": {}}

    # Global watchlist (optional)
    wl = build_watchlist(feed, TS_WATCH, NS_WATCH, WATCH_TOP)
    write_json(OUT_WATCHLIST, {
        "as_of_utc": feed.get("as_of_utc"),
        "criteria": {"TS_WATCH": TS_WATCH, "NS_WATCH": NS_WATCH, "TOP": WATCH_TOP},
        "items": wl
    })

    for pid, node in portfolios.items():
        cfg = mk_portfolio_config(pid, defaults, node)

        pos_path = str(Path(cfg.path) / cfg.positions_csv)
        cash, pos = read_positions_csv(pos_path)
        port_val = total_value(pos, feed)

        actions = propose_actions({k: v["qty"] for k, v in pos.items()}, feed, cfg.strategy)
        sized, cash_left = size_with_cash(actions, port_val, cash, cfg.sizing)

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
            "cash_left": cash_left
        }

    write_json(OUT_TRADES, out)
    print(f"[OK] wrote {OUT_TRADES} with {len(out['portfolios'])} portfolios.")

if __name__ == "__main__":
    main()
