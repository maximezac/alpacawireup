#!/usr/bin/env python3
from __future__ import annotations
import os, json, math, yaml
from pathlib import Path
from dataclasses import dataclass
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from typing import Dict, Any, List, Tuple, Literal

from scripts.strategy.momentum_phase import momentum_phase_and_size
from scripts.strategy.rotation_upgrade import pick_laggards_to_fund, RotationConfig

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

# -------- Feature toggles --------
ENABLE_MOMENTUM_PHASE   = os.environ.get("ENABLE_MOMENTUM_PHASE", "true").lower() == "true"
ENABLE_ROTATION_UPGRADE = os.environ.get("ENABLE_ROTATION_UPGRADE", "true").lower() == "true"

# How CDS maps to position size:
# - At buy_threshold → we shrink to CDS_MIN_SIZE_K of normal size.
# - At CDS_FULL_SIZE or above → full size (no downsize).
CDS_FULL_SIZE   = float(os.environ.get("CDS_FULL_SIZE", "0.55"))
CDS_MIN_SIZE_K  = float(os.environ.get("CDS_MIN_SIZE_K", "0.35"))

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

# ---------- NEW: guidance-aware sizing helper for SELLs ----------

def guidance_size_for_sell(
    symbol: str,
    price: float,
    signals: Dict[str, Any],
    ind_full: Dict[str, Any],
    guidance: Dict[str, Any],
    phase: str | None,
) -> Tuple[float, str]:
    """
    Returns (size_k_guidance, reason_suffix) for SELL trades.
    size_k_guidance in (0,1]; multiplies existing qty for SELLs.

    Intuition:
      - Avoid full sells when price is *truly* oversold (2 of 3: below dip, well under SMA20, low RSI).
      - Mid zone => partial trim.
      - Near/above trim zone => ok to sell full size.
    """
    if not guidance:
        return 1.0, "guidance:none"

    buy_on_dip = guidance.get("buy_on_dip_below")
    trim_above = guidance.get("trim_above")
    sma20      = ind_full.get("sma20")
    rsi        = ind_full.get("rsi") or ind_full.get("rsi14")

    gk = 1.0
    reasons: List[str] = []
    ph = phase or "mid"

    # --- 1) Oversold score: need 2 out of 3 signals ---
    score = 0
    if buy_on_dip is not None and price <= buy_on_dip:
        score += 1
    if sma20 and price <= sma20 * 0.95:      # tighter than 0.97
        score += 1
    if rsi is not None and rsi < 35:
        score += 1

    oversold = score >= 2

    # --- 2) Zones relative to guidance levels ---
    rich_zone = bool(trim_above and price >= trim_above)
    mid_zone  = bool(
        buy_on_dip and trim_above and buy_on_dip < price < trim_above
    )

    # Oversold: shrink sells a lot (avoid puking at the bottom)
    if oversold:
        gk *= 0.4
        reasons.append("oversold_avoid_full_sell")

    # Mid zone: partial trims if not oversold
    elif mid_zone:
        gk *= 0.7
        reasons.append("mid_zone_partial_trim")

    # Rich / trim zone: we’re happy with full planned size
    elif rich_zone:
        reasons.append("at_or_above_trim_zone")
        # optional: if late phase / hot RSI, just note it
        if ph == "late" or (rsi is not None and rsi > 70):
            reasons.append("late_or_hot_confirm_sell")

    # Floor/ceiling
    gk = max(0.1, min(1.0, gk))
    if not reasons:
        reasons.append("guidance_ok")

    return gk, "|".join(reasons)


# -----------------------------------------------------------------


# ---------- NEW: guidance-aware sizing helper for BUYs ----------  # <<< NEW
def guidance_size_for_buy(
    symbol: str,
    price: float,
    signals: Dict[str, Any],
    ind_full: Dict[str, Any],
    guidance: Dict[str, Any],
    phase: str | None,
) -> Tuple[float, str]:
    """
    Returns (size_k_guidance, reason_suffix).
    size_k_guidance in [0,1]; multiplies existing size_k for BUY trades.

    Intuition:
      - Prefer early-phase dips near/under buy_on_dip_below.
      - Be cautious when price is well above dip level.
      - Be small when near/above trim_above or in late phase.
    """
    if not guidance:
        return 1.0, "guidance:none"

    buy_on_dip = guidance.get("buy_on_dip_below")
    trim_above = guidance.get("trim_above")
    sma20      = ind_full.get("sma20")
    rsi        = ind_full.get("rsi")

    gk = 1.0
    reasons: List[str] = []

    ph = phase or "mid"

    # Late-phase: we don't want to be aggressive even if CDS says BUY
    if ph == "late":
        gk *= 0.5
        reasons.append("late_phase")

    # If price is above the preferred dip level, shrink size
    if buy_on_dip is not None:
        if price > buy_on_dip * 1.01:
            gk *= 0.7
            reasons.append("above_dip_level")

    # If price is near/above trim zone, be very conservative
    if trim_above is not None:
        if price >= trim_above:
            gk *= 0.4
            reasons.append("near_or_above_trim_zone")

    # Additional safety: if price is far above SMA20 and RSI is hot, shrink more
    if sma20 and price > sma20 * 1.07 and (rsi is not None and rsi > 70):
        gk *= 0.7
        reasons.append("extended_vs_sma20_hot_rsi")

    # Clamp and build reason string
    gk = max(0.0, min(1.0, gk))
    if not reasons:
        reasons.append("guidance_ok")

    return gk, "|".join(reasons)
# -----------------------------------------------------------------  # <<< NEW

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

        # Per-portfolio feature toggles (YAML overrides env)
        enable_momentum_phase   = bool(node.get("enable_momentum_phase", defaults.get("enable_momentum_phase", ENABLE_MOMENTUM_PHASE)))
        enable_rotation_upgrade = bool(node.get("enable_rotation_upgrade", defaults.get("enable_rotation_upgrade", ENABLE_ROTATION_UPGRADE)))

        # Optional rotation params from YAML
        rot_params = (defaults.get("rotation", {}) | node.get("rotation", {})) if isinstance(defaults.get("rotation", {}), dict) else (node.get("rotation", {}) or {})

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

        # --- Enrich each trade with price, signals, indicators, guidance, and a few news items
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
            ind_full = n.get("indicators") or {}  # full set from publish_feed.py  # <<< NEW
            t["indicators"] = {
                "ema_fast":        ind_full.get("ema_fast", ind_full.get("ema12")),
                "ema_slow":        ind_full.get("ema_slow", ind_full.get("ema26")),
                "macd":            ind_full.get("macd"),
                "macd_signal":     ind_full.get("macd_signal"),
                "macd_hist":       ind_full.get("macd_hist"),
                "macd_hist_prev":  ind_full.get("macd_hist_prev"),   # <-- needed for slope
                "rsi14":           ind_full.get("rsi", ind_full.get("rsi14")),
                "sma20":           ind_full.get("sma20"),
            }

            # guidance from feed (if publish_feed was updated to emit it)  # <<< NEW
            guidance = n.get("guidance") or {}
            if guidance:
                t["guidance"] = guidance

            # Momentum + CDS-based sizing: only ever *downsize* buys
            if t.get("action") == "BUY":
                size_k = 1.0
                phase = None

                # --- Momentum phase part (optional) ---
                if enable_momentum_phase:
                    ind_for_phase = {
                        "rsi": t["indicators"]["rsi14"],
                        "macd_hist": t["indicators"]["macd_hist"],
                        "macd_hist_prev": t["indicators"]["macd_hist_prev"],
                    }
                    phase, mp_k = momentum_phase_and_size(ind_for_phase)
                    t["phase"] = phase
                    if mp_k is not None:
                        size_k *= float(mp_k)

                # --- CDS strength part ---
                sig_t = t.get("signals") or {}
                cds = float(sig_t.get("CDS") or 0.0)
                buy_thr = float(cfg.strategy.buy_threshold)

                if cds <= buy_thr:
                    # Barely a buy → small probe size
                    cds_k = CDS_MIN_SIZE_K
                elif cds >= CDS_FULL_SIZE:
                    # Strong conviction → full size
                    cds_k = 1.0
                else:
                    # Linearly interpolate between CDS_MIN_SIZE_K and 1.0
                    span = max(1e-6, CDS_FULL_SIZE - buy_thr)
                    frac = (cds - buy_thr) / span  # in (0,1)
                    cds_k = CDS_MIN_SIZE_K + frac * (1.0 - CDS_MIN_SIZE_K)

                size_k *= cds_k

                # --- NEW: guidance-aware modifier on top of TS/CDS/phase sizing ---  # <<< NEW
                px_f = float(t.get("px") or 0.0)
                if px_f > 0:
                    gk, g_reason = guidance_size_for_buy(
                        symbol=sym,
                        price=px_f,
                        signals=t["signals"],
                        ind_full=ind_full,
                        guidance=guidance,
                        phase=phase,
                    )
                    size_k *= gk
                    t["size_k_guidance"] = gk
                    # append reason hint (non-breaking)
                    base_reason = t.get("reason") or "buy"
                    t["reason"] = base_reason + f"|guidance:{g_reason}"

                # Never allow size_k > 1.0 (we only shrink, never boost)
                size_k = min(size_k, 1.0)
                t["size_k"] = size_k

                # --- Apply downsizing to qty ---
                q = float(t.get("qty") or 0.0)
                if q > 0 and size_k < 1.0:
                    if cfg.sizing.fractional_buys:
                        new_q = round(q * size_k, 6)
                    else:
                        new_q = int(q * size_k)

                    # Only update if it meaningfully changes size
                    if new_q < q:
                        t["qty_prev"] = q
                        t["qty"] = new_q
                        # Recompute notional to keep outputs consistent
                        if t.get("px") is not None:
                            t["notional"] = round(new_q * float(t["px"]), 2)
            
            elif t.get("action") == "SELL" and t.get("reason") != "rotation_upgrade":
                # Guidance-aware downsize for SELLs (we don't boost, only shrink / partial)
                size_k = 1.0
                phase = None

                # Optional: compute phase for sells too, so we recognize late-phase tops vs early bounces
                if enable_momentum_phase:
                    ind_for_phase = {
                        "rsi": t["indicators"]["rsi14"],
                        "macd_hist": t["indicators"]["macd_hist"],
                        "macd_hist_prev": t["indicators"]["macd_hist_prev"],
                    }
                    phase, _ = momentum_phase_and_size(ind_for_phase)
                    t["phase"] = phase

                px_f = float(t.get("px") or 0.0)
                if px_f > 0:
                    gk, g_reason = guidance_size_for_sell(
                        symbol=sym,
                        price=px_f,
                        signals=t["signals"],
                        ind_full=ind_full,
                        guidance=guidance,
                        phase=phase,
                    )
                    size_k *= gk
                    t["size_k_guidance"] = gk
                    base_reason = t.get("reason") or "sell"
                    t["reason"] = base_reason + f"|guidance:{g_reason}"

                # Never allow size_k > 1.0 (we only shrink)
                size_k = min(size_k, 1.0)
                t["size_k"] = size_k

                # Apply downsizing to SELL qty (partial trims instead of full exit)
                q = float(t.get("qty") or 0.0)
                if q > 0 and size_k < 1.0:
                    if cfg.sizing.fractional_buys:
                        new_q = round(q * size_k, 6)
                    else:
                        new_q = int(q * size_k)

                    if new_q < q:
                        t["qty_prev"] = q
                        t["qty"] = new_q
                        if t.get("px") is not None:
                            t["notional"] = round(new_q * float(t["px"]), 2)

            # short news preview
            t["news_preview"] = _news_preview(n.get("news"), max_items=3)

        # ---- Rotation upgrade: free cash from weak holds to fund strong buys ----
        if enable_rotation_upgrade and sized:
            rot_cfg = RotationConfig(
                trigger_cds=float(rot_params.get("trigger_cds", 0.50)),
                min_gap=float(rot_params.get("min_gap", 0.25)),
                max_turnover_frac=float(rot_params.get("max_turnover_frac", 0.10)),
                sell_step_frac=float(rot_params.get("sell_step_frac", 0.25)),
                protect_profitable=bool(rot_params.get("protect_profitable", True)),
            )

            # Strong BUYs among the planned trades
            strong_buys = [
                t for t in sized
                if t.get("action") == "BUY"
                and (t.get("signals", {}).get("CDS", 0.0) >= rot_cfg.trigger_cds)
            ]

            # Only rotate if we actually need cash and have strong buys to fund
            if strong_buys:
                equity_now = port_val  # already computed
                # Approximate remaining cash (post momentum downsize we usually have >= original cash_left)
                cash_available = float(cash_left or 0.0)

                # If we still have adequate cash, skip rotation
                # You can tune this threshold; here we only rotate if cash < one per-line target.
                per_line_dollars = equity_now * cfg.sizing.per_line_frac
                if cash_available < 0.5 * per_line_dollars:
                    # Build current holdings with CDS snapshots
                    current_holdings = []
                    for sym, p in (pos or {}).items():
                        node_sym = symbols_map.get(sym, {})
                        sig_sym  = node_sym.get("signals", {})
                        current_holdings.append({
                            "symbol": sym,
                            "qty": float(p.get("qty") or 0.0),
                            "px": float(node_sym.get("price") or 0.0),
                            "unrealized_pnl": float(p.get("unrealized_pnl", 0.0)),
                            "signals": {"CDS": float(sig_sym.get("CDS", 0.0))}
                        })

                    # Needed cash to bring all strong buys to their per-line target (minus what they already got)
                    want = 0.0
                    for b in strong_buys:
                        px = float(b.get("px") or 0.0)
                        if px <= 0:
                            continue
                        already = float(b.get("qty") or 0.0) * px
                        # If momentum-phase applied, use size_k-adjusted per-line target; else 1.0x
                        size_k = float(b.get("size_k", 1.0))
                        target = per_line_dollars * size_k
                        add = max(0.0, target - already)
                        want += add

                    need = max(0.0, want - cash_available)

                    # Ask rotation module which laggards to trim to raise 'need' (bounded by max_turnover_frac)
                    if need > 0.0 and current_holdings:
                        sells_plan = pick_laggards_to_fund(
                            candidates=[{
                                "symbol": b["symbol"],
                                "signals": {"CDS": float((b.get("signals") or {}).get("CDS", 0.0))},
                                "px": float(b.get("px", 0.0)),
                                "notional_target": per_line_dollars * float(b.get("size_k", 1.0))
                            } for b in strong_buys],
                            holdings=current_holdings,
                            needed_cash=need,
                            equity=equity_now,
                            cfg=rot_cfg
                        )

                        # Emit SELL trades (append to 'sized') and track freed cash
                        freed = 0.0
                        for sym, amt in sells_plan:
                            if amt <= 0:
                                continue
                            px = float(symbols_map.get(sym, {}).get("price") or 0.0)
                            if px <= 0:
                                continue
                            if cfg.sizing.fractional_buys:
                                q = round(amt / px, 6)
                            else:
                                q = int(amt // px)
                            if q <= 0:
                                continue
                            sized.append({
                                "symbol": sym,
                                "action": "SELL",
                                "px": px,
                                "qty": q,
                                "notional": round(q * px, 2),
                                "reason": "rotation_upgrade",
                                "signals": symbols_map.get(sym, {}).get("signals", {}),
                                "indicators": symbols_map.get(sym, {}).get("indicators", {}),
                                "news_preview": _news_preview(symbols_map.get(sym, {}).get("news"), max_items=3),
                            })
                            freed += q * px

                        # Top-up strong buys with freed cash (simple even split; respects fractional_buys)
                        if freed > 0 and strong_buys:
                            per_buy_budget = freed / len(strong_buys)
                            for b in strong_buys:
                                px = float(b.get("px") or 0.0)
                                if px <= 0:
                                    continue
                                add_notional = per_buy_budget
                                if add_notional <= 0:
                                    continue
                                if cfg.sizing.fractional_buys:
                                    add_q = round(add_notional / px, 6)
                                else:
                                    add_q = int(add_notional // px)
                                if add_q <= 0:
                                    continue
                                # Increase qty & notional in-place
                                prev_q = float(b.get("qty") or 0.0)
                                b["qty_prev"] = prev_q
                                b["qty"] = prev_q + add_q
                                b["notional"] = round(float(b.get("notional", prev_q * px)) + add_q * px, 2)
                                # Annotate reason
                                b["reason"] = (b.get("reason") or "buy") + "|topped_by_rotation"

        # --- Compute a simulated cash_left that reflects rotation/top-ups (optional but nice)
        cash_left_sim = float(cash_left or 0.0)
        for tr in sized:
            side = tr.get("action")
            px   = float(tr.get("px") or 0.0)
            q    = float(tr.get("qty") or 0.0)
            if side == "SELL":
                cash_left_sim += q * px
            elif side == "BUY":
                cash_left_sim -= q * px

        # --- Debug info: how buys were sized ---
        buy_debug = []
        sell_debug = []
        for tr in sized:
            sig_dbg = tr.get("signals") or {}
            if tr.get("action") == "BUY":
                buy_debug.append({
                    "symbol": tr.get("symbol"),
                    "cds": float(sig_dbg.get("CDS") or 0.0),
                    "size_k": float(tr.get("size_k", 1.0)),
                    "size_k_guidance": float(tr.get("size_k_guidance", 1.0)),
                    "phase": tr.get("phase"),
                    "qty": float(tr.get("qty") or 0.0),
                    "px": float(tr.get("px") or 0.0),
                    "notional": float(tr.get("notional") or 0.0),
                    "reason": tr.get("reason"),
                })
            elif tr.get("action") == "SELL":
                sell_debug.append({
                    "symbol": tr.get("symbol"),
                    "cds": float(sig_dbg.get("CDS") or 0.0),
                    "size_k": float(tr.get("size_k", 1.0)),
                    "size_k_guidance": float(tr.get("size_k_guidance", 1.0)),
                    "phase": tr.get("phase"),
                    "qty": float(tr.get("qty") or 0.0),
                    "px": float(tr.get("px") or 0.0),
                    "notional": float(tr.get("notional") or 0.0),
                    "reason": tr.get("reason"),
                })

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
            "cash_left": round(float(cash_left_sim), 2),  # <- simulated, reflects rotation and downsizing
            "debug": {
                "buy_sizing": buy_debug,
                "sell_sizing": sell_debug,
            },
        }

        # --- Also write a per-portfolio trade plan into its folder
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

        total_trades += len(sized)

    Path(OUT_TRADES).parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT_TRADES, out)
    print(f"[OK] wrote {OUT_TRADES} with {len(out['portfolios'])} portfolios, {total_trades} total trades.")

if __name__ == "__main__":
    main()
