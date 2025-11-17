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
OUT_TRADES     = os.environ.get("OUT_TRADES", "artifacts/recommended_trades_v3.json")
OUT_WATCHLIST  = os.environ.get("OUT_WATCHLIST", "data/watchlist_summary.json")
OUT_TRADES_HUMAN = os.environ.get("OUT_TRADES_HUMAN", "artifacts/recommended_trades_read.md")

# -------- Watchlist knobs (optional) --------
TS_WATCH  = float(os.environ.get("TS_WATCH", "0.50"))
NS_WATCH  = float(os.environ.get("NS_WATCH", "0.50"))
WATCH_TOP = int(os.environ.get("WATCH_TOP", "20"))

# -------- Feature toggles --------
ENABLE_MOMENTUM_PHASE   = os.environ.get("ENABLE_MOMENTUM_PHASE", "true").lower() == "true"
ENABLE_ROTATION_UPGRADE = os.environ.get("ENABLE_ROTATION_UPGRADE", "true").lower() == "true"

# How CDS maps to position size:
CDS_FULL_SIZE   = float(os.environ.get("CDS_FULL_SIZE", "0.55"))
CDS_MIN_SIZE_K  = float(os.environ.get("CDS_MIN_SIZE_K", "0.35"))

# -------- misc helpers --------

def write_text(path: str, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def _fmt_money(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "0"
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1000:
        return f"{sign}${v:,.0f}"
    return f"{sign}${v:,.2f}"

def _fmt_float(x, digits=2) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"

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

# ---------- NEW: indicators flattening helpers ----------

def _extract_indicators_views(node: dict) -> Tuple[dict, dict, dict]:
    """
    Returns (ind_daily, ind_intraday, ind_flat).

    - Works with both legacy flat indicators and the new
      {"daily": {...}, "intraday": {...}} shape from publish_feed.py.
    - ind_flat exposes:
        rsi / rsi14
        sma20
        ema_fast/ema_slow
        macd_hist/macd_hist_prev
        ema50_5m / ema200_5m
    """
    raw = node.get("indicators") or {}

    if isinstance(raw, dict) and ("daily" in raw or "intraday" in raw):
        ind_daily = raw.get("daily") or {}
        ind_intr  = raw.get("intraday") or {}
    else:
        # legacy flat shape
        ind_daily = raw
        ind_intr  = {}

    # pick best-available RSI (prefer intraday if present)
    rsi_val = (
        ind_intr.get("rsi")
        or ind_intr.get("rsi14")
        or ind_daily.get("rsi")
        or ind_daily.get("rsi14")
    )

    # pick MACD hist & prev (prefer intraday if present)
    macd_hist = ind_intr.get("macd_hist", ind_daily.get("macd_hist"))
    macd_hist_prev = ind_intr.get("macd_hist_prev", ind_daily.get("macd_hist_prev"))

    flat = {
        "rsi": rsi_val,
        "rsi14": rsi_val,
        "sma20": ind_daily.get("sma20"),
        "ema_fast": ind_daily.get("ema_fast", ind_daily.get("ema12")),
        "ema_slow": ind_daily.get("ema_slow", ind_daily.get("ema26")),
        "macd_hist": macd_hist,
        "macd_hist_prev": macd_hist_prev,
        # intraday EMAs for EMA gating
        "ema50_5m": ind_intr.get("ema50"),
        "ema200_5m": ind_intr.get("ema200"),
    }
    return ind_daily, ind_intr, flat

# ---------- EMA gating helpers (for BUYs; SELLs unchanged for now) ----------

def ema_size_for_buy(price: float, ema50: float | None, ema200: float | None) -> Tuple[float, str]:
    """
    Returns (k_ema, reason_suffix) to multiply BUY size_k.

    Rules:
      - If price is clearly below EMA200 → block new buys (k=0).
      - If price is below EMA50 but above EMA200 → reduce size (k ~ 0.6).
      - Otherwise → k=1.0.
    """
    if price is None or price <= 0:
        return 1.0, "ema:none"

    if ema50 is None and ema200 is None:
        return 1.0, "ema:none"

    reasons: List[str] = []
    k = 1.0

    if ema200 is not None and price < ema200 * 0.99:
        k = 0.0
        reasons.append("below_ema200_block_buy")
        return k, "|".join(reasons)

    if ema50 is not None and price < ema50 * 0.995:
        k *= 0.6
        reasons.append("below_ema50_reduce_buy")

    if not reasons:
        reasons.append("ema_ok")

    return k, "|".join(reasons)

def ema_size_for_sell(price: float, ema50: float | None, ema200: float | None) -> Tuple[float, str]:
    """
    Returns (k_ema, reason_suffix) to multiply SELL size_k.

    Intent:
      - If price is well ABOVE EMA50 in an uptrend → reduce trims (don’t fight a strong trend).
      - If price is below EMA200 → it's OK to sell; we just tag it as trend-broken.
      - Otherwise → neutral (k=1.0).
    """
    if price is None or price <= 0:
        return 1.0, "ema:none"

    if ema50 is None and ema200 is None:
        return 1.0, "ema:none"

    reasons: List[str] = []
    k = 1.0

    # Trend clearly broken → OK to sell normal size (no reduction, just a tag)
    if ema200 is not None and price < ema200 * 0.99:
        reasons.append("ema_below200_sell_ok")
        return k, "|".join(reasons)

    # Still strong uptrend → shrink the trim size
    if ema50 is not None and price > ema50 * 1.01:
        k *= 0.6
        reasons.append("ema_above50_reduce_sell")

    if not reasons:
        reasons.append("ema_ok")

    return k, "|".join(reasons)


# -------- watchlist --------

def build_watchlist(feed: Dict[str, Any], ts_thr: float, ns_thr: float, top_n: int):
    """
    Global watchlist:
    - Uses TS/NS/CDS
    - Shows RSI / SMA20 and guidance levels
    - Works with both legacy and new indicators shape.
    """
    out = []
    for sym, node in (feed.get("symbols") or {}).items():
        sig = node.get("signals") or {}
        ts  = float(sig.get("TS")  or 0.0)
        ns  = float(sig.get("NS")  or 0.0)
        cds = float(sig.get("CDS") or 0.0)

        if ts < ts_thr and ns < ns_thr:
            continue

        ind_daily, ind_intr, ind_flat = _extract_indicators_views(node)
        rsi = ind_flat.get("rsi")
        sma20 = ind_flat.get("sma20")
        price = _latest_px(node)
        guidance = node.get("guidance") or {}
        buy_on_dip = guidance.get("buy_on_dip_below")
        trim_above = guidance.get("trim_above")

        note_bits: List[str] = []

        if rsi is not None:
            r = float(rsi)
            if r < 30:
                note_bits.append("RSI oversold")
            elif r < 40:
                note_bits.append("RSI weak")
            elif r > 70:
                note_bits.append("RSI hot")
            elif r > 60:
                note_bits.append("RSI strong")

        if sma20 and price:
            if price >= sma20 * 1.05:
                note_bits.append("extended above 20d")
            elif price <= sma20 * 0.95:
                note_bits.append("below 20d trend")

        if buy_on_dip is not None:
            note_bits.append(f"dip< {_fmt_money(buy_on_dip)}")
        if trim_above is not None:
            note_bits.append(f"trim> {_fmt_money(trim_above)}")

        note = "; ".join(note_bits) if note_bits else ""

        out.append({
            "symbol": sym,
            "price": price,
            "TS": ts,
            "NS": ns,
            "CDS": cds,
            "rsi14": rsi,
            "sma20": sma20,
            "buy_on_dip_below": buy_on_dip,
            "trim_above": trim_above,
            "note": note,
        })

    out.sort(key=lambda r: (abs(r["CDS"]), r["TS"]), reverse=True)
    return out[:top_n]

# -------- portfolio config helpers --------

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
    p = Path(pos_path)
    if not p.exists():
        return 0.0, {}
    return read_positions_csv(pos_path)

# ---------- guidance-aware sizing helpers (unchanged) ----------

def guidance_size_for_sell(
    symbol: str,
    price: float,
    signals: Dict[str, Any],
    ind_full: Dict[str, Any],
    guidance: Dict[str, Any],
    phase: str | None,
) -> Tuple[float, str]:
    if not guidance:
        return 1.0, "guidance:none"

    buy_on_dip = guidance.get("buy_on_dip_below")
    trim_above = guidance.get("trim_above")
    sma20      = ind_full.get("sma20")
    rsi        = ind_full.get("rsi") or ind_full.get("rsi14")

    gk = 1.0
    reasons: List[str] = []
    ph = phase or "mid"

    score = 0
    if buy_on_dip is not None and price <= buy_on_dip:
        score += 1
    if sma20 and price <= sma20 * 0.95:
        score += 1
    if rsi is not None and rsi < 35:
        score += 1

    oversold = score >= 2

    rich_zone = bool(trim_above and price >= trim_above)
    mid_zone  = bool(
        buy_on_dip and trim_above and buy_on_dip < price < trim_above
    )

    if oversold:
        gk *= 0.4
        reasons.append("oversold_avoid_full_sell")
    elif mid_zone:
        gk *= 0.7
        reasons.append("mid_zone_partial_trim")
    elif rich_zone:
        reasons.append("at_or_above_trim_zone")
        if ph == "late" or (rsi is not None and rsi > 70):
            reasons.append("late_or_hot_confirm_sell")

    gk = max(0.1, min(1.0, gk))
    if not reasons:
        reasons.append("guidance_ok")

    return gk, "|".join(reasons)

def guidance_size_for_buy(
    symbol: str,
    price: float,
    signals: Dict[str, Any],
    ind_full: Dict[str, Any],
    guidance: Dict[str, Any],
    phase: str | None,
) -> Tuple[float, str]:
    if not guidance:
        return 1.0, "guidance:none"

    buy_on_dip = guidance.get("buy_on_dip_below")
    trim_above = guidance.get("trim_above")
    sma20      = ind_full.get("sma20")
    rsi        = ind_full.get("rsi") or ind_full.get("rsi14")

    gk = 1.0
    reasons: List[str] = []

    ph = phase or "mid"

    if ph == "late":
        gk *= 0.5
        reasons.append("late_phase")

    if buy_on_dip is not None:
        if price > buy_on_dip * 1.01:
            gk *= 0.7
            reasons.append("above_dip_level")

    if trim_above is not None:
        if price >= trim_above:
            gk *= 0.4
            reasons.append("near_or_above_trim_zone")

    if sma20 and price > sma20 * 1.07 and (rsi is not None and rsi > 70):
        gk *= 0.7
        reasons.append("extended_vs_sma20_hot_rsi")

    gk = max(0.0, min(1.0, gk))
    if not reasons:
        reasons.append("guidance_ok")

    return gk, "|".join(reasons)

# ---------- human-readable trade explainer (unchanged) ----------

def describe_trade(tr: Dict[str, Any]) -> str:
    side   = tr.get("action")
    sym    = tr.get("symbol")
    qty    = tr.get("qty")
    px     = tr.get("px")
    notional = tr.get("notional")
    if notional is None and px is not None and qty is not None:
        try:
            notional = float(px) * float(qty)
        except Exception:
            notional = None

    sig    = tr.get("signals") or {}
    ind    = tr.get("indicators") or {}
    guidance = tr.get("guidance") or {}
    phase  = tr.get("phase")
    size_k = tr.get("size_k")
    size_g = tr.get("size_k_guidance", 1.0)
    reason = tr.get("reason") or ""

    cds = sig.get("CDS")
    ts  = sig.get("TS")
    ns  = sig.get("NS")

    rsi = ind.get("rsi") or ind.get("rsi14")
    sma20 = ind.get("sma20")
    price = float(px or 0.0) if px is not None else 0.0

    header = f"- {side} {qty} {sym} @ {_fmt_money(px)}"
    if notional is not None:
        header += f" (~{_fmt_money(notional)})"

    metrics_bits: List[str] = []
    if cds is not None:
        metrics_bits.append(f"CDS={_fmt_float(cds, 3)}")
    if ts is not None:
        metrics_bits.append(f"TS={_fmt_float(ts, 3)}")
    if ns is not None:
        metrics_bits.append(f"NS={_fmt_float(ns, 3)}")
    if rsi is not None:
        metrics_bits.append(f"RSI={_fmt_float(rsi, 1)}")
    if sma20 is not None and price:
        metrics_bits.append(f"SMA20={_fmt_money(sma20)}")
    if phase:
        metrics_bits.append(f"phase={phase}")
    if size_k is not None:
        metrics_bits.append(f"size_k={_fmt_float(size_k, 2)}")
    if size_g not in (None, 1.0):
        metrics_bits.append(f"guidance_k={_fmt_float(size_g, 2)}")

    metrics_line = ""
    if metrics_bits:
        metrics_line = "  - " + " | ".join(metrics_bits)

    why_bits: List[str] = []

    if side == "BUY" and cds is not None:
        if cds >= 0.55:
            why_bits.append("Strong CDS supports an aggressive buy.")
        elif cds >= 0.35:
            why_bits.append("CDS is moderately positive; buy with some caution.")
        else:
            why_bits.append("CDS only slightly above buy threshold; treat as a small probe.")
    elif side == "SELL" and cds is not None:
        if cds <= -0.4:
            why_bits.append("Strongly negative CDS; favors selling/derisking.")
        elif cds <= -0.25:
            why_bits.append("CDS below sell threshold; partial derisking is reasonable.")

    if rsi is not None:
        r = float(rsi)
        if r < 30:
            why_bits.append("RSI is oversold, so price may be washed-out short term.")
        elif r < 40:
            why_bits.append("RSI is weak, indicating soft momentum.")
        elif r > 70:
            why_bits.append("RSI is overbought, meaning the move is hot/extended.")
        elif r > 60:
            why_bits.append("RSI is strong, suggesting positive momentum.")

    if sma20 and price:
        if price <= sma20 * 0.95:
            why_bits.append(
                f"Price {_fmt_money(price)} is well below the 20-day SMA "
                f"({_fmt_money(sma20)}), showing near-term pressure."
            )
        elif price >= sma20 * 1.05:
            why_bits.append(
                f"Price {_fmt_money(price)} is well above the 20-day SMA "
                f"({_fmt_money(sma20)}), showing an extended push."
            )

    if "oversold_avoid_full_sell" in reason:
        why_bits.append("Guidance marked this as oversold, so we only do a partial sell instead of a full exit.")
    if "mid_zone_partial_trim" in reason:
        why_bits.append("Price is between dip and trim levels; treating this as a partial trim.")
    if "at_or_above_trim_zone" in reason or "near_or_above_trim_zone" in reason:
        why_bits.append("Price is near/above the trim zone, so trimming/taking profits is justified.")
    if side == "BUY" and "above_dip_level" in reason:
        why_bits.append("Price is above the ideal dip level, so size is reduced to avoid chasing.")
    if "below_ema200_block_buy" in reason:
        why_bits.append("Buy was blocked or heavily reduced because price is below the 200-period EMA (5m trend broken).")
    if "below_ema50_reduce_buy" in reason:
        why_bits.append("Buy size was reduced because price is under the 50-period EMA on 5m (testing near-term support).")
    if "ema_below200_sell_ok" in reason:
        why_bits.append("Selling aligns with a broken 5m trend (price below EMA200).")
    if "ema_above50_reduce_sell" in reason:
        why_bits.append("Sell size reduced because price is above the 5m EMA50 (uptrend still strong).")


    if phase == "early":
        why_bits.append("Momentum phase: early in a potential move.")
    elif phase == "mid":
        why_bits.append("Momentum phase: mid, in the body of the move.")
    elif phase == "late":
        why_bits.append("Momentum phase: late; move may be getting tired.")

    if size_k is not None and float(size_k) < 1.0:
        why_bits.append(
            f"Final position size is scaled down to ~{_fmt_float(float(size_k)*100, 0)}% of the original plan."
        )

    if not why_bits:
        why_bits.append("Trade generated by CDS/TS/NS rules without additional strong flags.")

    why_line = "  - Why: " + " ".join(why_bits)

    return "\n".join([header, metrics_line, why_line])

# -----------------------------------------------------------------

def main():
    feed = read_json(INPUT_FINAL)
    defaults, portfolios = load_portfolios_config(PORTFOLIOS_YML)

    out: Dict[str, Any] = {
        "as_of_utc": feed.get("as_of_utc"),
        "portfolios": {}
    }

    human_lines: List[str] = []
    human_lines.append(f"# Trade plan as of {feed.get('as_of_utc')}\n")

    # Global watchlist
    wl = build_watchlist(feed, TS_WATCH, NS_WATCH, WATCH_TOP)
    write_json(OUT_WATCHLIST, {
        "as_of_utc": feed.get("as_of_utc"),
        "criteria": {"TS_WATCH": TS_WATCH, "NS_WATCH": NS_WATCH, "TOP": WATCH_TOP},
        "items": wl
    })

    if wl:
        human_lines.append("## Global Watchlist\n")
        human_lines.append("| Symbol | Price | CDS | TS | NS | RSI | SMA20 | Notes |")
        human_lines.append("|--------|-------|-----|----|----|-----|-------|-------|")
        for item in wl:
            human_lines.append(
                f"| {item['symbol']} | {_fmt_money(item['price'])} | "
                f"{_fmt_float(item['CDS'],3)} | {_fmt_float(item['TS'],3)} | {_fmt_float(item['NS'],3)} | "
                f"{_fmt_float(item.get('rsi14'),1)} | {_fmt_money(item.get('sma20'))} | {item.get('note','')} |"
            )
        human_lines.append("")

    symbols_map = (feed or {}).get("symbols", {})

    total_trades = 0
    for pid, node in portfolios.items():
        cfg = mk_portfolio_config(pid, defaults, node)

        enable_momentum_phase   = bool(node.get("enable_momentum_phase", defaults.get("enable_momentum_phase", ENABLE_MOMENTUM_PHASE)))
        enable_rotation_upgrade = bool(node.get("enable_rotation_upgrade", defaults.get("enable_rotation_upgrade", ENABLE_ROTATION_UPGRADE)))

        rot_params = (defaults.get("rotation", {}) or {}) | (node.get("rotation", {}) or {})

        if not cfg.path:
            base = Path("data/portfolios") / pid
        else:
            base = Path(cfg.path)
        pos_path = str(base / cfg.positions_csv)

        cash, pos = _safe_read_positions(pos_path)
        port_val  = total_value(pos, feed)

        qtys = {k: v["qty"] for k, v in pos.items()}
        actions = propose_actions(qtys, feed, cfg.strategy)
        sized, cash_left = size_with_cash(actions, port_val, cash, cfg.sizing)

        # --- Enrich each trade with price, signals, indicators (incl intraday), guidance, and news
        for t in sized:
            sym = t.get("symbol")
            n = symbols_map.get(sym, {})

            if "px" not in t or t["px"] is None:
                t["px"] = n.get("price")

            sig = n.get("signals") or {}
            t["signals"] = {
                "TS":  sig.get("TS"),
                "NS":  sig.get("NS"),
                "CDS": sig.get("CDS"),
                "wT":  sig.get("wT"),
                "wN":  sig.get("wN"),
            }

            ind_daily, ind_intr, ind_flat = _extract_indicators_views(n)
            t["indicators"] = ind_flat

            guidance = n.get("guidance") or {}
            if guidance:
                t["guidance"] = guidance

            if t.get("action") == "BUY":
                size_k = 1.0
                phase = None

                if enable_momentum_phase:
                    ind_for_phase = {
                        "rsi": ind_flat.get("rsi"),
                        "macd_hist": ind_flat.get("macd_hist"),
                        "macd_hist_prev": ind_flat.get("macd_hist_prev"),
                    }
                    phase, mp_k = momentum_phase_and_size(ind_for_phase)
                    t["phase"] = phase
                    if mp_k is not None:
                        size_k *= float(mp_k)

                sig_t = t.get("signals") or {}
                cds = float(sig_t.get("CDS") or 0.0)
                buy_thr = float(cfg.strategy.buy_threshold)

                if cds <= buy_thr:
                    cds_k = CDS_MIN_SIZE_K
                elif cds >= CDS_FULL_SIZE:
                    cds_k = 1.0
                else:
                    span = max(1e-6, CDS_FULL_SIZE - buy_thr)
                    frac = (cds - buy_thr) / span
                    cds_k = CDS_MIN_SIZE_K + frac * (1.0 - CDS_MIN_SIZE_K)

                size_k *= cds_k

                px_f = float(t.get("px") or 0.0)

                # Guidance sizing
                if px_f > 0:
                    gk, g_reason = guidance_size_for_buy(
                        symbol=sym,
                        price=px_f,
                        signals=t["signals"],
                        ind_full=ind_flat,
                        guidance=guidance,
                        phase=phase,
                    )
                    size_k *= gk
                    t["size_k_guidance"] = gk
                    base_reason = t.get("reason") or "buy"
                    t["reason"] = base_reason + f"|guidance:{g_reason}"

                # EMA gating (5m) on BUYs
                ema50_5m = ind_flat.get("ema50_5m")
                ema200_5m = ind_flat.get("ema200_5m")
                if px_f > 0 and (ema50_5m is not None or ema200_5m is not None):
                    ema_k, ema_reason = ema_size_for_buy(px_f, ema50_5m, ema200_5m)
                    size_k *= ema_k
                    base_reason = t.get("reason") or "buy"
                    t["reason"] = base_reason + f"|ema:{ema_reason}"

                size_k = min(size_k, 1.0)
                t["size_k"] = size_k

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

            elif t.get("action") == "SELL" and t.get("reason") != "rotation_upgrade":
                size_k = 1.0
                phase = None
            
                if enable_momentum_phase:
                    ind_for_phase = {
                        "rsi": ind_flat.get("rsi"),
                        "macd_hist": ind_flat.get("macd_hist"),
                        "macd_hist_prev": ind_flat.get("macd_hist_prev"),
                    }
                    phase, _ = momentum_phase_and_size(ind_for_phase)
                    t["phase"] = phase
            
                px_f = float(t.get("px") or 0.0)
                if px_f > 0:
                    # 1) Guidance sizing (dip/trim/SMA20/RSI)
                    gk, g_reason = guidance_size_for_sell(
                        symbol=sym,
                        price=px_f,
                        signals=t["signals"],
                        ind_full=ind_flat,
                        guidance=guidance,
                        phase=phase,
                    )
                    size_k *= gk
                    t["size_k_guidance"] = gk
                    base_reason = t.get("reason") or "sell"
                    t["reason"] = base_reason + f"|guidance:{g_reason}"
            
                    # 2) EMA-based sizing (5m)
                    ema50_5m = ind_flat.get("ema50_5m")
                    ema200_5m = ind_flat.get("ema200_5m")
                    if ema50_5m is not None or ema200_5m is not None:
                        ema_k, ema_reason = ema_size_for_sell(px_f, ema50_5m, ema200_5m)
                        size_k *= ema_k
                        base_reason = t.get("reason") or "sell"
                        t["reason"] = base_reason + f"|ema:{ema_reason}"
            
                size_k = min(size_k, 1.0)
                t["size_k"] = size_k
            
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

            t["news_preview"] = _news_preview(n.get("news"), max_items=3)

        # ---- Rotation upgrade (unchanged) ----
        if enable_rotation_upgrade and sized:
            rot_cfg = RotationConfig(
                trigger_cds=float(rot_params.get("trigger_cds", 0.50)),
                min_gap=float(rot_params.get("min_gap", 0.25)),
                max_turnover_frac=float(rot_params.get("max_turnover_frac", 0.10)),
                sell_step_frac=float(rot_params.get("sell_step_frac", 0.25)),
                protect_profitable=bool(rot_params.get("protect_profitable", True)),
            )

            strong_buys = [
                t for t in sized
                if t.get("action") == "BUY"
                and (t.get("signals", {}).get("CDS", 0.0) >= rot_cfg.trigger_cds)
            ]

            if strong_buys:
                equity_now = port_val
                cash_available = float(cash_left or 0.0)

                per_line_dollars = equity_now * cfg.sizing.per_line_frac
                if cash_available < 0.5 * per_line_dollars:
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

                    want = 0.0
                    for b in strong_buys:
                        px = float(b.get("px") or 0.0)
                        if px <= 0:
                            continue
                        already = float(b.get("qty") or 0.0) * px
                        size_k = float(b.get("size_k", 1.0))
                        target = per_line_dollars * size_k
                        add = max(0.0, target - already)
                        want += add

                    need = max(0.0, want - cash_available)

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
                                prev_q = float(b.get("qty") or 0.0)
                                b["qty_prev"] = prev_q
                                b["qty"] = prev_q + add_q
                                b["notional"] = round(float(b.get("notional", prev_q * px)) + add_q * px, 2)
                                b["reason"] = (b.get("reason") or "buy") + "|topped_by_rotation"

        cash_left_sim = float(cash_left or 0.0)
        for tr in sized:
            side = tr.get("action")
            px   = float(tr.get("px") or 0.0)
            q    = float(tr.get("qty") or 0.0)
            if side == "SELL":
                cash_left_sim += q * px
            elif side == "BUY":
                cash_left_sim -= q * px

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
            "cash_left": round(float(cash_left_sim), 2),
            "debug": {
                "buy_sizing": buy_debug,
                "sell_sizing": sell_debug,
            },
        }

        # ---- Human-readable section ----
        human_lines.append(f"## {cfg.title} ({pid})\n")
        human_lines.append(
            f"Portfolio value ≈ {_fmt_money(port_val)} | "
            f"Cash: {_fmt_money(cash)} → {_fmt_money(cash_left_sim)}\n"
        )

        buys = [tr for tr in sized if tr.get("action") == "BUY"]
        sells = [tr for tr in sized if tr.get("action") == "SELL"]

        def _notional(tr):
            try:
                if tr.get("notional") is not None:
                    return float(tr["notional"])
            except (TypeError, ValueError):
                pass
            try:
                return float(tr.get("qty", 0.0)) * float(tr.get("px", 0.0))
            except (TypeError, ValueError):
                return 0.0

        buys.sort(key=_notional, reverse=True)
        sells.sort(key=_notional, reverse=True)

        if sells:
            human_lines.append("### SELLS\n")
            for tr in sells:
                human_lines.append(describe_trade(tr))
                human_lines.append("")
        if buys:
            human_lines.append("### BUYS\n")
            for tr in buys:
                human_lines.append(describe_trade(tr))
                human_lines.append("")

        total_trades += len(sized)

    Path(OUT_TRADES).parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT_TRADES, out)

    human_text = "\n".join(human_lines) if human_lines else "# No trades generated.\n"
    write_text(OUT_TRADES_HUMAN, human_text)

    print(f"[OK] wrote {OUT_TRADES} with {len(out['portfolios'])} portfolios, {total_trades} total trades.")
    print(f"[OK] wrote {OUT_TRADES_HUMAN} (human-readable).")

if __name__ == "__main__":
    main()
