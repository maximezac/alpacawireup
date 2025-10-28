# minify_feed.py
# Drop-in helper to shrink your trading feed while keeping everything needed for CDS/decisions.

from __future__ import annotations
from typing import Dict, Any, Optional

# ---- Tunables ---------------------------------------------------------------

# Keep only these indicators (everything else gets dropped).
KEEP_INDICATORS = {
    "ema12", "ema26", "macd", "macd_signal", "macd_hist",
    "rsi14", "zclose50", "rel_strength20", "atr14",
    # optional but cheap:
    "sma20"
}

# Round all floats to this many decimals to save bytes.
ROUND = 4

# If true, replace large news arrays with a count + last timestamp.
STRIP_NEWS = True

# If true, drop any per-symbol "df" / "history" arrays entirely.
DROP_HISTORY_KEYS = {"df", "history", "bars", "ohlc", "prices", "intraday"}

# Optional: cap how many symbols to emit (None = all)
SYMBOL_LIMIT: Optional[int] = None

# ---- Core -------------------------------------------------------------------

def _r(x: Any) -> Any:
    """Round floats recursively; leave everything else as-is."""
    if isinstance(x, float):
        return round(x, ROUND)
    if isinstance(x, dict):
        return {k: _r(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_r(v) for v in x]
    return x

def _compact_indicators(ind: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(ind, dict):
        return {}
    out = {}
    for k in KEEP_INDICATORS:
        if k in ind:
            out[k] = ind[k]
    # Include ns_decay if present (needed for continuity of NS)
    if "ns_decay" in ind:
        out["ns_decay"] = ind["ns_decay"]
    return _r(out)

def _compact_signals(sig: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(sig, dict):
        return {}
    # Keep only the top-level decision outputs (diagnostics dropped)
    keep = {}
    for k in ("TS", "NS", "CDS", "decision", "rationale"):
        if k in sig:
            keep[k] = sig[k]
    return _r(keep)

def _compact_symbol(sym: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(sym, dict):
        return {}

    # Base core fields
    core = {}
    for k in ("price", "volume", "ts", "risk_class"):
        if k in sym:
            core[k] = sym[k]

    # Indicators
    core["indicators"] = _compact_indicators(sym.get("indicators", {}))

    # Signals (final outputs only)
    if "signals" in sym:
        core["signals"] = _compact_signals(sym["signals"])

    # Strip heavy internals
    for k in list(sym.keys()):
        if k in DROP_HISTORY_KEYS:
            continue  # implicitly dropped

    # Optional: squash news arrays into tiny metadata
    if STRIP_NEWS:
        news = sym.get("news")
        if isinstance(news, list) and news:
            last_ts = None
            try:
                last_ts = max((n.get("ts") or n.get("date") or n.get("published_at") or ""), default="")
            except Exception:
                last_ts = ""
            core["news_meta"] = {"count": len(news), "last_ts": last_ts}

    return _r(core)

def compact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a minimized, execution-ready feed:
    {
      "as_of_utc": "...",
      "feed": "...",
      "timeframe": "...",
      "indicators_window": 200,
      "run": { "version": "...", "market_regime": 0.42 },
      "symbols": { ...minimized per symbol... }
    }
    """
    if not isinstance(payload, dict):
        return {}

    out: Dict[str, Any] = {}

    # Root metadata to keep
    for k in ("as_of_utc", "feed", "timeframe", "indicators_window"):
        if k in payload:
            out[k] = payload[k]

    # Run info (version + market_regime recommended)
    run_in = payload.get("run", {})
    run_out = {}
    for k in ("version", "market_regime"):
        if k in run_in:
            run_out[k] = run_in[k]
    if run_out:
        out["run"] = _r(run_out)

    # Symbols
    symbols = payload.get("symbols", {})
    if not isinstance(symbols, dict):
        out["symbols"] = {}
        return _r(out)

    compacted_syms: Dict[str, Any] = {}
    count = 0
    for ticker, data in symbols.items():
        if SYMBOL_LIMIT is not None and count >= SYMBOL_LIMIT:
            break
        compacted_syms[ticker] = _compact_symbol(data or {})
        count += 1

    out["symbols"] = compacted_syms
    return _r(out)

# ---- CLI convenience --------------------------------------------------------

if __name__ == "__main__":
    """
    Usage:
      python minify_feed.py < input.json > prices.json

    Or in your publisher:
      from minify_feed import compact_payload
      minimized = compact_payload(full_payload_dict)
    """
    import sys, json
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except Exception:
        print("{}", end="")
        sys.exit(0)

    minimized = compact_payload(data)
    sys.stdout.write(json.dumps(minimized, separators=(",", ":"), ensure_ascii=False))
