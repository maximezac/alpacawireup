from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class RotationConfig:
    trigger_cds: float = 0.50
    min_gap: float = 0.25
    max_turnover_frac: float = 0.10
    sell_step_frac: float = 0.25
    protect_profitable: bool = True

def pick_laggards_to_fund(candidates: List[Dict], holdings: List[Dict],
                          needed_cash: float, equity: float,
                          cfg: RotationConfig = RotationConfig()) -> List[Tuple[str, float]]:
    if needed_cash <= 0: return []
    budget = min(needed_cash, equity * cfg.max_turnover_frac)
    strong_buys = [c for c in candidates if c["signals"]["CDS"] >= cfg.trigger_cds]
    if not strong_buys: return []
    strongest_cds = max(c["signals"]["CDS"] for c in strong_buys)

    def sort_key(h):
        cds = h.get("signals", {}).get("CDS", 0.0)
        pnl = h.get("unrealized_pnl", 0.0)
        return (cds, pnl if cfg.protect_profitable else 0.0)

    sells = []
    for h in sorted(holdings, key=sort_key):  # weakest first
        if budget <= 0: break
        held_cds = h.get("signals", {}).get("CDS", 0.0)
        if (strongest_cds - held_cds) < cfg.min_gap: continue
        sellable = h["qty"] * h["px"] * cfg.sell_step_frac
        amt = min(sellable, budget)
        if amt > 0:
            sells.append((h["symbol"], amt))
            budget -= amt
    return sells
