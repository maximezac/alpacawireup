from dataclasses import dataclass

@dataclass
class PhaseConfig:
    rsi_early_max: float = 40.0      # early if below 40
    rsi_late_min: float = 65.0       # late if above 65
    slope_up_min: float = 0.001      # histogram ticking up
    slope_down_max: float = -0.001   # histogram ticking down
    early_size: float = 1.00
    mid_size: float = 0.75
    late_size: float = 0.50

def momentum_phase_and_size(ind: dict):
    rsi = ind.get("rsi")
    h   = ind.get("macd_hist")
    hp  = ind.get("macd_hist_prev")
    if rsi is None or h is None or hp is None:
        return "mid", PhaseConfig().mid_size

    slope = h - hp
    cfg = PhaseConfig()

    # EARLY: oversold or low RSI and slope turning upward
    if rsi < cfg.rsi_early_max and slope > cfg.slope_up_min:
        return "early", cfg.early_size

    # LATE: overbought and slope turning downward
    if rsi > cfg.rsi_late_min and slope < cfg.slope_down_max:
        return "late", cfg.late_size

    # Otherwise mid
    return "mid", cfg.mid_size
