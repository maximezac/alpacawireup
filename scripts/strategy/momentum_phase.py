from dataclasses import dataclass

@dataclass
class PhaseConfig:
    rsi_early_max: float = 70.0
    rsi_late_min: float = 75.0
    macd_hist_slope_min: float = 0.0
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
    if rsi < cfg.rsi_early_max and slope > cfg.macd_hist_slope_min:
        return "early", cfg.early_size
    if rsi >= cfg.rsi_late_min and slope <= cfg.macd_hist_slope_min:
        return "late", cfg.late_size
    return "mid", cfg.mid_size
