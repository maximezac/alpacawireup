import pandas as pd
import json
from pathlib import Path

def check_history():
    p = Path('backtest-results-exp_latenight/artifacts/exp_latenight_baseline/port_backfill_5each_aggressive/history.csv')
    if not p.exists():
        print('HISTORY MISSING:', p)
        return
    df = pd.read_csv(p, parse_dates=['date']).sort_values('date')
    print('HISTORY ROWS:', len(df))
    print('START:', df['date'].iloc[0])
    print('END:', df['date'].iloc[-1])
    span = (df['date'].iloc[-1] - df['date'].iloc[0]).days
    print('DAYS SPAN:', span)
    rets = df['equity'].pct_change().dropna()
    nz = rets[rets.abs()>1e-12]
    print('N_RETURNS:', len(rets), 'N_NONZERO:', len(nz))
    print('RETURNS MEAN,STD,ANN STD:', rets.mean(), rets.std(), rets.std()*(252**0.5))
    print('NONZERO MEAN,STD,ANN STD:', nz.mean(), nz.std(), nz.std()*(252**0.5))
    print('PERIOD RETURN (final/initial -1):', df['equity'].iloc[-1]/df['equity'].iloc[0]-1)

def recompute_sharpe():
    p = Path('backtest-results-exp_latenight/artifacts/exp_latenight_baseline/port_backfill_5each_aggressive/history.csv')
    df = pd.read_csv(p, parse_dates=['date']).sort_values('date')
    rets = df['equity'].pct_change().dropna()
    nz = rets[rets.abs()>1e-12]
    use = nz if len(nz) >= 60 else rets
    if len(use) == 0:
        print('No returns to compute Sharpe')
        return
    ann_ret = (1+use).prod()**(252/len(use)) - 1
    ann_vol = use.std(ddof=0) * (252**0.5)
    rf = 0.04
    sh = (ann_ret - rf)/ann_vol if ann_vol>0 else None
    print('RECOMPUTED ann_ret, ann_vol, sharpe:', ann_ret, ann_vol, sh)

def compare_with_summary():
    sfile = Path('backtest-results-exp_latenight/artifacts/exp_latenight_baseline/port_backfill_5each_aggressive/experiment_summary.json')
    if not sfile.exists():
        print('SUMMARY MISSING:', sfile)
        return
    s = json.loads(sfile.read_text())
    print('SUMMARY:', json.dumps(s, indent=2))

if __name__ == '__main__':
    print('--- CHECK HISTORY ---')
    check_history()
    print('\n--- RECOMPUTE SHARPE ---')
    recompute_sharpe()
    print('\n--- SUMMARY JSON ---')
    compare_with_summary()
