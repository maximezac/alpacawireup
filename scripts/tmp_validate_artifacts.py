#!/usr/bin/env python3
import sys, json, math
from pathlib import Path
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv)>1 else Path('backtest-results-exp_latenight/artifacts/exp_latenight_baseline')
print('Validating artifacts under', ROOT)

for p in sorted((ROOT / 'portfolios').iterdir() if (ROOT / 'portfolios').exists() else []):
    if not p.is_dir():
        continue
    pid = p.name
    print('\n---', pid, '---')
    sumf = p / 'experiment_summary.json'
    histf = p / 'history.csv'
    hist_en = p / 'history_enriched.csv'
    pvf = p / 'portfolio_value.json'
    tradesf = p / 'trades.csv'
    ledgerf = p / 'trades_ledger.csv'

    print('summary exists', sumf.exists())
    if sumf.exists():
        s = json.loads(sumf.read_text())
    else:
        s = None
    print('history exists', histf.exists(), 'enriched', hist_en.exists())
    if histf.exists():
        df = pd.read_csv(histf, parse_dates=['date']).sort_values('date')
        print('history rows', len(df), 'start', df['date'].iloc[0].date(), 'end', df['date'].iloc[-1].date())
        rets = df['equity'].pct_change().dropna()
        nz = rets[rets.abs()>1e-12]
        print('n_returns', len(rets), 'n_nonzero', len(nz))
        ann_vol = rets.std(ddof=0)*(252**0.5) if len(rets)>0 else None
        # recompute ann_ret as geometric
        ann_ret = (1+rets).prod()**(252/len(rets)) - 1 if len(rets)>0 else None
        # recompute sharpe
        sh = (ann_ret - 0.04)/ann_vol if ann_vol and ann_vol>0 else None
        print('recomputed ann_ret, ann_vol, sharpe:', ann_ret, ann_vol, sh)
        if s:
            print('summary total_return', s.get('total_return'), 'cagr', s.get('cagr'), 'sharpe', s.get('sharpe'))
            # compare final equity
            hist_final = float(df['equity'].iloc[-1])
            summ_final = s.get('final_equity')
            if summ_final is not None:
                diff = abs(hist_final - float(summ_final))
                print('final equity diff', diff)
    else:
        print('no history for', pid)

    print('portfolio_value.json exists', pvf.exists())
    if pvf.exists():
        pv = json.loads(pvf.read_text())
        print('pv equity', pv.get('equity'))
    print('trades.csv exists', tradesf.exists(), 'ledger exists', ledgerf.exists())
    if tradesf.exists():
        td = pd.read_csv(tradesf)
        print('n trades rows', len(td))
    if ledgerf.exists():
        ld = pd.read_csv(ledgerf)
        print('ledger rows', len(ld))

print('\n--- global checks ---')
# manifest and summary
mf = ROOT.parent / f'experiment_manifest_{ROOT.name}.json'
if mf.exists():
    print('manifest', mf)
else:
    print('manifest not found at', mf)
sf = ROOT.parent / f'summary_for_gpt_{ROOT.name}.txt'
if sf.exists():
    print('summary_for_gpt', sf)
else:
    print('summary_for_gpt not found at', sf)

print('\nValidation complete')
