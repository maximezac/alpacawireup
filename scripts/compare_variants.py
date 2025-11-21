#!/usr/bin/env python3
"""
Compare two experiment runs for a given portfolio.
Usage:
  python scripts/compare_variants.py EXP1 EXP2 portfolio_id
Creates artifacts/compare_<EXP1>_vs_<EXP2>_<portfolio>.md and a plot.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 4:
    print("Usage: compare_variants.py EXP1 EXP2 portfolio_id")
    sys.exit(1)
exp1, exp2, pid = sys.argv[1], sys.argv[2], sys.argv[3]
ART = Path('artifacts')
outdir = ART / f'compare_{exp1}_vs_{exp2}_{pid}'
outdir.mkdir(parents=True, exist_ok=True)

f1 = ART / exp1 / pid / 'history.csv'
f2 = ART / exp2 / pid / 'history.csv'
if not f1.exists() or not f2.exists():
    print('history CSV missing for one of the runs')
    sys.exit(1)

df1 = pd.read_csv(f1, parse_dates=['date']).sort_values('date')
df2 = pd.read_csv(f2, parse_dates=['date']).sort_values('date')

plt.figure(figsize=(10,4))
plt.plot(df1['date'], df1['equity'], label=exp1)
plt.plot(df2['date'], df2['equity'], label=exp2)
plt.legend()
plt.title(f'Equity: {pid} {exp1} vs {exp2}')
plt.grid(True)
plt.tight_layout()
plot_png = outdir / 'equity_compare.png'
plt.savefig(plot_png)

md = []
md.append(f"# Compare {exp1} vs {exp2} for {pid}\n")
md.append(f"![equity]({plot_png.name})\n")
# small metrics
md.append('## Final equity\n')
md.append(f"- {exp1}: {df1['equity'].iloc[-1]}\n")
md.append(f"- {exp2}: {df2['equity'].iloc[-1]}\n")

(outdir / 'compare.md').write_text('\n'.join(md))
print('Wrote', outdir)
