#!/usr/bin/env python3
"""
scripts/compare_all_variants.py

Generate a consolidated CSV and markdown report comparing all variant experiment runs
for a given base experiment id. It expects artifacts/<exp>_<variant>/<portfolio>/experiment_summary.json
for each variant run.

Usage:
  python scripts/compare_all_variants.py --exp exp_v1

Outputs:
  artifacts/compare_<exp>_per_portfolio.csv
  artifacts/compare_<exp>.md
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict, Any

ART = Path('artifacts')


def gather(exp_base: str) -> List[Dict[str, Any]]:
    rows = []
    # find directories matching exp_base + '_' + variant
    for d in ART.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if not name.startswith(exp_base + '_'):
            continue
        variant = name[len(exp_base)+1:]
        # iterate portfolios under this variant
        for p in d.iterdir():
            if not p.is_dir():
                continue
            summary = p / 'experiment_summary.json'
            if not summary.exists():
                continue
            data = json.loads(summary.read_text(encoding='utf-8'))
            row = {
                'experiment': exp_base,
                'variant': variant,
                'portfolio': p.name,
                'initial_equity': data.get('initial_equity'),
                'final_equity': data.get('final_equity'),
                'total_return': data.get('total_return'),
                'cagr': data.get('cagr'),
                'sharpe': data.get('sharpe'),
                'sortino': data.get('sortino'),
                'max_drawdown': data.get('max_drawdown'),
                'volatility_annual': data.get('volatility_annual'),
                'beta': data.get('beta'),
                'turnover_annual': data.get('turnover_annual'),
                'win_rate': data.get('win_rate'),
                'avg_gain_per_trade': data.get('avg_gain_per_trade'),
                'n_trades': data.get('n_trades')
            }
            rows.append(row)
    return rows


def write_csv(rows, out_path: Path):
    import csv
    if not rows:
        print('[WARN] no rows to write')
        return
    keys = list(rows[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_md(rows, out_md: Path, exp_base: str):
    md = []
    md.append(f'# Comparison for experiment {exp_base}\n')
    variants = sorted(set(r['variant'] for r in rows))
    portfolios = sorted(set(r['portfolio'] for r in rows))
    md.append('## Variants\n')
    md.append(', '.join(variants) + '\n')
    md.append('## Summary table per portfolio (rows=portfolio, columns=variant total_return)\n')
    # table header
    header = ['portfolio'] + variants
    md.append('|' + '|'.join(header) + '|')
    md.append('|' + '|'.join(['---']*len(header)) + '|')
    for p in portfolios:
        line = [p]
        for v in variants:
            # find row
            val = ''
            for r in rows:
                if r['portfolio']==p and r['variant']==v:
                    val = r.get('total_return')
                    if val is None:
                        val = ''
                    else:
                        val = f"{float(val):.4f}"
                    break
            line.append(str(val))
        md.append('|' + '|'.join(line) + '|')
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text('\n'.join(md), encoding='utf-8')
    print('Wrote', out_md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    args = ap.parse_args()
    rows = gather(args.exp)
    csv_out = ART / f'compare_{args.exp}_per_portfolio.csv'
    md_out = ART / f'compare_{args.exp}.md'
    write_csv(rows, csv_out)
    write_md(rows, md_out, args.exp)

if __name__ == '__main__':
    main()
