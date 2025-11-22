#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

ROOT = Path('backtest-results-exp_latenight/artifacts')
exp_base = 'exp_latenight'

# find all summary files under artifacts
summaries = list(ROOT.rglob('experiment_summary.json'))
if not summaries:
    print('No experiment_summary.json files found under', ROOT)
    raise SystemExit(1)

# map variant -> {portfolio_base: summary_path}
variants_map = {}
all_portfolios = set()
for s in summaries:
    # find the variant directory name (first ancestor starting with exp_base_)
    variant = None
    for anc in s.parents:
        if anc.name.startswith(exp_base + '_'):
            variant = anc.name
            break
    if not variant:
        # fallback to parent of summary
        variant = s.parents[1].name if len(s.parents) > 1 else 'unknown'
    # portfolio dir name is parent of summary
    port_dir = s.parent.name
    # normalize base portfolio name by removing known suffix (variant label) if present
    base_name = port_dir
    if base_name.endswith('_' + variant.replace(exp_base + '_','')):
        base_name = base_name[:-(len(variant.replace(exp_base + '_',''))+1)]
    variants_map.setdefault(variant, {})[base_name] = s
    all_portfolios.add(base_name)

variants = sorted(variants_map.keys())
all_portfolios = sorted(all_portfolios)

mat_return = pd.DataFrame(index=all_portfolios, columns=variants)
mat_cagr = pd.DataFrame(index=all_portfolios, columns=variants)
mat_final = pd.DataFrame(index=all_portfolios, columns=variants)
mat_dq = pd.DataFrame(index=all_portfolios, columns=variants)

for v in variants:
    for p in all_portfolios:
        spath = variants_map.get(v, {}).get(p)
        if spath and spath.exists():
            j = json.loads(spath.read_text())
            mat_return.loc[p, v] = j.get('total_return')
            mat_cagr.loc[p, v] = j.get('cagr')
            mat_final.loc[p, v] = j.get('final_equity')
            mat_dq.loc[p, v] = ','.join(j.get('data_quality') or [])
        else:
            mat_return.loc[p, v] = None
            mat_cagr.loc[p, v] = None
            mat_final.loc[p, v] = None
            mat_dq.loc[p, v] = 'not_present'

out_csv = ROOT / f'compare_{exp_base}_matrix_total_return.csv'
mat_return.to_csv(out_csv)
print('Wrote', out_csv)

out_csv2 = ROOT / f'compare_{exp_base}_matrix_data_quality.csv'
mat_dq.to_csv(out_csv2)
print('Wrote', out_csv2)

# md
md_lines = [f'# Comparison matrix for {exp_base} (Total Return)\n']
header = ['Portfolio'] + variants
md_lines.append('| ' + ' | '.join(header) + ' |')
md_lines.append('|' + '|'.join(['---'] * len(header)) + '|')
for p in all_portfolios:
    row = [p]
    for v in variants:
        val = mat_return.loc[p, v]
        if pd.isna(val) or val is None:
            row.append('')
        else:
            try:
                row.append(f'{float(val):.4f}')
            except Exception:
                row.append(str(val))
    md_lines.append('| ' + ' | '.join(row) + ' |')
md_path = ROOT / f'compare_{exp_base}_matrix.md'
md_path.write_text('\n'.join(md_lines), encoding='utf-8')
print('Wrote', md_path)

print('Done')
