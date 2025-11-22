#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

ROOT = Path('backtest-results-exp_latenight/artifacts')
exp_base = 'exp_latenight'

# find variant dirs matching exp_base_*
variants = sorted([d.name for d in ROOT.iterdir() if d.is_dir() and d.name.startswith(exp_base + '_')])
if not variants:
    print('No variant dirs found under', ROOT)
    raise SystemExit(1)

# gather portfolios across variants
all_portfolios = set()
variant_port_map = {}
# We'll normalize portfolio names by stripping known variant suffixes when present
variant_suffixes = [v.replace(exp_base + '_','') if v.startswith(exp_base + '_') else v for v in variants]
for v in variants:
    vdir = ROOT / v
    ports = []
    pdir = vdir / 'portfolios'
    if pdir.exists():
        cand = [d for d in pdir.iterdir() if d.is_dir()]
    else:
        cand = [d for d in vdir.iterdir() if d.is_dir()]
    for d in cand:
        name = d.name
        # normalize: if name ends with a known variant suffix, strip it
        base_name = name
        for suf in variant_suffixes:
            if suf and base_name.endswith('_' + suf):
                base_name = base_name[:-(len(suf)+1)]
                break
        ports.append((name, base_name))
        all_portfolios.add(base_name)
    variant_port_map[v] = ports

all_portfolios = sorted(all_portfolios)
# build matrices for total_return, cagr, final_equity, data_quality
mat_return = pd.DataFrame(index=all_portfolios, columns=variants)
mat_cagr = pd.DataFrame(index=all_portfolios, columns=variants)
mat_final = pd.DataFrame(index=all_portfolios, columns=variants)
mat_dq = pd.DataFrame(index=all_portfolios, columns=variants)

for v in variants:
    vdir = ROOT / v
    ports_dir = vdir / 'portfolios' if (vdir / 'portfolios').exists() else vdir
    ports = variant_port_map.get(v, [])
    # ports is list of tuples (actual_dir_name, base_name)
    name_map = {base: actual for actual, base in ports}
    for p in all_portfolios:
        actual = name_map.get(p)
        if actual:
            ppath = ports_dir / actual
            summary = ppath / 'experiment_summary.json'
            if summary.exists():
                try:
                    j = json.loads(summary.read_text())
                    mat_return.loc[p, v] = j.get('total_return')
                    mat_cagr.loc[p, v] = j.get('cagr')
                    mat_final.loc[p, v] = j.get('final_equity')
                    mat_dq.loc[p, v] = ','.join(j.get('data_quality') or [])
                except Exception as e:
                    mat_return.loc[p, v] = f'ERR:{e}'
            else:
                mat_return.loc[p, v] = None
                mat_cagr.loc[p, v] = None
                mat_final.loc[p, v] = None
                mat_dq.loc[p, v] = 'missing_summary'
        else:
            # portfolio not present in this variant
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

# also produce a simple human-readable markdown table for total_return
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

print('\nDone')
