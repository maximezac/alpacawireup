#!/usr/bin/env python3
"""Assemble and tidy artifacts for a run.

Usage: python scripts/assemble_artifacts.py <run_dir>

- Ensures manifest.json and summary_for_gpt.txt exist under run_dir
- Creates trades.csv from trades_ledger.csv when missing
- Writes README.txt describing files & units
- Produces a CSV report of data_quality flags across portfolios
"""
from __future__ import annotations
import sys, json, csv
from pathlib import Path
from typing import Dict, Any


def load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def write_json(p: Path, obj: Dict[str, Any]):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def build_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest = {"run": run_dir.name, "files": {}}
    files = []
    for f in sorted(run_dir.rglob('*')):
        if f.is_file():
            files.append(str(f.relative_to(run_dir)))
    manifest['files']['all'] = files
    return manifest


def ensure_trades_csv(port_dir: Path):
    trades_csv = port_dir / 'trades.csv'
    ledger_csv = port_dir / 'trades_ledger.csv'
    if trades_csv.exists():
        return False
    if not ledger_csv.exists():
        return False
    # read ledger and write simplified trades.csv
    rows = []
    with ledger_csv.open('r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                'datetime_utc': row.get('datetime_utc',''),
                'portfolio_id': row.get('portfolio_id',''),
                'symbol': (row.get('symbol') or '').upper(),
                'action': row.get('action',''),
                'qty': row.get('qty',''),
                'price_exec': row.get('price_exec',''),
                'gross_amount': row.get('gross_amount','')
            })
    if rows:
        with trades_csv.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return True
    return False


def write_readme(run_dir: Path, exp_id: str):
    rd = run_dir / 'README.txt'
    lines = []
    lines.append(f"Experiment: {exp_id}")
    lines.append("")
    lines.append("Files in this run (key ones):")
    lines.append("- portfolios/<portfolio>/history.csv : per-day equity history")
    lines.append("- portfolios/<portfolio>/history_enriched.csv : equity + drawdown + benchmark cum (if available)")
    lines.append("- portfolios/<portfolio>/experiment_summary.json : per-portfolio metrics (total_return, cagr, sharpe, volatility, data_quality)")
    lines.append("- portfolios/<portfolio>/trades_ledger.csv : full per-portfolio ledger (all executed trades)")
    lines.append("- portfolios/<portfolio>/trades.csv : simplified trades (datetime, symbol, action, qty, price_exec, gross_amount)")
    lines.append("- recommended_trades_v3.json : machine-readable trade plan (per-run)")
    lines.append("- recommended_trades_read.md : human-readable trade plan (good for LLM)" )
    lines.append("")
    lines.append("Units and notes:")
    lines.append("- turnover_annual_dollars: gross notional turnover annualized in USD/year.")
    lines.append("- turnover_fraction_of_mean_equity: turnover_annual_dollars / mean(equity) (dimensionless)")
    lines.append("- sharpe: annualized Sharpe using geometric annualization of returns and annual vol. Null if insufficient non-zero returns.")
    lines.append("- For short runs or sparse trades, prefer total_return (period return) over Sharpe/CAGR.")
    lines.append("")
    rd.write_text('\n'.join(lines), encoding='utf-8')


def collect_data_quality(run_dir: Path, out_csv: Path):
    rows = []
    portfolios_dir = run_dir / 'portfolios'
    if not portfolios_dir.exists():
        # maybe portfolios are directly under run_dir
        portfolios = [d for d in run_dir.iterdir() if d.is_dir()]
    else:
        portfolios = [d for d in portfolios_dir.iterdir() if d.is_dir()]
    for p in portfolios:
        summary = p / 'experiment_summary.json'
        if not summary.exists():
            continue
        j = load_json(summary)
        rows.append({
            'portfolio': p.name,
            'initial_equity': j.get('initial_equity'),
            'final_equity': j.get('final_equity'),
            'period_days': j.get('period_days'),
            'total_return': j.get('total_return'),
            'cagr': j.get('cagr'),
            'sharpe': j.get('sharpe'),
            'sharpe_se': j.get('sharpe_se'),
            'data_quality': ';'.join(j.get('data_quality') or [])
        })
    # write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = list(rows[0].keys())
        with out_csv.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)


def ensure_manifest_and_summary(run_dir: Path, exp_id: str):
    # manifest at run_dir/manifest.json
    mpath = run_dir / 'manifest.json'
    if not mpath.exists():
        manifest = build_manifest(run_dir)
        write_json(mpath, manifest)
    # also write top-level manifest
    top = run_dir.parent / f'experiment_manifest_{exp_id}.json'
    if not top.exists():
        write_json(top, load_json(mpath))
    # summary_for_gpt
    summary_src = Path('artifacts') / f'summary_for_gpt_{exp_id}.txt'
    summary_dst = run_dir / 'summary_for_gpt.txt'
    if summary_src.exists() and not summary_dst.exists():
        summary_dst.write_text(summary_src.read_text(), encoding='utf-8')
    elif not summary_dst.exists():
        # create minimal summary
        summary_dst.write_text(f"Experiment: {exp_id}\nSee manifest.json for files.", encoding='utf-8')


def main():
    if len(sys.argv) < 2:
        print('Usage: scripts/assemble_artifacts.py <run_dir> [exp_id]')
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    exp_id = sys.argv[2] if len(sys.argv)>2 else run_dir.name

    print('Run dir:', run_dir)
    # ensure trades.csv created where missing
    portfolios_dir = run_dir / 'portfolios'
    if portfolios_dir.exists():
        ports = [d for d in portfolios_dir.iterdir() if d.is_dir()]
    else:
        ports = [d for d in run_dir.iterdir() if d.is_dir()]
    print('Found portfolios:', [p.name for p in ports])
    for p in ports:
        changed = ensure_trades_csv(p)
        if changed:
            print('Wrote trades.csv for', p.name)

    # ensure manifest + summary
    ensure_manifest_and_summary(run_dir, exp_id)

    # write README
    write_readme(run_dir, exp_id)

    # collect data quality CSV
    out_csv = run_dir / f'data_quality_report_{exp_id}.csv'
    collect_data_quality(run_dir, out_csv)
    print('Wrote data quality report to', out_csv)

    print('Done')

if __name__ == '__main__':
    main()
