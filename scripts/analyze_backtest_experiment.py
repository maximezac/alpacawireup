#!/usr/bin/env python3
from __future__ import annotations

import os, sys, json, math, csv
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml


# ----------------------------------------------------------------------
# Enhanced analyzer for backtest experiments
# ----------------------------------------------------------------------
# Usage:
#   python scripts/analyze_backtest_experiment.py PRICES_BACKFILL CONFIG_PORTFOLIOS
# If CONFIG_PORTFOLIOS omitted, defaults to config/portfolios.yml
# ----------------------------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: analyze_backtest_experiment.py PRICES_BACKFILL [CONFIG_PORTFOLIOS]", file=sys.stderr)
    sys.exit(1)

PRICES_PATH = Path(sys.argv[1])
CONFIG_PORTFOLIOS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(os.getenv('CONFIG_PORTFOLIOS','config/portfolios.yml'))

EXPERIMENT_ID = os.getenv('EXPERIMENT_ID','exp_v1')
BENCHMARK = os.getenv('BENCHMARK_SYMBOL','SPY').upper()
RISK_FREE = float(os.getenv('RISK_FREE_RATE_ANNUAL','0.04'))
ARTIFACT_DIR = Path('artifacts')
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
BASE_ARTIFACT_DIR = ARTIFACT_DIR / EXPERIMENT_ID
BASE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# Load backfill prices
prices = json.loads(PRICES_PATH.read_text(encoding='utf-8'))
syms = prices.get('symbols',{})

# Build date->close maps
def build_close_series(sym):
    node = syms.get(sym,{})
    bars = node.get('bars',[]) or []
    if not bars:
        return pd.Series(dtype=float)
    data = {b['t'].split('T',1)[0]: float(b.get('c') or 0.0) for b in bars}
    s = pd.Series(data)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return s

bench_series = build_close_series(BENCHMARK) if BENCHMARK in syms else pd.Series(dtype=float)

# Load portfolios config and identify portfolios with tag 'backtest'
port_cfg = {}
if CONFIG_PORTFOLIOS.exists():
    port_cfg = yaml.safe_load(CONFIG_PORTFOLIOS.read_text()) or {}
else:
    print(f"[WARN] portfolios config not found: {CONFIG_PORTFOLIOS}")

defaults = (port_cfg.get('defaults') or {}) if isinstance(port_cfg, dict) else {}
portfolios = (port_cfg.get('portfolios') or {}) if isinstance(port_cfg, dict) else {}

# select portfolios with 'backtest' tag
selected = []
for pid, node in portfolios.items():
    tags = [t.lower() for t in (node.get('tags') or [])]
    if 'backtest' in tags:
        selected.append(pid)

if not selected:
    print('[WARN] No portfolios tagged backtest found in config; exiting')
    sys.exit(0)

# Helper functions

def annualize_return(first, last, days):
    if first <= 0 or days <= 0:
        return 0.0
    years = days / 365.25
    return (last / first) ** (1.0 / years) - 1.0


def max_drawdown(equity_series: pd.Series):
    roll_max = equity_series.cummax()
    dd = equity_series / roll_max - 1.0
    mdd = dd.min()
    return float(mdd)


def sharpe_ratio(daily_returns: pd.Series, rf_annual: float):
    # rf_annual -> daily rf
    if daily_returns.dropna().empty:
        return None
    rf_daily = (1+rf_annual)**(1/252.0) - 1
    excess = daily_returns - rf_daily
    ann_ret = (1+daily_returns).prod()**(252/len(daily_returns)) -1 if len(daily_returns)>0 else 0.0
    ann_vol = daily_returns.std(ddof=0) * math.sqrt(252)
    if ann_vol == 0:
        return None
    return (ann_ret - rf_annual) / ann_vol


def sortino_ratio(daily_returns: pd.Series, rf_annual: float):
    if daily_returns.dropna().empty:
        return None
    rf_daily = (1+rf_annual)**(1/252.0) - 1
    downside = daily_returns.copy()
    downside[downside > rf_daily] = 0
    downside_std = downside.std(ddof=0) * math.sqrt(252)
    ann_ret = (1+daily_returns).prod()**(252/len(daily_returns)) -1 if len(daily_returns)>0 else 0.0
    if downside_std == 0:
        return None
    return (ann_ret - rf_annual) / downside_std


def beta_vs_benchmark(daily_returns: pd.Series, bench_returns: pd.Series):
    # align series
    df = pd.concat([daily_returns, bench_returns], axis=1, join='inner').dropna()
    if df.shape[0] < 2:
        return None
    cov = df.iloc[:,0].cov(df.iloc[:,1])
    var = df.iloc[:,1].var()
    if var == 0:
        return None
    return float(cov / var)

# Process each portfolio
all_portfolio_summaries = []
per_symbol_rows = []
per_sector_agg = {}

for pid in selected:
    print(f"[INFO] analyzing portfolio {pid}")
    hist_path = Path(portfolios[pid].get('path', f"data/portfolios/{pid}")) / portfolios[pid].get('positions_csv','positions.csv')
    # history.csv path
    history_csv = Path(portfolios[pid].get('path', f"data/portfolios/{pid}")) / 'history.csv'
    ledger_csv = Path(portfolios[pid].get('path', f"data/portfolios/{pid}")) / 'trades_ledger.csv'

    if not history_csv.exists():
        print(f"[WARN] history not found for {pid}: {history_csv}")
        continue

    # load history
    df_hist = pd.read_csv(history_csv, parse_dates=['date'])
    df_hist = df_hist.sort_values('date')
    df_hist = df_hist.dropna(subset=['equity'])
    eq = df_hist['equity']
    dates = df_hist['date']
    days = (dates.iloc[-1] - dates.iloc[0]).days if len(dates)>1 else 1

            # --- safer returns & annualization ---
    daily_returns = eq.pct_change().dropna()
    n_returns = len(daily_returns)
    # filter out exact-zero returns (no-change snapshots) to avoid vol suppression
    non_zero_returns = daily_returns[ daily_returns.abs() > 1e-12 ]
    n_nonzero = len(non_zero_returns)

    total_return = None
    if len(eq) > 1 and eq.iloc[0] > 0:
        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)

    # Guarded CAGR: require at least 7 days and some returns
    cagr = None
    if days >= 7 and n_returns >= 5 and total_return is not None:
        cagr = annualize_return(float(eq.iloc[0]), float(eq.iloc[-1]), days)

    # Volatility / Sharpe / Sortino: require reasonable non-zero sample size
    vol = None
    sharpe = None
    sortino = None
    sharpe_se = None
    # prefer non_zero_returns for volatility estimate when available
    if n_nonzero >= 60:
        ann_vol = float(non_zero_returns.std(ddof=0) * math.sqrt(252))
        vol = ann_vol
        if ann_vol > 1e-12:
            sharpe = sharpe_ratio(non_zero_returns, RISK_FREE)
        else:
            sharpe = None
        sortino = sortino_ratio(non_zero_returns, RISK_FREE)
        # sharpe standard error (Lo 2002)
        if sharpe is not None:
            try:
                sharpe_se = math.sqrt((1.0 + 0.5 * (float(sharpe)**2)) / float(max(n_nonzero,1)))
            except Exception:
                sharpe_se = None
    else:
        # fallback: if some returns exist but few non-zero, compute best-effort vol on all returns
        if n_returns > 0:
            vol = float(daily_returns.std(ddof=0) * math.sqrt(252))
            # don't compute Sharpe for small non-zero sample; mark as short_history

    mdd = max_drawdown(eq)

    # benchmark returns aligned more robustly
    bench_close = build_close_series(BENCHMARK) if BENCHMARK in syms else pd.Series(dtype=float)
    beta = None
    bench_cagr = None
    bench_annual_vol = None
    tracking_error = None
    info_ratio = None
    alpha = None
    if not bench_close.empty:
        # align benchmark prices to portfolio history dates with forward-fill
        bench_prices = bench_close.reindex(df_hist['date']).ffill()
        bench_prices = bench_prices.dropna()
        if len(bench_prices) >= 2:
            # bench returns
            bench_returns = bench_prices.pct_change().dropna()
            if len(bench_returns) >= 2:
                bench_annual_vol = float(bench_returns.std(ddof=0) * math.sqrt(252))
                bench_cagr = annualize_return(float(bench_prices.iloc[0]), float(bench_prices.iloc[-1]), (bench_prices.index[-1] - bench_prices.index[0]).days)
            # align returns for beta / tracking_error
            common = pd.concat([non_zero_returns if len(non_zero_returns)>0 else daily_returns, bench_returns], axis=1, join='inner').dropna()
            if not common.empty and common.shape[0] >= 2:
                # common.iloc[:,0] = portfolio returns, common.iloc[:,1] = bench returns
                cov = common.iloc[:,0].cov(common.iloc[:,1])
                var = common.iloc[:,1].var()
                if var > 0:
                    beta = float(cov / var)
                excess = common.iloc[:,0] - common.iloc[:,1]
                tracking_error = float(excess.std(ddof=0) * math.sqrt(252)) if len(excess) >= 2 else None
                info_ratio = float((excess.mean() * 252) / tracking_error) if tracking_error and tracking_error > 1e-12 else None
                # alpha as ann_ret - beta*bench_ann_ret (if cagr present)
                if cagr is not None and bench_cagr is not None and beta is not None:
                    alpha = float(cagr - beta * bench_cagr)


    # trades analysis: replay ledger to compute per-trade realized pnl and turnover
    trades_list = []
    pos = {}
    total_turnover = 0.0
    realized_pnls = []
    wins = 0
    losses = 0
    sell_trades = 0

    if ledger_csv.exists():
        df_ldr = pd.read_csv(ledger_csv, parse_dates=['datetime_utc'])
        df_ldr = df_ldr[df_ldr['portfolio_id'] == pid].sort_values('datetime_utc')
        # replay per symbol
        for _, r in df_ldr.iterrows():
            sym = str(r['symbol']).upper()
            action = str(r['action']).upper()
            qty = float(r.get('qty') or 0.0)
            px = float(r.get('price_exec') or 0.0)
            gross = float(r.get('gross_amount') or 0.0)
            # compute turnover
            total_turnover += abs(gross)
            if sym not in pos:
                pos[sym] = {'qty':0.0,'avg_cost':0.0}
            cur = pos[sym]
            if action == 'BUY':
                cost = qty * px
                new_qty = cur['qty'] + qty
                new_avg = ((cur['qty']*cur['avg_cost']) + cost)/new_qty if new_qty>0 else cur['avg_cost']
                cur['qty'] = new_qty
                cur['avg_cost'] = new_avg
                trades_list.append({'datetime': r['datetime_utc'], 'symbol':sym, 'action':'BUY','qty':qty,'px':px,'pnl':None})
            elif action == 'SELL':
                sell_qty = min(qty, cur['qty'])
                if sell_qty<=0:
                    continue
                realized = (px - cur['avg_cost'])*sell_qty
                cur['qty'] = cur['qty'] - sell_qty
                if cur['qty']<=0:
                    cur['avg_cost'] = 0.0
                trades_list.append({'datetime': r['datetime_utc'],'symbol':sym,'action':'SELL','qty':sell_qty,'px':px,'pnl':realized})
                realized_pnls.append(realized)
                sell_trades += 1
                if realized>0:
                    wins += 1
                else:
                    losses += 1
    else:
        df_ldr = pd.DataFrame()

    avg_gain = float(np.mean([p for p in realized_pnls if p>0])) if [p for p in realized_pnls if p>0] else None
    win_rate = float(wins / sell_trades) if sell_trades>0 else None
    avg_turnover = total_turnover / ((days/365.25) if days>0 else 1)
    turnover_fraction = None
    mean_equity = float(eq.mean()) if len(eq)>0 and not math.isnan(float(eq.mean())) else None
    if mean_equity and mean_equity>0:
        turnover_fraction = avg_turnover / mean_equity


    # sector attribution using per_symbol computed earlier (best-effort)
    # Build per-symbol realized/unrealized results (reuse previous logic if available)
    # We'll compute simple sector sums from per_symbol if present
    per_symbol_path = ARTIFACT_DIR / f"experiment_summary_{EXPERIMENT_ID}_per_symbol.csv"

    # Build portfolio summary
    data_quality = []
    if n_returns < 10:
        data_quality.append('short_history')
    if len(trades_list) < 3:
        data_quality.append('few_trades')

    p_summary = {
        'portfolio_id': pid,
        'initial_equity': float(eq.iloc[0]) if len(eq)>0 else None,
        'final_equity': float(eq.iloc[-1]) if len(eq)>0 else None,
        'period_days': int(days),
        'total_return': float(total_return) if total_return is not None else None,
        'cagr': float(cagr) if cagr is not None else None,
        'max_drawdown': float(mdd) if mdd is not None else None,
        'volatility_annual': float(vol) if vol is not None else None,
        'sharpe': float(sharpe) if sharpe is not None else None,
        'sharpe_se': float(sharpe_se) if sharpe_se is not None else None,
        'sortino': float(sortino) if sortino is not None else None,
        'beta': float(beta) if beta is not None else None,
        'bench_cagr': float(bench_cagr) if bench_cagr is not None else None,
        'bench_annual_vol': float(bench_annual_vol) if bench_annual_vol is not None else None,
        'tracking_error': float(tracking_error) if tracking_error is not None else None,
        'info_ratio': float(info_ratio) if info_ratio is not None else None,
        'alpha': float(alpha) if alpha is not None else None,
        'turnover_annual_dollars': float(avg_turnover) if avg_turnover is not None else None,
        'turnover_fraction_of_mean_equity': float(turnover_fraction) if turnover_fraction is not None else None,
        'win_rate': win_rate,
        'avg_gain_per_trade': float(avg_gain) if avg_gain is not None else None,
        'n_trades': len(trades_list),
        'data_quality': data_quality
    }

    # Verify portfolio_value snapshot consistency (best-effort)
    try:
        pv_path = BASE_ARTIFACT_DIR / pid / 'portfolio_value.json'
        if not pv_path.exists():
            # try data/portfolios path
            pv_path = Path(portfolios[pid].get('path', f"data/portfolios/{pid}")) / 'portfolio_value.json'
        if pv_path.exists():
            pv = json.loads(pv_path.read_text(encoding='utf-8'))
            pv_equity = pv.get('equity')
            if pv_equity is not None and p_summary['final_equity'] is not None:
                diff = abs(float(p_summary['final_equity']) - float(pv_equity))
                if diff > max(1e-6, 0.01 * (abs(float(p_summary['final_equity'])) if p_summary['final_equity'] else 1.0)):
                    p_summary.setdefault('data_quality', []).append('portfolio_value_mismatch')
                    print(f"[WARN] equity mismatch for {pid}: history {p_summary['final_equity']} vs portfolio_value {pv_equity}")
    except Exception as e:
        print(f"[WARN] failed to verify portfolio_value for {pid}: {e}")



    # write per-portfolio artifacts
    port_art = BASE_ARTIFACT_DIR / pid
    port_art.mkdir(parents=True, exist_ok=True)
    (port_art / 'experiment_summary.json').write_text(json.dumps(p_summary, indent=2))
    # history and trades export
    df_hist.to_csv(port_art / 'history.csv', index=False)
    if len(trades_list):
        pd.DataFrame(trades_list).to_csv(port_art / 'trades.csv', index=False)

    all_portfolio_summaries.append(p_summary)

# Write consolidated per-portfolio CSV
if all_portfolio_summaries:
    dfp = pd.DataFrame(all_portfolio_summaries)
    dfp.to_csv(BASE_ARTIFACT_DIR / f"experiment_summary_{EXPERIMENT_ID}_per_portfolio.csv", index=False)

# Aggregate per-symbol across selected portfolios (aggregate realized/sell/buy/market_value)
per_symbol_agg = {}
for pid in selected:
    node = portfolios.get(pid, {})
    port_path = Path(node.get('path', f"data/portfolios/{pid}"))
    ledger_path = port_path / 'trades_ledger.csv'
    if not ledger_path.exists():
        continue
    with ledger_path.open('r', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            s = r.get('symbol','').upper()
            if not s:
                continue
            rec = per_symbol_agg.setdefault(s, {'symbol':s,'buy_volume':0.0,'sell_volume':0.0,'realized_pnl':0.0,'trades':0,'first_trade':None,'last_trade':None})
            qty = float(r.get('qty') or 0.0)
            gross = float(r.get('gross_amount') or 0.0)
            # gross_amount was defined as proceeds for sells, negative for buys in our ledger convention earlier
            # interpret buy vs sell
            if r.get('action','').upper()=='BUY':
                rec['buy_volume'] += abs(gross)
            else:
                rec['sell_volume'] += abs(gross)
            # try to extract realized pnl from signal or trades_applied if present
            # if 'net_amount' present, no direct realized pnl; we will approximate later
            rec['trades'] += 1
            dt = r.get('datetime_utc')
            try:
                dti = datetime.fromisoformat(dt.replace('Z','+00:00'))
                if rec['first_trade'] is None or dti < rec['first_trade']:
                    rec['first_trade'] = dti
                if rec['last_trade'] is None or dti > rec['last_trade']:
                    rec['last_trade'] = dti
            except Exception:
                pass

# try to attach mark & unrealized using prices map
for s, rec in per_symbol_agg.items():
    node = syms.get(s,{})
    bars = node.get('bars',[]) or []
    if bars:
        last_close = float(bars[-1].get('c') or 0.0)
    else:
        last_close = 0.0
    rec['mark'] = last_close

# write per-symbol CSV
if per_symbol_agg:
    df_sym = pd.DataFrame(list(per_symbol_agg.values()))
    df_sym.to_csv(BASE_ARTIFACT_DIR / f"experiment_summary_{EXPERIMENT_ID}_per_symbol.csv", index=False)

# Generate CSV/markdown summaries per portfolio and global (no PNG plots)

md_lines = []
md_lines.append(f"# Experiment summary: {EXPERIMENT_ID}\n")
# Date range: attempt to use last processed history 'dates' if available; fall back to n/a
if 'dates' in locals() and len(dates) > 0:
    try:
        md_lines.append(f"Date range: {dates.iloc[0].strftime('%Y-%m-%d')} → {dates.iloc[-1].strftime('%Y-%m-%d')}\n")
    except Exception:
        md_lines.append(f"Date range: {dates[0]} → {dates[-1]}\n")
else:
    md_lines.append("Date range: n/a\n")

for p in all_portfolio_summaries:
    pid = p['portfolio_id']
    port_art = BASE_ARTIFACT_DIR / pid

    hist_file = port_art / 'history.csv'
    if not hist_file.exists():
        continue
    dfh = pd.read_csv(hist_file, parse_dates=['date'])
    dfh = dfh.sort_values('date')

    # drawdown series
    roll_max = dfh['equity'].cummax()
    dfh['drawdown'] = (dfh['equity'] / roll_max - 1.0)

    # bench series aligned to dates (forward-fill)
    bench_close = build_close_series(BENCHMARK) if BENCHMARK in syms else pd.Series(dtype=float)
    if not bench_close.empty:
        bench_prices = bench_close.reindex(dfh['date']).ffill()
        bench_prices = bench_prices.dropna()
        if len(bench_prices) >= 2:
            bench_returns = bench_prices.pct_change().dropna()
            # scale benchmark cumulative to portfolio start equity for comparison
            bench_cum = (1 + bench_returns).cumprod().reindex(dfh['date'], method='ffill') * dfh['equity'].iloc[0]
            dfh['bench_cum'] = bench_cum.values
        else:
            dfh['bench_cum'] = [None] * len(dfh)
    else:
        dfh['bench_cum'] = [None] * len(dfh)

    # write enriched history CSV (date,equity,drawdown,bench_cum)
    out_hist = port_art / 'history_enriched.csv'
    dfh[['date','equity','drawdown','bench_cum']].to_csv(out_hist, index=False)

    # per-portfolio markdown summary
    pm = []
    pm.append(f"# Portfolio {pid} — Experiment {EXPERIMENT_ID}\n")
    pm.append(f"- Initial equity: {p.get('initial_equity')}\n")
    pm.append(f"- Final equity: {p.get('final_equity')}\n")
    pm.append(f"- Period days: {p.get('period_days')}\n")
    pm.append(f"- Total return: {p.get('total_return')}\n")
    pm.append(f"- CAGR: {p.get('cagr')}\n")
    pm.append(f"- Sharpe: {p.get('sharpe')} (SE={p.get('sharpe_se')})\n")
    pm.append(f"- Sortino: {p.get('sortino')}\n")
    pm.append(f"- Max drawdown: {p.get('max_drawdown')}\n")
    pm.append(f"- Volatility (annual): {p.get('volatility_annual')}\n")
    pm.append(f"- Beta: {p.get('beta')}\n")
    pm.append('\n')
    pm.append('## Notes\n')
    dq = p.get('data_quality') or []
    if dq:
        pm.append('Data quality flags: ' + ', '.join(dq) + '\n')
    pm.append('\n')

    (port_art / 'summary.md').write_text('\n'.join(pm), encoding='utf-8')

    md_lines.append(f"## Portfolio {pid}\n")
    md_lines.append(f"- initial_equity: {p.get('initial_equity')}\n")
    md_lines.append(f"- final_equity: {p.get('final_equity')}\n")
    md_lines.append(f"- total_return: {p.get('total_return')}\n")
    md_lines.append(f"- sharpe: {p.get('sharpe')}\n")
    md_lines.append('\n')


# Write global markdown
md_path = ARTIFACT_DIR / f"experiment_summary_{EXPERIMENT_ID}.md"
md_path.write_text('\n'.join(md_lines), encoding='utf-8')

# ---- Generate a manifest and a short summary intended for downstream summarizers (GPT, reviewers) ----
try:
    import time
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "base_artifact_dir": str(BASE_ARTIFACT_DIR),
        "files": {}
    }

    def _gather(pattern_root, rel_prefix=None):
        out = []
        p_root = Path(pattern_root)
        if p_root.exists():
            for p in sorted([x for x in p_root.rglob('*') if x.is_file()]):
                rel = p.relative_to(Path.cwd())
                out.append(str(rel))
        return out

    # Per-portfolio summaries
    per_portfolio = {}
    for pid in sorted([d.name for d in BASE_ARTIFACT_DIR.iterdir() if d.is_dir()]):
        basep = BASE_ARTIFACT_DIR / pid
        files = [str(x.relative_to(Path.cwd())) for x in sorted(basep.glob('*')) if x.is_file()]
        per_portfolio[pid] = files

    manifest['files']['per_portfolio'] = per_portfolio

    # recommended trades + human-readable
    recs = []
    for p in Path('artifacts').glob('recommended_trades*'):
        if p.is_file():
            recs.append(str(p.relative_to(Path.cwd())))
    manifest['files']['recommended_trades'] = recs

    # perf timeseries and charts
    perf_files = _gather('data', 'data') + _gather('artifacts/perf_charts', 'artifacts/perf_charts')
    manifest['files']['perf'] = perf_files

    # compare outputs
    comp_files = _gather('artifacts', 'artifacts')
    manifest['files']['others'] = comp_files

    # write manifest to artifacts root and to base artifact dir
    MANIFEST_PATH = ARTIFACT_DIR / f"experiment_manifest_{EXPERIMENT_ID}.json"
    BASE_MANIFEST_PATH = BASE_ARTIFACT_DIR / 'manifest.json'
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    BASE_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    # short, human-friendly summary advising which files to upload to a summarizer
    summary_lines = []
    summary_lines.append(f"Experiment: {EXPERIMENT_ID}")
    summary_lines.append(f"Generated: {manifest['generated_at']}")
    summary_lines.append("")
    summary_lines.append("Recommended files to upload to an LLM (in this order):")
    summary_lines.append("1) artifacts/recommended_trades_read.md — human-readable trade plan for all portfolios.")
    summary_lines.append("2) artifacts/recommended_trades_v3.json — machine-readable trade plan (per-portfolio).")
    summary_lines.append("3) artifacts/experiment_summary_<EXPERIMENT_ID>_per_portfolio.csv — consolidated metrics (CAGR, Sharpe, etc.).")
    summary_lines.append("4) artifacts/<EXPERIMENT_ID>/portfolios/<PORT>/history.csv — per-portfolio equity history (time series).")
    summary_lines.append("5) data/perf_timeseries.csv — consolidated time series across portfolios (if present).")
    summary_lines.append("6) artifacts/perf_charts/* — charts (png) for quick visual context.")
    summary_lines.append("")
    summary_lines.append("Manifest file created at: " + str(MANIFEST_PATH))
    summary_lines.append("")
    summary_lines.append("Notes:")
    summary_lines.append("- For best LLM summaries, upload the human-readable plan first, then per-portfolio summaries and histories.")
    summary_lines.append("- If you want a single zip for the model, include: recommended_trades_read.md, experiment_summary_*_per_portfolio.csv, data/perf_timeseries.csv, and the per-portfolio history CSVs.")

    summary_path = ARTIFACT_DIR / f"summary_for_gpt_{EXPERIMENT_ID}.txt"
    summary_path.write_text('\n'.join(summary_lines), encoding='utf-8')
    (BASE_ARTIFACT_DIR / 'summary_for_gpt.txt').write_text('\n'.join(summary_lines), encoding='utf-8')

    print(f"[OK] Wrote manifest and summary_for_gpt to {ARTIFACT_DIR} and {BASE_ARTIFACT_DIR}")
except Exception as e:
    print(f"[WARN] failed to write manifest/summary: {e}")

print(f"[OK] Analysis complete. Artifacts in {ARTIFACT_DIR}")

