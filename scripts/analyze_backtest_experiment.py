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

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = annualize_return(float(eq.iloc[0]), float(eq.iloc[-1]), days)
    daily_returns = eq.pct_change().fillna(0)
    vol = daily_returns.std(ddof=0) * math.sqrt(252)
    sharpe = sharpe_ratio(daily_returns, RISK_FREE)
    sortino = sortino_ratio(daily_returns, RISK_FREE)
    mdd = max_drawdown(eq)

    # benchmark returns aligned
    bench_close = build_close_series(BENCHMARK) if BENCHMARK in syms else pd.Series(dtype=float)
    bench_daily = bench_close.pct_change().reindex(df_hist['date']).fillna(0)
    beta = beta_vs_benchmark(daily_returns, bench_daily)

        # trades analysis: replay ledger to compute per-trade realized pnl and turnover
    trades_list = []
    # initialize accumulators even if ledger missing to avoid NameError later
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

    avg_gain = float(np.mean([p for p in realized_pnls if p>0])) if realized_pnls else 0.0
    win_rate = float(wins / sell_trades) if sell_trades>0 else None
    avg_turnover = total_turnover / ((days/365.25) if days>0 else 1)


    # sector attribution using per_symbol computed earlier (best-effort)
    # Build per-symbol realized/unrealized results (reuse previous logic if available)
    # We'll compute simple sector sums from per_symbol if present
    per_symbol_path = ARTIFACT_DIR / f"experiment_summary_{EXPERIMENT_ID}_per_symbol.csv"

    # Build portfolio summary
    p_summary = {
        'portfolio_id': pid,
        'initial_equity': float(eq.iloc[0]),
        'final_equity': float(eq.iloc[-1]),
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': mdd,
        'volatility_annual': vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'beta': beta,
        'turnover_annual': avg_turnover,
        'win_rate': win_rate,
        'avg_gain_per_trade': avg_gain,
        'n_trades': len(trades_list)
    }

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

# Generate charts and markdown summaries per portfolio and global
from scripts.plot_helpers import plot_equity_curve, plot_drawdown, plot_benchmark_compare

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
    dates_str = [d.strftime('%Y-%m-%d') for d in dfh['date']]
    equities = list(dfh['equity'])

    eq_png = port_art / 'equity_curve.png'
    dd_png = port_art / 'drawdown.png'
    bench_png = port_art / 'benchmark_compare.png'

    # drawdown series
    roll_max = dfh['equity'].cummax()
    drawdown = (dfh['equity'] / roll_max - 1.0).tolist()

    # bench series aligned
    bench_close = build_close_series(BENCHMARK) if BENCHMARK in syms else pd.Series(dtype=float)
    bench_aligned = None
    if not bench_close.empty:
        bench_aligned = bench_close.reindex(dfh['date']).ffill().pct_change().fillna(0)
        bench_cum = (1+bench_aligned).cumprod().fillna(method='ffill') * (dfh['equity'].iloc[0])
        bench_series_vals = list(bench_cum)
    else:
        bench_series_vals = [None]*len(dates_str)

    # plot
    plot_equity_curve(dates_str, equities, str(eq_png))
    plot_drawdown(dates_str, drawdown, str(dd_png))
    if bench_series_vals[0] is not None:
        plot_benchmark_compare(dates_str, equities, bench_series_vals, str(bench_png))

    # per-portfolio markdown
    pm = []
    pm.append(f"# Portfolio {pid} — Experiment {EXPERIMENT_ID}\n")
    pm.append(f"- Initial equity: {p.get('initial_equity')}\n")
    pm.append(f"- Final equity: {p.get('final_equity')}\n")
    pm.append(f"- Total return: {p.get('total_return')}\n")
    pm.append(f"- CAGR: {p.get('cagr')}\n")
    pm.append(f"- Sharpe: {p.get('sharpe')}\n")
    pm.append(f"- Sortino: {p.get('sortino')}\n")
    pm.append(f"- Max drawdown: {p.get('max_drawdown')}\n")
    pm.append(f"- Volatility (annual): {p.get('volatility_annual')}\n")
    pm.append(f"- Beta: {p.get('beta')}\n")
    pm.append('\n')
    pm.append('## Charts\n')
    pm.append(f'![Equity curve]({eq_png.name})\n')
    pm.append(f'![Drawdown]({dd_png.name})\n')
    if bench_series_vals[0] is not None:
        pm.append(f'![Benchmark compare]({bench_png.name})\n')

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

print(f"[OK] Analysis complete. Artifacts in {ARTIFACT_DIR}")
