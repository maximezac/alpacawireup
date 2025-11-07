#!/usr/bin/env python3
from __future__ import annotations
import csv, os
from pathlib import Path
from typing import Dict, List
import json
import matplotlib.pyplot as plt  # std lib in actions runner

HIST_GLOB = "data/portfolios/*/history.csv"
OUT_CSV   = Path("data/perf_timeseries.csv")
OUT_JSON  = Path("data/perf_timeseries.json")
CHART_DIR = Path("artifacts/perf_charts")

def read_history(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # keep only the columns we care about
            rows.append({
                "date": row.get("date"),
                "portfolio": row.get("portfolio"),
                "cash": float(row.get("cash") or 0),
                "market_value": float(row.get("market_value") or 0),
                "equity": float(row.get("equity") or 0),
                "daily_pnl": float(row.get("daily_pnl") or 0),
                "daily_return": float(row.get("daily_return") or 0),
                "cumulative_return": float(row.get("cumulative_return") or 0),
            })
    # sort by date just in case
    rows.sort(key=lambda x: (x["portfolio"] or "", x["date"] or ""))
    return rows

def write_consolidated(rows: List[Dict]):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date","portfolio","cash","market_value","equity","daily_pnl","daily_return","cumulative_return"])
        for r in rows:
            w.writerow([r["date"], r["portfolio"], r["cash"], r["market_value"], r["equity"], r["daily_pnl"], r["daily_return"], r["cumulative_return"]])
    OUT_JSON.write_text(json.dumps(rows, indent=2))

def chart_portfolio(rows: List[Dict], portfolio: str):
    pts = [r for r in rows if r["portfolio"] == portfolio and r["date"]]
    if not pts:
        return
    dates = [r["date"] for r in pts]
    eq = [r["equity"] for r in pts]
    mv = [r["market_value"] for r in pts]
    cash = [r["cash"] for r in pts]

    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # Equity chart
    plt.figure()
    plt.plot(dates, eq, label="Equity")
    plt.xticks(rotation=30, ha="right")
    plt.title(f"{portfolio} — Equity over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / f"{portfolio}_equity.png")
    plt.close()

    # MV vs Cash chart
    plt.figure()
    plt.plot(dates, mv, label="Market Value")
    plt.plot(dates, cash, label="Cash")
    plt.xticks(rotation=30, ha="right")
    plt.title(f"{portfolio} — Market Value & Cash")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / f"{portfolio}_mv_cash.png")
    plt.close()

def main():
    import glob
    all_rows: List[Dict] = []
    portfolios = set()

    for p in glob.glob(HIST_GLOB):
        path = Path(p)
        try:
            rows = read_history(path)
            all_rows.extend(rows)
            for r in rows:
                if r["portfolio"]:
                    portfolios.add(r["portfolio"])
        except Exception as e:
            print(f"[WARN] failed reading {path}: {e}")

    if not all_rows:
        print("[WARN] no history rows found; did update_performance_v3 write history.csv?")
        return

    write_consolidated(all_rows)
    for pid in sorted(portfolios):
        chart_portfolio(all_rows, pid)

    print(f"[OK] wrote {OUT_CSV} and {OUT_JSON}, charts in {CHART_DIR}/")

if __name__ == "__main__":
    main()
