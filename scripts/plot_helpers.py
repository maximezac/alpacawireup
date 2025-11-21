#!/usr/bin/env python3
from __future__ import annotations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List


def plot_equity_curve(dates: List[str], equities: List[float], out_path: str):
    d = [datetime.fromisoformat(x) for x in dates]
    plt.figure(figsize=(10,4))
    plt.plot(d, equities, lw=2)
    plt.title('Equity Curve')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_drawdown(dates: List[str], drawdowns: List[float], out_path: str):
    d = [datetime.fromisoformat(x) for x in dates]
    plt.figure(figsize=(10,4))
    plt.plot(d, drawdowns, lw=2, color='red')
    plt.title('Drawdown')
    plt.ylabel('Drawdown')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_benchmark_compare(dates: List[str], port_series: List[float], bench_series: List[float], out_path: str):
    d = [datetime.fromisoformat(x) for x in dates]
    plt.figure(figsize=(10,4))
    plt.plot(d, port_series, label='Portfolio')
    plt.plot(d, bench_series, label='Benchmark')
    plt.legend()
    plt.title('Portfolio vs Benchmark')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
