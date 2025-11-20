#!/usr/bin/env python3
import json, math, statistics, sys
from datetime import datetime, timezone, timedelta

# -------- CONFIG --------
F = "data/prices_final_snapshot_20230210.json"   # <-- adjust file name if needed
SYMBOL = "TSLA"                                   # <-- symbol to debug
# ------------------------


def parse_dt(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


# -------- Load snapshot --------
try:
    f = json.load(open(F, "r", encoding="utf-8"))
except Exception as e:
    print("ERROR: failed to load", F, e)
    sys.exit(2)

asof_str = f.get("as_of_utc")
asof = parse_dt(asof_str) or datetime.now(timezone.utc)
print("AS_OF:", asof.isoformat())

node = f.get("symbols", {}).get(SYMBOL, {})
news_all = node.get("news", []) or []

print("\n--- Symbol:", SYMBOL, "---")
print("TOTAL_NEWS_FOR", SYMBOL, "=", len(news_all))

# Timestamps
dts = [n.get("ts") for n in news_all if n.get("ts")]
print("first_ts:", dts[0] if dts else None)
print("last_ts :", dts[-1] if dts else None)


# -------- Show first 20 news --------
print("\nSAMPLE first 20 entries (ts, tone):")
for i, n in enumerate(news_all[:20]):
    print(f"{i+1:02d}. {n.get('ts')}   tone={n.get('tone')}")


# -------- Count news by recency windows --------
for cutoff_days in (5, 7, 14, 30, 90):
    cut_dt = asof - timedelta(days=cutoff_days)
    recent = [
        n for n in news_all
        if n.get("ts") and parse_dt(n["ts"]) and parse_dt(n["ts"]) >= cut_dt
    ]
    print(f"within_{cutoff_days}d:", len(recent))


# -------- NS table for multiple half-lives + windows --------
half_lives = [12.0, 24.0, 72.0, 168.0, 720.0, 1440.0]
cutoffs = [None, 5, 14, 30]   # None = use ALL news

print("\nNS table: rows=cutoff_days (None=ALL)  columns=half-life hours:")
for cut in cutoffs:
    if cut is None:
        news = [
            n for n in news_all
            if n.get("tone") is not None and n.get("ts")
        ]
        label = "ALL"
    else:
        cut_dt = asof - timedelta(days=cut)
        news = [
            n for n in news_all
            if n.get("tone") is not None and n.get("ts")
               and parse_dt(n["ts"]) >= cut_dt
        ]
        label = f"{cut}d"

    row = []
    for hl in half_lives:
        lam = math.log(2.0) / hl
        num = 0.0
        den = 0.0
        for n in news:
            tone = float(n.get("tone"))
            dt = parse_dt(n["ts"])
            if not dt:
                continue
            h = max(0.0, (asof - dt).total_seconds() / 3600.0)
            w = math.exp(-lam * h)
            num += tone * w
            den += w

        ns = 0.0 if den == 0 else num / den
        row.append(round(ns, 6))

    print(f"{label}: {row}")


# -------- Global tone stats --------
all_tones = [
    float(n.get("tone"))
    for node in f.get("symbols", {}).values()
    for n in (node.get("news") or [])
    if n.get("tone") is not None
]

print("\nGLOBAL raw tone count:", len(all_tones),
      "mean:", (statistics.fmean(all_tones) if all_tones else None))


# -------- Global NS (example half-life) --------
hl = 72.0
lam = math.log(2.0) / hl
ns_list = []

for sym, node in f.get("symbols", {}).items():
    news = node.get("news") or []
    num = 0.0
    den = 0.0
    for n in news:
        t = n.get("tone")
        ts = n.get("ts")
        if t is None or not ts:
            continue
        dt = parse_dt(ts)
        if not dt:
            continue
        h = max(0.0, (asof - dt).total_seconds() / 3600.0)
        w = math.exp(-lam * h)
        num += float(t) * w
        den += w
    if den > 0:
        ns_list.append(num / den)

print("global avg NS (hl=72h):",
      (statistics.fmean(ns_list) if ns_list else None),
      "symbols_with_ns", len(ns_list))


# -------- Top recent contributors (last 14 days) --------
cut_dt = asof - timedelta(days=14)
candidates = []

hl = 72.0
lam = math.log(2.0) / hl

for n in news_all:
    t = n.get("tone")
    ts = n.get("ts")
    if t is None or not ts:
        continue
    dt = parse_dt(ts)
    if not dt or dt < cut_dt:
        continue
    h = max(0.0, (asof - dt).total_seconds() / 3600.0)
    w = math.exp(-lam * h)
    candidates.append((abs(w * float(t)), w, float(t), ts, (n.get("headline") or "")[:120]))

candidates.sort(reverse=True)

print("\nTop recent contributors (score, weight, tone, ts, headline snippet):")
for c in candidates[:20]:
    print(c)
