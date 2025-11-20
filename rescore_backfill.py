# scripts/rescore_backfill.py
#!/usr/bin/env python3
"""
Rescore (recompute) news tone for an existing backfill file.

Produces a backup: data/prices_backfill.json.bak and writes a refreshed
data/prices_backfill.json with new `tone` values for each news item.

Uses VADER + simple LM overlay + source weights (same heuristics as fetch_news),
but DOES NOT apply age-based decay (we want raw tone stored and let snapshots
apply decay relative to the snapshot date).
"""
from __future__ import annotations
import json, shutil, math, re
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

IN_PATH = Path("data/prices_backfill.json")
BACKUP = Path("data/prices_backfill.json.bak")

if not IN_PATH.exists():
    raise SystemExit(f"backfill file not found: {IN_PATH}")

# --- Configuration copied from fetch_news defaults (keeps behavior consistent) ---
NEWS_SOURCE_WEIGHTS_DEFAULT = {
    "benzinga": 0.90,
    "mtnewswires": 0.95,
    "marketbeat": 0.90,
    "seekingalpha": 0.90,
    "tipranks": 0.80,
    "motleyfool": 0.75,
    "zacksinvestmentresearch": 0.75,
    "yahoofinance": 0.85,
    "barrons": 0.95,
    "bloomberg": 0.98,
    "reuters": 0.98,
    "wsj": 0.98,
    "investorsbusinessdaily": 0.90,
    "googlerss": 0.70,
    "reddit": 0.35,
    "finnhub": 1.00,
    "newsapi": 0.30,
}

_SRC_NORM_RE = re.compile(r"[^a-z0-9]+")
def norm_source(s):
    if not s:
        return "unknown"
    return _SRC_NORM_RE.sub("", s.lower())

# LM overlay lists (copied from fetch_news for parity)
LM_POS = {
    "accretive","advantage","benefit","boost","bullish","contract","efficiency",
    "exceed","growth","improve","innovation","lead","opportunity","profitable",
    "record","tailwind","undersupplied","upgrade","win","wins"
}
LM_NEG = {
    "adverse","antitrust","bankrupt","bearish","breach","caution","costly",
    "cutback","dilution","downturn","headwind","impair","layoff","litigation",
    "miss","recall","shortfall","slowdown","warning","downgrade"
}

def lm_overlay_score(text: str) -> float:
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z]+", text)]
    pos = sum(1 for t in tokens if t in LM_POS)
    neg = sum(1 for t in tokens if t in LM_NEG)
    total = pos + neg
    if total == 0:
        return 0.0
    return max(-0.25, min(0.25, 0.25 * (pos - neg) / total))

analyzer = SentimentIntensityAnalyzer()

def rescore_text(headline: str, summary: str, src_token: str, via_token: str | None = None) -> float:
    text = " ".join([headline or "", summary or ""]).strip()
    if not text:
        return 0.0
    comp = analyzer.polarity_scores(text).get("compound", 0.0)
    comp *= 1.5
    # speedup idioms not included here — keep parity minimally
    comp += lm_overlay_score(text)
    comp = max(-1.0, min(1.0, comp))
    src = norm_source(src_token)
    via = norm_source(via_token) if via_token else ""
    w = NEWS_SOURCE_WEIGHTS_DEFAULT.get(src, NEWS_SOURCE_WEIGHTS_DEFAULT.get(via, 1.0))
    return max(-1.0, min(1.0, comp * float(w)))

def main():
    print(f"[INFO] backing up {IN_PATH} -> {BACKUP}")
    shutil.copy2(IN_PATH, BACKUP)

    obj = json.loads(IN_PATH.read_text(encoding="utf-8"))
    syms = obj.get("symbols", {}) or {}

    total = 0
    changed = 0
    for sym, node in syms.items():
        news = node.get("news") or []
        total += len(news)
        for n in news:
            headline = n.get("headline") or ""
            summary = n.get("summary") or ""
            src = n.get("source") or ""
            via = n.get("via") or ""
            # recompute raw tone (no time-decay here)
            try:
                tone = rescore_text(headline, summary, src, via)
            except Exception:
                tone = 0.0
            n["tone"] = tone
            changed += 1

    # write file back (overwriting)
    IN_PATH.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"[OK] rescored {changed} news items (total before: {total}). Backup at {BACKUP}")

if __name__ == "__main__":
    main()