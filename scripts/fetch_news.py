#!/usr/bin/env python3
"""
scripts/fetch_news.py
Reads data/prices.json (or INPUT_PATH), fetches recent news from Alpaca v1beta1/news
for each symbol, attaches to the ticker's "news" array, and writes back to the same file.

Env (required):
- ALPACA_KEY_ID
- ALPACA_SECRET_KEY

Env (optional):
- INPUT_PATH:  default "data/prices.json"
- OUTPUT_PATH: default "data/prices.json" (in-place update)
- NEWS_LOOKBACK_DAYS: default "7"
- NEWS_LIMIT_PER_SYMBOL: default "25"
- NEWS_SOURCES: optional comma-separated (e.g., "benzinga,mtnewswires")  # leave empty to use all
"""
import os, sys, json
from datetime import datetime, timedelta, timezone
import requests

API_NEWS = "https://data.alpaca.markets/v1beta1/news"

ALPACA_KEY_ID = os.environ.get("ALPACA_KEY_ID")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

INPUT_PATH  = os.environ.get("INPUT_PATH", "data/prices.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", INPUT_PATH)
LOOKBACK_DAYS = int(os.environ.get("NEWS_LOOKBACK_DAYS", "7"))
NEWS_LIMIT = int(os.environ.get("NEWS_LIMIT_PER_SYMBOL", "25"))
NEWS_SOURCES = os.environ.get("NEWS_SOURCES", "").strip()  # optional

if not ALPACA_KEY_ID or not ALPACA_SECRET_KEY:
    print("Missing ALPACA_KEY_ID / ALPACA_SECRET_KEY", file=sys.stderr)
    sys.exit(1)

headers = {
    "APCA-API-KEY-ID": ALPACA_KEY_ID,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

def fetch_news_for_symbol(symbol: str, start_iso: str, end_iso: str):
    """
    Fetch up to NEWS_LIMIT stories between start and end for a single symbol.
    Returns a list of {headline, summary, ts, source, relevance}.
    """
    params = {
        "symbols": symbol,
        "start": start_iso,
        "end": end_iso,
        "limit": NEWS_LIMIT,
        "sort": "desc",
        # "include_content": "false",  # (default false, responses include headline+summary/description)
    }
    if NEWS_SOURCES:
        params["source"] = NEWS_SOURCES  # comma-separated string

    r = requests.get(API_NEWS, headers=headers, params=params, timeout=20)
    if r.status_code != 200:
        raise requests.HTTPError(f"{r.status_code} {r.text}")

    data = r.json() or {}
    items = data.get("news") or data.get("data") or []  # API schemas sometimes differ
    out = []
    for it in items:
        # Alpaca fields: "headline", "summary" (sometimes "summary" or "description"), "created_at", "source"
        headline = it.get("headline") or it.get("title") or ""
        summary  = it.get("summary") or it.get("description") or ""
        ts       = it.get("created_at") or it.get("updated_at") or it.get("published_at")
        source   = it.get("source") or it.get("author") or "unknown"

        # Skip empty headlines
        if not (headline.strip() or summary.strip()):
            continue

        out.append({
            "headline": headline,
            "summary": summary,
            "ts": ts,
            "source": source,
            "relevance": 1.0,   # publish_feed.py will weight & decay; Alpaca doesn't give per-ticker relevance.
        })
    return out

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        prices = json.load(f)

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=LOOKBACK_DAYS)
    start_iso = start.isoformat()
    end_iso = now.isoformat()

    symbols = prices.get("symbols", {})
    if not symbols:
        print(f"[WARN] No symbols in {INPUT_PATH}; nothing to do.", file=sys.stderr)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(prices, f, indent=2)
        return

    for sym, node in symbols.items():
        try:
            news_items = fetch_news_for_symbol(sym, start_iso, end_iso)
            # Attach/overwrite news array
            node["news"] = news_items
            print(f"[OK] {sym}: {len(news_items)} news items")
        except Exception as e:
            print(f"[WARN] {sym}: news fetch failed: {e}", file=sys.stderr)
            # Leave existing news as-is (or ensure it's at least an empty list)
            node.setdefault("news", [])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(prices, f, indent=2)
    print(f"[DONE] Wrote news into {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
