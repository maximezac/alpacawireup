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
from email.utils import parsedate_to_datetime
import requests
import feedparser

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

# ==========================================================
# Source weighting map (tunable)
# ==========================================================
NEWS_SOURCE_WEIGHTS = {
    "finnhub": 1.00,
    "benzinga": 0.90,
    "mtnewswires": 0.95,
    "google_rss": 0.50,
    "reddit": 0.35,
    "newsapi": 0.30
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
    }

    r = requests.get(API_NEWS, headers=headers, params=params, timeout=20)
    if r.status_code != 200:
        raise requests.HTTPError(f"{r.status_code} {r.text}")

    data = r.json() or {}
    items = data.get("news") or data.get("data") or []  # Alpaca sometimes varies schema

    # If user provided NEWS_SOURCES, filter manually
    source_filter = []
    if NEWS_SOURCES:
        source_filter = [s.strip().lower() for s in NEWS_SOURCES.split(",") if s.strip()]

    out = []
    for it in items:
        headline = it.get("headline") or it.get("title") or ""
        summary = it.get("summary") or it.get("description") or ""
        ts = it.get("created_at") or it.get("updated_at") or it.get("published_at")
        source = (it.get("source") or it.get("author") or "unknown").lower()

        # Skip empty headlines
        if not (headline.strip() or summary.strip()):
            continue

        # If filtering by NEWS_SOURCES, drop anything not in the list
        if source_filter and source not in source_filter:
            continue

        out.append({
            "headline": headline,
            "summary": summary,
            "ts": ts,
            "source": source,
            "relevance": 1.0,
        })

    return out

# ==========================================================
# Additional news adapters
# ==========================================================
def get_news_from_newsapi(symbol: str, api_key: str) -> list[dict]:
    url = "https://newsapi.org/v2/everything"
    params = {"q": symbol, "apiKey": api_key, "sortBy": "publishedAt", "language": "en", "pageSize": 10}
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        return []
    data = r.json().get("articles", [])
    out = []
    for a in data:
        out.append({
            "headline": a.get("title"),
            "summary": a.get("description") or "",
            "ts": a.get("publishedAt"),
            "source": "newsapi",
            "relevance": 1.0
        })
    return out

def get_news_from_finnhub(symbol: str, api_key: str) -> list[dict]:
    now = datetime.utcnow().date()
    start = (now - timedelta(days=LOOKBACK_DAYS)).isoformat()
    end = now.isoformat()
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start}&to={end}&token={api_key}"

    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        return []
    items = r.json()
    return [{
        "headline": it.get("headline"),
        "summary": it.get("summary") or "",
        "ts": datetime.utcfromtimestamp(it.get("datetime")).isoformat() if it.get("datetime") else None,
        "source": "finnhub",
        "relevance": 1.0
    } for it in items]

import feedparser

def get_news_from_google(symbol: str) -> list[dict]:
    """Fetch latest Google News RSS items for a symbol/company."""
    query = f"{symbol}+stock"
    feed_url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(feed_url)
    out = []
    for entry in feed.entries[:10]:
        out.append({
            "headline": entry.title,
            "summary": entry.get("summary", ""),
            "ts": entry.get("published_parsed") and datetime(*entry.published_parsed[:6]).isoformat() or entry.get("published", ""),
            "source": "google_rss",
            "url": entry.link
        })
    return out

def get_news_from_reddit(symbol: str) -> list[dict]:
    """Fetch recent Reddit posts mentioning the symbol."""
    url = f"https://www.reddit.com/search.json?q={symbol}&limit=25&t=day"
    headers = {"User-Agent": "alpacawireup/1.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return []
        posts = r.json().get("data", {}).get("children", [])
        out = []
        for p in posts:
            data = p.get("data", {})
            out.append({
                "headline": data.get("title"),
                "summary": data.get("selftext") or "",
                "ts": datetime.utcfromtimestamp(data.get("created_utc", 0)).isoformat(),
                "source": "reddit",
                "url": f"https://reddit.com{data.get('permalink','')}",
                "relevance": 0.8 if data.get("score", 0) > 50 else 0.4
            })
        return out
    except Exception as e:
        print(f"[WARN] Reddit fetch failed for {symbol}: {e}")
        return []


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
            alpaca_news = fetch_news_for_symbol(sym, start_iso, end_iso)

            extra_news = []
            NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
            FINNHUB_KEY = os.getenv("FINNHUB_KEY")

            sources = [s.strip().lower() for s in NEWS_SOURCES.split(",") if s.strip()]

            if NEWSAPI_KEY:
                extra_news += get_news_from_newsapi(sym, NEWSAPI_KEY)
            if FINNHUB_KEY:
                extra_news += get_news_from_finnhub(sym, FINNHUB_KEY)
            if "google_rss" in sources:
                extra_news += get_news_from_google(sym)
            if "reddit" in sources:
                extra_news += get_news_from_reddit(sym)



            news_items = alpaca_news + extra_news

            # ----------------------------------------------------------
            # 🧹 Deduplicate and trim news items
            # ----------------------------------------------------------
            # Remove duplicate headlines across all sources (case-insensitive)
            seen = set()
            deduped = []
            for a in news_items:
                key = (a.get("headline", "").strip().lower(), a.get("source", "").lower())
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(a)

            # Sort by timestamp (newest first)
            from dateutil import parser
            def _safe_parse(ts):
                try:
                    return parser.isoparse(ts)
                except Exception:
                    return datetime.min.replace(tzinfo=timezone.utc)

            deduped.sort(key=lambda x: _safe_parse(x.get("ts") or x.get("time") or ""), reverse=True)

            # Apply global cap per symbol
            MAX_ARTICLES_TOTAL = 50
            news_items = deduped[:MAX_ARTICLES_TOTAL]

            # Optional: drop stale articles older than lookback window (redundant safeguard)
            cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
            filtered = []
            for a in news_items:
                ts = _safe_parse(a.get("ts") or a.get("time") or "")
                if ts >= cutoff:
                    filtered.append(a)
                if ts < cutoff:
                    print(f"[DEBUG] Dropped old {a.get('source')} article for {sym}: {a.get('ts')}")
            news_items = filtered

            # ----------------------------------------------------------
            # ✅ Summary print for debugging
            # ----------------------------------------------------------
            print(
                f"   [•] {sym}: {len(news_items)} deduped & filtered articles "
                f"(from {len(alpaca_news)+len(extra_news)} raw)"
            )
            

            # Attach/overwrite news array
            node["news"] = news_items
            print(f"[OK] {sym}: {len(news_items)} news items")
            if extra_news:
                print(f"   [+] Added {len(extra_news)} extra articles from non-Alpaca sources.")
        except Exception as e:
            print(f"[WARN] {sym}: news fetch failed: {e}", file=sys.stderr)
            # Leave existing news as-is (or ensure it's at least an empty list)
            node.setdefault("news", [])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(prices, f, indent=2)
    print(f"[DONE] Wrote news into {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
