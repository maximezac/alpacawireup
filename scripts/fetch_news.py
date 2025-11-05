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
import os, sys, json, re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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

analyzer = SentimentIntensityAnalyzer()

_src_norm_re = re.compile(r"[^a-z0-9]+")
def norm_source(s: str) -> str:
    """Normalize a source string to a lowercase token without spaces/punct, e.g. 'MT Newswires' -> 'mtnewswires'."""
    if not s:
        return "unknown"
    return _src_norm_re.sub("", s.lower())

def to_utc_iso(ts: str | None) -> str | None:
    if not ts:
        return None
    try:
        # Try robust parse (RFC3339/ISO) with fallback to email/date style
        try:
            dt = parsedate_to_datetime(ts)
        except Exception:
            # last resort: dateutil if available via publish step; else naive parse
            from dateutil import parser
            dt = parser.isoparse(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None

def score_tone(headline: str, summary: str, src_token: str) -> float:
    """Return tone in [-1, 1], VADER compound * source weight."""
    text = " ".join([headline or "", summary or ""]).strip()
    if not text:
        return 0.0
    comp = analyzer.polarity_scores(text).get("compound", 0.0)
    w = NEWS_SOURCE_WEIGHTS.get(src_token, 1.0)
    val = comp * w
    # clamp
    return max(-1.0, min(1.0, val))

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
        source_filter = [norm_source(s) for s in NEWS_SOURCES.split(",") if s.strip()]

    out = []
    for it in items:
        headline = it.get("headline") or it.get("title") or ""
        summary = it.get("summary") or it.get("description") or ""
        ts_raw = it.get("created_at") or it.get("updated_at") or it.get("published_at")
        ts = to_utc_iso(ts_raw)
        source_raw = it.get("source") or it.get("author") or "unknown"
        source = norm_source(source_raw)

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
            "tone": score_tone(headline, summary, source),
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
        ts = None
        if getattr(entry, "published_parsed", None):
            ts = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
        else:
            ts = to_utc_iso(entry.get("published", ""))
        out.append({
            "headline": entry.title,
            "summary": entry.get("summary", ""),
            "ts": ts,
            "source": "google_rss",
            "url": entry.link,
            "tone": score_tone(entry.title, entry.get("summary",""), "google_rss"),
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
            title = data.get("title") or ""
            body = data.get("selftext") or ""
            tiso = datetime.utcfromtimestamp(data.get("created_utc", 0)).replace(tzinfo=timezone.utc).isoformat()
            out.append({
                "headline": data.get("title"),
                "summary": body,
                "ts": tiso,
                "source": "reddit",
                "url": f"https://reddit.com{data.get('permalink','')}",
                "relevance": 0.8 if data.get("score", 0) > 50 else 0.4,
                "tone": score_tone(title, body, "reddit"),
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
                    dt = parser.isoparse(ts)
                    # 🔧 Ensure timezone-aware datetime (UTC fallback)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except Exception:
                    return datetime.min.replace(tzinfo=timezone.utc)

            deduped.sort(key=lambda x: _safe_parse(x.get("ts") or x.get("time") or ""), reverse=True)

            # Apply global cap per symbol
            MAX_ARTICLES_TOTAL = 50
            news_items = deduped[:MAX_ARTICLES_TOTAL]

            # Optional: drop stale articles older than lookback window (redundant safeguard)
            cutoff = datetime.now(timezone.utc).replace(tzinfo=timezone.utc) - timedelta(days=LOOKBACK_DAYS)
            filtered = []
            for a in news_items:
                ts = _safe_parse(a.get("ts") or a.get("time") or "")
                if ts >= cutoff:
                    filtered.append(a)
                else:
                    print(f"[DEBUG] Dropped old {a.get('source')} article for {sym}: {a.get('ts')}")
            news_items = filtered


            # ----------------------------------------------------------
            # ✅ Summary print for debugging
            # ----------------------------------------------------------
            print(f"   [•] {sym}: {len(news_items)} deduped/filtered (raw={len(alpaca_news)+len(extra_news)})")
            

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
    total_news = sum(len((n or {}).get("news") or []) for n in prices.get("symbols", {}).values())
    print(f"[DONE] Wrote news into {OUTPUT_PATH} (total items: {total_news})")

if __name__ == "__main__":
    main()
