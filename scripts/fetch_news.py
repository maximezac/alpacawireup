#!/usr/bin/env python3
"""
scripts/fetch_news.py

Reads INPUT_PATH (default: data/prices.json), fetches recent news per symbol
(Alpaca + optional adapters), attaches to each ticker's "news" array, and
writes back to OUTPUT_PATH (default: same file).

Required env:
- ALPACA_KEY_ID
- ALPACA_SECRET_KEY

Optional env:
- INPUT_PATH:  default "data/prices.json"
- OUTPUT_PATH: default "data/prices.json" (in-place)
- NEWS_LOOKBACK_DAYS: default "7"
- NEWS_LIMIT_PER_SYMBOL: default "25"
- NEWS_SOURCES: comma-separated adapters/sources to allow/filter, e.g.
    "benzinga,mtnewswires,google_rss,finnhub,reddit,newsapi"
  Notes:
    - For Alpaca items, we client-side filter by normalized `source`.
    - For adapters: include "google_rss", "reddit", "newsapi", "finnhub" to enable them.
- NEWSAPI_KEY: optional (enables NewsAPI adapter)
- FINNHUB_KEY: optional (enables Finnhub adapter)
"""

from __future__ import annotations

import os
import sys
import re
import json
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List

import requests
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dateutil import parser as dtparser

API_NEWS = "https://data.alpaca.markets/v1beta1/news"

ALPACA_KEY_ID = os.environ.get("ALPACA_KEY_ID", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

INPUT_PATH = os.environ.get("INPUT_PATH", "data/prices.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", INPUT_PATH)
LOOKBACK_DAYS = int(os.environ.get("NEWS_LOOKBACK_DAYS", "7"))
NEWS_LIMIT = int(os.environ.get("NEWS_LIMIT_PER_SYMBOL", "25"))
NEWS_SOURCES = os.environ.get("NEWS_SOURCES", "").strip()  # adapters + allowlist for Alpaca sources

if not ALPACA_KEY_ID or not ALPACA_SECRET_KEY:
    print("Missing ALPACA_KEY_ID / ALPACA_SECRET_KEY", file=sys.stderr)
    sys.exit(1)

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY_ID,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

# Tunable per-source weights used to scale sentiment tone
NEWS_SOURCE_WEIGHTS: Dict[str, float] = {
    "finnhub": 1.00,
    "benzinga": 0.90,
    "mtnewswires": 0.95,
    "google_rss": 0.50,
    "googlerss": 0.50,   # normalized alias
    "reddit": 0.35,
    "newsapi": 0.30,
}

analyzer = SentimentIntensityAnalyzer()
_src_norm_re = re.compile(r"[^a-z0-9]+")


def norm_source(s: str) -> str:
    """Normalize to lowercase token w/o spaces/punct: 'MT Newswires' -> 'mtnewswires'."""
    if not s:
        return "unknown"
    return _src_norm_re.sub("", s.lower())


def to_utc_iso(ts: str | None) -> str | None:
    if not ts:
        return None
    try:
        # Try RFC2822 first, then ISO
        try:
            dt = parsedate_to_datetime(ts)
        except Exception:
            dt = dtparser.isoparse(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def score_tone(headline: str, summary: str, src_token: str) -> float:
    """Return tone in [-1, 1]. VADER compound * per-source weight."""
    text = " ".join([headline or "", summary or ""]).strip()
    if not text:
        return 0.0
    comp = analyzer.polarity_scores(text).get("compound", 0.0)
    w = NEWS_SOURCE_WEIGHTS.get(src_token, NEWS_SOURCE_WEIGHTS.get(norm_source(src_token), 1.0))
    val = comp * w
    return max(-1.0, min(1.0, val))


def fetch_news_for_symbol(symbol: str, start_iso: str, end_iso: str, allow_sources: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch up to NEWS_LIMIT Alpaca stories for a symbol between start/end.
    Returns normalized items with tone.
    """
    params = {
        "symbols": symbol,
        "start": start_iso,
        "end": end_iso,
        "limit": NEWS_LIMIT,
        "sort": "desc",
        # IMPORTANT: do not send 'source' (singular). If you want server-side filter,
        # you could try 'sources' (plural) CSV; many accounts don't support it reliably.
    }

    r = requests.get(API_NEWS, headers=HEADERS, params=params, timeout=20)
    if r.status_code != 200:
        raise requests.HTTPError(f"{r.status_code} {r.text}")

    data = r.json() or {}
    items = data.get("news") or data.get("data") or []

    out: List[Dict[str, Any]] = []
    for it in items:
        headline = it.get("headline") or it.get("title") or ""
        summary = it.get("summary") or it.get("description") or ""
        ts_raw = it.get("created_at") or it.get("updated_at") or it.get("published_at")
        ts = to_utc_iso(ts_raw)
        source_raw = it.get("source") or it.get("author") or "unknown"
        source = norm_source(source_raw)

        if not (headline.strip() or summary.strip()):
            continue

        # Client-side filter by allowlist (if provided)
        if allow_sources and source not in allow_sources:
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


# ---- Additional adapters -----------------------------------------------------

def get_news_from_newsapi(symbol: str, api_key: str) -> List[Dict[str, Any]]:
    url = "https://newsapi.org/v2/everything"
    params = {"q": symbol, "apiKey": api_key, "sortBy": "publishedAt", "language": "en", "pageSize": 10}
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json().get("articles", [])
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for a in data:
        title = a.get("title") or ""
        desc = a.get("description") or ""
        ts = to_utc_iso(a.get("publishedAt"))
        src_token = "newsapi"
        out.append({
            "headline": title,
            "summary": desc,
            "ts": ts,
            "source": src_token,
            "relevance": 1.0,
            "tone": score_tone(title, desc, src_token),
        })
    return out


def get_news_from_finnhub(symbol: str, api_key: str) -> List[Dict[str, Any]]:
    now = datetime.utcnow().date()
    start = (now - timedelta(days=LOOKBACK_DAYS)).isoformat()
    end = now.isoformat()
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start}&to={end}&token={api_key}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return []
        items = r.json()
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for it in items:
        head = it.get("headline") or ""
        summ = it.get("summary") or ""
        ts = None
        if it.get("datetime") is not None:
            ts = datetime.utcfromtimestamp(it["datetime"]).replace(tzinfo=timezone.utc).isoformat()
        src_token = "finnhub"
        out.append({
            "headline": head,
            "summary": summ,
            "ts": ts,
            "source": src_token,
            "relevance": 1.0,
            "tone": score_tone(head, summ, src_token),
        })
    return out


def get_news_from_google(symbol: str) -> List[Dict[str, Any]]:
    query = f"{symbol}+stock"
    feed_url = f"https://news.google.com/rss/search?q={query}"
    try:
        feed = feedparser.parse(feed_url)
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for entry in feed.entries[:10]:
        title = getattr(entry, "title", "") or ""
        summary = entry.get("summary", "") if isinstance(entry, dict) else getattr(entry, "summary", "") or ""
        if getattr(entry, "published_parsed", None):
            ts = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
        else:
            ts = to_utc_iso(entry.get("published") if isinstance(entry, dict) else getattr(entry, "published", None))
        src_token = "google_rss"
        out.append({
            "headline": title,
            "summary": summary,
            "ts": ts,
            "source": src_token,
            "url": entry.link if hasattr(entry, "link") else entry.get("link", ""),
            "relevance": 1.0,
            "tone": score_tone(title, summary, src_token),
        })
    return out


def get_news_from_reddit(symbol: str) -> List[Dict[str, Any]]:
    url = f"https://www.reddit.com/search.json?q={symbol}&limit=25&t=day"
    headers = {"User-Agent": "alpacawireup/1.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return []
        posts = r.json().get("data", {}).get("children", [])
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for p in posts:
        data = p.get("data", {})
        title = data.get("title") or ""
        body = data.get("selftext") or ""
        tiso = datetime.utcfromtimestamp(data.get("created_utc", 0)).replace(tzinfo=timezone.utc).isoformat()
        score = data.get("score", 0)
        src_token = "reddit"
        out.append({
            "headline": title,
            "summary": body,
            "ts": tiso,
            "source": src_token,
            "url": f"https://reddit.com{data.get('permalink','')}",
            "relevance": 0.8 if score > 50 else 0.4,
            "tone": score_tone(title, body, src_token),
        })
    return out


def main() -> None:
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

    # Normalize NEWS_SOURCES list for:
    #  - Alpaca allowlist by source token
    #  - Which adapters to enable (google_rss, reddit, newsapi, finnhub)
    raw_sources = [s for s in (NEWS_SOURCES.split(",") if NEWS_SOURCES else []) if s.strip()]
    norm_sources = [norm_source(s) for s in raw_sources]
    allowlist = set(norm_sources)  # used to filter Alpaca items by source token
    enable_google = ("googlerss" in allowlist) or ("google_rss" in allowlist)
    enable_reddit = ("reddit" in allowlist)
    enable_newsapi = ("newsapi" in allowlist) and bool(os.getenv("NEWSAPI_KEY"))
    enable_finnhub = ("finnhub" in allowlist) and bool(os.getenv("FINNHUB_KEY"))

    total_raw = 0
    total_kept = 0

    for sym, node in symbols.items():
        try:
            # Alpaca news (client-side filtered by allowlist if provided)
            alpaca_items = fetch_news_for_symbol(sym, start_iso, end_iso, list(allowlist) if allowlist else [])
            total_raw += len(alpaca_items)

            extra_news: List[Dict[str, Any]] = []
            if enable_newsapi:
                extra_news += get_news_from_newsapi(sym, os.environ["NEWSAPI_KEY"])
            if enable_finnhub:
                extra_news += get_news_from_finnhub(sym, os.environ["FINNHUB_KEY"])
            if enable_google:
                extra_news += get_news_from_google(sym)
            if enable_reddit:
                extra_news += get_news_from_reddit(sym)

            news_items = alpaca_items + extra_news

            # ---- Deduplicate by (headline, source) case-insensitive
            seen = set()
            deduped: List[Dict[str, Any]] = []
            for a in news_items:
                key = (str(a.get("headline", "")).strip().lower(), str(a.get("source", "")).strip().lower())
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(a)

            # ---- Sort newest first
            def _safe_parse(ts: str | None) -> datetime:
                try:
                    dt = dtparser.isoparse(ts) if ts else None
                    if dt is None:
                        raise ValueError("no ts")
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except Exception:
                    return datetime.min.replace(tzinfo=timezone.utc)

            deduped.sort(key=lambda x: _safe_parse(x.get("ts")), reverse=True)

            # ---- Cap and drop outside lookback (safety)
            MAX_ARTICLES_TOTAL = 50
            deduped = deduped[:MAX_ARTICLES_TOTAL]
            cutoff = now - timedelta(days=LOOKBACK_DAYS)
            filtered: List[Dict[str, Any]] = []
            for a in deduped:
                dt = _safe_parse(a.get("ts"))
                if dt >= cutoff:
                    filtered.append(a)
                else:
                    # optional debug
                    pass

            total_kept += len(filtered)
            node["news"] = filtered

            raw_count = len(alpaca_items) + len(extra_news)
            print(f"   [•] {sym}: kept {len(filtered)} (raw={raw_count})")

        except Exception as e:
            print(f"[WARN] {sym}: news fetch failed: {e}", file=sys.stderr)
            node.setdefault("news", [])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(prices, f, indent=2)

    # Tiny summary
    total_attached = sum(len((n or {}).get("news") or []) for n in prices.get("symbols", {}).values())
    print(f"[DONE] Wrote news into {OUTPUT_PATH} (attached={total_attached}, raw_seen={total_raw}, kept_total={total_kept})")


if __name__ == "__main__":
    main()
