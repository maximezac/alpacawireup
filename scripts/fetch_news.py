#!/usr/bin/env python3
"""
scripts/fetch_news.py

Reads data/prices.json (or INPUT_PATH), fetches recent news from Alpaca v1beta1/news
for each symbol, attaches to the ticker's "news" array, and writes back to the same file.

Required env:
- ALPACA_KEY_ID
- ALPACA_SECRET_KEY

Optional env:
- INPUT_PATH  (default: data/prices.json)
- OUTPUT_PATH (default: INPUT_PATH)
- NEWS_LOOKBACK_DAYS         (default: 7)
- NEWS_LIMIT_PER_SYMBOL      (default: 25)
- NEWS_SOURCES               (comma-separated; e.g. "benzinga,mtnewswires,google_rss,finnhub,reddit")
- RECENCY_HALFLIFE_HOURS     (default: 36)  # relevance recency decay half-life
- MAX_ARTICLES_TOTAL         (default: 50)  # final per-symbol cap after dedupe/filter
"""

from __future__ import annotations

import os, sys, re, json, math
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import List, Dict, Any
from urllib.parse import urlparse

import requests
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -----------------------------
# Config / ENV
# -----------------------------
API_NEWS = "https://data.alpaca.markets/v1beta1/news"

ALPACA_KEY_ID     = os.environ.get("ALPACA_KEY_ID")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
if not ALPACA_KEY_ID or not ALPACA_SECRET_KEY:
    print("Missing ALPACA_KEY_ID / ALPACA_SECRET_KEY", file=sys.stderr)
    sys.exit(1)

INPUT_PATH  = os.environ.get("INPUT_PATH",  "data/prices.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", INPUT_PATH)
LOOKBACK_DAYS = int(os.environ.get("NEWS_LOOKBACK_DAYS", "7"))
NEWS_LIMIT    = int(os.environ.get("NEWS_LIMIT_PER_SYMBOL", "25"))
NEWS_SOURCES  = os.environ.get("NEWS_SOURCES", "").strip()
RECENCY_HALFLIFE_HOURS = float(os.environ.get("RECENCY_HALFLIFE_HOURS", "36"))
MAX_ARTICLES_TOTAL     = int(os.environ.get("MAX_ARTICLES_TOTAL", "50"))

headers = {
    "APCA-API-KEY-ID": ALPACA_KEY_ID,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

# -----------------------------
# Weights / heuristics
# -----------------------------

# Baseline source weights; google_rss kept reasonably high per request.
NEWS_SOURCE_WEIGHTS = {
    "finnhub":     0.90,
    "mtnewswires": 0.95,
    "benzinga":    0.80,
    "google_rss":  0.60,   # fallback when we can't map to the real publisher
    "newsapi":     0.30,
    "reddit":      0.25,
}

# Per-publisher overrides when we can infer the true site from Google RSS links
PUBLISHER_WEIGHTS = {
    "yahoo":        0.80,
    "bloomberg":    0.95,
    "reuters":      0.98,
    "wsj":          0.95,
    "barrons":      0.90,
    "seekingalpha": 0.70,
    "tipranks":     0.60,
    "motleyfool":   0.60,
    "marketwatch":  0.75,
    "investors":    0.75,  # IBD
    "nasdaq":       0.75,
}

PUBLISHER_DOMAIN_MAP = {
    "finance.yahoo.com": "yahoo",
    "yahoo.com": "yahoo",
    "bloomberg.com": "bloomberg",
    "reuters.com": "reuters",
    "wsj.com": "wsj",
    "barrons.com": "barrons",
    "seekingalpha.com": "seekingalpha",
    "tipranks.com": "tipranks",
    "fool.com": "motleyfool",
    "investors.com": "investors",
    "marketwatch.com": "marketwatch",
    "nasdaq.com": "nasdaq",
}

# Headline heuristics
NEG_TRIGGERS = re.compile(
    r"\b(drops?|down\s?\d+%|plunge[sd]?|tumble[sd]?|fall[s]?\b|sell[\s-]?off|downgraded?|cut to neutral|miss|warning|halted?)\b",
    re.IGNORECASE
)
POS_TRIGGERS = re.compile(
    r"\b(beat[s]?|surge[sd]?|soar[s]?|raises? guidance|upgrade[sd]?|wins? (deal|contract)|available|launch(?:es|ed)?)\b",
    re.IGNORECASE
)
LISTICLE_CLICKBAIT = re.compile(r"(which|should you buy|millionaire|best|top\s?\d+|\?$)", re.IGNORECASE)

SYMBOL_ORGS = {
    # Help specificity detection for synonyms/companies
    "QBTS": [r"\bQBTS\b", r"\bD-?Wave\b", r"\bDwave\b", r"\bD Wave\b"],
    # Add more per symbol here if you like
}

analyzer = SentimentIntensityAnalyzer()
_src_norm_re = re.compile(r"[^a-z0-9]+")

def norm_source(s: str) -> str:
    if not s:
        return "unknown"
    return _src_norm_re.sub("", s.lower())

def strip_html(text: str) -> str:
    if not text:
        return ""
    # Remove <...> tags and unescape the most common HTML entities
    no_tags = re.sub(r"<[^>]+>", "", text)
    return (no_tags.replace("&nbsp;", " ")
                   .replace("&amp;", "&")
                   .replace("&quot;", '"')
                   .replace("&#39;", "'")
                   .replace("&lt;", "<")
                   .replace("&gt;", ">")).strip()

def to_utc_iso(ts: str | None) -> str | None:
    if not ts:
        return None
    try:
        try:
            dt = parsedate_to_datetime(ts)
        except Exception:
            from dateutil import parser
            dt = parser.isoparse(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None

def recency_weight(ts_iso: str, ref: datetime) -> float:
    try:
        from dateutil import parser
        dt = parser.isoparse(ts_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        hours = (ref - dt.astimezone(timezone.utc)).total_seconds() / 3600.0
        # exponential half-life decay
        return 0.5 ** max(0.0, hours / RECENCY_HALFLIFE_HOURS)
    except Exception:
        return 1.0

def specificity_weight(symbol: str, headline: str, summary: str) -> float:
    pats = SYMBOL_ORGS.get(symbol, [rf"\b{re.escape(symbol)}\b"])
    text = f"{headline} {summary}"
    for p in pats:
        if re.search(p, text, re.IGNORECASE):
            return 1.0
    # If competitors in title but not our symbol, de-weight
    if re.search(r"\b(IONQ|RGTI|QUBT)\b", text, re.IGNORECASE):
        return 0.6
    # Sector-level generic
    if re.search(r"\b(quantum( computing)? stocks?)\b", text, re.IGNORECASE):
        return 0.4
    return 0.6

def map_domain_to_publisher(url: str) -> str | None:
    try:
        netloc = urlparse(url).netloc.lower()
        # strip www.
        if netloc.startswith("www."):
            netloc = netloc[4:]
        # direct mapping
        if netloc in PUBLISHER_DOMAIN_MAP:
            return PUBLISHER_DOMAIN_MAP[netloc]
        # collapse subdomains to base
        parts = netloc.split(".")
        base = ".".join(parts[-2:]) if len(parts) >= 2 else netloc
        for k, v in PUBLISHER_DOMAIN_MAP.items():
            if k.endswith(base):
                return v
        # fallback: token from domain
        token = base.split(".")[0]
        return norm_source(token)
    except Exception:
        return None

def base_weight_for_source(src: str) -> float:
    return NEWS_SOURCE_WEIGHTS.get(src, 0.60 if src == "google_rss" else 0.70)

def score_tone_finance_aware(headline: str, summary: str, src_token: str) -> float:
    """VADER compound with finance-aware adjustments and source weight."""
    text = " ".join([headline or "", summary or ""]).strip()
    if not text:
        return 0.0
    comp = analyzer.polarity_scores(text).get("compound", 0.0)

    # Heuristics
    title = (headline or "")
    lower = title.lower()

    # % move detection with sign verbs
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s?%", lower)
    if pct_match:
        if re.search(r"\b(drop|fell|fall|down|plunge|tumble|decline|slide|sink)\b", lower):
            comp = min(comp, -0.25)
        elif re.search(r"\b(rise|rose|up|surge|soar|jump|rally)\b", lower):
            comp = max(comp, 0.25)

    if NEG_TRIGGERS.search(title):
        comp = min(comp, -0.25)
    if POS_TRIGGERS.search(title):
        comp = max(comp, 0.25)
    if LISTICLE_CLICKBAIT.search(title):
        comp *= 0.5

    # Source weight multiplier
    w = base_weight_for_source(src_token)
    comp *= w
    # clamp
    return max(-1.0, min(1.0, comp))

# -----------------------------
# Fetchers
# -----------------------------
def fetch_news_for_symbol(symbol: str, start_iso: str, end_iso: str) -> List[Dict[str, Any]]:
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
    items = data.get("news") or data.get("data") or []

    # Manual filter on NEWS_SOURCES if provided
    allowed = []
    if NEWS_SOURCES:
        allowed = [norm_source(s) for s in NEWS_SOURCES.split(",") if s.strip()]

    out = []
    for it in items:
        headline = it.get("headline") or it.get("title") or ""
        summary  = it.get("summary")  or it.get("description") or ""
        ts_raw   = it.get("created_at") or it.get("updated_at") or it.get("published_at")
        ts       = to_utc_iso(ts_raw)
        source   = norm_source(it.get("source") or it.get("author") or "unknown")
        url      = it.get("url")

        if allowed and source not in allowed:
            continue

        out.append({
            "headline": headline,
            "summary": summary,
            "ts": ts,
            "source": source,
            "url": url,
        })
    return out

def get_news_from_newsapi(symbol: str, api_key: str) -> list[dict]:
    url = "https://newsapi.org/v2/everything"
    params = {"q": symbol, "apiKey": api_key, "sortBy": "publishedAt", "language": "en", "pageSize": 10}
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        return []
    out = []
    for a in r.json().get("articles", []):
        out.append({
            "headline": a.get("title"),
            "summary": a.get("description") or "",
            "ts": a.get("publishedAt"),
            "source": "newsapi",
            "url": a.get("url"),
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
    out = []
    for it in r.json():
        out.append({
            "headline": it.get("headline"),
            "summary": it.get("summary") or "",
            "ts": datetime.utcfromtimestamp(it.get("datetime")).replace(tzinfo=timezone.utc).isoformat()
                  if it.get("datetime") else None,
            "source": "finnhub",
            "url": it.get("url"),
        })
    return out

def get_news_from_google(symbol: str) -> list[dict]:
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

        # Map publisher from link if possible
        link = entry.link
        pub = map_domain_to_publisher(link) or "google_rss"

        out.append({
            "headline": entry.title,
            "summary": strip_html(entry.get("summary", "")),
            "ts": ts,
            "source": pub,   # key change: use real publisher token when possible
            "url": link
        })
    return out

def get_news_from_reddit(symbol: str) -> list[dict]:
    url = f"https://www.reddit.com/search.json?q={symbol}&limit=25&t=day"
    try:
        r = requests.get(url, headers={"User-Agent": "alpacawireup/1.0"}, timeout=15)
        if r.status_code != 200:
            return []
        posts = r.json().get("data", {}).get("children", [])
        out = []
        for p in posts:
            d = p.get("data", {})
            tiso = datetime.utcfromtimestamp(d.get("created_utc", 0)).replace(tzinfo=timezone.utc).isoformat()
            out.append({
                "headline": d.get("title"),
                "summary": d.get("selftext") or "",
                "ts": tiso,
                "source": "reddit",
                "url": f"https://reddit.com{d.get('permalink','')}",
            })
        return out
    except Exception as e:
        print(f"[WARN] Reddit fetch failed for {symbol}: {e}")
        return []

# -----------------------------
# Main
# -----------------------------
def main():
    prices = json.load(open(INPUT_PATH, "r", encoding="utf-8"))
    now = datetime.now(timezone.utc)
    start_iso = (now - timedelta(days=LOOKBACK_DAYS)).isoformat()
    end_iso   = now.isoformat()

    symbols = prices.get("symbols") or {}
    if not symbols:
        print(f"[WARN] No symbols in {INPUT_PATH}; nothing to do.", file=sys.stderr)
        json.dump(prices, open(OUTPUT_PATH, "w", encoding="utf-8"), indent=2)
        return

    # Parse optional list for non-Alpaca adapters
    sources = [s.strip().lower() for s in NEWS_SOURCES.split(",") if s.strip()]
    use_google = "google_rss" in sources
    use_reddit = "reddit" in sources

    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
    FINNHUB_KEY = os.getenv("FINNHUB_KEY")

    total_attached = 0
    raw_seen_total = 0
    kept_total = 0

    for sym, node in symbols.items():
        try:
            # --- Fetch from Alpaca
            alpaca_news = fetch_news_for_symbol(sym, start_iso, end_iso)

            # --- Extra sources
            extra_news = []
            if NEWSAPI_KEY:
                extra_news += get_news_from_newsapi(sym, NEWSAPI_KEY)
            if FINNHUB_KEY:
                extra_news += get_news_from_finnhub(sym, FINNHUB_KEY)
            if use_google:
                extra_news += get_news_from_google(sym)
            if use_reddit:
                extra_news += get_news_from_reddit(sym)

            raw_items = alpaca_news + extra_news
            raw_seen_total += len(raw_items)

            # --- Annotate tone + relevance
            annotated = []
            for a in raw_items:
                headline = a.get("headline") or ""
                summary  = a.get("summary")  or ""
                ts_iso   = a.get("ts")
                source   = norm_source(a.get("source") or "unknown")
                url      = a.get("url")

                tone = score_tone_finance_aware(headline, summary, source)
                # recency + specificity
                rw = recency_weight(ts_iso, now) if ts_iso else 1.0
                sw = specificity_weight(sym, headline, summary)
                base_w = base_weight_for_source(source)
                relevance = base_w * rw * sw

                annotated.append({
                    "headline": headline,
                    "summary": summary,
                    "ts": ts_iso,
                    "source": source,
                    "url": url,
                    "tone": round(float(tone), 6),
                    "relevance": round(float(relevance), 6),
                })

            # --- Dedupe by (headline_lower, source)
            seen = set()
            deduped = []
            for a in annotated:
                key = (a["headline"].strip().lower(), a["source"])
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(a)

            # --- Sort newest first
            from dateutil import parser
            def _safe_parse(x):
                try:
                    d = parser.isoparse(x)
                    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
                except Exception:
                    return datetime.min.replace(tzinfo=timezone.utc)

            deduped.sort(key=lambda x: _safe_parse(x.get("ts") or ""), reverse=True)

            # --- Cap and stale cut
            cutoff = now - timedelta(days=LOOKBACK_DAYS)
            filtered = [a for a in deduped if (a.get("ts") and _safe_parse(a["ts"]) >= cutoff)]
            kept = filtered[:MAX_ARTICLES_TOTAL]
            kept_total += len(kept)

            node["news"] = kept
            total_attached += len(kept)

            print(f"   [•] {sym}: kept {len(kept)} (raw={len(raw_items)})")

        except Exception as e:
            print(f"[WARN] {sym}: news fetch failed: {e}", file=sys.stderr)
            node.setdefault("news", [])

    prices["as_of_utc"] = now.isoformat()
    json.dump(prices, open(OUTPUT_PATH, "w", encoding="utf-8"), indent=2)
    print(f"[DONE] Wrote news into {OUTPUT_PATH} (attached={total_attached}, raw_seen={raw_seen_total}, kept_total={kept_total})")

if __name__ == "__main__":
    main()
