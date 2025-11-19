#!/usr/bin/env python3
"""
scripts/fetch_news.py

Reads INPUT_PATH (default data/prices.json), fetches recent news from:
- Alpaca v1beta1/news,
- (optional) NewsAPI,
- (optional) Finnhub,
- Google News RSS,
- Reddit search,

Dedupes, sorts, trims, scores sentiment, and writes back to OUTPUT_PATH.

Env (required):
- ALPACA_KEY_ID
- ALPACA_SECRET_KEY

Env (optional):
- INPUT_PATH:  default "data/prices.json"
- OUTPUT_PATH: default same as INPUT_PATH
- NEWS_LOOKBACK_DAYS: default "7"
- NEWS_START_ISO: optional explicit ISO start (overrides lookback start if set)
- NEWS_END_ISO:   optional explicit ISO end (defaults to now if not set)
- NEWS_LIMIT_PER_SYMBOL: default "25"  (per Alpaca API call)
- NEWS_MAX_ARTICLES_TOTAL: default "50"
    * If set to "0", there is **no cap** after dedupe (good for backtests).
- NEWS_SOURCES: comma list to keep (e.g. "benzinga,mtnewswires,google_rss,finnhub,reddit")
  * NOTE: Items include `source` (publisher token) and `via` (transport like "google_rss").
    Filtering allows either to match.
- NEWS_SOURCE_WEIGHTS_JSON: path to a JSON dict overriding source weights
- USE_FINBERT: "1" to enable optional FinBERT fallback on low-confidence
- FINBERT_TRIGGER: abs(VADER_score) below this (default 0.20) triggers fallback
- FINBERT_FRACTION: blend weight for finbert (default 0.30)
- NEWSAPI_KEY, FINNHUB_KEY

Backfill knobs:
- NEWS_BACKFILL: "1" to indicate backfill mode (you'll usually drive multiple date windows)
- BACKFILL_START: optional YYYY-MM-DD (hint for your loops; window is still controlled
                  by NEWS_START_ISO / NEWS_END_ISO or LOOKBACK_DAYS)

Output item fields per story:
- headline, summary, ts (ISO UTC), source (publisher token), via (optional),
- url (if available), relevance (float), tone ([-1,1])
"""
from __future__ import annotations
import os, sys, json, re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import math

import requests
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from vaderSentiment import vaderSentiment as vs

API_NEWS = "https://data.alpaca.markets/v1beta1/news"

# ---------- Env ----------
ALPACA_KEY_ID     = os.environ.get("ALPACA_KEY_ID")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

INPUT_PATH    = os.environ.get("INPUT_PATH", "data/prices.json")
OUTPUT_PATH   = os.environ.get("OUTPUT_PATH", INPUT_PATH)
LOOKBACK_DAYS = int(os.environ.get("NEWS_LOOKBACK_DAYS", "7"))
NEWS_LIMIT    = int(os.environ.get("NEWS_LIMIT_PER_SYMBOL", "25"))
NEWS_SOURCES  = os.environ.get("NEWS_SOURCES", "").strip()

# Backfill / volume knobs
NEWS_MAX_ARTICLES_TOTAL = int(os.environ.get("NEWS_MAX_ARTICLES_TOTAL", "50"))
NEWS_BACKFILL = os.environ.get("NEWS_BACKFILL", "0") == "1"
BACKFILL_START = os.environ.get("BACKFILL_START")  # e.g. "2021-01-01"
BACKFILL_END   = os.environ.get("BACKFILL_END")    # e.g. "2025-11-18"

# Optional explicit time window (for backfill / replay)
NEWS_START_ISO = os.environ.get("NEWS_START_ISO")
NEWS_END_ISO   = os.environ.get("NEWS_END_ISO")

# If running in BACKFILL mode, be conservative with caps and respect BACKFILL_START/END
if NEWS_BACKFILL:
    # Default to no cap when backfilling so historical windows accumulate fully
    # Only override the cap if the environment did not explicitly set it.
    if "NEWS_MAX_ARTICLES_TOTAL" not in os.environ:
        NEWS_MAX_ARTICLES_TOTAL = 0
    # Allow BACKFILL_START / BACKFILL_END to drive explicit window if NEWS_START_ISO/END not set
    if not NEWS_START_ISO and BACKFILL_START:
        # assume date only YYYY-MM-DD -> start of day
        NEWS_START_ISO = BACKFILL_START + "T00:00:00+00:00"
    if not NEWS_END_ISO and BACKFILL_END:
        NEWS_END_ISO = BACKFILL_END + "T23:59:59+00:00"



USE_FINBERT       = os.environ.get("USE_FINBERT", "0") == "1"
FINBERT_TRIGGER   = float(os.environ.get("FINBERT_TRIGGER", "0.30"))
FINBERT_FRACTION  = float(os.environ.get("FINBERT_FRACTION", "0.40"))
NEWSAPI_KEY       = os.environ.get("NEWSAPI_KEY")
FINNHUB_KEY       = os.environ.get("FINNHUB_KEY")
SRC_WEIGHTS_JSON  = os.environ.get("NEWS_SOURCE_WEIGHTS_JSON")  # optional path

if not ALPACA_KEY_ID or not ALPACA_SECRET_KEY:
    print("Missing ALPACA_KEY_ID / ALPACA_SECRET_KEY", file=sys.stderr)
    sys.exit(1)

# ---------- Helpers ----------
_src_norm_re = re.compile(r"[^a-z0-9]+")
def norm_source(s: str | None) -> str:
    if not s:
        return "unknown"
    return _src_norm_re.sub("", s.lower())

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

# ---------- Source weights ----------
NEWS_SOURCE_WEIGHTS_DEFAULT = {
    # publishers (normalized)
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

    # transports
    "googlerss": 0.70,
    "reddit": 0.35,
    "finnhub": 1.00,
    "newsapi": 0.30,
}

def load_source_weights():
    weights = dict(NEWS_SOURCE_WEIGHTS_DEFAULT)
    if SRC_WEIGHTS_JSON and os.path.exists(SRC_WEIGHTS_JSON):
        try:
            with open(SRC_WEIGHTS_JSON, "r", encoding="utf-8") as f:
                override = json.load(f) or {}
            for k, v in override.items():
                weights[norm_source(k)] = float(v)
        except Exception as e:
            print(f"[WARN] failed to load weights {SRC_WEIGHTS_JSON}: {e}", file=sys.stderr)
    return weights

NEWS_SOURCE_WEIGHTS = load_source_weights()

# ---------- VADER + domain fixes ----------
analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update({
    "efficiency": 1.6, "time-to-market": 1.8, "latency": -0.5,
    "downtime": -1.8, "cost-cutting": 0.8, "costs": -0.2,
    "profit-taking": -0.6, "beat": 1.2, "miss": -1.2,
    "guidance": 0.4, "dilution": -1.6, "impairment": -1.8,
    "spin-off": 0.6, "spinoff": 0.6, "divestiture": 0.4,
    "tailwind": 1.2, "headwind": -1.2,
    "merger": 0.6, "acquisition": 0.5, "acquire": 0.5,
    "raised guidance": 1.4, "lowered guidance": -1.4,
})
vs.SENTIMENT_LADEN_IDIOMS.update({
    "cut time": 2.2, "reduce latency": 2.2, "lower costs": 2.0,
    "from hours to seconds": 3.0, "from minutes to seconds": 2.6,
    "from days to hours": 2.2, "beat expectations": 2.5,
    "miss expectations": -2.5, "raises guidance": 2.2, "cuts guidance": -2.2,
    "tops estimates": 2.0, "beats estimates": 2.0,
    "misses estimates": -2.0, "falls short": -1.8
})
_SPEEDUP_RE = re.compile(
    r"(from\s+(\d+(\.\d+)?)(\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?))\s+to\s+(\d+(\.\d+)?)(\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)))",
    re.I,
)

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

_finbert = None
def finbert_score(text: str) -> float:
    global _finbert
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except Exception:
        return 0.0
    if _finbert is None:
        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _finbert = (tok, mdl)
    tok, mdl = _finbert
    inputs = tok(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = mdl(**inputs).logits.squeeze(0)
    probs = logits.softmax(dim=-1).tolist()  # [neg, neu, pos]
    return float(probs[2] - probs[0])  # [-1,1]

def _parse_iso_dt(ts: str | None):
    if not ts:
        return None
    try:
        from dateutil import parser
        dt = parser.isoparse(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def score_tone(headline: str, summary: str, ts_iso: str | None, src_token: str, via_token: str | None = None) -> float:
    text = " ".join([headline or "", summary or ""]).strip()
    if not text:
        return 0.0

    comp = analyzer.polarity_scores(text).get("compound", 0.0)
    comp *= 1.5

    if _SPEEDUP_RE.search(text):
        comp = max(comp, 0.35)

    comp += lm_overlay_score(text)
    comp = max(-1.0, min(1.0, comp))

    dt = _parse_iso_dt(ts_iso)
    if dt is not None:
        age_hours = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)
        decay = math.exp(-age_hours / 48.0)
        comp *= decay

    if USE_FINBERT and abs(comp) < FINBERT_TRIGGER:
        fb = finbert_score(headline + " " + summary)
        comp = (1 - FINBERT_FRACTION) * comp + FINBERT_FRACTION * fb
        comp = max(-1.0, min(1.0, comp))

    w = NEWS_SOURCE_WEIGHTS.get(src_token, NEWS_SOURCE_WEIGHTS.get(via_token or "", 1.0))
    return max(-1.0, min(1.0, comp * float(w)))

# ---------- Fetchers ----------
HEADERS_ALPACA = {
    "APCA-API-KEY-ID": ALPACA_KEY_ID,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

def fetch_news_for_symbol(symbol: str, start_iso: str, end_iso: str):
    params = {
        "symbols": symbol,
        "start": start_iso,
        "end": end_iso,
        "limit": NEWS_LIMIT,
        "sort": "desc",
    }
    r = requests.get(API_NEWS, headers=HEADERS_ALPACA, params=params, timeout=20)
    if r.status_code != 200:
        raise requests.HTTPError(f"{r.status_code} {r.text}")
    data = r.json() or {}
    items = data.get("news") or data.get("data") or []

    out = []
    for it in items:
        headline = it.get("headline") or it.get("title") or ""
        summary  = it.get("summary") or it.get("description") or ""
        ts       = to_utc_iso(it.get("created_at") or it.get("updated_at") or it.get("published_at"))
        publisher_raw = it.get("source") or it.get("author") or "unknown"
        source   = norm_source(publisher_raw)
        via      = None
        url      = it.get("url") or it.get("link")

        out.append({
            "headline": headline,
            "summary": summary,
            "ts": ts,
            "source": source,
            "via": via,
            "url": url,
            "relevance": 1.0,
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
        publisher = norm_source(((a.get("source") or {}).get("name")) or "newsapi")
        out.append({
            "headline": a.get("title"),
            "summary":  a.get("description") or "",
            "ts": to_utc_iso(a.get("publishedAt")),
            "source": publisher,
            "via": "newsapi",
            "url": a.get("url"),
            "relevance": 1.0,
        })
    return out

def get_news_from_finnhub(symbol: str, api_key: str, start_dt: datetime, end_dt: datetime) -> list[dict]:
    # Respect the same window as Alpaca (start_dt / end_dt)
    start = start_dt.date().isoformat()
    end = end_dt.date().isoformat()
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start}&to={end}&token={api_key}"
    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        return []
    items = r.json() or []
    out = []
    for it in items:
        publisher = norm_source(it.get("source") or "finnhub")
        out.append({
            "headline": it.get("headline"),
            "summary":  it.get("summary") or "",
            "ts": datetime.utcfromtimestamp(it.get("datetime", 0)).replace(tzinfo=timezone.utc).isoformat(),
            "source": publisher,
            "via": "finnhub",
            "url": it.get("url"),
            "relevance": 1.0,
        })
    return out

_PUBLISHER_SPLIT_RE = re.compile(r"\s+-\s+")
def _split_publisher_from_title(title: str) -> tuple[str, str | None]:
    parts = _PUBLISHER_SPLIT_RE.split(title.strip())
    if len(parts) >= 2:
        headline = " - ".join(parts[:-1]).strip()
        publisher = parts[-1].strip()
        return headline, publisher
    return title, None

def get_news_from_google(symbol: str) -> list[dict]:
    query = f"{symbol}+stock"
    feed_url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(feed_url)
    out = []
    for entry in feed.entries[:50]:
        title_raw = entry.title
        headline, publisher = _split_publisher_from_title(title_raw)
        publisher_token = norm_source(publisher) if publisher else "googlerss"

        if getattr(entry, "published_parsed", None):
            ts = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
        else:
            ts = to_utc_iso(getattr(entry, "published", None))

        out.append({
            "headline": headline,
            "summary": getattr(entry, "summary", "") or "",
            "ts": ts,
            "source": publisher_token,
            "via": "googlerss",
            "url": entry.link,
            "relevance": 1.0,
        })
    return out

def get_news_from_reddit(symbol: str) -> list[dict]:
    url = f"https://www.reddit.com/search.json?q={symbol}&limit=25&t=day"
    headers = {"User-Agent": "alpacawireup/1.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return []
        posts = r.json().get("data", {}).get("children", [])
        out = []
        for p in posts:
            d = p.get("data", {})
            title = d.get("title") or ""
            body  = d.get("selftext") or ""
            tiso  = datetime.utcfromtimestamp(d.get("created_utc", 0)).replace(tzinfo=timezone.utc).isoformat()
            out.append({
                "headline": title,
                "summary": body,
                "ts": tiso,
                "source": "reddit",
                "via": "reddit",
                "url": f"https://reddit.com{d.get('permalink','')}",
                "relevance": 0.8 if d.get("score", 0) > 50 else 0.4,
            })
        return out
    except Exception as e:
        print(f"[WARN] Reddit fetch failed for {symbol}: {e}", file=sys.stderr)
        return []

# ---------- Main ----------
def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        prices = json.load(f)

    # Determine time window
    if NEWS_START_ISO or NEWS_END_ISO:
        # Explicit window mode (backfill / custom)
        end_dt = _parse_iso_dt(NEWS_END_ISO) or datetime.now(timezone.utc)
        if NEWS_START_ISO:
            start_dt = _parse_iso_dt(NEWS_START_ISO)
        else:
            start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)
        start = start_dt
        end = end_dt
    else:
        # Default: last LOOKBACK_DAYS from "now"
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=LOOKBACK_DAYS)
        end = now

    start_iso = start.isoformat()
    end_iso   = end.isoformat()

    print(f"[INFO] Fetching news window {start_iso} → {end_iso} "
          f"(LOOKBACK_DAYS={LOOKBACK_DAYS}, MAX_TOTAL={NEWS_MAX_ARTICLES_TOTAL}, BACKFILL={NEWS_BACKFILL})")

    symbols = prices.get("symbols", {})
    if not symbols:
        print(f"[WARN] No symbols in {INPUT_PATH}; nothing to do.", file=sys.stderr)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(prices, f, indent=2)
        return

    source_filter = []
    if NEWS_SOURCES:
        source_filter = [norm_source(s) for s in NEWS_SOURCES.split(",") if s.strip()]

    kept_total = 0
    raw_seen   = 0

    for sym, node in symbols.items():
        try:
            bucket = []

            # 1) Existing news so multiple runs accumulate when backfilling
            existing = node.get("news") or []
            if existing:
                bucket.extend(existing)

            # 2) Alpaca core
            alpaca_news = fetch_news_for_symbol(sym, start_iso, end_iso)
            bucket += alpaca_news

            # 3) Extra sources
            if NEWSAPI_KEY:
                bucket += get_news_from_newsapi(sym, NEWSAPI_KEY)
            if FINNHUB_KEY:
                bucket += get_news_from_finnhub(sym, FINNHUB_KEY, start, end)
            if (not source_filter) or ("googlerss" in source_filter) or ("google_rss" in [s.replace("_","") for s in source_filter]):
                bucket += get_news_from_google(sym)
            if (not source_filter) or ("reddit" in source_filter):
                bucket += get_news_from_reddit(sym)

            raw_seen += len(bucket)

            # 4) Filter by NEWS_SOURCES against either publisher (source) or via
            if source_filter:
                tmp = []
                for a in bucket:
                    s = norm_source(a.get("source"))
                    v = norm_source(a.get("via"))
                    if (s in source_filter) or (v in source_filter):
                        tmp.append(a)
                bucket = tmp

            # 5) Deduplicate by (headline lower, publisher)
            seen = set()
            deduped = []
            for a in bucket:
                key = ((a.get("headline", "").strip().lower()), norm_source(a.get("source")))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(a)

            # 6) Score tone
            for a in deduped:
                src = norm_source(a.get("source"))
                via = norm_source(a.get("via"))
                a["tone"] = score_tone(a.get("headline", ""), a.get("summary", ""), a.get("ts"), src, via)

            # 7) Sort newest first
            from dateutil import parser
            def _safe_parse(ts):
                try:
                    dt = parser.isoparse(ts)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except Exception:
                    return datetime.min.replace(tzinfo=timezone.utc)

            deduped.sort(key=lambda x: _safe_parse(x.get("ts") or ""), reverse=True)

            # 8) Cap (or not) by NEWS_MAX_ARTICLES_TOTAL
            if NEWS_MAX_ARTICLES_TOTAL and NEWS_MAX_ARTICLES_TOTAL > 0:
                news_items = deduped[:NEWS_MAX_ARTICLES_TOTAL]
            else:
                news_items = deduped

            kept_total += len(news_items)
            print(f"   [•] {sym}: kept {len(news_items)} (raw={len(bucket)})")

            node["news"] = news_items

        except Exception as e:
            print(f"[WARN] {sym}: news fetch failed: {e}", file=sys.stderr)
            node.setdefault("news", [])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(prices, f, indent=2)

    print(f"[DONE] Wrote news into {OUTPUT_PATH} (kept_total={kept_total}, raw_seen={raw_seen})")

if __name__ == "__main__":
    main()
