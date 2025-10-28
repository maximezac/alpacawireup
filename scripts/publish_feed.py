# scripts/publish_feed.py
import os, json, math, time, requests, re
from datetime import datetime, timezone, timedelta
import pandas as pd

# ========= Config / Env =========
ALPACA_KEY    = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
ALPACA_FEED   = os.getenv("ALPACA_FEED", "iex")         # "iex" or "sip"
IND_TF        = os.getenv("INDICATOR_TIMEFRAME", "1Day")# 1Day, 5Min, 1Min, etc.
HIST_LIMIT    = int(os.getenv("HIST_LIMIT", "200"))      # bars used for indicators
DRY_RUN       = os.getenv("DRY_RUN", "false").lower() == "true"

# Publishing targets
GIST_ID       = os.getenv("GIST_ID")
GIST_TOKEN    = os.getenv("GIST_TOKEN")

DATA_BASE = "https://data.alpaca.markets"
H = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

# ========= Small HTTP helper with retries =========
def _get(url, headers=None, params=None, timeout=25, tries=3, backoff=0.6):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code == 429:
                time.sleep(backoff * (i+1))
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            time.sleep(backoff * (i+1))
    raise last

# ========= Helpers: load symbols =========
def _load_list(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip().upper() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    return []

def load_universe():
    base  = _load_list("tickers.txt")
    watch = _load_list("watchlist.txt")
    env   = [s.strip().upper() for s in os.getenv("SYMBOLS","").split(",") if s.strip()]
    seen, out = set(), []
    for s in base + watch + env:
        if s not in seen:
            seen.add(s); out.append(s)
    # Ensure SPY present for rel_strength baseline if any symbols exist
    if out and "SPY" not in out:
        out.append("SPY")
    return out

def load_portfolio_b_symbols():
    syms = _load_list("portfolio_b.txt")
    if syms:
        print(f"📁 Portfolio B tickers (file): {len(syms)}")
        return syms
    env = os.getenv("PORTFOLIO_B_TICKERS","")
    if env.strip():
        syms = [s.strip().upper() for s in env.split(",") if s.strip()]
        print(f"🌐 Portfolio B tickers (env): {len(syms)}")
        return syms
    print("⚠️ Portfolio B list missing (no portfolio_b.txt and no PORTFOLIO_B_TICKERS).")
    return []

SYMBOLS = load_universe()
print(f"🧾 Loaded {len(SYMBOLS)} total tickers: {', '.join(SYMBOLS[:25])}{'...' if len(SYMBOLS)>25 else ''}")

def iso(dt):  # RFC3339/ISO8601 Zulu
    return dt.replace(microsecond=0, tzinfo=timezone.utc).isoformat().replace("+00:00","Z")

# ========= Fetch latest bars (snapshot) =========
def fetch_latest_bars(symbols):
    url = f"{DATA_BASE}/v2/stocks/bars/latest"
    params = {"symbols": ",".join(symbols), "feed": ALPACA_FEED}
    print(f"🔄 Fetching latest bars for {len(symbols)} tickers...")
    r = _get(url, headers=H, params=params, timeout=25)
    print(f"📡 Bars response: {r.status_code}")
    data = r.json().get("bars", {})
    got = list(data.keys())
    missing = [s for s in symbols if s not in got]
    print(f"✅ Retrieved {len(got)} latest bars" + (f" | ⚠️ Missing {len(missing)}: {', '.join(missing[:15])}{'...' if len(missing)>15 else ''}" if missing else ""))
    return data

# ========= Fetch historical bars =========
def fetch_hist_bars(symbol, timeframe="1Day", limit=200):
    url = f"{DATA_BASE}/v2/stocks/{symbol}/bars"
    lookback_days = max(90, limit + 80)  # more cushion for ATR/vol windows
    start_dt = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    params = {
        "timeframe": timeframe,
        "limit": limit,
        "start": iso(start_dt),
        "adjustment": "raw",
        "feed": ALPACA_FEED,
    }
    try:
        r = _get(url, headers=H, params=params, timeout=25)
        bars = r.json().get("bars", [])
    except Exception:
        bars = []

    # Fallback IEX
    if not bars and ALPACA_FEED.lower() != "iex":
        params["feed"] = "iex"
        try:
            r2 = _get(url, headers=H, params=params, timeout=25)
            bars = r2.json().get("bars", [])
            if bars:
                print(f"🔁 Fallback to IEX worked for {symbol}")
        except Exception as e:
            print(f"⚠️ Fallback IEX fetch {symbol} failed: {e}")

    if not bars:
        print(f"⚠️ No hist bars for {symbol} (feed={params['feed']}, tf={timeframe}, start={params['start']}, limit={limit})")
        return None

    df = pd.DataFrame(bars)
    need = {'t','o','h','l','c','v'}
    if not need.issubset(df.columns):
        print(f"⚠️ Unexpected schema for {symbol}: {df.columns.tolist()}")
        return None
    return df.sort_values('t').reset_index(drop=True)

# ========= Indicators =========
def ema(series, span): return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up  = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    dn  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs  = up / (dn.replace(0, 1e-12))
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    h = df['h'].astype(float); l = df['l'].astype(float); c = df['c'].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return float(atr.iloc[-1]) if len(atr.dropna()) else None

def compute_indicators(df, bench_df=None):
    px = df['c'].astype(float)
    ema12 = ema(px, 12)
    ema26 = ema(px, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    rsi14 = rsi(px, 14)
    sma20 = px.rolling(20).mean()

    # NEW: volume metrics
    vol = df['v'].astype(float)
    vol_sma20 = vol.rolling(20).mean()
    vol_ratio = (vol.iloc[-1] / vol_sma20.iloc[-1]) if len(vol_sma20.dropna()) else None

    # NEW: ATR14
    atr14 = compute_atr(df, 14)

    # NEW: z-score of close vs 50D mean/std
    mean50 = px.rolling(50).mean()
    std50  = px.rolling(50).std(ddof=0)
    zclose50 = ((px.iloc[-1] - mean50.iloc[-1]) / (std50.iloc[-1] if std50.iloc[-1] else float('nan'))) if len(std50.dropna()) else None

    # NEW: relative strength vs SPY over 20 bars
    rel_strength20 = None
    if bench_df is not None and len(px) >= 20 and len(bench_df) >= 20:
        r_sym = (px.iloc[-1] / px.iloc[-20]) - 1.0
        pxb = bench_df['c'].astype(float)
        r_bm = (pxb.iloc[-1] / pxb.iloc[-20]) - 1.0
        rel_strength20 = float(r_sym - r_bm)

    return {
        "ema12": round(float(ema12.iloc[-1]), 4) if len(ema12) else None,
        "ema26": round(float(ema26.iloc[-1]), 4) if len(ema26) else None,
        "macd": round(float(macd_line.iloc[-1]), 4) if len(macd_line) else None,
        "macd_signal": round(float(signal.iloc[-1]), 4) if len(signal) else None,
        "macd_hist": round(float(hist.iloc[-1]), 4) if len(hist) else None,
        "rsi14": round(float(rsi14.iloc[-1]), 2) if len(rsi14) else None,
        "sma20": None if pd.isna(sma20.iloc[-1]) else round(float(sma20.iloc[-1]), 4),
        "atr14": None if atr14 is None or math.isnan(atr14) else round(float(atr14), 4),
        "vol_sma20": None if len(vol_sma20.dropna())==0 else round(float(vol_sma20.iloc[-1]), 2),
        "vol_ratio": None if (vol_ratio is None or math.isnan(vol_ratio)) else round(float(vol_ratio), 3),
        "zclose50": None if (zclose50 is None or pd.isna(zclose50)) else round(float(zclose50), 3),
        "rel_strength20": None if rel_strength20 is None or math.isnan(rel_strength20) else round(float(rel_strength20), 4)
    }

# ========= Simple News Sentiment + Decay =========
_POS = re.compile(r"\b(beat|surge|record|raise|upgrade|win|contract|stake|funding|partnership|approval|profit|buyback|guidance\s+raise)\b", re.I)
_NEG = re.compile(r"\b(miss|downgrade|delay|probe|lawsuit|recall|guidance\s+cut|bankruptcy|loss|weak|halt|shortfall|fraud)\b", re.I)

def score_news_items(items):
    """Return naive sentiment in [-1, 1] from headlines; 0 if none."""
    if not items: return 0.0
    s = 0.0; n = 0
    for it in items:
        h = (it.get("headline") or "") + " " + (it.get("summary") or "")
        if not h: continue
        n += 1
        p = 1.0 if _POS.search(h) else 0.0
        q = 1.0 if _NEG.search(h) else 0.0
        s += (p - q)
    return max(-1.0, min(1.0, s / max(n,1)))

def fetch_news_global(symbols, limit=20):
    url = f"{DATA_BASE}/v1beta1/news"
    r = _get(url, headers=H, params={"symbols": ",".join(symbols), "limit": limit}, timeout=25)
    print(f"📰 Global news response: {r.status_code}, limit={limit}")
    return r.json().get("news", [])

def fetch_news_for_symbol(symbol, limit=3):
    url = f"{DATA_BASE}/v1beta1/news"
    try:
        r = _get(url, headers=H, params={"symbols": symbol, "limit": limit}, timeout=20)
        items = r.json().get("news", [])[:limit]
    except Exception:
        print(f"⚠️ News fetch failed for {symbol}")
        items = []
    return [{
        "headline": n.get("headline"),
        "source": n.get("source"),
        "url": n.get("url"),
        "created_at": n.get("created_at"),
        "summary": n.get("summary")
    } for n in items]

def build_news_map_for_pfB(pf_syms, per_symbol_limit=3):
    out = {}
    for s in pf_syms:
        out[s] = fetch_news_for_symbol(s, per_symbol_limit)
    print(f"🗂️ Per-symbol news built for {len(pf_syms)} Portfolio B tickers (≤{per_symbol_limit} each)")
    return out

# ========= Gist read/write (for NS_decay + prior carry) =========
def load_previous_feed_from_gist(gist_id, token, filename="prices.json"):
    if not (gist_id and token): return None
    url = f"https://api.github.com/gists/{gist_id}"
    try:
        r = _get(url, headers={"Authorization": f"Bearer {token}", "Accept":"application/vnd.github+json"}, timeout=25)
        files = r.json().get("files", {})
        if filename in files and files[filename].get("content"):
            return json.loads(files[filename]["content"])
    except Exception as e:
        print(f"⚠️ Could not load previous feed: {e}")
    return None

def publish_to_gist(gist_id, token, filename, content_str):
    url = f"https://api.github.com/gists/{gist_id}"
    payload = {"files": {filename: {"content": content_str}}}
    headers = {"Authorization": f"Bearer {token}", "Accept":"application/vnd.github+json"}
    r = requests.patch(url, headers=headers, json=payload, timeout=25)
    print(f"💾 Gist update status: {r.status_code}")
    r.raise_for_status()
    return r.json()

def write_to_repo(path, content_str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content_str)
    print(f"💾 Wrote {path}")

# ========= Main =========
def main():
    if not (ALPACA_KEY and ALPACA_SECRET):
        raise RuntimeError("Missing ALPACA_KEY / ALPACA_SECRET")

    prev_doc = load_previous_feed_from_gist(GIST_ID, GIST_TOKEN)  # may be None

    # 1) Latest bars
    latest = fetch_latest_bars(SYMBOLS)

    # 2) Historical + indicators (need SPY for rel_strength)
    dfs = {}
    for s in SYMBOLS:
        df = fetch_hist_bars(s, timeframe=IND_TF, limit=HIST_LIMIT)
        if df is not None:
            dfs[s] = df
    print(f"📚 Hist dataframes ready: {len(dfs)} symbols")

    bench_df = dfs.get("SPY")  # could be None; compute_indicators handles that
    indicators = {}
    for s, df in dfs.items():
        try:
            indicators[s] = compute_indicators(df, bench_df=bench_df)
        except Exception as e:
            print(f"⚠️ Indicator calc failed for {s}: {e}")

    # 3) News
    pfB = load_portfolio_b_symbols()
    news_global = fetch_news_global(SYMBOLS, limit=20)
    news_map   = build_news_map_for_pfB(pfB, per_symbol_limit=3)

    # 4) Assemble JSON (+ freshness, NS, NS_decay)
    now_utc = datetime.now(timezone.utc)
    doc = {
        "as_of_utc": now_utc.isoformat(),
        "feed": ALPACA_FEED,
        "timeframe": IND_TF,
        "indicators_window": HIST_LIMIT,
        "freshness": {"max_age_sec": 60*60*6, "symbols_with_stale_data": []},  # 6h guard
        "symbols": {}
    }

    # previous NS_decay map (if any)
    prev_decay = {}
    if prev_doc and "symbols" in prev_doc:
        for k, v in prev_doc["symbols"].items():
            ind = v.get("indicators") or {}
            if "ns_decay" in ind:
                prev_decay[k] = ind["ns_decay"]

    for s in SYMBOLS:
        b = latest.get(s) or {}
        ts = b.get("t")
        # freshness check per symbol
        age_sec = None
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
                age_sec = max(0, (now_utc - dt).total_seconds())
            except Exception:
                age_sec = None
        if age_sec is not None and age_sec > 60*60*6:
            doc["freshness"]["symbols_with_stale_data"].append({"symbol": s, "age_sec": int(age_sec)})

        # base entry
        doc["symbols"][s] = {
            "price": b.get("c"),
            "volume": b.get("v"),
            "ts": ts,
            "indicators": indicators.get(s)
        }

        # attach per-symbol news (for Portfolio B)
        if s in news_map:
            items = news_map[s]
            doc["symbols"][s]["news"] = items
            # naive sentiment & decay
            ns_raw = score_news_items(items)
            prev = prev_decay.get(s, 0.0)
            ns_decay = 0.5 * float(prev) + 0.5 * float(ns_raw)
            # store inside indicators for one-stop read
            if doc["symbols"][s]["indicators"] is None:
                doc["symbols"][s]["indicators"] = {}
            doc["symbols"][s]["indicators"]["ns_raw"] = round(ns_raw, 3)
            doc["symbols"][s]["indicators"]["ns_decay"] = round(ns_decay, 3)

    # global news passthrough
    doc["news"] = [{
        "symbols": n.get("symbols"),
        "headline": n.get("headline"),
        "summary": n.get("summary"),
        "source": n.get("source"),
        "url": n.get("url"),
        "created_at": n.get("created_at")
    } for n in news_global]

    content = json.dumps(doc, separators=(",", ":"), ensure_ascii=False)

    if DRY_RUN:
        print("🧪 DRY_RUN — first 900 chars:")
        print(content[:900])
        return

    # 5) Publish
    if GIST_ID and GIST_TOKEN:
        publish_to_gist(GIST_ID, GIST_TOKEN, "prices.json", content)
    else:
        write_to_repo("data/prices.json", content)

    print("✅ Completed successfully.")

if __name__ == "__main__":
    main()
