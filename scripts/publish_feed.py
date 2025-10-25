# scripts/publish_feed.py
import os, json, math, requests
from datetime import datetime, timezone, timedelta
import pandas as pd

# ========= Config / Env =========
ALPACA_KEY    = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
ALPACA_FEED   = os.getenv("ALPACA_FEED", "iex")        # "iex" or "sip"
IND_TF        = os.getenv("INDICATOR_TIMEFRAME", "1Day")# 1Day, 5Min, 1Min, etc.
HIST_LIMIT    = int(os.getenv("HIST_LIMIT", "200"))     # bars used for indicators
DRY_RUN       = os.getenv("DRY_RUN", "false").lower() == "true"

# Publishing targets
GIST_ID       = os.getenv("GIST_ID")
GIST_TOKEN    = os.getenv("GIST_TOKEN")

DATA_BASE = "https://data.alpaca.markets"
H = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

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

# ========= Fetch latest bars (snapshot) =========
def fetch_latest_bars(symbols):
    url = f"{DATA_BASE}/v2/stocks/bars/latest"
    params = {"symbols": ",".join(symbols), "feed": ALPACA_FEED}
    print(f"🔄 Fetching latest bars for {len(symbols)} tickers...")
    r = requests.get(url, headers=H, params=params, timeout=25)
    print(f"📡 Bars response: {r.status_code}")
    r.raise_for_status()
    data = r.json().get("bars", {})
    got = list(data.keys())
    missing = [s for s in symbols if s not in got]
    print(f"✅ Retrieved {len(got)} latest bars" + (f" | ⚠️ Missing {len(missing)}: {', '.join(missing[:15])}{'...' if len(missing)>15 else ''}" if missing else ""))
    return data
    
def iso(dt):  # RFC3339/ISO8601 Zulu
    return dt.replace(microsecond=0, tzinfo=timezone.utc).isoformat().replace("+00:00","Z")
    
# ========= Fetch historical bars (per symbol) for indicators =========
def fetch_hist_bars(symbol, timeframe="1Day", limit=200):
    url = f"{DATA_BASE}/v2/stocks/{symbol}/bars"

    # daily bars often need an explicit start; give plenty of history
    lookback_days = max(60, limit + 50)  # safety margin
    start_dt = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    params = {
        "timeframe": timeframe,
        "limit": limit,             # still keep a cap
        "start": iso(start_dt),     # <<< important for 1Day
        "adjustment": "raw",
        "feed": ALPACA_FEED,        # "iex" or "sip"
    }

    r = requests.get(url, headers=H, params=params, timeout=25)
    if r.status_code != 200:
        print(f"⚠️ Hist fetch {symbol} -> HTTP {r.status_code}")
        return None

    bars = r.json().get("bars", [])
    if not bars and ALPACA_FEED.lower() != "iex":
        # Auto-fallback: try IEX if SIP returns nothing
        params["feed"] = "iex"
        r2 = requests.get(url, headers=H, params=params, timeout=25)
        if r2.status_code == 200:
            bars = r2.json().get("bars", [])
            if bars:
                print(f"🔁 Fallback to IEX worked for {symbol}")
        else:
            print(f"⚠️ Fallback IEX fetch {symbol} -> HTTP {r2.status_code}")

    if not bars:
        print(f"⚠️ No hist bars for {symbol} (feed={params['feed']}, tf={timeframe}, start={params['start']}, limit={limit})")
        return None

    df = pd.DataFrame(bars)
    if not {'t','o','h','l','c','v'}.issubset(df.columns):
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

def compute_indicators(df):
    px = df['c'].astype(float)
    ema12 = ema(px, 12)
    ema26 = ema(px, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    rsi14 = rsi(px, 14)
    sma20 = px.rolling(20).mean()
    return {
        "ema12": round(float(ema12.iloc[-1]), 4) if len(ema12) else None,
        "ema26": round(float(ema26.iloc[-1]), 4) if len(ema26) else None,
        "macd": round(float(macd_line.iloc[-1]), 4) if len(macd_line) else None,
        "macd_signal": round(float(signal.iloc[-1]), 4) if len(signal) else None,
        "macd_hist": round(float(hist.iloc[-1]), 4) if len(hist) else None,
        "rsi14": round(float(rsi14.iloc[-1]), 2) if len(rsi14) else None,
        "sma20": None if math.isnan(sma20.iloc[-1]) else round(float(sma20.iloc[-1]), 4)
    }

# ========= News: global (20) + per-symbol (Portfolio B, up to 3 each) =========
def fetch_news_global(symbols, limit=20):
    url = f"{DATA_BASE}/v1beta1/news"
    r = requests.get(url, headers=H, params={"symbols": ",".join(symbols), "limit": limit}, timeout=25)
    print(f"📰 Global news response: {r.status_code}, limit={limit}")
    r.raise_for_status()
    return r.json().get("news", [])

def fetch_news_for_symbol(symbol, limit=3):
    url = f"{DATA_BASE}/v1beta1/news"
    r = requests.get(url, headers=H, params={"symbols": symbol, "limit": limit}, timeout=20)
    if r.status_code != 200:
        print(f"⚠️ News fetch failed for {symbol}: {r.status_code}")
        return []
    items = r.json().get("news", [])[:limit]
    return [{
        "headline": n.get("headline"),
        "source": n.get("source"),
        "url": n.get("url"),
        "created_at": n.get("created_at")
    } for n in items]

def build_news_map_for_pfB(pf_syms, per_symbol_limit=3):
    out = {}
    for s in pf_syms:
        out[s] = fetch_news_for_symbol(s, per_symbol_limit)
    print(f"🗂️ Per-symbol news built for {len(pf_syms)} Portfolio B tickers (≤{per_symbol_limit} each)")
    return out

# ========= Publish helpers =========
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
    # sanity
    if not (ALPACA_KEY and ALPACA_SECRET):
        raise RuntimeError("Missing ALPACA_KEY / ALPACA_SECRET")

    # 1) Latest bars
    latest = fetch_latest_bars(SYMBOLS)

    # 2) Historical + indicators
    indicators = {}
    count_hist = 0
    for s in SYMBOLS:
        df = fetch_hist_bars(s, timeframe=IND_TF, limit=HIST_LIMIT)
        if df is None: 
            continue
        try:
            indicators[s] = compute_indicators(df)
            count_hist += 1
        except Exception as e:
            print(f"⚠️ Indicator calc failed for {s}: {e}")
    print(f"📚 Indicators computed for {count_hist} symbols (tf={IND_TF}, window={HIST_LIMIT})")

    # 3) News
    pfB = load_portfolio_b_symbols()
    news_global = fetch_news_global(SYMBOLS, limit=20)
    news_map   = build_news_map_for_pfB(pfB, per_symbol_limit=3)

    # 4) Assemble JSON
    doc = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "feed": ALPACA_FEED,
        "timeframe": IND_TF,
        "indicators_window": HIST_LIMIT,
        "symbols": {}
    }
    for s in SYMBOLS:
        b = latest.get(s) or {}
        doc["symbols"][s] = {
            "price": b.get("c"),
            "volume": b.get("v"),
            "ts": b.get("t"),
            "indicators": indicators.get(s)
        }
        if s in news_map:
            doc["symbols"][s]["news"] = news_map[s]

    doc["news"] = [
        {
            "symbols": n.get("symbols"),
            "headline": n.get("headline"),
            "summary": n.get("summary"),
            "source": n.get("source"),
            "url": n.get("url"),
            "created_at": n.get("created_at")
        } for n in news_global
    ]

    content = json.dumps(doc, separators=(",", ":"), ensure_ascii=False)

    if DRY_RUN:
        print("🧪 DRY_RUN — first 600 chars:")
        print(content[:600])
        return

    # 5) Publish (Gist if creds provided, else write to repo)
    if GIST_ID and GIST_TOKEN:
        publish_to_gist(GIST_ID, GIST_TOKEN, "prices.json", content)
    else:
        write_to_repo("data/prices.json", content)

    print("✅ Completed successfully.")

if __name__ == "__main__":
    main()
