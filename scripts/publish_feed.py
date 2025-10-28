# scripts/publish_feed.py
import os, json, math, time, requests, re
from datetime import datetime, timezone, timedelta
import pandas as pd

# ========= Config / Env =========
ALPACA_KEY    = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
ALPACA_FEED   = os.getenv("ALPACA_FEED", "iex")          # "iex" or "sip"
IND_TF        = os.getenv("INDICATOR_TIMEFRAME", "1Day") # 1Day, 5Min, 1Min, etc.
HIST_LIMIT    = int(os.getenv("HIST_LIMIT", "200"))      # bars used for indicators
DRY_RUN       = os.getenv("DRY_RUN", "false").lower() == "true"

# Publishing targets
GIST_ID       = os.getenv("GIST_ID")
GIST_TOKEN    = os.getenv("GIST_TOKEN")

DATA_BASE = "https://data.alpaca.markets"
H = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

# ========= v2.2 Config (signals) =========
BENCH     = os.getenv("BENCHMARK", "SPY")     # used for regime + rel strength reference
BENCH_ALT = os.getenv("BENCHMARK_ALT", "QQQ")
NS_ALPHA  = float(os.getenv("NS_ALPHA", "0.5"))  # 0.5 => 50% half-life per day

# Risk classes help pick ATR trails and stops
RISK_CLASS = {
    # ETFs / megacaps
    "SPY":"etf","QQQ":"etf","VOO":"etf","VTI":"etf","DIA":"etf","IWM":"etf",
    "AAPL":"mega","MSFT":"mega","GOOGL":"mega","META":"mega","AMZN":"mega","NVDA":"mega",
    # growth
    "SNOW":"growth","SHOP":"growth","NOW":"growth","PLTR":"growth","TSLA":"growth","AMD":"growth","AVGO":"growth","ASML":"growth","CRM":"growth",
    # speculative / small-cap tech & quantum
    "QBTS":"spec","RGTI":"spec","QUBT":"spec","IONQ":"spec","RKLB":"spec",
    # finance/dividends
    "JPM":"mega","BAC":"mega","WFC":"mega","ARCC":"mega",
    # add/override freely...
}
def get_risk_class(symbol: str) -> str:
    return RISK_CLASS.get(symbol, "growth")  # default to 'growth'

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
    # Normalize: ensure o/h/l/c/v/t are present (Alpaca returns them already)
    norm = {}
    for s, b in (data or {}).items():
        norm[s] = {
            "o": b.get("o"), "h": b.get("h"), "l": b.get("l"),
            "c": b.get("c"), "v": b.get("v"), "t": b.get("t")
        }
    return norm

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

def compute_rel_strength_20(df_sym: pd.DataFrame, df_bench: pd.DataFrame):
    try:
        if df_sym is None or df_bench is None: return None
        if len(df_sym) < 21 or len(df_bench) < 21: return None
        cs = df_sym['c'].astype(float)
        cb = df_bench['c'].astype(float)
        rs_sym = cs.iloc[-1] / cs.iloc[-21] - 1.0
        rs_b   = cb.iloc[-1] / cb.iloc[-21] - 1.0
        return round(float(rs_sym - rs_b), 4)
    except Exception:
        return None

def compute_zclose50(df: pd.DataFrame):
    try:
        if df is None or len(df) < 50: return None
        px = df['c'].astype(float)
        m = px.rolling(50).mean()
        s = px.rolling(50).std()
        z = (px - m) / (s.replace(0, 1e-9))
        return round(float(z.iloc[-1]), 3)
    except Exception:
        return None

def compute_volumes(df: pd.DataFrame, window=20):
    try:
        v = df['v'].astype(float)
        vsma = v.rolling(window).mean().iloc[-1]
        last = float(v.iloc[-1])
        return float(vsma), (last / vsma) if vsma and not math.isnan(vsma) else None
    except Exception:
        return None, None

def compute_indicators(df: pd.DataFrame, bench_df: pd.DataFrame = None):
    px = df['c'].astype(float)
    ema12_series = ema(px, 12)
    ema26_series = ema(px, 26)
    macd_line = ema12_series - ema26_series
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    rsi14_series = rsi(px, 14)
    sma20_series = px.rolling(20).mean()

    def last2(s):
        if len(s) >= 2:
            return float(s.iloc[-1]), float(s.iloc[-2])
        return (float(s.iloc[-1]), float(s.iloc[-1])) if len(s) else (None, None)

    ema12, ema12_prev = last2(ema12_series)
    ema26, ema26_prev = last2(ema26_series)
    macd_hist, macd_hist_prev = last2(hist)
    rsi14 = float(rsi14_series.iloc[-1]) if len(rsi14_series) else None
    sma20_last = sma20_series.iloc[-1] if len(sma20_series) else float("nan")

    atr14 = compute_atr(df, 14)
    vol_sma20, vol_ratio = compute_volumes(df, 20)
    z50 = compute_zclose50(df)
    rel_s20 = compute_rel_strength_20(df, bench_df) if bench_df is not None else None

    out = {
        "ema12": round(ema12, 4) if ema12 is not None else None,
        "ema26": round(ema26, 4) if ema26 is not None else None,
        "macd": round(float(macd_line.iloc[-1]), 4) if len(macd_line) else None,
        "macd_signal": round(float(signal.iloc[-1]), 4) if len(signal) else None,
        "macd_hist": round(macd_hist, 4) if macd_hist is not None else None,
        "macd_hist_prev": round(macd_hist_prev, 4) if macd_hist_prev is not None else None,
        "ema12_prev": round(ema12_prev, 4) if ema12_prev is not None else None,
        "ema26_prev": round(ema26_prev, 4) if ema26_prev is not None else None,
        "rsi14": round(rsi14, 2) if rsi14 is not None else None,
        "sma20": None if math.isnan(sma20_last) else round(float(sma20_last), 4),
        "atr14": None if atr14 is None else round(float(atr14), 4),
        "vol_sma20": None if vol_sma20 is None else round(float(vol_sma20), 2),
        "vol_ratio": None if vol_ratio is None else round(float(vol_ratio), 3),
        "zclose50": z50,
        "rel_strength20": rel_s20,
        # placeholders for news sentiment; filled later
        "ns_raw": 0.0,
        "ns_decay": 0.0
    }
    return out

def get_prev_close(df: pd.DataFrame):
    if df is None or len(df) < 2:
        return None
    return float(df['c'].iloc[-2])

def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))

def compute_market_regime(sym_spy: dict, sym_qqq: dict):
    """
    MR = 40% sign(MACD_hist_SPY) + 30% (RSI_SPY-50)/50 + 30% sign(zclose50_SPY)
    If SPY missing fields, fall back to QQQ.
    """
    def part(sym):
        ind = (sym or {}).get("indicators") or {}
        rsi = ind.get("rsi14"); hist = ind.get("macd_hist"); z = ind.get("zclose50")
        if rsi is None or hist is None or z is None:
            return None
        return 0.4 * (1 if hist > 0 else (-1 if hist < 0 else 0)) \
             + 0.3 * clamp((rsi - 50) / 50.0) \
             + 0.3 * (1 if z > 0 else (-1 if z < 0 else 0))
    m = part(sym_spy)
    if m is None:
        m = part(sym_qqq) if sym_qqq else 0
    return clamp(m if m is not None else 0)

def compute_ts(symbol: str, sym_blob: dict):
    ind = sym_blob.get("indicators") or {}
    price = sym_blob.get("price")
    atr = (ind.get("atr14") or 0) if price else 0
    atr_norm = max(atr, 0.01 * (price or 1.0))

    macd_hist = ind.get("macd_hist"); macd_hist_prev = ind.get("macd_hist_prev")
    macd_slope = 0.0
    if macd_hist is not None and macd_hist_prev is not None:
        macd_slope = (macd_hist - macd_hist_prev) / atr_norm

    ema12 = ind.get("ema12"); ema12_prev = ind.get("ema12_prev")
    ema12_slope = 0.0
    if ema12 is not None and ema12_prev is not None:
        ema12_slope = (ema12 - ema12_prev) / atr_norm

    rsi = ind.get("rsi14")
    rsi_term = ((rsi - 50) / 50.0) if rsi is not None else 0.0

    rel_strength20 = ind.get("rel_strength20")
    rs_term = rel_strength20 if rel_strength20 is not None else 0.0

    z50 = ind.get("zclose50")
    z_term = (z50 / 3.0) if z50 is not None else 0.0

    ts = (0.30 * clamp(macd_slope, -2, 2)
        + 0.20 * clamp(rsi_term, -1, 1)
        + 0.20 * clamp(ema12_slope, -2, 2)
        + 0.15 * clamp(rs_term, -1, 1)
        + 0.15 * clamp(z_term, -1, 1))
    return clamp(ts, -1, 1)

def compute_ns(sym_blob: dict, sector_ns: float = None):
    ind = sym_blob.get("indicators") or {}
    n_t = ind.get("ns_decay")
    n_s = sector_ns if sector_ns is not None else 0.0
    raw = 0.70 * (n_t or 0.0) + 0.30 * (n_s or 0.0)
    # tanh without numpy
    e2x = math.exp(2 * raw)
    ns = (e2x - 1) / (e2x + 1)
    return clamp(ns, -1, 1)

def compute_cds(ts: float, ns: float):
    wT, wN = (0.4, 0.6) if abs(ns) >= 0.7 else (0.6, 0.4)
    return clamp(wT * ts + wN * ns, -1, 1), wT, wN

def decision_from_scores(cds: float, mr: float, sym_blob: dict):
    # thresholds shift with market regime
    entry_bias = 0.10 if mr < -0.3 else (-0.05 if mr > 0.3 else 0.0)
    buy_th = 0.35 + entry_bias
    sell_th = -0.25

    ind = sym_blob.get("indicators") or {}
    price = sym_blob.get("price")
    ema12 = ind.get("ema12"); ema26 = ind.get("ema26")
    rsi = ind.get("rsi14"); rs20 = ind.get("rel_strength20")
    z50 = ind.get("zclose50")

    # confirmations for buys
    buy_confirm = (price is not None and ema12 is not None and price > ema12) \
                  and (rs20 is None or rs20 >= -0.02)
    extended = (z50 is not None and z50 > 2.0 and not (mr > 0.5 and (ind.get("ns_decay") or 0) > 0.6))

    # fast-fail
    fast_fail = (ema26 is not None and price is not None and price < ema26 and (rsi or 100) < 45) \
                or ((ind.get("ns_decay") or 0) < -0.7)

    if fast_fail or cds < sell_th:
        return "SELL/TRIM"
    if cds > buy_th and buy_confirm and not extended:
        return "BUY/ADD"
    return "HOLD"

def intraday_metrics(latest_bar: dict, prev_close: float):
    o = latest_bar.get("o"); h = latest_bar.get("h"); l = latest_bar.get("l"); c = latest_bar.get("c")
    out = {"intraday_range_pct": None, "gap_pct": None}
    try:
        if all(v is not None for v in (h, l, c)) and c:
            out["intraday_range_pct"] = round((h - l) / c, 4)
        if prev_close and o:
            out["gap_pct"] = round((o - prev_close) / prev_close, 4)
    except Exception:
        pass
    return out

def weekly_tailwind(df_daily: pd.DataFrame):
    """
    Light multi-timeframe nudge using daily -> weekly resample.
    Returns modifier in [-0.08, +0.05].
    """
    if df_daily is None or df_daily.empty:
        return 0.0
    # build weekly close from last close of each week
    df = df_daily.copy()
    df['t'] = pd.to_datetime(df['t'])
    df = df.set_index('t').resample('W-FRI').last().dropna()
    if len(df) < 4:
        return 0.0
    # compute TS-like on weekly close only
    px = df['c'].astype(float)
    ema12_w = ema(px, 12); ema26_w = ema(px, 26)
    macd_hist_w = (ema12_w - ema26_w) - ema(ema12_w - ema26_w, 9)
    rsi_w = rsi(px, 14)
    z_w = (px - px.rolling(50).mean()) / (px.rolling(50).std() + 1e-9)

    ts_w = 0.4 * (1 if (macd_hist_w.iloc[-1] or 0) > 0 else -1 if (macd_hist_w.iloc[-1] or 0) < 0 else 0) \
         + 0.3 * ((rsi_w.iloc[-1] - 50) / 50.0) \
         + 0.3 * (1 if (z_w.iloc[-1] or 0) > 0 else -1 if (z_w.iloc[-1] or 0) < 0 else 0)
    ts_w = clamp(ts_w, -1, 1)
    # Map to small modifier
    return 0.05 if ts_w > 0.25 else (-0.08 if ts_w < -0.25 else 0.0)

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

# ========= Gist read/write (for NS_decay carry & freshness) =========
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

def build_freshness(latest_map, max_age_sec=6*3600):
    out = []
    now = datetime.now(timezone.utc)
    for s, b in (latest_map or {}).items():
        t_iso = b.get("t")
        try:
            t = datetime.fromisoformat(t_iso.replace("Z","+00:00")) if t_iso else None
        except Exception:
            t = None
        if t:
            age = (now - t).total_seconds()
            if age > max_age_sec:
                out.append({"symbol": s, "age_sec": int(age)})
    return {"max_age_sec": max_age_sec, "symbols_with_stale_data": out}

# ========= Main =========
def main():
    if not (ALPACA_KEY and ALPACA_SECRET):
        raise RuntimeError("Missing ALPACA_KEY / ALPACA_SECRET")

    prev_doc = load_previous_feed_from_gist(GIST_ID, GIST_TOKEN)  # may be None

    # 1) Latest bars
    latest = fetch_latest_bars(SYMBOLS)

    # 2) Historical + indicators (compute once, cache)
    dfs = {}
    for s in SYMBOLS:
        df = fetch_hist_bars(s, timeframe=IND_TF, limit=HIST_LIMIT)
        if df is not None:
            dfs[s] = df
    print(f"📚 Hist dataframes ready: {len(dfs)} symbols")

    bench_df = next((dfs[s] for s in (BENCH, BENCH_ALT) if s in dfs), None)

    indicators = {}
    for s, df in dfs.items():
        try:
            indicators[s] = compute_indicators(df, bench_df=bench_df)
        except Exception as e:
            print(f"⚠️ Indicator calc failed for {s}: {e}")

    # 3) News (global + per-symbol for Portfolio B)
    pfB = load_portfolio_b_symbols()
    news_global = fetch_news_global(SYMBOLS, limit=20)
    news_map   = build_news_map_for_pfB(pfB, per_symbol_limit=3)

    # 4) Compose per-symbol data blobs
    symdata = {}
    for s in SYMBOLS:
        b  = latest.get(s) or {}
        df = dfs.get(s)
        prev_c = get_prev_close(df) if df is not None else None
        ind = indicators.get(s, {}).copy()

        # news sentiment & decay carry
        ns_raw = score_news_items(news_map.get(s)) if s in news_map else 0.0
        prev_ns = None
        if prev_doc and prev_doc.get("symbols", {}).get(s, {}).get("indicators"):
            prev_ns = prev_doc["symbols"][s]["indicators"].get("ns_decay")
        ns_decay = (1.0 - NS_ALPHA) * (prev_ns or 0.0) + NS_ALPHA * (ns_raw or 0.0)
        ind["ns_raw"]   = round(ns_raw, 3)
        ind["ns_decay"] = round(ns_decay, 3)

        blob = {
            "price": b.get("c"), "volume": b.get("v"), "ts": b.get("t"),
            "o": b.get("o"), "h": b.get("h"), "l": b.get("l"), "c": b.get("c"),
            "indicators": ind, "df": df, "prev_close": prev_c
        }
        blob.update(intraday_metrics(b, prev_c))
        blob["risk_class"] = get_risk_class(s)
        symdata[s] = blob

    # 5) Market regime from BENCH / BENCH_ALT
    mr = compute_market_regime(symdata.get(BENCH, {}), symdata.get(BENCH_ALT, {}))

    # 6) Assemble JSON
    doc = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "feed": ALPACA_FEED,
        "timeframe": IND_TF,
        "indicators_window": HIST_LIMIT,
        "freshness": build_freshness(latest, max_age_sec=6*3600),
        "run": {"version":"v2.2", "bench": BENCH, "bench_alt": BENCH_ALT, "ns_decay": f"{NS_ALPHA}", "market_regime": round(mr, 3)},
        "symbols": {}
    }

    for s in SYMBOLS:
        blob = symdata[s]
        ind  = blob.get("indicators") or {}

        wt = weekly_tailwind(blob.get("df"))
        ts = compute_ts(s, blob)
        ns = compute_ns(blob)  # sector_ns unknown => 0.0
        cds, wT, wN = compute_cds(ts + (wt or 0.0), ns)
        dec = decision_from_scores(cds, mr, blob)

        entry = {
            "price": blob.get("price"),
            "volume": blob.get("volume"),
            "ts": blob.get("ts"),
            "o": blob.get("o"),
            "h": blob.get("h"),
            "l": blob.get("l"),
            "intraday_range_pct": blob.get("intraday_range_pct"),
            "gap_pct": blob.get("gap_pct"),
            "prev_close": blob.get("prev_close"),
            "risk_class": blob.get("risk_class"),
            "indicators": ind,
            "signals": {
                "MR": round(mr, 3),
                "TS": round(ts, 3),
                "NS": round(ns, 3),
                "wT": round(wT, 2),
                "wN": round(wN, 2),
                "CDS": round(cds, 3),
                "weekly_tailwind": round(wt or 0.0, 3),
                "decision": dec
            }
        }
        if s in news_map:
            entry["news"] = news_map[s]
        doc["symbols"][s] = entry

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

    # 7) Publish
    if GIST_ID and GIST_TOKEN:
        publish_to_gist(GIST_ID, GIST_TOKEN, "prices.json", content)
    else:
        write_to_repo("data/prices.json", content)

    print("✅ Completed successfully.")

if __name__ == "__main__":
    main()
