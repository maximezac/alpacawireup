# scripts/publish_feed.py
import os, json, math, time, re, hashlib
from datetime import datetime, timezone, timedelta
import pandas as pd
import requests

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
NS_DECAY_HALF_LIFE_DAYS = 1.0                 # ~50% per day

# Risk classes help pick ATR trails and stops / mean-revert behavior
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
}

def get_risk_class(symbol: str) -> str:
    return RISK_CLASS.get(symbol, "growth")  # default bucket


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
    # Ensure SPY present if any symbols exist (for rel strength baseline)
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
    if not symbols:
        return {}
    url = f"{DATA_BASE}/v2/stocks/bars/latest"
    params = {"symbols": ",".join(symbols), "feed": ALPACA_FEED}
    print(f"🔄 Fetching latest bars for {len(symbols)} tickers...")
    r = _get(url, headers=H, params=params, timeout=25)
    print(f"📡 Bars response: {r.status_code}")
    data = r.json().get("bars", {}) or {}
    got = list(data.keys())
    missing = [s for s in symbols if s not in got]
    print(f"✅ Retrieved {len(got)} latest bars" + (f" | ⚠️ Missing {len(missing)}: {', '.join(missing[:15])}{'...' if len(missing)>15 else ''}" if missing else ""))

    # Normalize: ensure o/h/l/c/v/t are present
    norm = {}
    for s, b in data.items():
        norm[s] = {
            "o": b.get("o"), "h": b.get("h"), "l": b.get("l"),
            "c": b.get("c"), "v": b.get("v"), "t": b.get("t")
        }
    return norm


# ========= Fetch historical bars =========
def fetch_hist_bars(symbol, timeframe="1Day", limit=200):
    url = f"{DATA_BASE}/v2/stocks/{symbol}/bars"
    lookback_days = max(90, limit + 80)  # cushion for ATR/vol windows
    start_dt = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    params = {
        "timeframe": timeframe,
        "limit": limit,
        "start": iso(start_dt),
        "adjustment": "raw",
        "feed": ALPACA_FEED,
    }
    bars = []
    try:
        r = _get(url, headers=H, params=params, timeout=25)
        bars = r.json().get("bars", []) or []
    except Exception:
        bars = []

    # Fallback IEX if needed
    if not bars and ALPACA_FEED.lower() != "iex":
        params["feed"] = "iex"
        try:
            r2 = _get(url, headers=H, params=params, timeout=25)
            bars = r2.json().get("bars", []) or []
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


# ========= Indicators & utilities =========
def ema(series, span): 
    return series.ewm(span=span, adjust=False).mean()

def rsi_series(series, period=14):
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

def bollinger_bandwidth(px: pd.Series, window=20):
    sma = px.rolling(window).mean()
    std = px.rolling(window).std()
    upper = sma + 2*std
    lower = sma - 2*std
    bw = (upper - lower) / (sma.replace(0, 1e-9))
    return bw

def clamp(x, lo=-1.0, hi=1.0): 
    return max(lo, min(hi, x))

def compute_indicators(df: pd.DataFrame, bench_df: pd.DataFrame = None):
    px = df['c'].astype(float)
    vol = df['v'].astype(float)

    ema12_series = ema(px, 12)
    ema26_series = ema(px, 26)
    macd_line = ema12_series - ema26_series
    signal = ema(macd_line, 9)
    hist = macd_line - signal

    rsi14_series = rsi_series(px, 14)
    sma20_series = px.rolling(20).mean()
    atr14_val = compute_atr(df, 14)
    vol_sma20 = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else None
    vol_ratio = None
    if vol_sma20 and vol_sma20 > 0:
        vol_ratio = float(vol.iloc[-1]) / vol_sma20

    # 50-day z-score of close
    mean50 = px.rolling(50).mean()
    std50  = px.rolling(50).std()
    zclose50 = None
    if len(px) >= 50 and std50.iloc[-1] and not math.isnan(std50.iloc[-1]):
        zclose50 = float((px.iloc[-1] - mean50.iloc[-1]) / (std50.iloc[-1] + 1e-12))

    # 20-day relative strength vs bench (excess return)
    rel_strength20 = None
    if len(px) >= 21:
        ret_sym = px.iloc[-1] / px.iloc[-21] - 1.0
        if bench_df is not None and len(bench_df) >= 21:
            bpx = bench_df['c'].astype(float)
            ret_b = bpx.iloc[-1] / bpx.iloc[-21] - 1.0
            rel_strength20 = float(ret_sym - ret_b)
        else:
            rel_strength20 = float(ret_sym)

    # Bollinger bandwidth
    bw20 = bollinger_bandwidth(px, 20)
    bb_bw20 = None
    if len(bw20) and not math.isnan(bw20.iloc[-1]):
        bb_bw20 = float(bw20.iloc[-1])

    # convenient accessor for previous values
    def last2(s):
        if len(s) >= 2:
            return float(s.iloc[-1]), float(s.iloc[-2])
        return (float(s.iloc[-1]), float(s.iloc[-1])) if len(s) else (None, None)

    ema12, ema12_prev = last2(ema12_series)
    ema26, ema26_prev = last2(ema26_series)
    macd_hist, macd_hist_prev = last2(hist)
    rsi14 = float(rsi14_series.iloc[-1]) if len(rsi14_series) else None
    sma20_last = sma20_series.iloc[-1] if len(sma20_series) else float("nan")

    return {
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
        "atr14": round(atr14_val, 4) if atr14_val is not None else None,
        "vol_sma20": round(vol_sma20, 2) if vol_sma20 is not None else None,
        "vol_ratio": round(vol_ratio, 3) if vol_ratio is not None else None,
        "zclose50": round(zclose50, 3) if zclose50 is not None else None,
        "rel_strength20": round(float(rel_strength20), 4) if rel_strength20 is not None else None,
        "bb_bw20": round(bb_bw20, 4) if bb_bw20 is not None else None
    }


# ========= Divergence & news helpers =========
def simple_divergence(px: pd.Series, rsi14: pd.Series, lookback=20):
    """
    Lightweight divergence on last bar:
    - bearish if price last > prior peak while RSI last <= prior RSI peak
    - bullish if price last < prior trough while RSI last >= prior RSI trough
    """
    if len(px) < lookback+2 or len(rsi14) < lookback+2:
        return None
    window = lookback // 2 if lookback >= 4 else lookback
    last = px.iloc[-1]
    prev_peak = px.iloc[-window:-1].max()
    last_rsi = rsi14.iloc[-1]
    prev_rsi_peak = rsi14.iloc[-window:-1].max()
    prev_trough = px.iloc[-window:-1].min()
    prev_rsi_trough = rsi14.iloc[-window:-1].min()

    if last > prev_peak and last_rsi <= prev_rsi_peak - 0.1:
        return "bearish"
    if last < prev_trough and last_rsi >= prev_rsi_trough + 0.1:
        return "bullish"
    return None

_HIGH_QUALITY = ("sec.gov","ir.", "investor.", "prnewswire", "globenewswire", "reuters", "bloomberg", "wsj")
_EVENT_CAP_WORDS = ("offering","atm","convertible","secondary","dilution")

def source_weight(url: str, default=1.0):
    if not url:
        return default
    u = url.lower()
    for kw in _HIGH_QUALITY:
        if kw in u:
            return 1.15
    return 1.0

def event_cap_present(text: str):
    if not text:
        return False
    t = text.lower()
    return any(w in t for w in _EVENT_CAP_WORDS)


# ========= News fetching & scoring =========
_POS = re.compile(r"\b(beat|surge|record|raise|upgrade|win|contract|stake|funding|partnership|approval|profit|buyback|guidance\s+raise)\b", re.I)
_NEG = re.compile(r"\b(miss|downgrade|delay|probe|lawsuit|recall|guidance\s+cut|bankruptcy|loss|weak|halt|shortfall|fraud)\b", re.I)

def score_news_items(items):
    """Naive sentiment in [-1, 1] with source weighting and event-cap damping."""
    if not items:
        return 0.0
    s = 0.0; n = 0
    for it in items:
        head = (it.get("headline") or "")
        summ = (it.get("summary") or "")
        txt = f"{head} {summ}".strip()
        if not txt:
            continue
        n += 1
        pos = 1.0 if _POS.search(txt) else 0.0
        neg = 1.0 if _NEG.search(txt) else 0.0
        w = source_weight(it.get("url"), 1.0)
        val = (pos - neg) * w
        if event_cap_present(txt):
            val = min(val, 0.2)  # offerings/ATMs seldom bullish
        s += val
    return clamp(s / max(n,1), -1, 1)

def fetch_news_global(symbols, limit=20):
    if not symbols:
        return []
    url = f"{DATA_BASE}/v1beta1/news"
    r = _get(url, headers=H, params={"symbols": ",".join(symbols), "limit": limit}, timeout=25)
    print(f"📰 Global news response: {r.status_code}, limit={limit}")
    return r.json().get("news", []) or []

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


# ========= Gist read/write (for NS decay carry) =========
def load_previous_feed_from_gist(gist_id, token, filename="prices.json"):
    if not (gist_id and token):
        return None
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


# ========= Misc metrics =========
def get_prev_close(df: pd.DataFrame):
    if df is None or len(df) < 2:
        return None
    return float(df['c'].iloc[-2])

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
    df = df_daily.copy()
    df['t'] = pd.to_datetime(df['t'])
    df = df.set_index('t').resample('W-FRI').last().dropna()
    if len(df) < 4:
        return 0.0
    px = df['c'].astype(float)
    ema12_w = ema(px, 12); ema26_w = ema(px, 26)
    macd_hist_w = (ema12_w - ema26_w) - ema(ema12_w - ema26_w, 9)
    rsi_w = rsi_series(px, 14)
    z_w = (px - px.rolling(50).mean()) / (px.rolling(50).std() + 1e-9)

    def sgn(v): 
        return 1 if v > 0 else (-1 if v < 0 else 0)

    ts_w = 0.4 * sgn(float(macd_hist_w.iloc[-1])) \
         + 0.3 * clamp(float(rsi_w.iloc[-1] - 50) / 50.0, -1, 1) \
         + 0.3 * sgn(float(z_w.iloc[-1]))
    ts_w = clamp(ts_w, -1, 1)
    return 0.05 if ts_w > 0.25 else (-0.08 if ts_w < -0.25 else 0.0)


# ========= Market regime & scoring =========
def compute_market_regime(sym_spy: dict, sym_qqq: dict):
    """
    MR = 40% sign(MACD_hist_SPY) + 30% (RSI_SPY-50)/50 + 30% sign(zclose50_SPY)
    If SPY missing fields, fall back to QQQ.
    """
    def part(sym):
        if not sym:
            return None
        ind = sym.get("indicators") or {}
        rsi = ind.get("rsi14"); hist = ind.get("macd_hist"); z = ind.get("zclose50")
        if rsi is None or hist is None or z is None:
            return None
        s_hist = 1 if hist > 0 else (-1 if hist < 0 else 0)
        s_z = 1 if z > 0 else (-1 if z < 0 else 0)
        return 0.4 * s_hist + 0.3 * clamp((rsi - 50) / 50.0, -1, 1) + 0.3 * s_z
    m = part(sym_spy)
    if m is None:
        m = part(sym_qqq) if sym_qqq else 0
    return clamp(m if m is not None else 0)

def compute_ts(symbol: str, sym_blob: dict):
    """
    Enhanced TS with volume confirmation, squeeze/expansion, divergence,
    and tiny mean-revert on mega/etf. Returns ts, subterms, tags.
    """
    ind = sym_blob.get("indicators") or {}
    price = sym_blob.get("price")
    risk_class = sym_blob.get("risk_class", "growth")

    # Normalizers
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

    rsi_val = ind.get("rsi14")
    rsi_term = ((rsi_val - 50) / 50.0) if rsi_val is not None else 0.0

    rs_term = ind.get("rel_strength20") or 0.0
    z50 = ind.get("zclose50")
    z_term = (z50 / 3.0) if z50 is not None else 0.0

    # Base TS (your weights)
    ts_core = (0.30 * clamp(macd_slope, -2, 2)
             + 0.20 * clamp(rsi_term, -1, 1)
             + 0.20 * clamp(ema12_slope, -2, 2)
             + 0.15 * clamp(rs_term, -1, 1)
             + 0.15 * clamp(z_term, -1, 1))

    tags = []

    # Volume confirmation
    vol_ratio = ind.get("vol_ratio") or 1.0
    vol_adj = 0.85 if vol_ratio < 0.8 else (1.10 if vol_ratio > 1.5 else 1.0)
    if vol_adj != 1.0:
        tags.append(f"vol_adj:{vol_adj:.2f}")

    # Squeeze / expansion via Bollinger bandwidth
    bw = ind.get("bb_bw20")
    if bw is not None:
        if bw < 0.10:
            tags.append("squeeze")
        elif bw > 0.20 and macd_slope > 0:
            ts_core += 0.05
            tags.append("squeeze_break")

    # Divergence check (recompute RSI series if needed from df)
    div = None
    df = sym_blob.get("df")
    if df is not None and not df.empty:
        px = df['c'].astype(float)
        rsi_ser = rsi_series(px, 14)
        div = simple_divergence(px, rsi_ser, lookback=20)
        if div == "bearish":
            ts_core -= 0.05
            tags.append("div_bear")
        elif div == "bullish":
            ts_core += 0.05
            tags.append("div_bull")

    # Tiny mean-revert on mega/etf to avoid chasing extended moves
    meanrev_term = 0.0
    if risk_class in ("mega","etf"):
        meanrev_term = - (z_term or 0.0) * 0.05
        ts_core += meanrev_term

    ts = clamp(ts_core * vol_adj, -1, 1)

    subterms = {
        "macd_slope": round(macd_slope, 4),
        "ema12_slope": round(ema12_slope, 4),
        "rsi_term": round(rsi_term, 4),
        "rs_term": round(rs_term, 4),
        "z_term": round(z_term, 4),
        "vol_adj": round(vol_adj, 3),
        "meanrev_term": round(meanrev_term, 4),
        "bb_bw20": bw,
        "divergence": div,
    }
    return ts, subterms, tags

def compute_ns(sym_blob: dict, sector_ns: float = None):
    """Symbol-level NS from decayed news + optional sector NS."""
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


# ========= NS decay carry =========
def compute_ns_decay(prev_doc, symbol, fresh_ns_raw, now_dt):
    """Blend previous decayed score with fresh news score. ~50% daily half-life."""
    prev_decay = 0.0
    prev_ts = None
    if prev_doc and isinstance(prev_doc, dict):
        try:
            prev_asof = prev_doc.get("as_of_utc")
            if prev_asof:
                prev_ts = datetime.fromisoformat(prev_asof.replace("Z","+00:00"))
            prev_sym = (prev_doc.get("symbols") or {}).get(symbol) or {}
            prev_ind = prev_sym.get("indicators") or {}
            prev_decay = float(prev_ind.get("ns_decay") or 0.0)
        except Exception:
            prev_decay = 0.0

    # decay factor by elapsed days
    if prev_ts:
        dt_days = max(0.0, (now_dt - prev_ts).total_seconds() / 86400.0)
    else:
        dt_days = 1.0  # assume ~1 day separation if unknown

    decay_factor = 0.5 ** (dt_days / NS_DECAY_HALF_LIFE_DAYS)
    carried = prev_decay * decay_factor

    # simple blend: equal mix of carried and new
    ns_decay = clamp(0.5 * carried + 0.5 * (fresh_ns_raw or 0.0), -1, 1)
    return ns_decay


# ========= Main =========
def main():
    if not (ALPACA_KEY and ALPACA_SECRET):
        raise RuntimeError("Missing ALPACA_KEY / ALPACA_SECRET")

    prev_doc = load_previous_feed_from_gist(GIST_ID, GIST_TOKEN)  # may be None

    # 1) Latest bars
    latest = fetch_latest_bars(SYMBOLS)

    # 2) Historical + indicators (need SPY for rel_strength baseline)
    dfs = {}
    for s in SYMBOLS:
        df = fetch_hist_bars(s, timeframe=IND_TF, limit=HIST_LIMIT)
        if df is not None:
            dfs[s] = df
    print(f"📚 Hist dataframes ready: {len(dfs)} symbols")

    bench_df = dfs.get("SPY")
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

    # 4) Build per-symbol working blobs
    symdata = {}
    now_utc = datetime.now(timezone.utc)
    for s in SYMBOLS:
        b  = latest.get(s) or {}
        df = dfs.get(s)
        prev_c = get_prev_close(df) if df is not None else None
        ind = indicators.get(s, {}).copy()

        # Per-symbol news -> ns_raw + ns_decay (carry)
        sym_news = news_map.get(s, [])
        ns_raw = score_news_items(sym_news) if sym_news else 0.0
        ns_decay = compute_ns_decay(prev_doc, s, ns_raw, now_utc)
        ind["ns_raw"] = round(ns_raw, 3)
        ind["ns_decay"] = round(ns_decay, 3)

        # Blob
        blob = {
            "price": b.get("c"), "volume": b.get("v"), "ts": b.get("t"),
            "o": b.get("o"), "h": b.get("h"), "l": b.get("l"), "c": b.get("c"),
            "indicators": ind, "df": df, "prev_close": prev_c,
            "risk_class": get_risk_class(s)
        }
        blob.update(intraday_metrics(b, prev_c))
        symdata[s] = blob

    # 5) Market regime from BENCH / BENCH_ALT
    mr = compute_market_regime(symdata.get(BENCH, {}), symdata.get(BENCH_ALT, {}))

    # 6) Freshness metadata
    freshness = {"max_age_sec": 6*60*60, "symbols_with_stale_data": []}
    def _age_sec(ts_str):
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z","+00:00"))
            return int((now_utc - dt).total_seconds())
        except Exception:
            return None

    for s, blob in symdata.items():
        age = _age_sec(blob.get("ts")) if blob.get("ts") else None
        if age is not None and age > freshness["max_age_sec"]:
            freshness["symbols_with_stale_data"].append({"symbol": s, "age_sec": age})

    # 7) Assemble JSON (+ explainability / backtest hooks)
    params = {
        "ts_weights": {"macd_slope":0.30,"rsi_term":0.20,"ema12_slope":0.20,"rs_term":0.15,"z_term":0.15},
        "vol_adj_low":0.85,"vol_adj_high":1.10,
        "squeeze_bw_low":0.10,"squeeze_bw_high":0.20,
        "divergence_nudge":0.05,
        "meanrev_on": ["mega","etf"], "meanrev_coeff":0.05
    }

    doc = {
        "as_of_utc": now_utc.isoformat(),
        "feed": ALPACA_FEED,
        "timeframe": IND_TF,
        "indicators_window": HIST_LIMIT,
        "freshness": freshness,
        "run": {
            "version":"v2.2",
            "bench": BENCH,
            "bench_alt": BENCH_ALT,
            "ns_decay": "0.5/day",
            "market_regime": round(mr, 3),
            "params_hash": hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
        },
        "symbols": {}
    }

    for s in SYMBOLS:
        blob = symdata[s]
        ind = blob.get("indicators") or {}

        # Tailwind, TS/NS/CDS/decision
        wt = weekly_tailwind(blob.get("df"))
        ts, subterms, tags = compute_ts(s, blob)
        ns = compute_ns(blob)                # sector_ns placeholder -> 0.0
        cds, wT, wN = compute_cds(ts + (wt or 0.0), ns)
        dec = decision_from_scores(cds, mr, blob)

        # Explainability string
        checks = []
        if ind.get("macd_hist") is not None and ind.get("macd_hist_prev") is not None and ind["macd_hist"] > ind["macd_hist_prev"]:
            checks.append("MACD↑")
        if blob.get("price") and ind.get("ema12") and ind.get("ema26") and blob["price"] > ind["ema12"] > ind["ema26"]:
            checks.append("stackedMAs")
        if (ind.get("rel_strength20") or 0) >= 0:
            checks.append("RS≥0")
        if ind.get("vol_ratio") and ind["vol_ratio"] > 1.5:
            checks.append("vol>1.5×")
        rationale = f"TS:{ts:.2f} NS:{ns:.2f} MR:{mr:.2f} → CDS:{cds:.2f} | " + ",".join(checks or ["no-confirms"])

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
                "decision": dec,
                "rationale": rationale,
                "tags": tags,
                "subterms": subterms
            }
        }

        # keep your per-symbol news if present
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

    # 8) Publish
    if GIST_ID and GIST_TOKEN:
        publish_to_gist(GIST_ID, GIST_TOKEN, "prices.json", content)
    else:
        write_to_repo("data/prices.json", content)

    print("✅ Completed successfully.")


if __name__ == "__main__":
    main()
