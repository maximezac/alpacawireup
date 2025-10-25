import os, json, requests
from datetime import datetime, timezone

# --- Config ---
ALPACA_KEY    = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
FEED          = os.getenv("ALPACA_FEED", "iex")
GIST_ID       = os.getenv("GIST_ID")
GITHUB_TOKEN  = os.getenv("GIST_TOKEN")
DRY_RUN       = os.getenv("DRY_RUN", "false").lower() == "true"

DATA_BASE = "https://data.alpaca.markets"
H = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

# ---------- LOAD SYMBOLS ----------
def load_list(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip().upper() for ln in f if ln.strip() and not ln.strip().startswith("#")]
            return lines
    return []

def load_symbols():
    base = load_list("tickers.txt")
    watch = load_list("watchlist.txt")
    env = [s.strip().upper() for s in os.getenv("SYMBOLS", "").split(",") if s.strip()]
    all_syms = base + watch + env
    unique_syms = []
    seen = set()
    for s in all_syms:
        if s not in seen:
            seen.add(s)
            unique_syms.append(s)
    return unique_syms

SYMBOLS = load_symbols()
print(f"🧾 Loaded {len(SYMBOLS)} total tickers (tickers.txt + watchlist.txt + env):")
print(", ".join(SYMBOLS[:25]) + ("..." if len(SYMBOLS) > 25 else ""))

# ---------- FETCHING ----------
def latest_bars(symbols):
    url = f"{DATA_BASE}/v2/stocks/bars/latest"
    print(f"🔄 Fetching latest bars for {len(symbols)} tickers...")
    r = requests.get(url, headers=H, params={"symbols": ",".join(symbols), "feed": FEED}, timeout=20)
    print(f"📡 Bars response status: {r.status_code}")
    r.raise_for_status()
    data = r.json().get("bars", {})
    retrieved = list(data.keys())
    print(f"✅ Retrieved {len(retrieved)} bars")
    missing = [s for s in symbols if s not in retrieved]
    if missing:
        print(f"⚠️ Missing {len(missing)} tickers: {', '.join(missing[:15])}{'...' if len(missing)>15 else ''}")
    return data

def latest_news(symbols, limit=12):
    url = f"{DATA_BASE}/v1beta1/news"
    print(f"📰 Fetching latest news for {len(symbols)} tickers...")
    r = requests.get(url, headers=H, params={"symbols": ",".join(symbols), "limit": limit}, timeout=20)
    print(f"📡 News response: {r.status_code}")
    r.raise_for_status()
    return r.json().get("news", [])

# ---------- GIST PUBLISH ----------
def publish_to_gist(gist_id, token, filename, content_str):
    url = f"https://api.github.com/gists/{gist_id}"
    payload = {"files": {filename: {"content": content_str}}}
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    r = requests.patch(url, headers=headers, json=payload, timeout=15)
    print(f"💾 Gist update status: {r.status_code}")
    r.raise_for_status()
    return r.json()

# ---------- MAIN ----------
def main():
    bars = latest_bars(SYMBOLS)
    news = latest_news(SYMBOLS, limit=12)
    doc = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "feed": FEED,
        "symbols": {
            s: {
                "price": (bars.get(s) or {}).get("c"),
                "volume": (bars.get(s) or {}).get("v"),
                "ts": (bars.get(s) or {}).get("t")
            } for s in SYMBOLS
        },
        "news": [
            {
                "symbols": n.get("symbols"),
                "headline": n.get("headline"),
                "summary": n.get("summary"),
                "source": n.get("source"),
                "url": n.get("url"),
                "created_at": n.get("created_at")
            } for n in news
        ]
    }

    content = json.dumps(doc, separators=(",", ":"), ensure_ascii=False)
    if DRY_RUN:
        print("🧪 Dry run mode — showing first 400 chars of output:")
        print(content[:400])
        return

    publish_to_gist(GIST_ID, GITHUB_TOKEN, "prices.json", content)
    print("✅ Completed successfully.")

if __name__ == "__main__":
    for var in ["ALPACA_KEY", "ALPACA_SECRET"]:
        if not os.getenv(var):
            raise RuntimeError(f"Missing required env var: {var}")
    main()
