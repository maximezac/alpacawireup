# scripts/publish_feed.py
import os, json, requests
from datetime import datetime, timezone

ALPACA_KEY    = os.getenv("ALPACA_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
FEED          = os.getenv("ALPACA_FEED", "iex")  # "iex" (free) or "sip" (paid)

# replace the SYMBOLS line with this:
def load_symbols():
    p = "tickers.txt"
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return [line.strip().upper() for line in f if line.strip()]
    return [s.strip().upper() for s in os.getenv("SYMBOLS","RKLB,ASTS,VTI,VOO,TTWO").split(",")]

SYMBOLS = load_symbols()


GIST_ID       = os.getenv("GIST_ID")             # hex ID from your gist URL
GITHUB_TOKEN  = os.getenv("GIST_TOKEN")          # PAT with 'gist' scope

DATA_BASE = "https://data.alpaca.markets"
H = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}

def latest_bars(symbols):
    url = f"{DATA_BASE}/v2/stocks/bars/latest"
    r = requests.get(url, headers=H, params={"symbols":",".join(symbols),"feed":FEED}, timeout=15)
    r.raise_for_status()
    return r.json().get("bars", {})

def latest_news(symbols, limit=12):
    url = f"{DATA_BASE}/v1beta1/news"
    r = requests.get(url, headers=H, params={"symbols":",".join(symbols), "limit": limit}, timeout=15)
    r.raise_for_status()
    return r.json().get("news", [])

def publish_to_gist(gist_id, token, filename, content_str):
    url = f"https://api.github.com/gists/{gist_id}"
    payload = {"files": {filename: {"content": content_str}}}
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    r = requests.patch(url, headers=headers, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()

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
    if not (GIST_ID and GITHUB_TOKEN):
        raise RuntimeError("Missing GIST_ID or GIST_TOKEN env vars.")
    content = json.dumps(doc, separators=(",", ":"), ensure_ascii=False)
    publish_to_gist(GIST_ID, GITHUB_TOKEN, "prices.json", content)

if __name__ == "__main__":
    for var in ["ALPACA_KEY","ALPACA_SECRET","GIST_ID","GIST_TOKEN"]:
        if not os.getenv(var):
            raise RuntimeError(f"Missing env var: {var}")
    main()
