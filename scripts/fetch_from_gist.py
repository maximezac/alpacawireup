#!/usr/bin/env python3
import json, os, sys, urllib.request

gist_id  = os.environ.get("GIST_ID")
token    = os.environ.get("GIST_TOKEN")

if not gist_id or not token:
    print("❌ Missing GIST_ID or GIST_TOKEN env vars", file=sys.stderr)
    sys.exit(1)

def fetch(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req) as r:
        return r.read()

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "User-Agent": "actions/market-feed"
}

try:
    meta = json.loads(fetch(f"https://api.github.com/gists/{gist_id}", headers))
except Exception as e:
    print(f"❌ Failed to read gist metadata: {e}", file=sys.stderr)
    sys.exit(1)

files = meta.get("files", {}) or {}

def try_download(name: str) -> bool:
    f = files.get(name)
    if not f or not f.get("raw_url"):
        return False
    try:
        raw = fetch(f["raw_url"], {"Authorization": f"Bearer {token}"})
        os.makedirs("data", exist_ok=True)
        with open("data/prices_full.json", "wb") as out:
            out.write(raw)
        print(f"✅ Downloaded {name} -> data/prices_full.json")
        return True
    except Exception as e:
        print(f"⚠️ Failed downloading {name}: {e}", file=sys.stderr)
        return False

if not (try_download("prices_full.json") or try_download("prices.json")):
    print("❌ Neither prices_full.json nor prices.json found in the Gist.", file=sys.stderr)
    sys.exit(2)
