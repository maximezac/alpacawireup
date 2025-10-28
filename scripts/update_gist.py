#!/usr/bin/env python3
"""
scripts/update_gist.py
Updates a GitHub Gist file with the latest OUTPUT_PATH JSON.

Env required:
- GH_TOKEN: GitHub token with gist scope (or use GITHUB_TOKEN + API, but GH_TOKEN is simplest)
- GIST_ID: the Gist ID to update
- GIST_FILENAME: target filename in the gist (e.g., prices_final.json)

Optional:
- OUTPUT_PATH: local path to publish (default data/prices_final.json)
"""
import os, sys, json, requests

GH_TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
GIST_ID = os.environ.get("GIST_ID")
GIST_FILENAME = os.environ.get("GIST_FILENAME", "prices_final.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "data/prices_final.json")

def main():
    if not GH_TOKEN or not GIST_ID:
        print("GH_TOKEN and GIST_ID are required.", file=sys.stderr)
        sys.exit(1)
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    url = f"https://api.github.com/gists/{GIST_ID}"
    headers = {"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github+json"}
    payload = {"files": {GIST_FILENAME: {"content": content}}}
    r = requests.patch(url, headers=headers, json=payload, timeout=20)
    if r.status_code >= 300:
        print(f"Failed to update gist: {r.status_code} {r.text}", file=sys.stderr)
        sys.exit(2)
    print(f"Gist {GIST_ID} updated file {GIST_FILENAME}.")

if __name__ == "__main__":
    main()
