import os, json, argparse, pathlib, datetime
from openai import OpenAI

def iso_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def ensure_out():
    outdir = pathlib.Path("out")
    outdir.mkdir(exist_ok=True)
    return outdir

def load_feed(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_simple(client: OpenAI, feed: dict, gist_url: str):
    # Sends JSON inline with a short instruction. No persistence/thread.
    system = (
        "You are GPT-5 Thinking acting as my trading simulator. "
        "A new prices.json feed is attached in the user message as text."
    )
    user = (
        "Use this feed snapshot for future analysis requests in this conversation. "
        "Verify metadata (as_of_utc, timeframe, indicators_window). "
        f"If you need to re-fetch later, use this URL: {gist_url}\n\n"
        f"{json.dumps(feed, separators=(',', ':'))}"
    )
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ],
        temperature=0.0
    )
    return resp.choices[0].message.content

def run_assistants(client: OpenAI, feed: dict, gist_url: str):
    """
    Posts to a dedicated Assistants thread so your uploads are kept together.
    Requires:
      - OPENAI_ASSISTANT_ID (created once in your account)
      - OPENAI_THREAD_ID (create once and save to secrets), or we create a new thread on first run
    """
    assistant_id = os.getenv("OPENAI_ASSISTANT_ID")
    if not assistant_id:
        raise RuntimeError("OPENAI_ASSISTANT_ID is required for --mode assistants")

    thread_id = os.getenv("OPENAI_THREAD_ID")
    if not thread_id:
        # Create a new thread on first run (optional); store ID in repo secrets later
        thread = client.beta.threads.create()
        thread_id = thread.id
        print("Created new thread_id:", thread_id)

    content = (
        "New market feed snapshot uploaded.\n"
        f"- Source (canonical): {gist_url}\n"
        f"- as_of_utc: {feed.get('as_of_utc')}\n"
        f"- timeframe: {feed.get('timeframe')} | indicators_window: {feed.get('indicators_window')}\n"
        "JSON payload attached below.\n\n"
        f"{json.dumps(feed, separators=(',', ':'))}"
    )

    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content
    )

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions=("Acknowledge receipt and briefly validate the feed metadata "
                      "so logs show the snapshot is stored.")
    )

    # Poll to completion (kept simple)
    while run.status in ("queued", "in_progress"):
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)

    # Pull messages
    msgs = client.beta.threads.messages.list(thread_id=thread_id)
    texts = []
    for m in msgs.data:
        if m.role == "assistant":
            for part in m.content:
                if part.type == "text":
                    texts.append(part.text.value)
    return thread_id, "\n\n---\n\n".join(reversed(texts)) if texts else "(no assistant text)"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed", required=True, help="Path to data/prices.json")
    parser.add_argument("--gist-url", required=True, help="Canonical raw Gist URL for this feed")
    parser.add_argument("--mode", choices=["simple","assistants"], default="assistants")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    feed = load_feed(args.feed)

    outdir = ensure_out()
    ts = iso_now().replace(":", "-")

    if args.mode == "simple":
        text = run_simple(client, feed, args.gist_url)
        p = outdir / f"upload_simple_{ts}.md"
        p.write_text(text, encoding="utf-8")
        print("Saved:", p)
    else:
        thread_id, text = run_assistants(client, feed, args.gist_url)
        p = outdir / f"upload_assistants_{ts}.md"
        p.write_text(f"(thread_id: {thread_id})\n\n{text}", encoding="utf-8")
        print("Saved:", p)

if __name__ == "__main__":
    main()
