#!/usr/bin/env python3
"""Reusable fetch helper. Saves RAW payload to raw/{source}/{name} BEFORE any
parsing (prime directive: every row must trace to a saved artifact)."""
import os, sys, json, hashlib
import datetime as dt
import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

def utcnow():
    # Date.now() analog; explicit UTC ISO string
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def fetch(url, source, name=None, extra_headers=None, timeout=45, save=True):
    """Fetch url, save raw bytes under raw/<source>/<name>. Returns dict with
    status, path, len, content-type, retrieved_at, and .text/.content."""
    h = dict(HEADERS)
    if extra_headers:
        h.update(extra_headers)
    r = requests.get(url, headers=h, timeout=timeout)
    ct = r.headers.get("Content-Type","")
    retrieved = utcnow()
    if name is None:
        # derive a filename
        base = hashlib.sha1(url.encode()).hexdigest()[:12]
        ext = ".json" if "json" in ct else (".html" if "html" in ct else ".bin")
        name = base + ext
    path = None
    if save and r.status_code == 200:
        d = os.path.join("raw", source)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name)
        with open(path,"wb") as f:
            f.write(r.content)
    return {
        "url": url, "status": r.status_code, "content_type": ct,
        "len": len(r.content), "path": path, "retrieved_at": retrieved,
        "text": r.text, "content": r.content, "headers": dict(r.headers),
    }

if __name__ == "__main__":
    # quick probe of URLs passed as args: python3 fetch_util.py URL SOURCE
    url = sys.argv[1]; source = sys.argv[2] if len(sys.argv)>2 else "probe"
    res = fetch(url, source)
    print(json.dumps({k:res[k] for k in ("url","status","content_type","len","path","retrieved_at")}, indent=2))
    print("--- first 800 chars ---")
    print(res["text"][:800])
