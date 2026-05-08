"""
Data ingestion for GitLab Handbook and Direction pages.

Fetches the public sitemap, downloads each HTML page, extracts the main
article content, and writes one JSON record per page to data/raw_pages.jsonl.

Usage:
    python -m src.ingest --limit 50            # quick test
    python -m src.ingest --limit 500           # demo-sized corpus
    python -m src.ingest                       # full ingest (slow, lots of pages)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.config import settings

# --- Constants -------------------------------------------------------------

SITEMAP_URL = "https://handbook.gitlab.com/sitemap.xml"
DIRECTION_SITEMAP_URL = "https://about.gitlab.com/sitemap.xml"

# Only keep direction pages from the about.gitlab.com sitemap
DIRECTION_PREFIX = "https://about.gitlab.com/direction/"

USER_AGENT = "genai-gitlab-handbook-chatbot/0.1 (educational project)"
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_REQUESTS = 0.2  # be polite

# Sitemap XML namespace
SM_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


# --- Data model ------------------------------------------------------------

@dataclass
class Page:
    url: str
    title: str
    content: str
    source: str  # "handbook" or "direction"


# --- Helpers ---------------------------------------------------------------

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _fetch_sitemap_urls(session: requests.Session, sitemap_url: str) -> list[str]:
    """Return all <loc> URLs from a sitemap, recursing into sitemap indexes."""
    resp = session.get(sitemap_url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    # Sitemap index → recurse
    if root.tag.endswith("sitemapindex"):
        urls: list[str] = []
        for sm in root.findall("sm:sitemap/sm:loc", SM_NS):
            if sm.text:
                urls.extend(_fetch_sitemap_urls(session, sm.text.strip()))
                time.sleep(SLEEP_BETWEEN_REQUESTS)
        return urls

    # Regular sitemap
    return [
        loc.text.strip()
        for loc in root.findall("sm:url/sm:loc", SM_NS)
        if loc.text
    ]


def _collect_urls(session: requests.Session) -> list[tuple[str, str]]:
    """Returns list of (url, source_label) tuples for handbook + direction."""
    print("Fetching handbook sitemap...")
    handbook_urls = _fetch_sitemap_urls(session, SITEMAP_URL)
    print(f"  {len(handbook_urls)} handbook URLs")

    print("Fetching direction sitemap...")
    direction_all = _fetch_sitemap_urls(session, DIRECTION_SITEMAP_URL)
    direction_urls = [u for u in direction_all if u.startswith(DIRECTION_PREFIX)]
    print(f"  {len(direction_urls)} direction URLs (filtered)")

    pairs: list[tuple[str, str]] = []
    pairs.extend((u, "handbook") for u in handbook_urls)
    pairs.extend((u, "direction") for u in direction_urls)

    # Dedupe while preserving order
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for u, src in pairs:
        if u not in seen:
            seen.add(u)
            unique.append((u, src))
    return unique


def _clean_text(text: str) -> str:
    """Collapse whitespace and strip."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_main_content(html: str) -> tuple[str, str]:
    """Extract (title, content_text) from a page's HTML."""
    soup = BeautifulSoup(html, "lxml")

    # Title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]):
        tag.decompose()

    # Try common main-content containers, fall back to <body>
    candidates = [
        soup.find("main"),
        soup.find("article"),
        soup.find(attrs={"role": "main"}),
        soup.find("div", class_=re.compile(r"(content|main|post|article)", re.I)),
    ]
    main = next((c for c in candidates if c is not None), soup.body or soup)

    content = _clean_text(main.get_text("\n"))
    return title, content


def _fetch_page(session: requests.Session, url: str, source: str) -> Page | None:
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return None
        # Skip non-HTML responses
        ctype = resp.headers.get("Content-Type", "")
        if "html" not in ctype.lower():
            return None
        title, content = _extract_main_content(resp.text)
        if len(content) < 200:  # very short pages = likely landing/index
            return None
        return Page(url=url, title=title, content=content, source=source)
    except (requests.RequestException, ET.ParseError):
        return None


# --- Main ------------------------------------------------------------------

def ingest(limit: int | None, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session = _session()

    all_urls = _collect_urls(session)
    if limit:
        all_urls = all_urls[:limit]

    print(f"\nFetching {len(all_urls)} pages...\n")

    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as f:
        for url, src in tqdm(all_urls, unit="page"):
            page = _fetch_page(session, url, src)
            if page is None:
                skipped += 1
                continue
            f.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
            written += 1
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    print(f"\nDone. Wrote {written} pages, skipped {skipped}.")
    print(f"Output: {output_path}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest GitLab Handbook + Direction pages.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of pages to fetch (default: all).")
    parser.add_argument("--output", type=Path,
                        default=settings.data_dir / "raw_pages.jsonl",
                        help="Output JSONL path.")
    args = parser.parse_args(argv)

    n = ingest(limit=args.limit, output_path=args.output)
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())