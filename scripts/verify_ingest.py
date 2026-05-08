"""
Step 2 verification — sanity-check the ingestion output.
Run from project root:  python scripts/verify_ingest.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings


def main() -> int:
    path = settings.data_dir / "raw_pages.jsonl"
    print("=" * 60)
    print("Step 2: Ingestion Verification")
    print("=" * 60)

    if not path.exists():
        print(f"  [FAIL]  File not found: {path}")
        print("          Run:  python -m src.ingest --limit 30")
        return 1

    pages = []
    bad_lines = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pages.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1

    if not pages:
        print("  [FAIL]  No valid pages found.")
        return 1

    sources = Counter(p.get("source", "?") for p in pages)
    lengths = [len(p.get("content", "")) for p in pages]
    avg_len = sum(lengths) // len(lengths)
    titled = sum(1 for p in pages if p.get("title"))

    print(f"\n  [PASS]  Pages parsed:           {len(pages)}")
    print(f"  [{'PASS' if bad_lines == 0 else 'WARN'}]  Malformed lines:        {bad_lines}")
    print(f"  [PASS]  Sources:                {dict(sources)}")
    print(f"  [PASS]  Avg content length:     {avg_len} chars")
    print(f"  [PASS]  Min / max length:       {min(lengths)} / {max(lengths)} chars")
    print(f"  [PASS]  Pages with title:       {titled} / {len(pages)}")

    # Show a sample
    print("\n  Sample record:")
    sample = pages[0]
    print(f"    url:     {sample.get('url')}")
    print(f"    title:   {sample.get('title')!r}")
    print(f"    source:  {sample.get('source')}")
    print(f"    content: {sample.get('content', '')[:200]!r}...")

    print("\n  Verification complete. Ready for Step 3 if results look reasonable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())