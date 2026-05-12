"""
Step 4 verification — runs the full RAG pipeline end-to-end.

  python scripts/verify_rag.py                           # canned test queries
  python scripts/verify_rag.py "your question here"      # ad-hoc query
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag import RagPipeline


CANNED_QUERIES = [
    "What are GitLab's core values?",
    "How does GitLab handle asynchronous communication?",
    "What is GitLab's product direction or vision?",
    # An out-of-scope question to test the refusal behavior
    "What is the capital of Mongolia?",
]


def _print_response(pipe: RagPipeline, question: str) -> None:
    print("\n" + "=" * 70)
    print(f"Q: {question}")
    print("=" * 70)

    resp = pipe.answer(question)

    print(f"\n[top similarity score: {resp.top_score:.3f}]")
    print(f"\nAnswer:\n{resp.answer}")

    if resp.sources:
        print("\nSources:")
        for s in resp.sources:
            print(f"  [{s.n}] ({s.source}, score={s.best_score:.3f}) "
                  f"{s.title[:70]}")
            print(f"      {s.url}")
    else:
        print("\n(no sources returned)")


def main() -> int:
    print("Loading RAG pipeline...")
    pipe = RagPipeline()
    print("Ready.\n")

    args = sys.argv[1:]
    if args:
        _print_response(pipe, " ".join(args))
        return 0

    for q in CANNED_QUERIES:
        _print_response(pipe, q)
    return 0


if __name__ == "__main__":
    sys.exit(main())