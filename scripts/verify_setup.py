"""
Step 1 verification — confirms project setup is working.
Run from project root:  python scripts/verify_setup.py
"""
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    print("=" * 60)
    print("Step 1: Setup Verification")
    print("=" * 60)

    checks = []

    # Python version
    py = sys.version_info
    py_ok = py.major == 3 and py.minor >= 10
    checks.append(("Python >= 3.10", py_ok, f"{py.major}.{py.minor}.{py.micro}"))

    # Required imports
    deps = [
        "streamlit", "openai", "dotenv", "faiss",
        "numpy", "tiktoken", "requests", "bs4",
        "lxml", "markdown_it", "tqdm",
    ]
    for d in deps:
        try:
            __import__(d)
            checks.append((f"import {d}", True, "ok"))
        except ImportError as e:
            checks.append((f"import {d}", False, str(e)))

    # Config / env loading
    try:
        from src.config import settings
        key = settings.openai_api_key
        key_ok = bool(key) and key.startswith("sk-") and key != "sk-replace-with-your-key"
        masked = (key[:6] + "..." + key[-4:]) if key_ok else "missing or placeholder"
        checks.append(("OPENAI_API_KEY loaded", key_ok, masked))
        checks.append(("Embedding model configured", True, settings.embedding_model))
        checks.append(("Chat model configured", True, settings.chat_model))
    except Exception as e:
        checks.append(("Config module", False, str(e)))

    # Print results
    print()
    all_ok = True
    for name, ok, info in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}]  {name:<35} {info}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All checks passed. Ready for Step 2.")
        return 0
    print("Some checks failed. Fix the issues above before proceeding.")
    return 1


if __name__ == "__main__":
    sys.exit(main())