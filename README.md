# GenAI GitLab Handbook Chatbot

An interactive RAG-based chatbot that answers questions about GitLab's public [Handbook](https://handbook.gitlab.com/handbook/) and [Direction](https://about.gitlab.com/direction/) pages, with inline source citations.

Built as a project submission demonstrating retrieval-augmented generation, guardrail design, and end-to-end deployment.

---

## Demo

![demo screenshot — replace with your own once deployed](docs/demo.png)

**Live URL:** _to be added once deployed — see "Deployment" below._

---

## What it does

- Ingests GitLab's public Handbook and Direction pages.
- Embeds the content into a FAISS vector index for fast semantic retrieval.
- Answers user questions by retrieving relevant passages and grounding an OpenAI chat model strictly in that context.
- Cites every claim with bracketed numbers `[1]`, `[2]` that link back to the original GitLab URLs.
- Refuses to answer when retrieval confidence is low, rather than guessing.

## Architecture

```
                              ┌──────────────────┐
   GitLab Handbook ──────────▶│   Ingestion      │
   GitLab Direction           │  (sitemap +      │
                              │   HTTP fetch)    │
                              └────────┬─────────┘
                                       │  data/raw_pages.jsonl
                                       ▼
                              ┌──────────────────┐
                              │   Chunking       │
                              │ (token-aware,    │
                              │  sentence-pref)  │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │   Embedding      │
                              │ (OpenAI          │
                              │  text-embed-3)   │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │   FAISS Index    │
                              │  (cosine sim     │
                              │   via IndexFlatIP│
                              │   on normalized) │
                              └────────┬─────────┘
                                       │
   User question  ─────────────────────┤
                                       ▼
                              ┌──────────────────┐
                              │   Retrieval +    │
                              │   Guardrails     │
                              │ (top-k, sim      │
                              │  threshold,      │
                              │  input checks)   │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │   OpenAI Chat    │
                              │  (grounded,      │
                              │   streamed,      │
                              │   cited)         │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │   Streamlit UI   │
                              └──────────────────┘
```

## Tech stack

| Layer       | Choice                                    | Why                                             |
|-------------|-------------------------------------------|-------------------------------------------------|
| UI          | Streamlit                                 | Native chat components, fast deploy             |
| LLM         | OpenAI `gpt-4o-mini`                      | Cheap, fast, strong on structured grounding     |
| Embeddings  | OpenAI `text-embedding-3-small`           | 1536-dim, low cost, solid retrieval quality     |
| Vector DB   | FAISS (`IndexFlatIP` on normalized vecs)  | In-process, no infra, cosine via inner product  |
| Ingestion   | `requests` + `BeautifulSoup` + sitemaps   | Canonical URLs for citations, no git clone      |
| Tokenizer   | `tiktoken` (cl100k_base)                  | Predictable chunk sizes in tokens, not chars    |

## Quick start

Requirements: Python 3.10+, an OpenAI API key, ~$0.20 of embedding credit for a one-time index build.

```bash
# 1. Clone and enter
git clone https://github.com/<your-username>/genai-gitlab-handbook-chatbot.git
cd genai-gitlab-handbook-chatbot

# 2. Create + activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
cp .env.example .env
# Open .env and replace the placeholder with your real OPENAI_API_KEY

# 5. Verify setup
python scripts/verify_setup.py

# 6. Ingest GitLab content (~10 min, free)
python -m src.ingest --limit 1500

# 7. Build the FAISS index (~5-10 min, ~$0.15 in embedding credits)
python -m src.build_index --rebuild

# 8. Verify the pipeline works
python scripts/verify_rag.py

# 9. Run the chatbot
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## Project structure

```
genai-gitlab-handbook-chatbot/
├── app.py                  Streamlit entrypoint
├── requirements.txt
├── .env.example            Template — copy to .env and fill in
├── data/
│   └── raw_pages.jsonl     Output of ingestion
├── index/
│   ├── faiss.index         Vector index
│   ├── chunks.jsonl        Aligned chunk metadata
│   └── meta.json           Index build summary
├── scripts/                One verifier per pipeline stage
│   ├── verify_setup.py
│   ├── verify_ingest.py
│   ├── verify_index.py
│   ├── verify_rag.py
│   └── verify_guardrails.py
└── src/
    ├── config.py           Loads .env, single source of truth
    ├── ingest.py           Sitemap-driven page fetcher
    ├── chunking.py         Token-aware, sentence-preferring splitter
    ├── embeddings.py       OpenAI embedding client with batching + retries
    ├── build_index.py      Chunks → embeds → writes FAISS index
    ├── retriever.py        Loads index, runs semantic search
    ├── guardrails.py       Input checks + similarity-threshold gating
    └── rag.py              End-to-end orchestration + streaming
```

## Key design decisions

**Sitemap-driven ingestion over git clone.** The full GitLab handbook repo is several GB. Using public sitemaps gives us canonical URLs (which become citations for free) without cloning anything large, and decouples ingestion from any specific repo layout.

**Seed URLs guaranteed in the corpus.** Sitemaps don't rank by importance. Without intervention, an arbitrarily limited corpus might miss the values page or communication page entirely. The ingester pins a list of high-value seed URLs that always land regardless of `--limit`.

**Token-aware chunking with sentence-boundary preference.** Character-based splitting cuts mid-sentence and makes "chunk size" meaningless for embedding context. We split on tokens using `tiktoken`, preferring paragraph then sentence boundaries within each window.

**Cosine similarity via normalized vectors + `IndexFlatIP`.** Inner product on L2-normalized vectors equals cosine similarity in `[-1, 1]`. This means the `SIMILARITY_THRESHOLD` env var has a meaningful, interpretable value that we can reason about and tune.

**Refusal before LLM call when retrieval is weak.** Low top-similarity queries (`< 0.30`) refuse without round-tripping to the LLM. Deterministic, cheaper, and removes a class of "smart-sounding but tangential" hallucinated answers.

**Confidence as a third state, not binary.** Scores between 0.30 and 0.45 land in a `low_confidence` bucket — we still answer but show a visible warning. More honest than pass/fail and more useful than always answering.

## Guardrails

This project ships with these guardrails, all centralized in [`src/guardrails.py`](src/guardrails.py):

- **Input length checks** — minimum 3 characters, maximum 1000.
- **Prompt-injection screening** — regex-based detection of common override-instruction patterns (`ignore previous instructions`, `act as`, `<system>`, etc.). Note: this is a deflection layer for casual cases, not a security boundary against a determined adversary.
- **PII detection (informational)** — emails and US-style phone numbers flagged but not blocked.
- **Similarity threshold gating** — queries below `SIMILARITY_THRESHOLD` refuse before the LLM is called.
- **Strict grounding via system prompt** — the LLM is instructed to answer only from retrieved passages and to use a fixed refusal phrase otherwise.
- **Source citation transparency** — every answer must cite numbered sources that link back to verifiable GitLab URLs. Users can audit any claim.
- **Visible disclaimer** — the UI banner makes the unofficial, LLM-generated nature unmissable.

### Known limitations

Honest about what is NOT addressed:

- **No defense against prompt injection in retrieved content** — if someone edited GitLab's public handbook to embed instructions, those would reach the LLM as part of context. Real concern in production RAG; out of scope here.
- **No multi-turn conversational context** — each user turn is treated independently. Follow-up questions like "what about that other one?" won't work as intended.
- **Coverage is limited to the ingested seed + sitemap subset** — currently ~1500 pages out of ~10,000+ available. Questions about niche handbook sections may not retrieve well.
- **No moderation model** — relies on the system prompt and OpenAI's own upstream filters.

## Configuration

All runtime config lives in `.env`. See `.env.example` for the full list.

| Variable               | Default                     | Notes                                       |
|------------------------|-----------------------------|---------------------------------------------|
| `OPENAI_API_KEY`       | _(required)_                | Your OpenAI key                             |
| `EMBEDDING_MODEL`      | `text-embedding-3-small`    | 1536-dim                                    |
| `CHAT_MODEL`           | `gpt-4o-mini`               | Used for response generation                |
| `CHUNK_SIZE`           | `500`                       | Tokens per chunk                            |
| `CHUNK_OVERLAP`        | `50`                        | Tokens of overlap between chunks            |
| `TOP_K`                | `5`                         | Chunks retrieved per query                  |
| `SIMILARITY_THRESHOLD` | `0.3`                       | Below this cosine sim → refuse              |

## Verification scripts

Each pipeline stage has a dedicated verifier so you can validate the project end-to-end without running the UI:

```bash
python scripts/verify_setup.py        # env, deps, API key
python scripts/verify_ingest.py       # ingestion output health
python scripts/verify_index.py        # FAISS index + sanity-query results
python scripts/verify_rag.py          # full RAG pipeline answers
python scripts/verify_guardrails.py   # guardrail unit tests
```

`verify_rag.py` doubles as a CLI: pass a question as an argument to query directly.

```bash
python scripts/verify_rag.py "How does GitLab handle remote work?"
```

## Shortcut commands

A `Makefile` is included for common operations:

```bash
make install    # install Python dependencies
make ingest     # fetch GitLab handbook + direction pages
make index      # build the FAISS index
make rebuild    # full pipeline: ingest + index from scratch
make verify     # run all verification scripts
make run        # launch the Streamlit app
make clean      # remove generated data and indexes
```

## Deployment

_To be added — planned deployment on [gaaten.com](https://gaateh.com)._

## Cost notes

One-time:

- Index build for ~1500 pages: ~$0.15 (one embedding call per chunk).

Per query:

- One embedding call (query) + one chat call (`gpt-4o-mini` with ~2k tokens of context): roughly $0.0004.

