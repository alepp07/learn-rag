# Ask My Documents — a RAG pipeline built from scratch

A small, dependency-light Retrieval-Augmented Generation (RAG) system: documents in, grounded answers out, with every step written in plain Python so nothing is hidden behind a framework.

Ask a question about any PDF, Markdown, or text file you drop into `data/`, and get an answer generated only from what's actually in your documents — with the exact source chunks and similarity scores shown alongside it.

<!-- Add a screenshot or two here once you have them, e.g.: -->
<!-- ![App screenshot](docs/screenshot.png) -->

## What it does

- Loads and chunks documents (`.txt`, `.md`, `.pdf`)
- Embeds each chunk locally (no API cost, runs on CPU)
- Retrieves the most relevant chunks for a question using cosine similarity
- Builds a grounded prompt and generates an answer with an LLM
- Renders it all in a chat interface with conversation history, markdown/table rendering, source citations, and a light/dark theme

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| PDF/text extraction | [PyMuPDF](https://pymupdf.readthedocs.io/) | More tolerant of malformed PDF structure than alternatives I tried first |
| Chunking | [tiktoken](https://github.com/openai/tiktoken) | Token-accurate splitting instead of naive character counts |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) | Runs locally, free, no API key |
| Vector storage | numpy | A plain array is enough at this scale — no vector DB needed yet |
| Generation | [Groq](https://console.groq.com/) API (`openai/gpt-oss-20b`) | Free tier, fast inference |
| UI | [Streamlit](https://streamlit.io/) | Fast to build a real chat interface in pure Python |

## Setup

```bash
git clone <your-repo-url>
cd learn-rag
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

Get a free API key from [console.groq.com](https://console.groq.com) (no card required), then set it:

```bash
$env:GROQ_API_KEY="your-key-here"    # PowerShell
# export GROQ_API_KEY="your-key-here"  # macOS/Linux
```

## Run it

**Command line:**
```bash
python main.py build
python main.py ask "What is this document about?"
```

**Web UI:**
```bash
streamlit run app.py
```

Drop your own `.pdf`/`.md`/`.txt` files into `data/`, rebuild the index from the sidebar, and start asking questions.

## Project structure

```
learn-rag/
├── data/              <- source documents
├── assets/            <- UI avatar image
├── src/
│   ├── ingest.py      <- load files, split into token-based chunks
│   ├── embed_store.py <- local embeddings, vector index, cosine search
│   └── generate.py    <- prompt construction, LLM call
├── app.py             <- Streamlit chat UI
├── main.py            <- CLI entry point
└── requirements.txt
```

---

## The learning plan I followed

This project started as a way to actually understand RAG rather than just read about it. Below is the day-by-day plan I used — kept here in case it's useful to you too.

Don't just run it — read each file in order and change one thing before moving to the next. That's what builds understanding.

### Day 1 — Ingestion & chunking (`src/ingest.py`)
Read `load_and_chunk_directory` and `chunk_text`. Run:
```bash
python src/ingest.py
```
**Try:** change `chunk_size` from 400 to 100, rerun, and look at how much more fragmented the sample chunk looks. This is the central chunking trade-off: small chunks are precise but lose surrounding context; large chunks keep context but blur together multiple ideas, hurting retrieval.

### Day 2 — Embeddings & the vector index (`src/embed_store.py`)
Read `embed_texts` and `cosine_similarity`. Run:
```bash
python main.py build
```
**Try:** in a Python shell, embed two similar sentences and two unrelated ones, and print their cosine similarity manually. Seeing the actual numbers (similar ≈ 0.5-0.9, unrelated ≈ 0.0-0.3) makes "semantic similarity" concrete instead of abstract.

### Day 3 — Retrieval quality
Run `search()` directly with a few questions and inspect what comes back *before* touching generation at all:
```python
from src.embed_store import search
for r in search("your question here"):
    print(r['score'], r['source'], r['text'][:100])
```
**Try:** ask a question your data does NOT cover. Notice retrieval still returns its top-k chunks with a score — cosine similarity always returns *something*, even mediocre matches. This is why retrieval quality checks matter.

### Day 4 — Augmentation & generation (`src/generate.py`)
Read `build_prompt`. This is the "AG" in RAG, and it's almost embarrassingly simple — that's the point.
**Try:** remove the "If the context doesn't contain the answer, say so honestly" instruction and ask an out-of-scope question again. Watch the model start guessing instead of admitting it doesn't know.

### Day 5 — Evaluate retrieval, not just vibes
Write 10-15 question/answer pairs you already know the answer to. For each, check whether the *correct* chunk was actually retrieved in the top-k — not whether the final answer sounded right. Most RAG bugs are retrieval bugs wearing a generation costume.

## Stretch goals

- **Semantic chunking** — split on paragraph/section boundaries instead of fixed token counts
- **Hybrid search** — combine embeddings with BM25 keyword search for exact-term queries (names, codes)
- **Reranking** — retrieve top-20 by embedding similarity, then rerank with a cross-encoder for better precision
- **A real vector store** — Chroma or FAISS once the corpus is too large to hold in memory
- **Contextual retrieval** — prepend an LLM-generated summary blurb to each chunk before embedding, to help disambiguate similar-looking chunks
