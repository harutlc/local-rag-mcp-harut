# Local RAG/MCP Knowledge Base Assistant

# 📋 The Problem

- **Growing Documentation**: Knowledge scattered across files
- **Information Retrieval**: Hard to find answers without keywords
- **Privacy Concerns**: Cloud solutions may not comply with policies

```
Users → Search → Answer = 😫
```

# 📋 The Problem with Vector-Only RAG

The first version searched **only** by vector similarity. Small local models
and small embedding models lose accuracy exactly where it hurts:

- Rare terms, exact abbreviations, file names, command names
- A question phrased differently from the wording in the documents
- One query = one search = one chance to hit

```
"what is the 4096 limit?" → one embedding → miss
```

# ✨ The Solution

A **local, intelligent Q&A system** using:

- **Hybrid Retrieval**: Vector search **+** BM25 full-text search, fused
- **Query Expansion**: An LLM writes alternative phrasings and keywords first
- **MCP**: Dynamic document access
- **Local LLM**: Privacy-preserving answers (Ollama)

# ✨ Key Benefits

- ✅ Privacy-first (runs locally)
- ✅ No API costs
- ✅ Semantic **and** lexical search
- ✅ Consensus ranking across many searches
- ✅ Degrades gracefully when the small model misbehaves
- ✅ Complete data control

# 🏗️ Architecture - Top Level

```
┌──────────────────────┐
│   User Interface     │ (CLI)
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
  [RAG]       [MCP]
  Hybrid      Tools
  Pipeline      │
     │           │
     └─────┬─────┘
           ▼
    [Ollama LLM]
```

# 🏗️ The Hybrid Retrieval Pipeline

```
                 User query
                     │
                     ▼
      ┌──────────────────────────────┐
      │ 1. LLM Query Expansion       │  rag/query_expansion.py
      │    alternatives + keywords   │
      └──────────────┬───────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌────────────────┐      ┌──────────────────┐
│ 2a. Vector     │      │ 2b. FTS / BM25   │   run concurrently
│     search     │      │     search       │   (asyncio.gather)
│  FAISS         │      │  SQLite FTS5     │
└────────┬───────┘      └────────┬─────────┘
         └────────────┬──────────┘
                      ▼
      ┌──────────────────────────────┐
      │ 3. Pool + dedupe by chunk id │  rag/rerank.py
      │ 4. Reciprocal Rank Fusion    │
      └──────────────┬───────────────┘
                     ▼
              Top-K chunks
                     │
                     ▼
      ┌──────────────────────────────┐
      │ 5. LLM answer (+ MCP tools)  │  assistant.py
      └──────────────────────────────┘
```

# 🔍 Step 1: Query Expansion

`rag/query_expansion.py` asks Ollama for **3 alternative phrasings** and
**6 keywords**, constrained by a JSON schema passed as Ollama's `format`
parameter so the reply is grammar-constrained rather than parsed hopefully.

```json
{
  "alternative_queries": ["...", "...", "..."],
  "keywords": ["...", "...", "..."]
}
```

> **Gotcha worth knowing:** the schema fields must be **required**. Ollama
> builds its decoding grammar from the schema, and optional fields let the
> model omit one entirely — it reliably returned keywords and no alternatives.

# 🔍 Step 2: Parallel Search

Every search runs concurrently under one `asyncio.gather`:

| List | Count | Source |
|---|---|---|
| `vector[i]` | 4 | original query + 3 alternatives |
| `fts[i]` | 4 | original query + 3 alternatives |
| `fts:keywords` | 1 | all keywords combined |

The vector fan-out is **one batched encode + one batched FAISS search**, not
four separate ones. Measured **2.24× faster** than running the same searches
sequentially, with 5 FTS searches in flight at peak.

# 🔍 Step 2b: Why SQLite FTS5

- Ships with Python — no extra dependency
- BM25 ranking built in (`bm25()`)
- Porter stemming: searching `requirement` matches "requirements"
- External-content table: chunk text stored once, not duplicated into the index

> **Gotcha:** raw user text is *not* a valid `MATCH` expression. `-`, `"`,
> `()`, `*`, `:` and bare `AND`/`OR`/`NOT` are operators. `build_match_query()`
> reduces every term to word characters and re-quotes it. Short terms stay
> phrases; long ones are split into OR'd words — phrase-matching a whole
> sentence returns **zero** hits.

# 🔍 Steps 3–4: Dedupe and RRF

A chunk found by 7 searches is **one** candidate, not seven. Pooling keys on
chunk id and keeps each list position as provenance.

```
RRF_Score(d) = Σ  1 / (k + rank_m(d))       k = 60
              m∈M
```

RRF ranks by **position**, not score — which is the only reason cosine
similarity and BM25 can be combined at all; their scales are not comparable.

**The effect:** a chunk ranked #2 by three searches beats a chunk ranked #1 by
one search. Consensus wins.

# 🔍 Step 5: Fallback for Small Models

A 0.6B model sometimes returns unusable keywords. The system never crashes and
never loses an answer:

```
expansion failed?
  ├── retrieval still found chunks → answer from the original query text
  └── nothing found at all        → LLM writes a "try rephrasing" message
                                     (rag/fallback.py)
        └── Ollama also down       → fixed static message
```

# 🏗️ Architecture - Storage

```
┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐
│  FAISS          │  │  SQLite          │  │  chunks.pkl    │
│  IndexIDMap2    │  │  chunks +        │  │  (vector-only  │
│  over IndexFlat │  │  chunks_fts FTS5 │  │   mode)        │
│  → vectors      │  │  → text + BM25   │  │                │
└────────┬────────┘  └────────┬─────────┘  └───────┬────────┘
         │      shared integer chunk id            │
         └────────────────────┬────────────────────┘
                         ┌────▼─────┐
                         │  docs/   │
                         └──────────┘
```

`IndexIDMap2` means `index.search()` returns **real chunk ids**, so the vector
and full-text retrievers speak one id space — exactly what RRF needs.

# 📁 Project Structure

```
src/
├── config.py                 Configuration
├── main.py                   CLI entry point
├── assistant.py              Orchestrator (RAG + fallback + MCP)
├── rag/
│   ├── ingest.py            Load .txt/.md/.pdf/.docx
│   ├── chunk.py             Token-based chunking
│   ├── embed.py             SentenceTransformers
│   ├── build_index.py       Build FAISS + SQLite + pkl
│   ├── query_expansion.py   ① LLM keywords/alternatives
│   ├── store.py             ② SQLite FTS5 store + BM25 search
│   ├── query.py             ② batched vector search
│   ├── search.py            ② parallel orchestration
│   ├── rerank.py            ③④ pool, dedupe, RRF
│   └── fallback.py          ⑤ fallback message node
├── mcp/
│   ├── server.py            MCP tool definitions
│   └── client.py            MCP client wrapper
├── benchmark/
│   ├── queries.json         Fixed query set
│   ├── run.py               Run + record results
│   ├── compare.py           Diff two runs
│   └── results/             hybrid.json, main.json, comparison.md
└── data/                    Generated index + stores (gitignored)
```

# 💻 Tech Stack

```
Language:      Python 3.10+
Vector DB:     FAISS (IndexIDMap2 / IndexFlatIP)
Full-text:     SQLite FTS5 + BM25
Embeddings:    SentenceTransformers (all-MiniLM-L6-v2)
LLM:           Ollama (local, Qwen 0.6B–3B)
Schemas:       Pydantic v2
MCP:           FastMCP
```

# ⚙️ Configuration Options

```python
# Chunking — sized for all-MiniLM-L6-v2's 256-token limit
CHUNK_SIZE = 220
CHUNK_OVERLAP = 40
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Models
OLLAMA_MODEL = "qwen3:0.6b"       # final answer
EXPANSION_MODEL = "qwen3:0.6b"    # keywords / alternatives
FALLBACK_MODEL = "qwen3:0.6b"     # fallback message

# Retrieval
USE_HYBRID_RETRIEVAL = True       # False = plain single vector search
TOP_K = 5                         # chunks passed to the LLM
FTS_TOP_K = 10                    # FTS over-fetches for fusion
RRF_K = 60                        # RRF damping constant

# Expansion — temperature kept low to reduce hallucination
EXPANSION_NUM_ALTERNATIVES = 3
EXPANSION_NUM_KEYWORDS = 6
EXPANSION_TEMPERATURE = 0.1
EXPANSION_TIMEOUT = 30

# Debugging (both off by default — very verbose)
DEBUG_RETRIEVAL = False           # expansion + every search's hits
DEBUG_PROMPT = False              # the final prompt sent to the LLM
```

# 🚀 Setup

> **Run every command from the repository root.** `DOCUMENTS_DIR` is resolved
> relative to the working directory, so `build-index` finds nothing if run from
> inside `src/`. See [Known Issues](#-known-issues).

```bash
# 1. Install Ollama and pull the model
ollama pull qwen3:0.6b

# 2. Install dependencies
pip install -r src/requirements.txt

# 3. Add documents to docs/  (.txt, .md, .pdf, .docx)

# 4. Build the index (also happens automatically on first query)
python src/main.py build-index
```

# 🚀 Usage

```bash
# Interactive Q&A
python src/main.py

# Inspect the whole pipeline for one query
python src/rag/search.py "your question here"

# Individual nodes
python src/rag/query_expansion.py "your question"
python src/rag/store.py "keyword one" "keyword two"
python src/rag/fallback.py "vague question"
python src/rag/rerank.py                      # worked RRF example
```

`rag/search.py` prints the expansion, every ranked list, the pooled count, the
fused Top-K, and a **vector-only baseline** for comparison.

# 📊 Benchmarking

```bash
python src/benchmark/run.py   # runs queries.json, saves results, auto-compares
```

Results are saved to `hybrid.json` or `main.json` based on
`USE_HYBRID_RETRIEVAL` — not the git branch, since either branch can run either
mode — and `compare.py` diffs them into `results/comparison.md`.

To produce a before/after pair: run once with `USE_HYBRID_RETRIEVAL = True`,
flip it to `False`, and run again. Both chunk stores are always built, so
switching modes never requires a reindex.

# 📊 Benchmark Results

10 queries, `qwen3:0.6b`, 0 errors, 0 fallbacks:

| Metric | vector-only | hybrid + RRF |
|---|---|---|
| avg time | 7.25s | 9.09s |
| max time | 13.00s | 12.69s |
| MCP tools used | 1 | 3 |

Hybrid answers are consistently more specific — e.g. "193 real estate
properties" vs "hundreds of real estate properties", and full direct quotes
where the vector-only run paraphrased.

**On the latency cost:** FTS itself adds ~10ms and runs concurrently with the
vector search. The extra ~2s is the query-expansion LLM call. Set
`USE_HYBRID_RETRIEVAL = False` or point `EXPANSION_MODEL` at a smaller model to
trade accuracy back for speed.

# 🔧 MCP - Model Context Protocol

MCP provides a **standardized interface** for LLM tool access:

```python
read_document(file_path)
list_documents()
search_documents(query)
```

After retrieval, the LLM decides whether it also needs a tool — for example to
read a full document rather than the retrieved chunks.

# ⚠️ Known Issues

**1. `DOCUMENTS_DIR` is the only path not anchored to the source tree.**
Storage paths resolve against `src/` via `Path(__file__).parent.parent`, but
`DOCUMENTS_DIR = "./docs"` resolves against the *working directory*. One cause,
three symptoms:

| Symptom | Why |
|---|---|
| `build-index` finds no documents when run from `src/` | `./docs` → empty `src/docs/` |
| Querying works from anywhere | index and stores are anchored, not CWD-relative |
| Every MCP tool returns "No documents found" | `mcp/client.py` spawns the server with `cwd=src/` |

Workaround: run everything from the repository root. Fix: resolve
`DOCUMENTS_DIR` against `Path(__file__).parent.parent` in `rag/ingest.py` and
`mcp/server.py`, as the storage paths already do.

**2. MCP results reach the LLM as a dict, not text.** `assistant.py` extracts
`result.get("result", "")`, which is the 4-key JSON-RPC envelope. The real text
is at `result["result"]["content"][0]["text"]`. This is also why the log line
reads "length: 4 chars" — that is the dict's key count.

RAG retrieval is unaffected by both; only the MCP tool path is broken.

# 🔐 Security - Local vs Cloud

**Cloud**: Data → Internet → Server
- ⚠️ Network transmission
- ⚠️ External storage
- ⚠️ Subscription costs

**Local**: Data → Local System
- ✅ No transmission
- ✅ Local storage only
- ✅ No costs

# 🔐 Implementation Safeguards

- **MCP Sandbox**: Path-containment check against `DOCUMENTS_DIR`
- **FTS Injection Safety**: Terms are tokenized and re-quoted, never interpolated
- **Local Storage**: Documents stay on device
- **No Telemetry**: No tracking
- **Offline Ready**: Works without internet once the model is pulled

# 🔮 Roadmap

**Phase 2 — Retrieval quality**
- ☐ Weighted RRF tuning (`weights={"vector": …, "fts": …}` is already supported)
- ☐ Cross-encoder reranking of the fused Top-K
- ☐ Incremental indexing instead of full rebuilds

**Phase 3 — Product**
- ☐ Web UI / API endpoints
- ☐ Conversation memory
- ☐ Metadata filtering
- ☐ Automated test suite

# 📊 Why This Works

| Aspect | Vector-only RAG | Hybrid + RRF |
|---|---|---|
| **Rare terms / IDs** | Often missed | BM25 catches them |
| **Phrasing mismatch** | One chance | 4 phrasings tried |
| **Ranking** | Single score | Consensus across 9 lists |
| **Small-model errors** | Crash or silence | Graceful fallback |
| **Candidates for Top-K** | 5 | ~25–35 pooled |

# 🙋 Quick Reference

```bash
# all from the repository root
python src/main.py build-index          # build FAISS + SQLite + pkl
python src/main.py                      # interactive Q&A
python src/rag/search.py "question"     # full pipeline trace
python src/benchmark/run.py             # benchmark + comparison
```

# 📚 Resources

- **Original**: MobilaName/local-rag-mcp
- **FAISS**: facebook/faiss
- **SQLite FTS5**: sqlite.org/fts5.html
- **RRF paper**: Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet"
- **Ollama**: ollama.ai
- **FastMCP**: github.com/jlowin/fastmcp
