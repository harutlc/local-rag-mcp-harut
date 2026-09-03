# Company Knowledge Base Assistant

An intelligent Q&A system that answers questions about company documentation using RAG (Retrieval-Augmented Generation) and MCP (Model Context Protocol) tools.

## Features

- **Hybrid retrieval**: FAISS vector search + SQLite FTS5/BM25, fused with RRF
- **Query expansion**: an LLM generates alternative phrasings and keywords first
- **Parallel search**: all vector and full-text searches run under one `asyncio.gather`
- **Graceful fallback**: a small model returning bad keywords never costs an answer
- **MCP tools**: Dynamic document reading and management
- **Local LLM**: Privacy-preserving answers using Ollama

## Setup

### 1. Install Dependencies

```bash
pip install -r src/requirements.txt
```

### 2. Set Up Documents

Create a `docs/` directory and add your company documentation files (`.txt`, `.md`, `.pdf`, `.docx`):

```bash
mkdir docs
# Add your company documentation files here
```

### 3. Configure

Edit `config.py` to set:
- `DOCUMENTS_DIR`: Path to your documentation directory
- `OLLAMA_MODEL`: Local LLM for the final answer (default: `qwen3:0.6b`)
- `EXPANSION_MODEL` / `FALLBACK_MODEL`: models for the expansion and fallback nodes
- `USE_HYBRID_RETRIEVAL`: `False` falls back to a plain single vector search
- `TOP_K`, `FTS_TOP_K`, `RRF_K`: retrieval and fusion tuning
- `DEBUG_RETRIEVAL`, `DEBUG_PROMPT`: verbose tracing, both off by default

### 4. Build Index (Optional)

> Run from the **repository root**, not from `src/` - `DOCUMENTS_DIR` is resolved
> against the working directory. See Troubleshooting.

The index will be built automatically on first use. To manually build it:

```bash
python src/main.py build-index
```

## Usage

### Interactive CLI

Run the interactive assistant:

```bash
python src/main.py
```

Then ask questions about your company documentation!

### Inspecting the pipeline

```bash
python src/rag/search.py "your question"      # expansion, every list, pooled, RRF Top-K
python src/rag/query_expansion.py "question"  # expansion node alone
python src/rag/store.py "keyword" "keyword"   # full-text search alone
python src/rag/fallback.py "vague question"   # fallback node alone
python src/rag/rerank.py                      # worked RRF example
```

### Benchmarking

```bash
python src/benchmark/run.py
```

Saves to `benchmark/results/hybrid.json` or `main.json` depending on
`USE_HYBRID_RETRIEVAL`, then writes `comparison.md`. Run once in each mode to
get a before/after pair - both chunk stores are always built, so switching
modes never requires a reindex.

## Project Structure

```
src/
├── config.py              # Configuration
├── main.py                # CLI entry point
├── assistant.py           # Main assistant class
├── rag/                   # RAG components
│   ├── ingest.py         # Document ingestion
│   ├── chunk.py          # Text chunking
│   ├── embed.py          # Embedding generation
│   ├── build_index.py    # Builds FAISS + SQLite + chunks.pkl
│   ├── query_expansion.py# LLM keyword / alternative-query generation
│   ├── store.py          # SQLite FTS5 chunk store + BM25 search
│   ├── query.py          # Batched vector search
│   ├── search.py         # Parallel vector + FTS orchestration
│   ├── rerank.py         # Pooling, dedup and Reciprocal Rank Fusion
│   └── fallback.py       # Fallback message node
├── mcp/                   # MCP components
│   ├── server.py         # MCP server with tools
│   └── client.py         # MCP client
├── benchmark/             # Before/after benchmark suite
│   ├── queries.json      # Fixed query set
│   ├── run.py            # Run and record results
│   ├── compare.py        # Diff two runs into comparison.md
│   └── results/          # hybrid.json, main.json, comparison.md
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## How It Works

**Indexing** (`python src/main.py build-index`)

1. **Document Ingestion**: Loads documents from the `docs/` directory
2. **Chunking**: Splits documents into token-based chunks with overlap
3. **Embedding**: Generates embeddings using SentenceTransformers
4. **Indexing**: Writes the FAISS index (`IndexIDMap2`, so search returns chunk
   ids), the SQLite chunk store with its FTS5 index, and `chunks.pkl`

**Querying** (`python src/main.py`)

1. **Query Expansion**: an LLM returns 3 alternative phrasings + 6 keywords
2. **Parallel Search**: 4 vector searches, 4 full-text searches and 1 keyword
   search all run concurrently
3. **Pool & Dedupe**: results are unioned by chunk id, keeping each list position
4. **RRF**: `score = Σ 1/(k + rank)` with `k = 60`; Top-K chunks are selected
5. **Answer**: the LLM optionally calls an MCP tool, then answers from the chunks

If expansion fails, retrieval proceeds on the original query text. Only when
nothing at all is found does the fallback node explain and suggest rephrasings.

## MCP Tools

The MCP server provides:
- `read_document`: Read a specific document
- `list_documents`: List all available documents
- `search_documents`: Search documents by name

## Troubleshooting

**Index not found**: Run `python src/main.py build-index` first

**Ollama not responding**: Make sure Ollama is running and the model is installed:
```bash
ollama pull qwen3:0.6b
```

**"No documents found" when building**: run from the repository root, not from
`src/`. `DOCUMENTS_DIR` is resolved against the working directory while the
index and stores are anchored to `src/`, so `cd src && python main.py
build-index` looks in the empty `src/docs/`.

**MCP tools always say "No documents found"**: same root cause - `mcp/client.py`
spawns the server with `cwd=src/`. RAG retrieval is unaffected.

**No documents found**: Check that `DOCUMENTS_DIR` in `config.py` points to your documents

## License

MIT
