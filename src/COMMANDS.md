# Setup and Run Commands

Run these commands in order to set up and use the Company Knowledge Base Assistant.

## Prerequisites

1. **Install Python 3.10+** (if not already installed)
2. **Install Ollama** and pull the model:
   ```bash
   # macOS
   brew install ollama
   
   # Or download from https://ollama.ai
   
   # Pull the model
   ollama pull qwen3:0.6b
   ```

## Setup Steps

### 1. Stay in the repository root

Run every command from the repository root. `DOCUMENTS_DIR` is resolved against
the working directory, so building from inside `src/` finds no documents.

### 2. Create a virtual environment (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r src/requirements.txt
```

### 4. Create documents directory
```bash
mkdir -p docs        # at the repository root
```

### 5. Add company documentation
Add your company documentation files (`.txt`, `.md`, `.pdf`, `.docx`) to the `docs/` directory:
```bash
# Example: Copy some sample documents
# cp /path/to/company/docs/* docs/
```

### 6. Update configuration (optional)
Edit `src/config.py` if needed:
- Set `DOCUMENTS_DIR` to your documents path (default: `./docs`)
- Change `OLLAMA_MODEL`, `EXPANSION_MODEL`, `FALLBACK_MODEL` to use other models
- Set `USE_HYBRID_RETRIEVAL = False` for a plain single vector search
- Adjust `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `FTS_TOP_K`, `RRF_K` as needed
- Turn on `DEBUG_RETRIEVAL` / `DEBUG_PROMPT` to trace a query

### 7. Build the index (Optional)

The index will be built automatically on first use. To manually build it:

```bash
python src/main.py build-index
```

This will:
- Load all documents from the `docs/` directory
- Chunk them into smaller pieces
- Generate embeddings
- Build the FAISS index and the SQLite full-text index
- Save `data/index.faiss`, `data/chunks.db` and `data/chunks.pkl`

Both chunk stores are written every time, so flipping `USE_HYBRID_RETRIEVAL`
never requires a reindex.

## Usage

### Interactive CLI Mode

Run the assistant interactively:
```bash
python src/main.py
```

Then ask questions like:
- "What is our vacation policy?"
- "How do I request time off?"
- "What are the company values?"

Type `exit` or `quit` to stop.

### Inspecting the retrieval pipeline

```bash
python src/rag/search.py "your question"      # expansion, all ranked lists, RRF Top-K
python src/rag/query_expansion.py "question"  # keyword / alternative generation
python src/rag/store.py "keyword" "keyword"   # BM25 full-text search
python src/rag/fallback.py "vague question"   # fallback message node
python src/rag/rerank.py                      # worked RRF example
```

### Benchmarking (before/after)

```bash
python src/benchmark/run.py
```

Runs `benchmark/queries.json`, saves to `benchmark/results/hybrid.json` or
`main.json` depending on `USE_HYBRID_RETRIEVAL`, and writes `comparison.md`.
Run once in each mode to produce the pair.

## Updating the Knowledge Base

When you add new documents or update existing ones:

1. Add/update files in the `docs/` directory
2. Rebuild the index:
   ```bash
   python src/main.py build-index
   ```

## Troubleshooting

### "Index not found" error
- The index will be built automatically on first use
- Or manually run `python src/main.py build-index`

### "No documents found"
- **Run from the repository root**, not from `src/`. `DOCUMENTS_DIR` is resolved
  against the working directory, so building from `src/` looks in the empty
  `src/docs/` instead of the repo-root `docs/`
- Check that `docs/` exists at the repository root and contains files
- Verify `DOCUMENTS_DIR` in `src/config.py` is correct
- Ensure files have supported extensions (`.txt`, `.md`, `.pdf`, `.docx`)

### Ollama connection errors
- Make sure Ollama is running: `ollama list`
- Verify the model is installed: `ollama pull qwen3:0.6b`
- Check `OLLAMA_URL` in `src/config.py` (default: `http://localhost:11434/api/generate`)

### MCP client errors
- MCP tools are optional - the assistant will work without them
- If MCP fails, RAG will still function
- Known issue: MCP tools always report "No documents found" because the server
  is spawned with `cwd=src/`, so `DOCUMENTS_DIR="./docs"` resolves to
  `src/docs/` instead of the repo-root `docs/` that RAG indexes

### Import errors
- Make sure you're in the repository root
- Verify virtual environment is activated
- Check that all dependencies are installed: `pip install -r requirements.txt`

## Quick Start Summary

```bash
# All commands run from the repository root

# 1. Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt

# 2. Prepare documents
mkdir -p docs
# Add your company documentation files to docs/

# 3. Build index
python src/main.py build-index

# 4. Run
python src/main.py

# 5. Optional: benchmark before/after
python src/benchmark/run.py
```
