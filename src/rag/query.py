import faiss
import pickle
import requests
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag import store
from config import (
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    OLLAMA_URL,
    OLLAMA_MODEL,
    TOP_K,
    USE_HYBRID_RETRIEVAL,
)

model = SentenceTransformer(EMBEDDING_MODEL)

# Global for the loaded FAISS index. Chunk text lives in SQLite (rag.store)
# when hybrid retrieval is on, or in the chunks.pkl cache below when it's off.
index = None
_pickle_chunks = None


def _load_pickle_chunks():
    """Load the chunks.pkl cache - a plain list where position == FAISS id."""
    global _pickle_chunks

    path = Path(__file__).parent.parent / CHUNKS_PATH
    if not path.exists():
        _pickle_chunks = []
        return

    with open(path, "rb") as f:
        _pickle_chunks = pickle.load(f)


def _get_chunks_by_ids_from_pickle(ids: list[int]) -> list[dict]:
    """Look up chunks by id from the pickle cache, in the given id order."""
    if _pickle_chunks is None:
        _load_pickle_chunks()

    results = []
    for i in ids:
        if 0 <= i < len(_pickle_chunks):
            chunk = dict(_pickle_chunks[i])
            chunk["id"] = i
            results.append(chunk)
    return results


def _load_index():
    """Load the FAISS index and the chunk store for the current retrieval mode."""
    global index

    index_path = Path(__file__).parent.parent / FAISS_INDEX_PATH
    if not index_path.exists():
        return False

    if USE_HYBRID_RETRIEVAL:
        if store.count() == 0:
            return False
    else:
        _load_pickle_chunks()
        if not _pickle_chunks:
            return False

    index = faiss.read_index(str(index_path))
    return True


def _ensure_index_exists():
    """Ensure the FAISS index and chunk database exist, build them if they don't."""
    try:
        if _load_index():
            return True
    except Exception as e:
        print(f"⚠️  Warning: Error loading existing index: {e}")
        print("Rebuilding index...")

    # Index or database missing/unreadable, build both
    print("📦 Index not found. Building index from documents...")
    try:
        from rag.build_index import build_index
        build_index()

        if _load_index():
            print("✅ Index built and loaded successfully")
            return True

        print("❌ Failed to build index. No documents found or error occurred.")
        from config import DOCUMENTS_DIR
        docs_path = Path(__file__).parent.parent / DOCUMENTS_DIR
        print(f"   Check that documents exist in: {docs_path}")
        return False
    except Exception as e:
        print(f"❌ Error building index: {e}")
        import traceback
        traceback.print_exc()
        return False


# Initialize index on module load
_ensure_index_exists()


def search_vector(queries: list[str], limit: int = TOP_K) -> list[list[dict]]:
    """Vector search for several queries at once.

    Returns one ranked result list per query, in the same order. The queries are
    encoded in a single batch and searched with one batched FAISS call, which is
    how the fan-out over alternative queries is parallelised - both the model
    forward pass and the index scan handle the whole batch at once.

    Each chunk carries `score` (cosine similarity, higher is better) and `rank`
    (1-based within its own list).
    """
    if isinstance(queries, str):
        queries = [queries]

    # Ensure index exists before retrieving
    if index is None:
        if not _ensure_index_exists():
            return [[] for _ in queries]

    if index is None or not queries:
        return [[] for _ in queries]

    q_embs = model.encode(queries)
    faiss.normalize_L2(q_embs)

    # IndexIDMap2 returns chunk ids, which are the SQLite primary keys
    all_scores, all_ids = index.search(q_embs, limit)

    results = []
    for row_ids, row_scores in zip(all_ids, all_scores):
        # FAISS pads short result sets with -1
        score_by_id = {
            int(i): float(s) for i, s in zip(row_ids, row_scores) if int(i) >= 0
        }
        ids = list(score_by_id)
        chunks = (
            store.get_chunks_by_ids(ids)
            if USE_HYBRID_RETRIEVAL
            else _get_chunks_by_ids_from_pickle(ids)
        )
        for rank, chunk in enumerate(chunks, start=1):
            chunk["score"] = score_by_id[chunk["id"]]
            chunk["rank"] = rank
        results.append(chunks)

    return results


def retrieve(query: str):
    """Retrieve relevant chunks for a query."""
    return search_vector([query], TOP_K)[0]


def build_prompt(query, contexts):
    """Build prompt with retrieved context."""
    if not contexts:
        return f"""
<role>You are a helpful assistant that answers questions about company information.</role>
<instructions>Answer the question based on your general knowledge. If you don't know, say so.</instructions>

<query>
{query}
</query>

<assistant>
"""

    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in contexts
    )

    return f"""
<role>You are a helpful assistant that answers questions about company information.</role>
<instructions>Answer the question ONLY based on the context provided below. If the answer is not in the context, say "I don't have that information in the knowledge base."</instructions>

<context>
{context_text}
</context>

<query>
{query}
</query>

<assistant>
"""


def ask_llm(prompt):
    """Query Ollama LLM."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


def ask(query: str):
    """Answer a question using RAG."""
    contexts = retrieve(query)
    prompt = build_prompt(query, contexts)
    return ask_llm(prompt), contexts


if __name__ == "__main__":
    while True:
        q = input("\n❓ Question: ")
        if q.lower() in {"exit", "quit"}:
            break
        print("\n🤖 Answer:\n")
        answer, sources = ask(q)
        print(answer)
        if sources:
            print("\n📚 Sources:")
            seen_sources = set()
            for src in sources:
                if src["source"] not in seen_sources:
                    print(f"  - {src['source']}")
                    seen_sources.add(src["source"])
