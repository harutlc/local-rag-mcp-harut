import faiss
import numpy as np
import pickle
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.ingest import ingest_documents
from rag.chunk import chunk_documents
from rag.embed import embed_chunks
from rag import store
from config import FAISS_INDEX_PATH, CHUNKS_PATH


def build_index():
    """Build the FAISS index, the SQLite chunk store and the chunks.pkl cache.

    Both chunk stores are always built together so flipping
    USE_HYBRID_RETRIEVAL never requires a reindex: SQLite (plus its FTS index)
    backs hybrid retrieval, chunks.pkl backs the plain single-vector-search
    path.
    """
    # Resolve paths relative to src directory
    src_dir = Path(__file__).parent.parent
    index_path = src_dir / FAISS_INDEX_PATH
    chunks_path = src_dir / CHUNKS_PATH

    print("📥 Loading documents...")
    documents = ingest_documents()

    if not documents:
        print("❌ No documents found. Please add documents to the docs directory.")
        return

    print("✂️ Chunking...")
    chunks = chunk_documents(documents)

    print("🧠 Generating embeddings...")
    embeddings = embed_chunks(chunks)

    print("🗄️ Storing chunks in SQLite...")
    ids = store.write_chunks(chunks)

    print("🥒 Caching chunks (pickle)...")
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print("📦 Creating FAISS index...")
    dim = embeddings.shape[1]
    # IndexIDMap2 so search returns chunk ids rather than row positions - the
    # vector and full-text retrievers then share one id space.
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
    faiss.normalize_L2(embeddings)
    index.add_with_ids(embeddings, np.array(ids, dtype=np.int64))

    print("💾 Saving...")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    print(f"✅ Indexing complete: {len(chunks)} chunks indexed")
    print(f"   Index saved to: {index_path}")
    print(f"   Chunks saved to: {store.db_path()} and {chunks_path}")


if __name__ == "__main__":
    build_index()
