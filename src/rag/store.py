import asyncio
import re
import sqlite3
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNKS_DB_PATH, FTS_TOP_K


SCHEMA = """
DROP TABLE IF EXISTS chunks_fts;
DROP TABLE IF EXISTS chunks;

CREATE TABLE chunks (
    id       INTEGER PRIMARY KEY,   -- global id; matches the FAISS vector id
    source   TEXT    NOT NULL,
    chunk_id INTEGER NOT NULL,      -- position within the source document
    text     TEXT    NOT NULL
);
CREATE INDEX idx_chunks_source ON chunks(source);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize="porter unicode61 remove_diacritics 2"
);
"""


def db_path() -> Path:
    """Absolute path to the chunk database (CHUNKS_DB_PATH is relative to src/)."""
    return Path(__file__).parent.parent / CHUNKS_DB_PATH


def db_exists() -> bool:
    return db_path().exists()


def _connect(readonly: bool = False) -> sqlite3.Connection:
    """Open a connection to the chunk database.

    A fresh connection per call on purpose: sqlite3 connections are not
    thread-safe, and searches run in worker threads via search_fts_async.
    """
    path = db_path()

    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)

    conn.row_factory = sqlite3.Row
    return conn


def _row_to_chunk(row: sqlite3.Row) -> dict:
    """Convert a DB row to the chunk dict shape the rest of the pipeline uses."""
    chunk = {
        "id": row["id"],
        "text": row["text"],
        "source": row["source"],
        "chunk_id": row["chunk_id"],
    }
    if "score" in row.keys():
        chunk["score"] = row["score"]
    return chunk


def write_chunks(chunks: list[dict]) -> list[int]:
    """Rebuild the database from scratch and store all chunks.

    Returns the assigned ids (0..N-1, in input order) so the caller can add the
    matching vectors to FAISS under the same ids.
    """
    ids = list(range(len(chunks)))

    with _connect() as conn:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO chunks (id, source, chunk_id, text) VALUES (?, ?, ?, ?)",
            [
                (i, c["source"], c["chunk_id"], c["text"])
                for i, c in zip(ids, chunks)
            ],
        )
        # External-content FTS table: populate once, after the bulk insert.
        conn.execute("INSERT INTO chunks_fts (rowid, text) SELECT id, text FROM chunks")

    return ids


def count() -> int:
    """Number of chunks stored, or 0 if the database does not exist."""
    if not db_exists():
        return 0

    try:
        with _connect(readonly=True) as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    except sqlite3.Error:
        return 0


def build_match_query(terms: list[str], phrase_max_words: int = 4) -> str:
    """Turn arbitrary search terms into a safe FTS5 MATCH expression.

    Raw user text is not valid FTS5 syntax: characters like - " ( ) * : and the
    bare words AND/OR/NOT are operators. Every term is reduced to word
    characters and re-quoted.

    Short terms (keywords) stay phrases so they match precisely; longer terms
    (full questions) are split into OR'd words, because phrase-matching a whole
    sentence almost never hits.

    Returns "" when nothing searchable survives - callers must not query with it.
    """
    parts = []

    for term in terms:
        words = re.findall(r"\w+", term or "", flags=re.UNICODE)
        if not words:
            continue
        if len(words) <= phrase_max_words:
            parts.append('"' + " ".join(words) + '"')
        else:
            parts.extend('"' + word + '"' for word in words)

    seen = set()
    unique = []
    for part in parts:
        key = part.lower()
        if key not in seen:
            seen.add(key)
            unique.append(part)

    return " OR ".join(unique)


def search_fts(terms: list[str], limit: int = FTS_TOP_K) -> list[dict]:
    """Full-text search over chunks, ranked by BM25. Best match first.

    `score` is the negated bm25 value, so higher is better - the same convention
    as the FAISS cosine scores. `rank` is 1-based within this result list.
    """
    if isinstance(terms, str):
        terms = [terms]

    match = build_match_query(terms)
    if not match or not db_exists():
        return []

    try:
        with _connect(readonly=True) as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.source, c.chunk_id, c.text, -bm25(chunks_fts) AS score
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
                """,
                (match, limit),
            ).fetchall()
    except sqlite3.Error as e:
        print(f"⚠️  Warning: full-text search failed ({e}).")
        return []

    results = [_row_to_chunk(row) for row in rows]
    for rank, chunk in enumerate(results, start=1):
        chunk["rank"] = rank
    return results


async def search_fts_async(terms: list[str], limit: int = FTS_TOP_K) -> list[dict]:
    """Async wrapper around search_fts - sqlite3 is blocking, so run it in a thread."""
    return await asyncio.to_thread(search_fts, terms, limit)


def get_chunks_by_ids(ids) -> list[dict]:
    """Fetch chunks by id, preserving the order of `ids`.

    FAISS pads short result sets with -1, which is skipped.
    """
    wanted = [int(i) for i in ids if int(i) >= 0]
    if not wanted or not db_exists():
        return []

    placeholders = ",".join("?" * len(wanted))
    with _connect(readonly=True) as conn:
        rows = conn.execute(
            f"SELECT id, source, chunk_id, text FROM chunks WHERE id IN ({placeholders})",
            wanted,
        ).fetchall()

    # SQL gives no ordering guarantee, so restore the requested order.
    by_id = {row["id"]: _row_to_chunk(row) for row in rows}
    return [by_id[i] for i in wanted if i in by_id]


if __name__ == "__main__":
    print(f"📚 {count()} chunks in {db_path()}")

    terms = sys.argv[1:] or ["conversation history", "start command"]
    results = search_fts(terms)
    print(f"\n🔍 Search: {terms}")
    print(f"   match: {build_match_query(terms)}")
    print(f"   {len(results)} results\n")

    for r in results:
        preview = " ".join(r["text"].split())[:80]
        print(f"  [{r['score']:.3f}] id={r['id']} {r['source']}#{r['chunk_id']}")
        print(f"          {preview}...")
