import asyncio
import sys
import time
from pathlib import Path
from pydantic import BaseModel, Field

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag import store
from rag.query import search_vector
from rag.query_expansion import QueryExpansion, expand_query
from rag.rerank import reciprocal_rank_fusion
from config import TOP_K, FTS_TOP_K


class RankedList(BaseModel):
    """One retriever's ranked results for one query.

    Rank fusion consumes a collection of these: each is an independent opinion
    about which chunks are relevant, and RRF cares about the positions.
    """
    name: str                       # e.g. "vector[0]", "fts:keywords"
    kind: str                       # "vector" or "fts"
    query: str                      # what was actually searched
    results: list[dict] = Field(default_factory=list)

    @property
    def ids(self) -> list[int]:
        return [c["id"] for c in self.results]


async def _vector_lists(queries: list[str], limit: int) -> list[RankedList]:
    """Run the vector fan-out in a worker thread and label each result list."""
    if not queries:
        return []

    # search_vector encodes the whole batch in one forward pass and issues one
    # batched FAISS search, so the fan-out is parallel inside the call. It is
    # blocking CPU work, so it goes to a thread to stay off the event loop.
    batches = await asyncio.to_thread(search_vector, queries, limit)

    return [
        RankedList(name=f"vector[{i}]", kind="vector", query=q, results=r)
        for i, (q, r) in enumerate(zip(queries, batches))
    ]


async def _fts_list(name: str, query: str, terms: list[str], limit: int) -> RankedList:
    """Run one full-text search in a worker thread and label it."""
    results = await store.search_fts_async(terms, limit)
    return RankedList(name=name, kind="fts", query=query, results=results)


async def hybrid_search(
    query: str,
    expansion: QueryExpansion | None = None,
    vector_limit: int = TOP_K,
    fts_limit: int = FTS_TOP_K,
) -> tuple[QueryExpansion, list[RankedList]]:
    """Expand a query, then run every vector and full-text search concurrently.

    Produces one ranked list per vector query, one per full-text query, and one
    more for the expansion keywords combined. Nothing is merged here - pooling,
    deduplication and rank fusion are the next steps.

    Returns the expansion used (so callers do not have to recompute it) and the
    ranked lists.
    """
    if expansion is None:
        expansion = await expand_query(query)

    queries = expansion.search_queries

    tasks = [_vector_lists(queries, vector_limit)]
    tasks += [
        _fts_list(f"fts[{i}]", q, [q], fts_limit)
        for i, q in enumerate(queries)
    ]
    if expansion.keywords:
        tasks.append(
            _fts_list(
                "fts:keywords",
                " / ".join(expansion.keywords),
                expansion.keywords,
                fts_limit,
            )
        )

    completed = await asyncio.gather(*tasks)

    # The first task returns a list of RankedList, the rest return one each.
    ranked_lists = []
    for item in completed:
        ranked_lists.extend(item) if isinstance(item, list) else ranked_lists.append(item)

    return expansion, ranked_lists


def hybrid_search_sync(query: str, **kwargs):
    """Blocking wrapper around hybrid_search for synchronous callers."""
    return asyncio.run(hybrid_search(query, **kwargs))


async def hybrid_retrieve(query: str, limit: int = TOP_K,
                          weights: dict | None = None) -> tuple[list[dict], QueryExpansion]:
    """Full retrieval pipeline: expand, search in parallel, pool, dedupe, fuse.

    Returns the fused chunks - the same dict shape query.retrieve() produces -
    together with the expansion that drove them, so callers can see whether the
    expansion actually worked (`expansion.failed`) and react to it.
    """
    expansion, ranked_lists = await hybrid_search(query)
    fused = reciprocal_rank_fusion(ranked_lists, limit=limit, weights=weights)
    return fused, expansion


def hybrid_retrieve_sync(query: str, limit: int = TOP_K,
                         weights: dict | None = None):
    """Blocking wrapper around hybrid_retrieve for synchronous callers."""
    return asyncio.run(hybrid_retrieve(query, limit=limit, weights=weights))


if __name__ == "__main__":
    from rag.query import retrieve

    q = " ".join(sys.argv[1:]) or "what does the telegram bot do when a user sends /start?"

    started = time.perf_counter()
    expansion, lists = hybrid_search_sync(q)
    fused = reciprocal_rank_fusion(lists, limit=TOP_K)
    elapsed = time.perf_counter() - started

    print(f"\n❓ Query: {q}")
    print("\n🔀 Expansion:")
    for alt in expansion.alternative_queries:
        print(f"   alt: {alt}")
    print(f"   keywords: {', '.join(expansion.keywords) or '(none)'}")

    print(f"\n📊 {len(lists)} ranked lists in {elapsed:.2f}s")
    for rl in lists:
        print(f"   {rl.name:<14} {rl.kind:<7} {len(rl.results):>2} hits  ids={rl.ids}")

    pooled = {c["id"] for rl in lists for c in rl.results}
    total_hits = sum(len(rl.results) for rl in lists)
    print(f"\n🧹 Pooled: {total_hits} hits -> {len(pooled)} distinct chunks after dedup")

    print(f"\n🏆 Top {len(fused)} after RRF:")
    for c in fused:
        preview = " ".join(c["text"].split())[:64]
        print(f"   #{c['rank']} id={c['id']:<3} rrf={c['rrf_score']:.5f} "
              f"found by {c['match_count']}/{len(lists)}  {c['source']}#{c['chunk_id']}")
        print(f"        {preview}...")

    baseline = [c["id"] for c in retrieve(q)]
    print(f"\n📉 vector-only baseline: {baseline}")
    print(f"📈 hybrid + RRF        : {[c['id'] for c in fused]}")
    new = [i for i in (c["id"] for c in fused) if i not in baseline]
    print(f"   {len(new)} chunk(s) the vector-only search missed: {new}")
