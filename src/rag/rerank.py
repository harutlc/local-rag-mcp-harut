import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RRF_K


def pool_and_deduplicate(ranked_lists) -> tuple[dict, dict]:
    """Pool every ranked list into one candidate set, keyed by chunk id.

    A chunk retrieved by several searches is one candidate, not several. Its
    position in each list is kept as provenance - that is what rank fusion
    scores, and it is also what makes a result explainable ("found by 6 of 9
    searches").

    Returns (chunks_by_id, ranks_by_id) where ranks_by_id maps a chunk id to
    {list_name: rank}.
    """
    chunks = {}
    ranks = {}

    for ranked in ranked_lists:
        for chunk in ranked.results:
            cid = chunk["id"]
            if cid not in chunks:
                # Copy so the pooled result never aliases a per-list entry, whose
                # `score`/`rank` are only meaningful inside that one list.
                chunks[cid] = {
                    "id": cid,
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                }
                ranks[cid] = {}
            ranks[cid][ranked.name] = chunk["rank"]

    return chunks, ranks


def reciprocal_rank_fusion(ranked_lists, k: int = RRF_K, limit: int | None = None,
                           weights: dict | None = None) -> list[dict]:
    """Fuse several ranked lists into one, using Reciprocal Rank Fusion.

        score(chunk) = sum over lists of  weight / (k + rank_in_that_list)

    RRF uses positions rather than scores, so it can combine retrievers whose
    scores are not comparable - here cosine similarity and BM25. A chunk ranked
    highly by several independent searches beats one ranked highly by a single
    search, which is exactly the consensus signal we want.

    `k` damps the influence of top positions; 60 is the value from the original
    paper and the usual default. `weights` optionally scales a retriever kind,
    e.g. {"vector": 1.0, "fts": 0.5}.

    Each returned chunk carries `rrf_score`, `matched_by` (the lists that found
    it), `match_count`, `ranks`, and its final 1-based `rank`.
    """
    lists = [rl for rl in ranked_lists if rl.results]
    if not lists:
        return []

    weight_of = {rl.name: (weights or {}).get(rl.kind, 1.0) for rl in lists}

    chunks, ranks = pool_and_deduplicate(lists)

    fused = []
    for cid, chunk in chunks.items():
        per_list = ranks[cid]
        chunk["rrf_score"] = sum(
            weight_of[name] / (k + rank) for name, rank in per_list.items()
        )
        chunk["ranks"] = per_list
        chunk["matched_by"] = sorted(per_list)
        chunk["match_count"] = len(per_list)
        fused.append(chunk)

    # RRF scores tie easily on small corpora. Break ties by how many searches
    # agreed, then by the best single position, then by id so the order is
    # deterministic across runs.
    fused.sort(key=lambda c: (-c["rrf_score"], -c["match_count"], min(c["ranks"].values()), c["id"]))

    if limit is not None:
        fused = fused[:limit]

    for rank, chunk in enumerate(fused, start=1):
        chunk["rank"] = rank

    return fused


if __name__ == "__main__":
    # Worked example: three lists, one chunk found by all three at middling
    # positions beats a chunk that is first in one list and absent elsewhere.
    from types import SimpleNamespace

    def chunk(i, rank):
        return {"id": i, "text": f"chunk {i}", "source": "demo.md",
                "chunk_id": i, "score": 0.0, "rank": rank}

    def rl(name, kind, ids):
        return SimpleNamespace(
            name=name, kind=kind, query="demo",
            results=[chunk(i, r) for r, i in enumerate(ids, start=1)],
        )

    lists = [
        rl("vector[0]", "vector", [10, 3, 7]),
        rl("vector[1]", "vector", [11, 3, 8]),
        rl("fts[0]", "fts", [12, 3, 9]),
    ]

    print("input lists:")
    for l in lists:
        print(f"  {l.name:<10} {[c['id'] for c in l.results]}")

    print(f"\nfused (k={RRF_K}):")
    for c in reciprocal_rank_fusion(lists):
        print(f"  #{c['rank']} id={c['id']:<3} rrf={c['rrf_score']:.5f} "
              f"matched_by={c['match_count']} {c['matched_by']}")
