import asyncio
import ollama
import sys
from pathlib import Path
from pydantic import BaseModel, Field

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OLLAMA_HOST,
    EXPANSION_MODEL,
    EXPANSION_NUM_ALTERNATIVES,
    EXPANSION_NUM_KEYWORDS,
    EXPANSION_TEMPERATURE,
    EXPANSION_TIMEOUT,
)


class ExpansionResponse(BaseModel):
    """Exactly the object shape requested from Ollama via the `format` schema.

    Both fields are required on purpose: Ollama builds its decoding grammar from
    this schema, and optional fields let the model skip one entirely. With
    defaults here it reliably returned keywords and no alternative_queries.
    """
    alternative_queries: list[str]
    keywords: list[str]


class QueryExpansion(BaseModel):
    """Node output: the original query carried alongside its expansion."""
    query: str
    alternative_queries: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    error: str | None = None        # why expansion failed, when it did

    @property
    def search_queries(self) -> list[str]:
        """Original query first, then the alternatives - the vector search fan-out."""
        return [self.query, *self.alternative_queries]

    @property
    def failed(self) -> bool:
        """True when expansion produced nothing usable.

        Covers both a hard failure (Ollama unreachable, timeout, malformed
        response) and a well-formed reply that came back with no alternative
        queries and no keywords - the model answered, but uselessly.
        """
        return bool(self.error) or not (self.alternative_queries or self.keywords)


SYSTEM_PROMPT = (
    "You rewrite search queries for a company knowledge base, expanding a "
    "question into phrasings and keywords for search. JSON only."
)


def _build_user_prompt(query: str) -> str:
    return f"""Question: {query}

Generate:
- alternative_queries: {EXPANSION_NUM_ALTERNATIVES} standalone rephrasings, using synonyms/domain terms.
- keywords: {EXPANSION_NUM_KEYWORDS} short (1-3 word) search terms, e.g. "annual leave".

Search terms only, no answer."""


def _normalize(values: list[str], limit: int, exclude: set[str] | None = None) -> list[str]:
    """Clean, drop empties/duplicates (case-insensitive), and cap to `limit`.

    Models occasionally emit identifier-style terms ("parental_leave"); underscores
    are turned back into spaces so the terms are usable for full-text search.
    """
    seen = set(exclude or set())
    result = []

    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = " ".join(value.replace("_", " ").split())
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) == limit:
            break

    return result


async def expand_query(query: str) -> QueryExpansion:
    """Expand a user query into alternative queries and keywords using Ollama.

    Never raises: on any failure it returns an empty expansion, so retrieval
    degrades to searching the original query alone.
    """
    try:
        client = ollama.AsyncClient(host=OLLAMA_HOST, timeout=EXPANSION_TIMEOUT)
        response = await client.chat(
            model=EXPANSION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(query)},
            ],
            format=ExpansionResponse.model_json_schema(),
            options={"temperature": EXPANSION_TEMPERATURE},
        )
        expansion = ExpansionResponse.model_validate_json(response["message"]["content"])

        return QueryExpansion(
            query=query,
            alternative_queries=_normalize(
                expansion.alternative_queries,
                EXPANSION_NUM_ALTERNATIVES,
                exclude={query.strip().lower()},
            ),
            keywords=_normalize(expansion.keywords, EXPANSION_NUM_KEYWORDS),
        )
    except Exception as e:
        print(f"⚠️  Warning: query expansion failed ({e}). Using original query only.")
        return QueryExpansion(query=query, error=str(e))


def expand_query_sync(query: str) -> QueryExpansion:
    """Blocking wrapper around expand_query for synchronous callers."""
    return asyncio.run(expand_query(query))


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "how many vacation days do I get?"
    print(expand_query_sync(q).model_dump_json(indent=2))
