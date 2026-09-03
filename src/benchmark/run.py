"""Run the fixed benchmark query set against the assistant and record results.

Each run is tagged with the current git branch/commit and a config snapshot,
so a run on this branch can later be diffed against a run on master (or any
other branch) to see how retrieval changes affect answers, sources and speed.
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from assistant import CompanyKBAssistant
from config import (
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
    EXPANSION_MODEL,
    USE_HYBRID_RETRIEVAL,
    TOP_K,
    FTS_TOP_K,
    RRF_K,
)

QUERIES_PATH = Path(__file__).parent / "queries.json"
RESULTS_DIR = Path(__file__).parent / "results"


def _git(*args) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return None


def _config_snapshot() -> dict:
    return {
        "ollama_model": OLLAMA_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "expansion_model": EXPANSION_MODEL,
        "use_hybrid_retrieval": USE_HYBRID_RETRIEVAL,
        "top_k": TOP_K,
        "fts_top_k": FTS_TOP_K,
        "rrf_k": RRF_K,
    }


def run_benchmark() -> dict:
    queries = json.loads(QUERIES_PATH.read_text())

    assistant = CompanyKBAssistant()
    results = []

    try:
        for item in queries:
            print(f"-> {item['id']}: {item['query']}")
            try:
                result = assistant.query(item["query"], verbose=False)
                error = None
            except Exception as e:
                result = {
                    "answer": None,
                    "sources": [],
                    "mcp_used": False,
                    "mcp_tool": None,
                    "fallback": False,
                    "timings": {},
                    "contexts": [],
                    "expansion": None,
                }
                error = str(e)

            results.append({
                "id": item["id"],
                "query": item["query"],
                "answer": result["answer"],
                "sources": result["sources"],
                "contexts": result["contexts"],
                "expansion": result["expansion"],
                "mcp_used": result["mcp_used"],
                "mcp_tool": result["mcp_tool"],
                "fallback": result["fallback"],
                "timings": result["timings"],
                "error": error,
            })
    finally:
        assistant.close()

    total_times = [r["timings"].get("total") for r in results if r["timings"].get("total") is not None]

    run = {
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": _config_snapshot(),
        "summary": {
            "num_queries": len(results),
            "num_errors": sum(1 for r in results if r["error"]),
            "num_fallback": sum(1 for r in results if r["fallback"]),
            "num_mcp_used": sum(1 for r in results if r["mcp_used"]),
            "avg_total_seconds": sum(total_times) / len(total_times) if total_times else None,
            "max_total_seconds": max(total_times) if total_times else None,
        },
        "results": results,
    }
    return run


def save_run(run: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = run["timestamp"].replace(":", "").replace("+00:00", "Z")
    branch = (run["branch"] or "unknown").replace("/", "-")
    path = RESULTS_DIR / f"{branch}__{run['commit'] or 'nogit'}__{ts}.json"
    path.write_text(json.dumps(run, indent=2))
    return path


def print_summary(run: dict):
    s = run["summary"]
    print(f"\n📊 Benchmark summary ({run['branch']}@{run['commit']})")
    print(f"   queries:     {s['num_queries']}")
    print(f"   errors:      {s['num_errors']}")
    print(f"   fallbacks:   {s['num_fallback']}")
    print(f"   mcp used:    {s['num_mcp_used']}")
    if s["avg_total_seconds"] is not None:
        print(f"   avg time:    {s['avg_total_seconds']:.2f}s")
        print(f"   max time:    {s['max_total_seconds']:.2f}s")


if __name__ == "__main__":
    run = run_benchmark()
    path = save_run(run)
    print_summary(run)
    print(f"\n💾 Saved to {path}")
