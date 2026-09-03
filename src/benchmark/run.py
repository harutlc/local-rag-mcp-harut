"""Run the fixed benchmark query set against the assistant and record results.

Each run is tagged with the current git branch/commit and a config snapshot,
and saved to hybrid.json or main.json depending on USE_HYBRID_RETRIEVAL - not
which branch it ran on, since either branch can run either mode - so a run
with hybrid retrieval on can be diffed against one with it off to see how
retrieval changes affect answers, sources and speed.
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from assistant import CompanyKBAssistant

QUERIES_PATH = Path(__file__).parent / "queries.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Not every config var exists on every branch (e.g. hybrid retrieval settings
# don't exist on pre-hybrid branches) - snapshot whatever is present so the
# runner works unmodified across branches.
CONFIG_VARS = [
    "OLLAMA_MODEL",
    "EMBEDDING_MODEL",
    "EXPANSION_MODEL",
    "USE_HYBRID_RETRIEVAL",
    "TOP_K",
    "FTS_TOP_K",
    "RRF_K",
]


def _git(*args) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return None


def _config_snapshot() -> dict:
    return {name.lower(): getattr(config, name, None) for name in CONFIG_VARS}


def run_benchmark() -> dict:
    queries = json.loads(QUERIES_PATH.read_text())

    assistant = CompanyKBAssistant()
    results = []

    try:
        for item in queries:
            print(f"-> {item['id']}: {item['query']}")
            wall_start = time.perf_counter()
            try:
                result = assistant.query(item["query"], verbose=False)
                error = None
            except Exception as e:
                result = {}
                error = str(e)
            wall_elapsed = time.perf_counter() - wall_start

            # Older assistant.py versions don't return timings/contexts/
            # expansion/fallback - default them so the schema stays uniform
            # across branches. Wall-clock time is measured here too so
            # "total" is always present, even without internal timing.
            timings = result.get("timings", {})
            timings.setdefault("total", wall_elapsed)

            results.append({
                "id": item["id"],
                "query": item["query"],
                "answer": result.get("answer"),
                "sources": result.get("sources", []),
                "contexts": result.get("contexts", []),
                "expansion": result.get("expansion"),
                "mcp_used": result.get("mcp_used", False),
                "mcp_tool": result.get("mcp_tool"),
                "fallback": result.get("fallback", False),
                "timings": timings,
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
    """Write the run to a fixed, mode-named file - hybrid.json or main.json.

    What's actually being compared is the retrieval mode, not the branch -
    both branches can now run either mode (USE_HYBRID_RETRIEVAL flips it), so
    the filename follows the config snapshot rather than the git branch name.
    Each run overwrites its mode's file rather than accumulating a new
    timestamped file every time (the run's own "timestamp" field still
    records when it happened).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = "hybrid" if run["config"].get("use_hybrid_retrieval") else "main"
    path = RESULTS_DIR / f"{name}.json"
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
