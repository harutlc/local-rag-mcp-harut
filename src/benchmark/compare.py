"""Build a side-by-side comparison table from results/main.json and
results/hybrid.json: one row per question, each branch's answer and time
next to each other.

Usage:
    python benchmark/compare.py
"""
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
MAIN_PATH = RESULTS_DIR / "main.json"
HYBRID_PATH = RESULTS_DIR / "hybrid.json"
OUT_PATH = RESULTS_DIR / "comparison.md"


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing {path} - run benchmark/run.py in that mode first.")
    return json.loads(path.read_text())


def _cell(text) -> str:
    """Escape a value for use inside a Markdown table cell."""
    if text is None:
        return ""
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _by_id(run: dict) -> dict:
    return {r["id"]: r for r in run["results"]}


def build_table(main: dict, hybrid: dict) -> str:
    main_by_id = _by_id(main)
    hybrid_by_id = _by_id(hybrid)
    # queries.json order, as it appears in either run
    ids = list(main_by_id) or list(hybrid_by_id)

    lines = [
        f"| Question | main ({main['commit']}) Answer | main Time (s) "
        f"| hybrid ({hybrid['commit']}) Answer | hybrid Time (s) |",
        "|---|---|---|---|---|",
    ]
    for qid in ids:
        m = main_by_id.get(qid)
        h = hybrid_by_id.get(qid)
        question = _cell(m["query"] if m else h["query"])

        m_answer = _cell(m and (m.get("error") and f"ERROR: {m['error']}" or m["answer"]))
        h_answer = _cell(h and (h.get("error") and f"ERROR: {h['error']}" or h["answer"]))
        m_time = m and m.get("timings", {}).get("total")
        h_time = h and h.get("timings", {}).get("total")

        lines.append(
            f"| {question} | {m_answer} | {f'{m_time:.2f}' if m_time is not None else ''} "
            f"| {h_answer} | {f'{h_time:.2f}' if h_time is not None else ''} |"
        )
    return "\n".join(lines)


def build_summary(main: dict, hybrid: dict) -> str:
    ms, hs = main["summary"], hybrid["summary"]
    lines = [
        "| Metric | main | hybrid |",
        "|---|---|---|",
        f"| model | {main['config'].get('ollama_model')} | {hybrid['config'].get('ollama_model')} |",
        f"| hybrid retrieval | {main['config'].get('use_hybrid_retrieval')} | {hybrid['config'].get('use_hybrid_retrieval')} |",
        f"| errors | {ms['num_errors']} | {hs['num_errors']} |",
        f"| fallbacks | {ms['num_fallback']} | {hs['num_fallback']} |",
        f"| mcp used | {ms['num_mcp_used']} | {hs['num_mcp_used']} |",
        f"| avg time (s) | {ms['avg_total_seconds']:.2f} | {hs['avg_total_seconds']:.2f} |",
        f"| max time (s) | {ms['max_total_seconds']:.2f} | {hs['max_total_seconds']:.2f} |",
    ]
    return "\n".join(lines)


def main():
    main_run = _load(MAIN_PATH)
    hybrid_run = _load(HYBRID_PATH)

    output = (
        f"# Benchmark comparison: main vs hybrid\n\n"
        f"## Summary\n\n{build_summary(main_run, hybrid_run)}\n\n"
        f"## Per-query\n\n{build_table(main_run, hybrid_run)}\n"
    )

    print(output)
    OUT_PATH.write_text(output)
    print(f"\n💾 Saved to {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
