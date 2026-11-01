"""Evaluate retrieval and answer quality on a small labeled question set.

Examples (run from the project root):
  python -m evaluation.run_eval                              # retrieval + answers
  python -m evaluation.run_eval --strategies vector hybrid hybrid_rerank
  python -m evaluation.run_eval --no-generation              # retrieval only, no LLM calls
  python -m evaluation.run_eval --tune-threshold             # find a good MIN_RELEVANCE

See EVALUATION.md for how to build your own question set.
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.output_parsers import StrOutputParser  # noqa: E402

import prompt as prompts  # noqa: E402
from config import Settings  # noqa: E402
from loaders import SUPPORTED_EXTENSIONS  # noqa: E402
from rag import RagEngine, format_context  # noqa: E402

# ---------------------------------------------------------------------- scoring helpers


def is_hit(chunk, question: dict) -> bool:
    """A retrieved chunk is a hit if it comes from the right file and contains the key fact."""
    if chunk.doc.metadata.get("source") != question["source"]:
        return False
    text = chunk.doc.page_content.lower()
    return any(e.lower() in text for e in question["evidence"])


def first_hit_rank(chunks: list, question: dict) -> Optional[int]:
    for rank, chunk in enumerate(chunks, 1):
        if is_hit(chunk, question):
            return rank
    return None


def keywords_ok(answer: str, keywords: list) -> bool:
    """Every keyword must appear. A keyword like '5 percent|5%' accepts either spelling."""
    text = answer.lower()
    return all(any(v.strip().lower() in text for v in kw.split("|")) for kw in keywords)


def percentile(values: list, p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return float(ordered[idx])


def summarize_retrieval(rows: list) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    if not answerable:
        return {"hit_rate": 0.0, "mrr": 0.0, "n": 0}
    hits = [r for r in answerable if r["rank"] is not None]
    return {
        "hit_rate": len(hits) / len(answerable),
        "mrr": sum(1 / r["rank"] for r in hits) / len(answerable),
        "n": len(answerable),
    }


def summarize_generation(rows: list) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    judged = [r["faithful"] for r in rows if "faithful" in r]
    latencies = [r["latency_ms"] for r in rows]
    return {
        "answer_accuracy": sum(r["correct"] for r in answerable) / len(answerable) if answerable else 0.0,
        "false_abstain_rate": sum(r["abstained"] for r in answerable) / len(answerable) if answerable else 0.0,
        "correct_abstain_rate": sum(r["abstained"] for r in unanswerable) / len(unanswerable) if unanswerable else 0.0,
        "faithfulness": (sum(judged) / len(judged)) if judged else None,
        "latency_avg_ms": statistics.mean(latencies) if latencies else 0.0,
        "latency_p95_ms": percentile(latencies, 95),
    }


def sweep_thresholds(rows: list, thresholds: list) -> list:
    """For each MIN_RELEVANCE value, how many questions would be handled correctly?

    Answerable questions should pass the gate, unanswerable ones should be blocked.
    """
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    out = []
    for t in thresholds:
        pass_rate = sum(r["top_score"] >= t for r in answerable) / len(answerable) if answerable else 1.0
        block_rate = sum(r["top_score"] < t for r in unanswerable) / len(unanswerable) if unanswerable else 1.0
        out.append(
            {
                "threshold": t,
                "answerable_pass_rate": pass_rate,
                "unanswerable_block_rate": block_rate,
                "balanced_accuracy": (pass_rate + block_rate) / 2,
            }
        )
    return out


def best_threshold(sweep: list) -> dict:
    # Best balanced accuracy. On a tie prefer not refusing real questions, then the lower value.
    return max(sweep, key=lambda r: (r["balanced_accuracy"], r["answerable_pass_rate"], -r["threshold"]))


# ---------------------------------------------------------------------- runners


def build_engine(corpus_dir: Path, tmp_dir: str) -> RagEngine:
    base = Settings.from_env()
    settings = replace(base, vectorstore_dir=Path(tmp_dir), collection_name="eval_run")
    engine = RagEngine(settings)
    files = [f for f in sorted(corpus_dir.iterdir()) if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        raise SystemExit(f"No supported files found in {corpus_dir}")
    for f in files:
        info = engine.ingest_file(f, f.name)
        print(f"  indexed {f.name}: {info['chunks']} chunks")
    return engine


def run_retrieval(engine: RagEngine, questions: list, strategy: str, top_k: int) -> list:
    rows = []
    for q in questions:
        res = engine.retrieve(q["question"], top_k=top_k, strategy=strategy)
        rows.append(
            {
                "id": q["id"],
                "answerable": q["answerable"],
                "rank": first_hit_rank(res.chunks, q) if q["answerable"] else None,
                "top_score": res.top_vector_score,
            }
        )
    return rows


def judge_faithful(engine: RagEngine, answer: str, chunks: list) -> bool:
    chain = prompts.JUDGE_PROMPT | engine.llm | StrOutputParser()
    reply = chain.invoke({"context": format_context(chunks), "answer": answer})
    return reply.strip().upper().startswith("YES")


def run_generation(engine: RagEngine, questions: list, strategy: str, top_k: int, judge: bool, sleep: float) -> list:
    rows = []
    for q in questions:
        retrieval = engine.retrieve(q["question"], top_k=top_k, strategy=strategy)
        final = engine.answer(q["question"], strategy=strategy, top_k=top_k, mode="strict")
        abstained = not final["grounded"]
        row = {
            "id": q["id"],
            "answerable": q["answerable"],
            "abstained": abstained,
            "latency_ms": final["latency_ms"],
            "answer": final["answer"],
        }
        if q["answerable"]:
            row["correct"] = (not abstained) and keywords_ok(final["answer"], q["answer_keywords"])
        if judge and not abstained:
            row["faithful"] = judge_faithful(engine, final["answer"], retrieval.chunks)
        rows.append(row)
        print(f"  {q['id']}: {'abstained' if abstained else 'answered'}")
        time.sleep(sleep)  # stay under free-tier rate limits
    return rows


# ---------------------------------------------------------------------- reporting


def pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def render_report(top_k: int, retrieval: dict, generation: Optional[tuple], sweep: Optional[list]) -> str:
    lines = [f"# Evaluation results ({datetime.now():%Y-%m-%d %H:%M})", ""]
    lines += [f"## Retrieval (top {top_k} passages)", "", "| Strategy | Hit rate | MRR |", "|---|---|---|"]
    for name, rows in retrieval.items():
        s = summarize_retrieval(rows)
        lines.append(f"| {name} | {pct(s['hit_rate'])} | {s['mrr']:.2f} |")

    if generation:
        strategy, rows = generation
        g = summarize_generation(rows)
        lines += [
            "",
            f"## Answers (strategy: {strategy}, strict mode)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Answer accuracy (answerable questions) | {pct(g['answer_accuracy'])} |",
            f"| Faithfulness (LLM judge) | {pct(g['faithfulness'])} |",
            f"| Correctly refused (unanswerable) | {pct(g['correct_abstain_rate'])} |",
            f"| Wrongly refused (answerable) | {pct(g['false_abstain_rate'])} |",
            f"| Latency average / p95 | {g['latency_avg_ms']:.0f} ms / {g['latency_p95_ms']:.0f} ms |",
        ]

    if sweep:
        best = best_threshold(sweep)
        lines += ["", "## MIN_RELEVANCE sweep", "", "| Threshold | Answerable pass | Unanswerable blocked | Balanced |", "|---|---|---|---|"]
        for r in sweep:
            mark = " (best)" if r is best else ""
            lines.append(
                f"| {r['threshold']:.2f}{mark} | {pct(r['answerable_pass_rate'])} | "
                f"{pct(r['unanswerable_block_rate'])} | {pct(r['balanced_accuracy'])} |"
            )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default=str(ROOT / "evaluation" / "corpus"))
    parser.add_argument("--questions", default=str(ROOT / "evaluation" / "eval_set.json"))
    parser.add_argument("--strategies", nargs="+", default=["vector", "hybrid"], choices=["vector", "hybrid", "hybrid_rerank"])
    parser.add_argument("--gen-strategy", default="hybrid", choices=["vector", "hybrid", "hybrid_rerank"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-generation", action="store_true", help="skip LLM answers (retrieval only)")
    parser.add_argument("--no-judge", action="store_true", help="skip the faithfulness judge")
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds between LLM calls")
    args = parser.parse_args(argv)

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if not args.no_generation and not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is not set. Set it, or run with --no-generation.")

    print("Building index...")
    with tempfile.TemporaryDirectory() as tmp:
        engine = build_engine(Path(args.corpus), tmp)

        retrieval = {}
        for strategy in args.strategies:
            print(f"Retrieval: {strategy}")
            retrieval[strategy] = run_retrieval(engine, questions, strategy, args.top_k)

        generation = None
        if not args.no_generation:
            print(f"Answers: {args.gen_strategy}")
            rows = run_generation(engine, questions, args.gen_strategy, args.top_k, not args.no_judge, args.sleep)
            generation = (args.gen_strategy, rows)

        sweep = None
        if args.tune_threshold:
            base_rows = next(iter(retrieval.values()))
            sweep = sweep_thresholds(base_rows, [round(0.10 + 0.05 * i, 2) for i in range(10)])

    report = render_report(args.top_k, retrieval, generation, sweep)
    print("\n" + report)

    out_dir = ROOT / "evaluation" / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"results_{stamp}.json").write_text(
        json.dumps({"retrieval": retrieval, "generation": generation, "sweep": sweep}, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "latest.md").write_text(report, encoding="utf-8")
    print(f"Saved to {out_dir}/latest.md")


if __name__ == "__main__":
    main()
