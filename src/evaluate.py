"""
evaluate.py — Run a fixed eval set and compute automated + manual-review metrics.

Usage:
    python -m src.evaluate               # print to stdout
    python -m src.evaluate --csv         # also write results/eval.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document

from dotenv import load_dotenv

from src.query import ask, build_chain, print_result

load_dotenv()

NOT_FOUND_PHRASE = "I could not find that information in the provided manuals."
CITATION_PATTERN = re.compile(r"\[source:\s*.+?,\s*p\.\d+\]", re.IGNORECASE)
K_EVAL = 5

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "what", "which", "who", "how",
    "when", "where", "and", "or", "not", "it", "its", "this", "that",
    "these", "those", "there",
}

EVAL_QUESTIONS = [
    "What is the operating temperature range of the evaporator?",
    "What is the minimum ambient temperature for operation?",
    "Which sensors are connected to the main controller?",
    "What are the steps to start the chiller for the first time?",
    "What fault code is triggered by low oil pressure?",
    "What refrigerant type is used in the Carrier 30XA?",
    "What are the recommended clearances for installation?",
    "How is the capacity control algorithm described?",
    "What is the maximum operating pressure of the refrigerant circuit?",
    "What type of compressor does the 30XA use?",
]

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _extract_keywords(question: str) -> set[str]:
    tokens = re.sub(r"[^a-z0-9\s]", "", question.lower()).split()
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


def _precision_at_k(question: str, docs: list["Document"], k: int = K_EVAL) -> float:
    """Keyword-based proxy: fraction of top-K chunks containing a question keyword."""
    keywords = _extract_keywords(question)
    if not keywords:
        return 0.0
    top_k = docs[:k]
    if not top_k:
        return 0.0
    hits = sum(
        1 for doc in top_k
        if any(kw in doc.page_content.lower() for kw in keywords)
    )
    return round(hits / len(top_k), 2)


def _automated_metrics(result: dict) -> dict:
    answer = result["answer"]
    sources = result["sources"]
    docs = result.get("docs", [])

    citation_present = bool(CITATION_PATTERN.search(answer))
    not_found = NOT_FOUND_PHRASE.lower() in answer.lower()

    unique_files = {s.split(" p.")[0].strip() for s in sources}
    unique_pages = {s for s in sources}
    pk = _precision_at_k(result["question"], docs)

    return {
        "citation_present": citation_present,
        "not_found": not_found,
        "unique_source_files": len(unique_files),
        "unique_pages_retrieved": len(unique_pages),
        "precision_at_k": pk,
    }


def _print_summary(results: list[dict]) -> None:
    n = len(results)
    citation_rate = sum(1 for r in results if r["citation_present"]) / n * 100
    not_found_rate = sum(1 for r in results if r["not_found"]) / n * 100
    avg_files = sum(r["unique_source_files"] for r in results) / n
    avg_pages = sum(r["unique_pages_retrieved"] for r in results) / n

    print("\n" + "=" * 60)
    print("AUTOMATED METRICS SUMMARY")
    print("=" * 60)
    print(f"  Citation presence rate : {citation_rate:.0f}%  (target: 100%)")
    print(f"  Not-found rate         : {not_found_rate:.0f}%  (lower is better)")
    avg_pk = sum(r["precision_at_k"] for r in results) / n
    print(f"  Avg unique source files: {avg_files:.1f}  (> 1 = cross-doc)")
    print(f"  Avg unique pages       : {avg_pages:.1f}")
    print(f"  Avg Precision@{K_EVAL}        : {avg_pk:.2f}  (keyword-based proxy)")
    print("=" * 60)
    print("\nManual review rubric (score each question 0-4):")
    print("  [Cited]    Every sentence has [source: ..., p.N]")
    print("  [Grounded] No fact absent from retrieved chunks")
    print("  [Correct]  Answer matches content of cited page")
    print("  [Complete] All parts of the question are addressed")
    print("  Target: >= 3/4 on all questions\n")


def run_eval(write_csv: bool = False) -> list[dict]:
    retriever_and_chain = build_chain()
    enriched = []
    print(f"Running {len(EVAL_QUESTIONS)} evaluation questions...\n")

    for q in EVAL_QUESTIONS:
        result = ask(retriever_and_chain, q)
        print_result(result)
        metrics = _automated_metrics(result)
        enriched.append({**result, **metrics})

    _print_summary(enriched)

    if write_csv:
        RESULTS_DIR.mkdir(exist_ok=True)
        out_path = RESULTS_DIR / "eval.csv"
        fieldnames = [
            "question", "answer", "sources",
            "citation_present", "not_found",
            "unique_source_files", "unique_pages_retrieved",
            "precision_at_k",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in enriched:
                writer.writerow({
                    "question": r["question"],
                    "answer": r["answer"],
                    "sources": "; ".join(r["sources"]),
                    "citation_present": r["citation_present"],
                    "not_found": r["not_found"],
                    "unique_source_files": r["unique_source_files"],
                    "unique_pages_retrieved": r["unique_pages_retrieved"],
                    "precision_at_k": r["precision_at_k"],
                })
        print(f"Results written to {out_path}")

    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the hybrid RAG pipeline")
    parser.add_argument("--csv", action="store_true", help="Write results to CSV")
    args = parser.parse_args()
    run_eval(write_csv=args.csv)


if __name__ == "__main__":
    main()
