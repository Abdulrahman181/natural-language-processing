import csv
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .metrics import (
    bootstrap_ci,
    first_relevant_rank,
    hits,
    latency_summary,
    ndcg_values,
    reciprocal_ranks,
)
from .provenance import run_manifest
from .questions import Question

KS = (1, 3, 5, 10)


def evaluate_retriever(retriever, questions: list[Question], ks: tuple[int, ...] = KS, warmup: int = 3):
    if not questions:
        raise ValueError("No questions to evaluate")
    depth = max(max(ks), 10)
    for question in questions[:warmup]:
        retriever.search(question.question, depth)
    ranks: list[int | None] = []
    latencies: list[float] = []
    previews: list[str] = []
    for question in questions:
        started = time.perf_counter()
        texts = retriever.search(question.question, depth)
        latencies.append((time.perf_counter() - started) * 1000.0)
        ranks.append(first_relevant_rank(texts, question.evidence))
        previews.append(texts[0][:160].replace("\n", " ") if texts else "")
    summary: dict = {"retriever": retriever.name, "questions": len(questions)}
    for k in ks:
        summary[f"recall@{k}"] = float(hits(ranks, k).mean())
    summary["mrr@10"] = float(reciprocal_ranks(ranks, 10).mean())
    summary["ndcg@10"] = float(ndcg_values(ranks, 10).mean())
    summary["recall@5_ci95"] = list(bootstrap_ci(hits(ranks, 5)))
    summary["mrr@10_ci95"] = list(bootstrap_ci(reciprocal_ranks(ranks, 10)))
    summary["latency_ms"] = latency_summary(latencies)
    return summary, ranks, previews


def evaluate_all(retrievers: list, questions: list[Question]):
    summaries = []
    per_query = []
    for retriever in retrievers:
        summary, ranks, previews = evaluate_retriever(retriever, questions)
        summaries.append(summary)
        for question, rank, preview in zip(questions, ranks, previews, strict=True):
            per_query.append(
                {
                    "question_id": question.id,
                    "retriever": retriever.name,
                    "rank": rank if rank is not None else "",
                    "top1_preview": preview,
                }
            )
    return summaries, per_query


def run_metadata(
    index_meta: dict,
    questions_in_file: int,
    questions_evaluated: int,
    verified_only: bool,
    benchmark_fingerprint: str | None = None,
) -> dict:
    metadata = run_manifest(index_meta, questions_in_file, questions_evaluated, verified_only)
    metadata.update(
        {
            "benchmark_fingerprint": benchmark_fingerprint,
            "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "platform": platform.platform(),
            "python_full": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "index": index_meta,
        }
    )
    return metadata


def format_table(summaries: list[dict]) -> str:
    header = "| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |"
    divider = "|---|---|---|---|---|---|---|---|---|"
    lines = [header, divider]
    for item in summaries:
        lines.append(
            "| {name} | {r1:.3f} | {r3:.3f} | {r5:.3f} | {r10:.3f} | {mrr:.3f} | {ndcg:.3f} | {p50:.1f} | {p95:.1f} |".format(
                name=item["retriever"],
                r1=item["recall@1"],
                r3=item["recall@3"],
                r5=item["recall@5"],
                r10=item["recall@10"],
                mrr=item["mrr@10"],
                ndcg=item["ndcg@10"],
                p50=item["latency_ms"]["p50"],
                p95=item["latency_ms"]["p95"],
            )
        )
    return "\n".join(lines)


def format_report(summaries: list[dict], meta: dict) -> str:
    index = meta["index"]
    lines = [
        "# Retrieval evaluation",
        "",
        f"- Questions evaluated: {meta['questions_evaluated']} of {meta['questions_in_file']} in the file "
        f"({'manually verified rows only' if meta['verified_only'] else 'includes unverified rows'})",
        f"- Corpus: {index.get('documents')} documents, {index.get('chunks')} chunks "
        f"({index.get('chunk_size_words')} words, {index.get('overlap_words')} overlap)",
        f"- Embedding model: {index.get('embedding_model')}",
        f"- Hardware and software: {meta['platform']}, {meta['cpu_count']} logical CPUs, Python {meta['python']}",
        f"- Run date (UTC): {meta['date_utc']}",
        "",
        format_table(summaries),
        "",
        "95% bootstrap confidence intervals (1000 resamples):",
        "",
    ]
    for item in summaries:
        recall_low, recall_high = item["recall@5_ci95"]
        mrr_low, mrr_high = item["mrr@10_ci95"]
        lines.append(
            f"- {item['retriever']}: Recall@5 {item['recall@5']:.3f} [{recall_low:.3f}, {recall_high:.3f}], "
            f"MRR@10 {item['mrr@10']:.3f} [{mrr_low:.3f}, {mrr_high:.3f}]"
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(out_dir: Path, summaries: list[dict], per_query: list[dict], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"meta": meta, "results": summaries}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "results.md").write_text(format_report(summaries, meta), encoding="utf-8")
    with (out_dir / "per_query.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_id", "retriever", "rank", "top1_preview"])
        writer.writeheader()
        writer.writerows(per_query)
