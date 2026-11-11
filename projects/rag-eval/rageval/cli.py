import argparse
import os
import sys
from pathlib import Path

from .corpus import (
    DEFAULT_USER_AGENT,
    export_text_files,
    fetch_wikipedia,
    load_corpus,
    load_titles,
    save_corpus,
)
from .evaluate import evaluate_all, format_table, run_metadata, write_reports
from .index import build_index, load_index
from .llm import ChatClient
from .provenance import questions_fingerprint
from .questions import generate, read_questions, write_questions
from .retrievers import Bm25Retriever, DenseRetriever, HybridRetriever, load_adapter


def _make_embedder(model: str, query_prefix: str, passage_prefix: str, max_seq_length, batch_size: int):
    from .dense import SentenceEmbedder

    return SentenceEmbedder(
        model,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
    )


def command_fetch(args) -> int:
    titles = load_titles(Path(args.titles))
    documents, skipped = fetch_wikipedia(
        titles, user_agent=args.user_agent, min_chars=args.min_chars, pause=args.pause
    )
    print(f"requested={len(titles)} saved={len(documents)} skipped={len(skipped)}")
    reasons: dict[str, int] = {}
    for _, reason in skipped:
        reasons[reason] = reasons.get(reason, 0) + 1
    for reason, count in sorted(reasons.items()):
        print(f"  skipped_{reason}={count}")
    if not documents:
        print("No documents saved. Check your connection and the titles file.")
        return 1
    save_corpus(documents, Path(args.out))
    return 0


def command_export(args) -> int:
    written = export_text_files(load_corpus(Path(args.corpus)), Path(args.out_dir))
    print(f"exported_files={len(written)} directory={args.out_dir}")
    return 0


def command_build(args) -> int:
    documents = load_corpus(Path(args.corpus))
    embedder = None
    if not args.no_dense:
        embedder = _make_embedder(args.model, args.query_prefix, args.passage_prefix, args.max_seq_length, args.batch_size)
    index = build_index(documents, Path(args.out), args.chunk_size, args.overlap, args.min_words, embedder)
    print(f"documents={index.meta['documents']} chunks={index.meta['chunks']} dense={index.meta['dense']}")
    return 0


def command_questions(args) -> int:
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    local = "localhost" in args.base_url or "127.0.0.1" in args.base_url
    if api_key is None and not local:
        print(f"Environment variable {args.api_key_env} is not set.")
        return 2
    index = load_index(Path(args.index))
    client = ChatClient(args.base_url, args.model, api_key=api_key)
    rows, stats = generate(
        index.chunks,
        client,
        target=args.n,
        per_doc=args.per_doc,
        min_words=args.min_words,
        seed=args.seed,
        max_run=args.max_run,
        pause=args.pause,
    )
    write_questions(Path(args.out), rows)
    print(f"written={len(rows)} target={args.n} file={args.out}")
    for key in sorted(stats):
        print(f"  {key}={stats[key]}")
    if len(rows) < args.n:
        print("Fewer questions than requested were accepted. Raise --n or lower the filters.")
    print("Next: review every row, set verified to 1 for correct ones, save as CSV UTF-8.")
    return 0


def command_evaluate(args) -> int:
    index = load_index(Path(args.index))
    questions, total = read_questions(Path(args.questions), verified_only=not args.include_unverified)
    if not questions:
        print("No questions to evaluate. Mark rows with verified=1 or pass --include-unverified.")
        return 2
    sparse = Bm25Retriever(index.chunks)
    retrievers: list = [sparse]
    if index.embeddings is not None and not args.no_dense:
        if index.embeddings.shape[0] != len(index.chunks):
            print("Embeddings and chunks are out of sync. Rebuild the index.")
            return 2
        meta = index.meta
        embedder = _make_embedder(
            meta["embedding_model"],
            meta.get("query_prefix", ""),
            meta.get("passage_prefix", ""),
            meta.get("max_seq_length"),
            args.batch_size,
        )
        dense = DenseRetriever(index.chunks, index.embeddings, embedder)
        retrievers.append(dense)
        retrievers.append(HybridRetriever(sparse, dense, pool=args.pool, rrf_k=args.rrf_k))
    for adapter_path in args.adapter:
        retrievers.append(load_adapter(Path(adapter_path)))
    summaries, per_query = evaluate_all(retrievers, questions)
    meta = run_metadata(
        index.meta,
        total,
        len(questions),
        verified_only=not args.include_unverified,
        benchmark_fingerprint=questions_fingerprint(questions),
    )
    write_reports(Path(args.out), summaries, per_query, meta)
    print(format_table(summaries))
    print(f"reports written to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rageval")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch")
    fetch.add_argument("--titles", default="data/titles_ar.txt")
    fetch.add_argument("--out", default="data/corpus.jsonl")
    fetch.add_argument("--min-chars", type=int, default=3000)
    fetch.add_argument("--pause", type=float, default=0.3)
    fetch.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    fetch.set_defaults(func=command_fetch)

    export = commands.add_parser("export-txt")
    export.add_argument("--corpus", default="data/corpus.jsonl")
    export.add_argument("--out-dir", default="data/txt")
    export.set_defaults(func=command_export)

    build = commands.add_parser("build-index")
    build.add_argument("--corpus", default="data/corpus.jsonl")
    build.add_argument("--out", default="artifacts")
    build.add_argument("--chunk-size", type=int, default=120)
    build.add_argument("--overlap", type=int, default=20)
    build.add_argument("--min-words", type=int, default=40)
    build.add_argument("--model", default="intfloat/multilingual-e5-small")
    build.add_argument("--query-prefix", default="query: ")
    build.add_argument("--passage-prefix", default="passage: ")
    build.add_argument("--max-seq-length", type=int, default=384)
    build.add_argument("--batch-size", type=int, default=16)
    build.add_argument("--no-dense", action="store_true")
    build.set_defaults(func=command_build)

    questions = commands.add_parser("gen-questions")
    questions.add_argument("--index", default="artifacts")
    questions.add_argument("--out", default="data/questions.csv")
    questions.add_argument("--n", type=int, default=150)
    questions.add_argument("--per-doc", type=int, default=2)
    questions.add_argument("--min-words", type=int, default=60)
    questions.add_argument("--seed", type=int, default=13)
    questions.add_argument("--max-run", type=int, default=3)
    questions.add_argument("--pause", type=float, default=2.0)
    questions.add_argument("--model", required=True)
    questions.add_argument("--base-url", default="https://api.groq.com/openai/v1")
    questions.add_argument("--api-key-env", default="GROQ_API_KEY")
    questions.set_defaults(func=command_questions)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--index", default="artifacts")
    evaluate.add_argument("--questions", default="data/questions.csv")
    evaluate.add_argument("--out", default="results")
    evaluate.add_argument("--adapter", action="append", default=[])
    evaluate.add_argument("--pool", type=int, default=50)
    evaluate.add_argument("--rrf-k", type=int, default=60)
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--no-dense", action="store_true")
    evaluate.add_argument("--include-unverified", action="store_true")
    evaluate.set_defaults(func=command_evaluate)
    return parser


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return args.func(args)
