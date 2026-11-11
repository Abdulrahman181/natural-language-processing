# rag-eval: retrieval-quality evaluation for Arabic RAG

Measures how well a retriever finds the passage that answers a question, so retrieval claims can be backed by numbers instead of impressions. The package is designed as an evaluation component that can be run locally, in CI, or behind an optional retrieval API.

It ships three reference retrievers (BM25, dense, hybrid) and an adapter that lets you score your own RAG project's retriever on exactly the same questions.

## How relevance is judged

Each test question is generated from one passage and stores a short evidence sentence copied verbatim from it. A retrieved chunk counts as relevant if it contains that evidence sentence after normalization. Because the check is on text, not on chunk ids, any retriever can be scored regardless of how it chunks documents.

Reported: Recall@1/3/5/10, MRR@10, nDCG@10 with bootstrap 95% confidence intervals, and latency p50/p95.

Every index and evaluation run records corpus/chunk/benchmark fingerprints, configuration, environment metadata, and the verified/unverified question policy. Loading an index rejects tampered chunks, missing dense embeddings, or mismatched embedding row counts.

## Pipeline

```
pip install -r requirements.txt

python -m rageval fetch --user-agent "rageval/0.1 (your-email@example.com)"
python -m rageval build-index
export GROQ_API_KEY=...
python -m rageval gen-questions --model <a Groq-hosted model that handles Arabic> --n 150
```

Then open `data/questions.csv` and review every row. Set `verified` to `1` for rows where the question is natural and answerable from the evidence sentence, leave the rest empty. Save as CSV with UTF-8. Only verified rows are evaluated.

```
python -m rageval evaluate
python -m rageval evaluate --adapter adapters/my_rag.py
```

The optional service can expose a built BM25 index:

```
python -m pip install -e '.[serve]'
RAGEVAL_INDEX_DIR=artifacts python -m rageval.api
```

It provides `/healthz`, `/readyz`, `/v1/metadata`, and `/v1/search`. A non-root Docker image is included for deployment after mounting a validated `artifacts/` directory.

Notes:

- `fetch` downloads about 120 Arabic Wikipedia articles listed in `data/titles_ar.txt`. Wikipedia asks for a descriptive User-Agent with a contact address, so pass your own with `--user-agent`.
- `gen-questions` takes any OpenAI-compatible endpoint through `--base-url` and `--api-key-env`. Check your provider's current model list for `--model`.
- `export-txt` writes the corpus as plain text files so you can ingest the same documents into your own RAG project.

## Scoring your own RAG

1. Ingest the exported text files into your RAG project with its normal chunking and embedding settings.
2. Copy `adapters/example_adapter.py` to `adapters/my_rag.py`, set `NAME`, and implement `search(query, k)` to return the text of the top-k retrieved chunks, best first.
3. Run `python -m rageval evaluate --adapter adapters/my_rag.py`.

The numbers describe your retriever only when produced this way. The reference retrievers' numbers describe those reference retrievers on this corpus and must not be presented as results of your project.

## Files to send back

- `results/results.md`, `results/results.json`, `results/per_query.csv`
- the reviewed `data/questions.csv`

## Limits

- Questions are LLM-generated and then human-verified; the quality of the benchmark is the quality of that review.
- The corpus is Wikipedia, which is cleaner than most enterprise documents, so absolute scores are optimistic for real deployments.
- Evidence matching is lexical after normalization; a retriever returning a correct passage that paraphrases the evidence is scored as a miss.
- The fetch and question-generation steps were tested with mocked network responses only, because the authoring environment could not reach Wikipedia. Report any failure with the full error text.

## Tests

`pytest -q` runs 70 tests covering normalization, BM25, ranking metrics, bootstrap intervals, corpus cleaning, the LLM client's retry behavior, question validation, provenance/integrity checks, API health/search, and the full pipeline on a synthetic Arabic corpus including negative controls (a retriever that always returns wrong chunks scores zero). CI also runs Ruff, Bandit, and a 70% coverage gate.
