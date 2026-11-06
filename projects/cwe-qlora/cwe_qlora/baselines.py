import time
from collections import Counter
from pathlib import Path

from .io_utils import read_jsonl, write_json, write_jsonl
from .similarity import max_train_similarity


def majority_label(labels: list[str]) -> str:
    return Counter(labels).most_common(1)[0][0]


def fit_tfidf_svm(texts: list[str], labels: list[str], c: float = 0.5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=500000),
        LinearSVC(C=c, dual="auto"),
    )
    pipeline.fit(texts, labels)
    return pipeline


def prediction_rows(test: list[dict], predictions: list[str]) -> list[dict]:
    return [
        {"id": row["id"], "gold": row["label"], "raw": pred, "pred": pred}
        for row, pred in zip(test, predictions)
    ]


def run_baselines(work_dir, out_dir) -> dict:
    work = Path(work_dir)
    out = Path(out_dir)
    train = read_jsonl(work / "train.jsonl")
    full_train = read_jsonl(work / "full_train.jsonl")
    test = read_jsonl(work / "test.jsonl")
    test_texts = [row["description"] for row in test]
    timings = {}

    prior = majority_label([row["label"] for row in full_train])
    write_jsonl(out / "pred_majority.jsonl", prediction_rows(test, [prior] * len(test)))

    started = time.perf_counter()
    matched = fit_tfidf_svm([row["description"] for row in train], [row["label"] for row in train])
    timings["tfidf_svm_matched_fit_seconds"] = time.perf_counter() - started
    write_jsonl(out / "pred_tfidf_svm_matched.jsonl", prediction_rows(test, list(matched.predict(test_texts))))

    started = time.perf_counter()
    full = fit_tfidf_svm([row["description"] for row in full_train], [row["label"] for row in full_train])
    timings["tfidf_svm_full_fit_seconds"] = time.perf_counter() - started
    write_jsonl(out / "pred_tfidf_svm_full.jsonl", prediction_rows(test, list(full.predict(test_texts))))

    started = time.perf_counter()
    similarities = max_train_similarity(test_texts, [row["description"] for row in full_train])
    timings["similarity_seconds"] = time.perf_counter() - started
    write_jsonl(
        out / "similarity.jsonl",
        [{"id": row["id"], "max_similarity": float(value)} for row, value in zip(test, similarities)],
    )
    meta = {"majority_label": prior, "timings": timings, "train_subset": len(train), "full_train": len(full_train)}
    write_json(out / "baselines_meta.json", meta)
    return meta
