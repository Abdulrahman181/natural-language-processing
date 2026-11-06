from pathlib import Path

from .io_utils import read_json, read_jsonl, write_json
from .metrics import (
    accuracy,
    bootstrap_ci,
    correct_vector,
    invalid_rate,
    macro_f1,
    paired_bootstrap_diff,
    per_class_scores,
    select,
    top_confusions,
)
from .parsing import INVALID

SYSTEM_LABELS = {
    "majority": "Majority class",
    "tfidf_svm_matched": "TF-IDF + linear SVM (same training subset)",
    "tfidf_svm_full": "TF-IDF + linear SVM (full train split)",
    "base_zero_shot": "Base model, zero-shot",
    "qlora_tuned": "Base model + QLoRA adapter",
}
COMPARISONS = [
    ("qlora_tuned", "base_zero_shot"),
    ("qlora_tuned", "tfidf_svm_matched"),
    ("qlora_tuned", "tfidf_svm_full"),
]
DEFAULT_NOVEL_THRESHOLD = 0.85
READING_NOTES = [
    "Validation and test labels in this dataset are agreement-filtered: only rows where the NVD label equals an LLM-assigned label are kept, "
    "so the test set is easier than raw NVD data and accuracies here will not transfer to ambiguous real-world records.",
    "The dataset split is random, not temporal. Templated or near-identical descriptions can appear on both sides, "
    "so the novel subset (no training description above the similarity threshold) is the more conservative number.",
    "Training labels were produced by an LLM relabeling pass described in the dataset card. These results measure agreement with those labels, not with human-verified ground truth.",
    "Single training run with one seed. Confidence intervals reflect test-set sampling only, not training variance.",
    "The dataset author reports 87.4% top-1 for a RoBERTa-base classifier on the full agreement-filtered test set. That figure is quoted for context and was not reproduced here.",
]


def load_predictions(path, test_rows: list[dict]) -> list[str]:
    records = read_jsonl(path)
    expected_ids = {row["id"] for row in test_rows}
    seen = set()
    duplicates = set()
    unexpected = set()
    by_id = {}
    for row in records:
        row_id = row.get("id")
        if row_id in seen:
            duplicates.add(row_id)
        seen.add(row_id)
        if row_id not in expected_ids:
            unexpected.add(row_id)
        by_id[row_id] = row.get("pred", INVALID)
    if duplicates:
        raise ValueError(f"Duplicate prediction ids in {path}: {sorted(duplicates)[:3]}")
    if unexpected:
        raise ValueError(f"Unexpected prediction ids in {path}: {sorted(unexpected)[:3]}")
    return [by_id.get(row["id"], INVALID) for row in test_rows]


def summarize(gold: list[str], pred: list[str], novel_mask: list[bool] | None) -> dict:
    correct = correct_vector(gold, pred)
    low, high = bootstrap_ci(correct)
    summary = {
        "n": len(gold),
        "accuracy": accuracy(gold, pred),
        "accuracy_ci95": [low, high],
        "macro_f1": macro_f1(gold, pred),
        "invalid_rate": invalid_rate(pred),
    }
    if novel_mask is not None:
        subset_gold = select(gold, novel_mask)
        subset_pred = select(pred, novel_mask)
        sub_low, sub_high = bootstrap_ci(correct_vector(subset_gold, subset_pred))
        summary["novel_subset"] = {
            "n": len(subset_gold),
            "accuracy": accuracy(subset_gold, subset_pred),
            "accuracy_ci95": [sub_low, sub_high],
            "macro_f1": macro_f1(subset_gold, subset_pred),
        }
    return summary


def build_report(
    work_dir,
    results_dir,
    threshold: float = DEFAULT_NOVEL_THRESHOLD,
    allow_dry_run: bool = False,
) -> dict:
    work = Path(work_dir)
    results = Path(results_dir)
    prepare_meta_path = work / "prepare_meta.json"
    prepare_meta = read_json(prepare_meta_path) if prepare_meta_path.exists() else None
    if prepare_meta and prepare_meta.get("run_mode") == "dry-run" and not allow_dry_run:
        raise ValueError(
            "Refusing to create a publishable report from dry-run artifacts. "
            "Use a full run, or pass --allow-dry-run only for debugging."
        )
    test = read_jsonl(work / "test.jsonl")
    gold = [row["label"] for row in test]
    novel_mask = None
    similarity_path = results / "similarity.jsonl"
    if similarity_path.exists():
        by_id = {row["id"]: row["max_similarity"] for row in read_jsonl(similarity_path)}
        novel_mask = [by_id.get(row["id"], 1.0) < threshold for row in test]
    systems = {}
    predictions = {}
    for name in SYSTEM_LABELS:
        path = results / f"pred_{name}.jsonl"
        if path.exists():
            predictions[name] = load_predictions(path, test)
            systems[name] = summarize(gold, predictions[name], novel_mask)
    if not systems:
        raise FileNotFoundError(f"No pred_*.jsonl files found in {results}")
    comparisons = []
    for first, second in COMPARISONS:
        if first in predictions and second in predictions:
            mean, low, high = paired_bootstrap_diff(
                correct_vector(gold, predictions[first]), correct_vector(gold, predictions[second])
            )
            comparisons.append(
                {"a": first, "b": second, "accuracy_difference": mean, "ci95": [low, high]}
            )
    focus = predictions.get("qlora_tuned") or next(iter(predictions.values()))
    focus_name = "qlora_tuned" if "qlora_tuned" in predictions else next(iter(predictions))
    weakest = sorted(
        (row for row in per_class_scores(gold, focus) if row["support"] >= 5),
        key=lambda row: row["f1"],
    )[:10]
    report = {
        "test_sample_size": len(test),
        "novel_similarity_threshold": threshold,
        "systems": systems,
        "paired_comparisons": comparisons,
        "confusions_for": focus_name,
        "top_confusions": top_confusions(gold, focus),
        "weakest_classes_with_support_5_plus": weakest,
        "prepare": prepare_meta,
        "training": read_json(work / "train_run.json") if (work / "train_run.json").exists() else None,
        "baselines": read_json(results / "baselines_meta.json") if (results / "baselines_meta.json").exists() else None,
        "reading_notes": READING_NOTES,
    }
    return report


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(report: dict) -> str:
    threshold = report["novel_similarity_threshold"]
    lines = [
        "# CVE-to-CWE: QLoRA fine-tuning evaluation",
        "",
        f"Test sample: {report['test_sample_size']} rows (seeded random sample of the dataset test split).",
        "",
        f"| System | Accuracy (95% CI) | Macro-F1 | Invalid outputs | Accuracy on novel subset (similarity < {threshold}) |",
        "|---|---|---|---|---|",
    ]
    for name, label in SYSTEM_LABELS.items():
        item = report["systems"].get(name)
        if item is None:
            continue
        low, high = item["accuracy_ci95"]
        novel = item.get("novel_subset")
        novel_text = (
            f"{percent(novel['accuracy'])} (n={novel['n']})" if novel is not None else "not computed"
        )
        lines.append(
            f"| {label} | {percent(item['accuracy'])} [{percent(low)}, {percent(high)}] | "
            f"{item['macro_f1']:.3f} | {percent(item['invalid_rate'])} | {novel_text} |"
        )
    if report["paired_comparisons"]:
        lines += ["", "## Paired accuracy differences (bootstrap, 95% CI)", ""]
        for row in report["paired_comparisons"]:
            low, high = row["ci95"]
            lines.append(
                f"- {SYSTEM_LABELS[row['a']]} minus {SYSTEM_LABELS[row['b']]}: "
                f"{row['accuracy_difference'] * 100:+.1f} points [{low * 100:+.1f}, {high * 100:+.1f}]"
            )
    training = report.get("training")
    if training:
        lines += [
            "",
            "## Training run",
            "",
            f"- Base model: {training.get('model')}",
            f"- Training examples used: {training.get('train_examples')} "
            f"(dropped for length: {training.get('dropped_train_too_long')})",
            f"- Epochs: {training.get('epochs')}, effective batch size: {training.get('effective_batch_size')}, "
            f"learning rate: {training.get('learning_rate')}",
            f"- LoRA rank {training.get('lora_r')}, alpha {training.get('lora_alpha')}, "
            f"trainable parameters: {training.get('trainable_parameters')}",
            f"- Hardware: {training.get('device')}, precision: {training.get('precision')}, "
            f"4-bit quantization: {training.get('quantized')}",
            f"- Wall-clock training time: {training.get('train_seconds')} seconds",
            f"- Library versions: {training.get('versions')}",
        ]
    prepare = report.get("prepare")
    if prepare:
        lines += [
            "",
            "## Data",
            "",
            f"- Dataset: {prepare.get('dataset')} at revision {prepare.get('revision')}",
            f"- Sizes: {prepare.get('sizes')}",
            f"- Classes: {prepare.get('classes')}",
            f"- Test rows whose label never appears in the training subset: "
            f"{prepare.get('test_rows_with_label_unseen_in_train_subset')}",
            f"- Test rows with an exact description duplicate in the full train split: "
            f"{prepare.get('test_exact_description_duplicates_in_full_train')}",
        ]
    if report["top_confusions"]:
        lines += ["", f"## Most frequent confusions ({report['confusions_for']})", ""]
        for row in report["top_confusions"][:10]:
            lines.append(f"- gold {row['gold']} predicted as {row['pred']}: {row['count']}")
    lines += ["", "## How to read these numbers", ""]
    lines += [f"- {note}" for note in report["reading_notes"]]
    lines.append("")
    return "\n".join(lines)


def write_report(
    work_dir,
    results_dir,
    threshold: float = DEFAULT_NOVEL_THRESHOLD,
    allow_dry_run: bool = False,
) -> dict:
    report = build_report(work_dir, results_dir, threshold, allow_dry_run=allow_dry_run)
    out = Path(results_dir)
    write_json(out / "results.json", report)
    (out / "results.md").write_text(render_markdown(report), encoding="utf-8")
    return report
