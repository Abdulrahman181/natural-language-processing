"""Train Arabic BERT sentiment model on the complete model-ready dataset.

Default behavior uses every row in data/processed/hotel_reviews_model_ready.csv.
For a smoke test only, set MAX_SAMPLES explicitly. Never use a smoke-test score as
an official result.
"""
from __future__ import annotations

import inspect
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/hotel_reviews_model_ready.csv"
ARTIFACTS = Path(os.getenv("ARTIFACTS_ROOT", str(ROOT / "artifacts")))
OUTPUT_DIR = ARTIFACTS / "arabert_full"
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
SEED = int(os.getenv("SEED", "42"))
MODEL_NAME = os.getenv("TRANSFORMER_MODEL", "aubmindlab/bert-base-arabertv02")
MAX_SAMPLES = os.getenv("MAX_SAMPLES")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "192"))
EPOCHS = float(os.getenv("EPOCHS", "2"))
TRAIN_BATCH = int(os.getenv("TRAIN_BATCH_SIZE", "8"))
EVAL_BATCH = int(os.getenv("EVAL_BATCH_SIZE", "16"))
GRAD_ACCUM = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "2"))
LR = float(os.getenv("LEARNING_RATE", "2e-5"))


def split_data(df: pd.DataFrame):
    from sklearn.model_selection import train_test_split
    if MAX_SAMPLES:
        n = min(int(MAX_SAMPLES), len(df))
        df = train_test_split(df, train_size=n, stratify=df["sentiment"], random_state=SEED)[0]
    train, test = train_test_split(df, test_size=0.20, stratify=df["sentiment"], random_state=SEED)
    train, val = train_test_split(train, test_size=0.125, stratify=train["sentiment"], random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def to_dataset(frame: pd.DataFrame) -> Dataset:
    data = frame[["review_clean", "sentiment"]].rename(columns={"review_clean": "text", "sentiment": "labels"}).copy()
    data["labels"] = data["labels"].map(LABEL2ID).astype("int64")
    return Dataset.from_pandas(data, preserve_index=False)


def metrics_fn(eval_pred):
    predictions = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    pred = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(eval_pred.label_ids, pred),
        "macro_f1": f1_score(eval_pred.label_ids, pred, average="macro"),
        "weighted_f1": f1_score(eval_pred.label_ids, pred, average="weighted"),
    }


def make_training_args():
    kwargs = dict(
        output_dir=str(OUTPUT_DIR),
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=min(4, os.cpu_count() or 1),
        fp16=bool(torch.cuda.is_available() and os.getenv("USE_FP16", "1") == "1"),
        bf16=bool(torch.cuda.is_available() and os.getenv("USE_BF16", "0") == "1"),
        optim="adamw_torch",
    )
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    # Keep the script portable across Transformers releases: older versions
    # expose fewer TrainingArguments fields than current versions.
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in params})


def make_trainer(model, args, train_ds, val_ds, tokenizer, collator):
    kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=metrics_fn,
    )
    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    ARTIFACTS.mkdir(exist_ok=True)
    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    train, val, test = split_data(df)
    print(f"rows: total={len(df)} train={len(train)} val={len(val)} test={len(test)}")
    print(f"device={'cuda' if torch.cuda.is_available() else 'cpu'} model={MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds, val_ds, test_ds = to_dataset(train), to_dataset(val), to_dataset(test)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )
    trainer = make_trainer(model, make_training_args(), train_ds, val_ds, tokenizer, collator)
    trainer.train()
    prediction = trainer.predict(test_ds)
    logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
    y_true = prediction.label_ids
    y_pred = np.argmax(logits, axis=1)
    report = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4, zero_division=0))
    metrics = {
        "model": "Arabic BERT Transformer",
        "model_name": MODEL_NAME,
        "data_rows": int(len(df)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "max_samples": None if not MAX_SAMPLES else int(MAX_SAMPLES),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS)))).tolist(),
    }
    metrics_path = ARTIFACTS / "transformer_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    final_dir = ARTIFACTS / "arabert_full_final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    (final_dir / "model_manifest.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics_saved={metrics_path}")
    print(f"model_saved={final_dir}")


if __name__ == "__main__":
    main()
