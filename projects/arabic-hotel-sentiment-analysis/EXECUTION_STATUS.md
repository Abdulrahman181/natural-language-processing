# Execution Status

## Completed on all 397,993 model-ready records

- TF-IDF + Linear SVM: completed. Accuracy 0.8572; Macro-F1 0.7894; Weighted-F1 0.8561.
- BiLSTM: completed with one efficient CPU epoch. Accuracy 0.8787; Macro-F1 0.8141; Weighted-F1 0.8744.

## Transformer

- The original full CPU attempt was stopped by the runtime limit after more than two hours without a completed metrics file.
- The Transformer implementation has since been rewritten as a robust standalone full-data pipeline in `scripts/train_transformer.py` and `05_transformer_arabic_bert.ipynb`.
- It defaults to `MAX_SAMPLES` unset, which means all 397,993 rows. It supports CUDA/FP16, dynamic padding, gradient accumulation, checkpointing, metrics, and old/new Transformers API compatibility.
- A complete smoke test succeeded end-to-end on a temporary 32-row sample. Its score is intentionally not included as an official project result.
- Run `make train-transformer` on the target machine to produce the official full-data Transformer score and `artifacts/arabert_full_final/`.

No Transformer score is reported until the official full-data run completes.
