# Results Summary

## Data audit

- Records: 409,562
- Columns: 7
- Encoding: UTF-16LE with BOM
- Malformed logical rows: 0
- Missing/empty values in the source columns: 0
- Rating counts: 1=14,382; 2=38,467; 3=80,326; 4=132,208; 5=144,179
- Sentiment mapping: 1–2 negative, 3 neutral, 4–5 positive
- Derived sentiment counts: negative=52,849; neutral=80,326; positive=276,387
- Repeated review groups: 2,693; extra repeated occurrences: 9,461

## Reproducible lightweight run

The notebook was executed successfully with a stratified sample of 12,000 rows: 8,400 train, 1,200 validation, and 2,400 test. The source file was read without modification.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Linear SVM | 0.8588 | 0.8054 | 0.7818 | 0.7925 | 0.8557 |
| BiLSTM | 0.8375 | 0.7828 | 0.7402 | 0.7575 | 0.8326 |

The SVM is the stronger model in this CPU-friendly run. Macro-F1 is emphasized because the positive class is substantially larger than the negative class.

## Transformer path

The notebook includes a complete Arabic BERT fine-tuning section using `aubmindlab/bert-base-arabertv02`. It is disabled by default (`RUN_TRANSFORMER=False`) because downloading model weights and fine-tuning requires considerably more memory/time. Set it to `True` on a machine with suitable resources; its test metrics will be added to the same comparison table.


## Full-data preprocessing output

Notebook 02 completed successfully on all 409,562 records and produced 30 columns. The original seven columns are present unchanged in the derived output, alongside cleaned fields and quality flags. The observed flags include: 154,406 rows with at least one missing metadata field, 72 suspicious user-type values matching room-description patterns, 14,586 rows belonging to repeated cleaned review texts, 9,937 reviews containing Latin characters, 32 reviews becoming empty after conservative normalization, and 40 reviews shorter than three characters after normalization. These are flags for analysis, not silent row deletion.


## Full cleaning completed

Notebook 02 now performs the complete documented cleaning workflow. It produced:

- Full traceable processed dataset: `data/processed/hotel_reviews_processed.csv` (409,562 rows).
- Final model-ready dataset: `data/processed/hotel_reviews_model_ready.csv` (397,993 rows, no duplicate normalized reviews, no empty/short reviews, and no invalid ratings).
- Quarantine dataset: `data/processed/hotel_reviews_quarantine.csv` (11,569 rows) with an explicit exclusion reason for every row.
- Cleaning log: `data/processed/model_ready_cleaning_log.json`.

The model-ready exclusions were: 11,529 normalized duplicate reviews, 32 empty reviews after normalization, and 8 too-short reviews. The 72 room-like `user type` values were corrected under the documented rule: moved into the room field only when the room field was missing, and the user field was set to `Unknown`. No original source row was overwritten; excluded rows are retained in quarantine.
