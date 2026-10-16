# Fake News Detection

This repository contains a reproducible **TF-IDF + logistic-regression** baseline for fake-news text classification. The notebook is intentionally small and auditable: preprocessing is fitted on the training partition only, the split is stratified and seeded, and the held-out test metrics are produced only when the local data is available.

## Data contract

The dataset is intentionally not committed because of size/licensing constraints. Provide both files below in one directory:

- `DataSet_Misinfo_TRUE.csv`
- `DataSet_Misinfo_FAKE.csv`

Each CSV must contain a `text` column. The loader removes pandas index columns, drops empty text, detects conflicting labels, and removes exact duplicate articles before splitting. Set `FAKE_NEWS_DATA_DIR` to the directory containing the files; it defaults to `data/` for local execution. If a required file or column is missing, the notebook raises an actionable error rather than inventing a result.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export FAKE_NEWS_DATA_DIR=/absolute/path/to/data
python validate_project.py
jupyter notebook detecting-fake-news.ipynb
```

Run cells from top to bottom. `validate_project.py` performs dependency-free JSON, AST, stale-output, portable-path, and methodology-marker checks. Generated models, caches, and local datasets remain untracked.

## Methodology and limitations

- A fixed `SEED=42` controls the Python, NumPy, split, and model randomness.
- The protocol is 60% train, 20% validation, and 20% held-out test, using stratification where scikit-learn can support it.
- TF-IDF vocabulary/statistics are learned from training text only; validation and test text are transform-only.
- Accuracy, balanced accuracy, precision, recall, F1, a classification report, and a held-out confusion matrix are reported. This repository contains no fabricated metric values.
- Results depend on the dataset, labels, split, preprocessing, and package versions. The dataset is absent from this repository, so a full model run is not claimed here.
- This is an educational/research baseline, not a production fact-checking or professional-advice system. For stronger claims, add a data manifest, repeated or time-based validation, and independent evaluation.
