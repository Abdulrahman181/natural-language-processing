# Engineering Review: `detecting-fake-news`

**Repository:** `Abdulrahman181/detecting-fake-news`  
**Review scope:** all tracked README, dependency, ignore, and notebook content, plus the repository's Python validation tooling  
**Review outcome:** repaired locally; no changes pushed to GitHub

## Executive summary

The repository contained one notebook and supporting documentation/dependency files, but no standalone Python source or tests. The notebook's original workflow had a syntax error, fragile data-column handling, missing-resource risk, nondeterministic training, and severe preprocessing leakage: Word2Vec and the Keras tokenizer were fit on the complete dataset before the train/test split. It also trained without an explicit validation/test protocol or stratification and depended on heavyweight, unpinned TensorFlow/Gensim/NLTK packages.

The notebook was replaced with a transparent TF-IDF + logistic-regression baseline. It now validates input paths/schema, removes exact duplicate articles before splitting, uses a fixed seed, performs stratified 60/20/20 train/validation/held-out-test splits, fits TF-IDF on training text only, and reports accuracy, balanced accuracy, precision, recall, F1, a classification report, and a held-out confusion matrix. No metric values are included in the repository because the dataset is not present.

## Findings and disposition

| Area | Finding | Disposition |
|---|---|---|
| Syntax | Original notebook cell contained `nos  < 1000`, which is invalid Python syntax. | Removed the broken exploratory cell; every new code cell is AST-parseable. |
| Undefined/fragile names | Original cells imported unused names, relied on notebook state, and used `nltk.sent_tokenize`/stopwords without ensuring NLTK resources. | Removed the unused NLP runtime path and use scikit-learn's self-contained text vectorizer. |
| Data paths | Original path default was usable but missing-file errors were raw `FileNotFoundError`s and schema assumptions were implicit. | Centralized `FAKE_NEWS_DATA_DIR`, used `pathlib`, checked both required files, checked `text`, and added actionable errors. |
| Data cleaning | Original `drop('Unnamed: 0', axis=1)` failed when that export artifact was absent; all-column `dropna` could discard unrelated fields. | Drop only index-like columns, select the required text field, remove empty text, detect conflicting duplicate labels, and deduplicate exact text. |
| Leakage | Original Word2Vec and tokenizer learned from all rows before splitting; the learned representation therefore saw the test corpus. | Split first; fit TF-IDF only on `text_train`; validation/test are transform-only. |
| Split/evaluation | Original `train_test_split` was not stratified, had no fixed test size, and used `validation_split` without an explicit reproducible validation protocol. | Fixed `SEED=42`, explicit 60/20/20 stratified partitions, and separate validation plus held-out-test evaluation. |
| Reproducibility | Original Word2Vec used multiple workers without a seed and the neural model had no global seed. | Python and NumPy seeds, seeded split, and seeded logistic regression; package ranges are documented. |
| Metrics | Original reported accuracy and a classification report only, with no balanced accuracy or confusion matrix and no guaranteed held-out methodology. | Added balanced accuracy, precision, recall, F1, classification report, and held-out confusion matrix. |
| Dependencies | Original requirements were unpinned and included TensorFlow/Gensim/NLTK for a workflow with avoidable compatibility and download risks. | Requirements now match the lightweight baseline with bounded major versions and Jupyter/nbformat tooling. |
| Notebook hygiene | Original notebook retained historical execution metadata and was not a clean, portable artifact. | Rebuilt with empty outputs and `execution_count=None`; static validator rejects stale outputs. |
| Repository hygiene | Ignore rules omitted common Python/test/tool caches. | Added `*.pyc`, pytest/mypy/Ruff caches, and logs while keeping the engineering report tracked. |

## Files changed

- `detecting-fake-news.ipynb` — repaired and simplified into a deterministic, leakage-safe baseline; outputs removed.
- `README.md` — documented the data contract, split methodology, validation command, and limitations.
- `requirements.txt` — replaced unbounded/heavy dependencies with bounded baseline dependencies.
- `.gitignore` — added common Python and validation caches/logs.
- `validate_project.py` — new dependency-free static checker for required files, nbformat shape, AST syntax, stale outputs/execution counts, portable paths, and methodology markers.
- `ENGINEERING_REVIEW.md` — this review.

## Validation performed

The following checks completed successfully in the sandbox:

1. `python3 validate_project.py` — 8 code cells compile; no stale outputs; required methodology markers present.
2. `python3 -m py_compile validate_project.py` — passed.
3. `python3 -m json.tool detecting-fake-news.ipynb` — valid JSON.
4. Notebook code cells extracted into a temporary Python file and checked with `ruff check --select E9,F63,F7,F82` — passed with no syntax or undefined-name errors.
5. `ruff check --select E9,F63,F7,F82 validate_project.py /tmp/fake_news_notebook.py` — passed.
6. `git diff --check` — run before commit; whitespace errors are not expected.

`nbformat` validation was attempted but skipped because the package is not installed in the current sandbox; it is declared in `requirements.txt`. No full notebook execution was claimed.

## Data and runtime caveat

The repository does not contain `DataSet_Misinfo_TRUE.csv` or `DataSet_Misinfo_FAKE.csv`, and no local dataset path was supplied. Therefore, the model was **not trained**, no metric values were generated, and no end-to-end notebook execution is claimed. The notebook instead fails clearly when the required data files are absent. Installing the declared dependencies and supplying the licensed/local dataset are prerequisites for a complete runtime check.

## Local-only delivery

A local commit was created in this repository only. Nothing was pushed to GitHub, and no other repository was changed.

## Second Senior Review (2026-10-16)

A second manual pass verified the notebook contract, portable data resolution, leakage-safe text fitting, clean execution state, and validator behavior. The missing regression-test layer was completed with `tests/test_validate_project.py`, covering validator execution, AST parsing, stale-output protection, and methodology markers. No dataset was present, so no training metrics were claimed.

Validation: `python3 validate_project.py`, `pytest -q` (2 passed), `compileall`, and `git diff --check` passed.
