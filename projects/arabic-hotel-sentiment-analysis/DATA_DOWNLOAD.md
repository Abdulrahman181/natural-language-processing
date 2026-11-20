# Data Distribution

The repository code, notebooks, reports, production service, and small model artifact are stored in Git. The raw and derived CSV files are intentionally excluded from ordinary Git history because GitHub rejects regular files over 100 MB.

The complete project bundle, including the original source and derived datasets, is attached to the private GitHub Release as:

`sentiment_analysis_company_ready_ubuntu.zip`

After downloading the release asset, extract it into the repository root so these paths exist:

- `data/Hotel Arabic-Reviews Dataset.txt`
- `data/processed/hotel_reviews_processed.csv`
- `data/processed/hotel_reviews_model_ready.csv`
- `data/processed/hotel_reviews_quarantine.csv`

The source file is preserved byte-for-byte and its SHA-256 is recorded in `DATA_INTEGRITY.txt`.
