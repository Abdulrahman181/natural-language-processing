from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/hotel_reviews_model_ready.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True, max_features=150000)),
        ("clf", LinearSVC(C=1.5, class_weight="balanced")),
    ])
    model.fit(df["review_clean"], df["sentiment"])
    artifact = ARTIFACTS / "tfidf_svm.joblib"
    joblib.dump(model, artifact, compress=3)
    metrics = {"model": "TF-IDF + Linear SVM", "training_rows": int(len(df)), "labels": sorted(df["sentiment"].unique().tolist())}
    (ARTIFACTS / "production_model_manifest.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved={artifact} rows={len(df)}")

if __name__ == "__main__":
    main()
