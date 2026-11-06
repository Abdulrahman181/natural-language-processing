import numpy as np


def max_train_similarity(test_texts: list[str], train_texts: list[str], chunk: int = 100) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
        max_df=0.1,
        min_df=1,
        max_features=600000,
    )
    train_matrix = vectorizer.fit_transform(train_texts).T.tocsr()
    test_matrix = vectorizer.transform(test_texts).tocsr()
    result = np.zeros(test_matrix.shape[0], dtype=np.float64)
    for start in range(0, test_matrix.shape[0], chunk):
        block = test_matrix[start:start + chunk] @ train_matrix
        result[start:start + chunk] = block.max(axis=1).toarray().ravel()
    return result


def near_duplicate_mask(similarities, threshold: float) -> list[bool]:
    return [bool(value >= threshold) for value in similarities]
