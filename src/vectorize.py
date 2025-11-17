import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

def join_tokens(tokens):
    return ' '.join(tokens)

def vectorize_text(df):
    df['joined_tokens'] = df['tokens'].apply(join_tokens)

    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b"
    )

    X_tfidf = vectorizer.fit_transform(df['joined_tokens'])
    numerical_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    X_numeric = csr_matrix(df[numerical_cols].values)
    X_final = hstack([X_tfidf, X_numeric])

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    return X_final, df['level']  # X, y
