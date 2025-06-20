import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

def join_tokens(tokens):
    """
    Join a list of tokens into a single string separated by spaces,
    suitable for TF-IDF vectorization.
    """
    return ' '.join(tokens)

def vectorize_text(df):
    """
    Convert tokenized text and numerical features from the DataFrame into
    a combined sparse feature matrix suitable for machine learning.
    Returns the feature matrix X and target labels y.
    """
    # Join tokens into strings for TF-IDF
    df['joined_tokens'] = df['tokens'].apply(join_tokens)

    # Initialize TF-IDF vectorizer with max 1000 features, unigrams and bigrams
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b"
    )
    # Fit the vectorizer and transform the joined token strings into vectors
    X_tfidf = vectorizer.fit_transform(df['joined_tokens'])

    # Select numeric columns from the DataFrame (int and float)
    numerical_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    # Convert numeric features to sparse matrix
    X_numeric = csr_matrix(df[numerical_cols].values)

    # Horizontally stack TF-IDF features and numeric features
    X_final = hstack([X_tfidf, X_numeric])

    # Save the fitted vectorizer for later use
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Return features and target labels
    return X_final, df['level']  # X, y
