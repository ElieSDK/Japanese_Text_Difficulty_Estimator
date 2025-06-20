import pandas as pd
from janome.tokenizer import Tokenizer

# Initialize the Janome tokenizer
tokenizer = Tokenizer()

def tokenize_japanese(text, max_length=10000):
    """
    Tokenize Japanese text using Janome.
    - Truncates input if it exceeds `max_length`
    - Returns an empty list if text is invalid
    - Catches and logs tokenizer errors
    """
    if not isinstance(text, str) or text.strip() == "":
        return []

    text = text[:max_length]  # Limit length to avoid Janome overflow

    try:
        return [token.surface for token in tokenizer.tokenize(text)]
    except Exception as e:
        print(f"[ERROR] Tokenization failed for: {text[:30]}... — {e}")
        return []

def apply_tokenization(df):
    """
    Apply the tokenizer on the 'text' column of a DataFrame.
    """
    df = df.copy()
    df['tokens'] = df['text'].apply(tokenize_japanese)
    return df
