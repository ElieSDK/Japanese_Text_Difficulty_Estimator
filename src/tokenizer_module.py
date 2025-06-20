import pandas as pd
from janome.tokenizer import Tokenizer

# Initialize tokenizer
tokenizer = Tokenizer()

def tokenize_japanese(text, max_length=8000):
    """
    Tokenize Japanese text.
    - Truncates long text to max_length.
    - Catches tokenization errors.
    """
    if not isinstance(text, str) or text.strip() == "":
        return []
    
    text = text.strip()[:max_length]  # Truncate long text

    try:
        return [token.surface for token in tokenizer.tokenize(text)]
    except Exception as e:
        print(f"[Janome ERROR] Tokenization failed: {e} — text: {text[:30]}...")
        return []

def apply_tokenization(df):
    df = df.copy()
    df['tokens'] = df['text'].apply(tokenize_japanese)
    return df
