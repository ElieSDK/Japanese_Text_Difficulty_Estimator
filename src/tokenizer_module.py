import pandas as pd
from janome.tokenizer import Tokenizer

# Initialize Janome tokenizer
tokenizer = Tokenizer()

def tokenize_japanese(text, max_length=5000):
    """
    Tokenize a Japanese string into a list of token surfaces.
    Handles empty, long, or malformed input safely.
    """
    if not isinstance(text, str):
        return []
    
    text = text.strip()
    if text == "":
        return []
    
    # Truncate long input to avoid Janome crashing
    if len(text) > max_length:
        text = text[:max_length]

    try:
        tokens = tokenizer.tokenize(text)
        return [token.surface for token in tokens]
    except Exception as e:
        print(f"[Tokenization Error] {e} — text was: {text[:30]}...")
        return []

def apply_tokenization(df):
    df = df.copy()
    df['tokens'] = df['text'].apply(tokenize_japanese)
    return df
