import pandas as pd
from janome.tokenizer import Tokenizer

# Initialize the Janome Japanese tokenizer
tokenizer = Tokenizer()

def tokenize_japanese(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return []
    try:
        return [token.surface for token in tokenizer.tokenize(text)]
    except Exception as e:
        print(f"Tokenization error for text: {text[:30]}... Error: {e}")
        return []

def apply_tokenization(df):
    """
    Apply Japanese tokenization to the 'text' column of the DataFrame,
    and store the resulting token lists in a new 'tokens' column.
    """
    df['tokens'] = df['text'].apply(tokenize_japanese)
    return df
