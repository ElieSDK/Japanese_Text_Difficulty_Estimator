import pandas as pd
from janome.tokenizer import Tokenizer

# Initialize the Janome Japanese tokenizer
tokenizer = Tokenizer()

def tokenize_japanese(text):
    """
    Tokenize a Japanese text string into a list of token surfaces.
    Returns an empty list if the input text is NaN.
    """
    if pd.isna(text):
        return []
    # Tokenize and return list of token surfaces (actual text parts)
    return [token.surface for token in tokenizer.tokenize(text)]

def apply_tokenization(df):
    """
    Apply Japanese tokenization to the 'text' column of the DataFrame,
    and store the resulting token lists in a new 'tokens' column.
    """
    df['tokens'] = df['text'].apply(tokenize_japanese)
    return df
