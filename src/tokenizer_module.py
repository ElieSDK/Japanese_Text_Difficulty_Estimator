import pandas as pd
from janome.tokenizer import Tokenizer

# Initialize Janome tokenizer
janome_tokenizer = Tokenizer()

def tokenize_japanese(text):
    if pd.isna(text):
        return []
    return [token.surface for token in janome_tokenizer.tokenize(text)]

def apply_tokenization(df):
    df['tokens'] = df['text'].apply(tokenize_japanese)
    return df
