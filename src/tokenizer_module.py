import pandas as pd
import MeCab

# Initialize MeCab tokenizer in wakati mode (word segmentation)
tagger = MeCab.Tagger("-Owakati")

def tokenize_japanese(text):
    """
    Tokenize a Japanese text string into a list of token surfaces using MeCab.
    Returns an empty list if the input text is NaN.
    """
    if pd.isna(text):
        return []
    return tagger.parse(text).strip().split()

def apply_tokenization(df):
    """
    Apply Japanese tokenization to the 'text' column of the DataFrame,
    and store the resulting token lists in a new 'tokens' column.
    """
    df['tokens'] = df['text'].apply(tokenize_japanese)
    return df