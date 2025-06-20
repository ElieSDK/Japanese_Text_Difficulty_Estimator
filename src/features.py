import re
from collections import Counter

import pandas as pd
from janome.tokenizer import Tokenizer

# Initialize the Japanese tokenizer from Janome
tokenizer = Tokenizer()

def clean_tokens(token_list):
    """
    Clean the list of tokens by stripping whitespace, removing newlines,
    and keeping only tokens that consist of Japanese characters
    (hiragana, katakana, kanji, and the prolonged sound mark).
    """
    cleaned = []
    for token in token_list:
        token = token.strip().replace('\n', '')
        # Keep token only if it matches Japanese script characters
        if token and re.fullmatch(r'[ぁ-んァ-ン一-龯ー]+', token):
            cleaned.append(token)
    return cleaned

def count_script_ratio(text):
    """
    Calculate the ratio of kanji characters in the text compared to the total
    number of Japanese script characters (kanji + hiragana + katakana).
    Returns 0 if text contains no Japanese characters.
    """
    kanji = re.findall(r'[\u4e00-\u9FFF]', text)
    hiragana = re.findall(r'[\u3040-\u309F]', text)
    katakana = re.findall(r'[\u30A0-\u30FF]', text)
    total = len(kanji) + len(hiragana) + len(katakana)
    return len(kanji) / total if total else 0

def pos_count_from_text(text):
    """
    Tokenize the text and count the occurrences of each part of speech (POS).
    Returns a dictionary with POS tags as keys and their counts as values.
    """
    tokens = tokenizer.tokenize(text)
    # Extract coarse POS (first part before comma)
    pos_list = [token.part_of_speech.split(',')[0] for token in tokens]
    pos_counts = Counter(pos_list)
    return dict(pos_counts)

def count_unique_kanji(text):
    """
    Count the number of unique kanji characters in the text.
    """
    kanji_list = re.findall(r'[\u4e00-\u9faf]', text)
    return len(set(kanji_list))

def count_katakana_words(text):
    """
    Count the number of katakana words in the text.
    A katakana word is defined here as a sequence of two or more katakana characters.
    """
    katakana_words = re.findall(r'[ァ-ンー]{2,}', text)
    return len(katakana_words)

def extract_features(df):
    """
    Given a DataFrame with columns 'text' and 'tokens', extract linguistic features:
    - Clean tokens
    - Count tokens
    - Count kanji characters
    - Calculate kanji ratio
    - Count parts of speech occurrences
    - Count unique kanji
    - Count katakana words
    Then remove unwanted columns and return the enriched DataFrame.
    """
    # Clean tokens in the DataFrame
    df['tokens'] = df['tokens'].apply(clean_tokens)

    # Count number of tokens per row
    df['tokens_nb'] = df['tokens'].apply(len)
    # Replace missing text with empty string
    df['text'] = df['text'].fillna('')
    # Count kanji characters in text
    df['kanji_count'] = df['text'].apply(re.compile(r'[\u4e00-\u9faf]').findall).apply(len)
    # Calculate ratio of kanji among all Japanese scripts
    df['kanji_ratio'] = df['text'].apply(count_script_ratio)
    # Get POS counts as a dictionary for each text
    df['pos_counts'] = df['text'].apply(pos_count_from_text)

    # Convert POS count dictionaries into separate columns with zeros for missing POS
    df_pos = df['pos_counts'].apply(pd.Series).fillna(0).astype(int)
    # Merge POS count columns back into the main DataFrame and drop the dict column
    df = pd.concat([df, df_pos], axis=1).drop(columns=['pos_counts'])

    # Rename Japanese POS columns to English names
    df.rename(columns={
        '名詞': 'noun',
        '動詞': 'verb',
        '形容詞': 'adjective',
        '副詞': 'adverb',
        '助詞': 'particle',
        '助動詞': 'auxiliary_verb',
        '連体詞': 'adnominal_adjective',
        '感動詞': 'interjection',
        '接続詞': 'conjunction',
        '接頭詞': 'prefix',
        '記号': 'symbol',
        'フィラー': 'filler',
        'その他': 'other'
    }, inplace=True)

    # Count unique kanji characters and katakana words
    df["unique_kanji_count"] = df["text"].apply(count_unique_kanji)
    df["katakana_word_count"] = df["text"].apply(count_katakana_words)

    # Drop unnecessary columns, ignoring errors if columns do not exist
    df = df.drop(columns=['filler', 'other', 'url', 'text'], errors='ignore')
    
    return df
