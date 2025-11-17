import re
from collections import Counter

import pandas as pd
from janome.tokenizer import Tokenizer
from sklearn.preprocessing import MinMaxScaler

# Initialize the Japanese tokenizer from Janome
tokenizer = Tokenizer()

def clean_tokens(token_list):
    cleaned = []
    for token in token_list:
        token = token.strip().replace('\n', '')
        # Keep token only if it matches Japanese script characters
        if token and re.fullmatch(r'[ぁ-んァ-ン一-龯ー]+', token):
            cleaned.append(token)
    return cleaned

def count_script_ratio(text):
    kanji = re.findall(r'[\u4e00-\u9FFF]', text)
    hiragana = re.findall(r'[\u3040-\u309F]', text)
    katakana = re.findall(r'[\u30A0-\u30FF]', text)
    total = len(kanji) + len(hiragana) + len(katakana)
    return len(kanji) / total if total else 0

def pos_count_from_text(text):
    tokens = tokenizer.tokenize(text)
    # Extract coarse POS (first part before comma)
    pos_list = [token.part_of_speech.split(',')[0] for token in tokens]
    pos_counts = Counter(pos_list)
    return dict(pos_counts)

def count_unique_kanji(text):
    kanji_list = re.findall(r'[\u4e00-\u9faf]', text)
    return len(set(kanji_list))

def count_katakana_words(text):
    katakana_words = re.findall(r'[ァ-ンー]{2,}', text)
    return len(katakana_words)

def extract_features(df):
    df['tokens'] = df['tokens'].apply(clean_tokens)
    df['tokens_nb'] = df['tokens'].apply(len)
    df['text'] = df['text'].fillna('')
    df['kanji_count'] = df['text'].apply(re.compile(r'[\u4e00-\u9faf]').findall).apply(len)
    df['kanji_ratio'] = df['text'].apply(count_script_ratio)
    df['pos_counts'] = df['text'].apply(pos_count_from_text)
    df_pos = df['pos_counts'].apply(pd.Series).fillna(0).astype(int)
    df = pd.concat([df, df_pos], axis=1).drop(columns=['pos_counts'])

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

    df["unique_kanji_count"] = df["text"].apply(count_unique_kanji)
    df["katakana_word_count"] = df["text"].apply(count_katakana_words)

    df = df.drop(columns=['filler', 'other', 'url', 'text'], errors='ignore')

    cols_to_normalize = [
        'tokens_nb', 'kanji_count', 'unique_kanji_count', 'katakana_word_count',
        'noun', 'verb', 'adjective', 'adverb', 'particle', 'auxiliary_verb',
        'adnominal_adjective', 'interjection', 'conjunction', 'prefix', 'symbol'
    ]

    existing_cols = [col for col in cols_to_normalize if col in df.columns]
    scaler = MinMaxScaler()
    df[existing_cols] = scaler.fit_transform(df[existing_cols])
    return df

