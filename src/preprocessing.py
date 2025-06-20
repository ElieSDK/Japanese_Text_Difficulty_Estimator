import pandas as pd

def preprocess_data():
    """
    Load JLPT reading exercise datasets, clean and preprocess the text data,
    split long texts into chunks, and merge datasets into a single DataFrame.
    """
    # Load existing JLPT reading exercises CSV
    df = pd.read_csv('outputs/jlpt_reading_exercises_n1_to_n5.csv', encoding='utf-8')

    # Load OCR extracted JLPT dataset CSV
    df2 = pd.read_csv('outputs/jlpt_dataset_from_pdfs.csv', encoding='utf-8-sig')

    # Remove non-Japanese characters (keep: hiragana, katakana, kanji, punctuation)
    df2['text'] = df2['text'].str.replace(r'[^\u3040-\u30FF\u4E00-\u9FFF\u3000-\u303F]', '', regex=True)

    new_rows = []
    for i, row in df2.iterrows():
        text = row['text']
        level = row['level']
        chunks = [text[j:j+1000] for j in range(0, len(text), 1000)]
        for chunk in chunks:
            new_rows.append({'level': level, 'text': chunk})

    df2_split = pd.DataFrame(new_rows)
    df = pd.concat([df, df2_split], ignore_index=True)

    # Clean text formatting
    df['text'] = df['text'].str.replace('\r', '', regex=True)
    df['text'] = df['text'].str.replace('\n+', '\n', regex=True)
    df['text'] = df['text'].str.strip()

    # Create a 'text_jp' column with only Japanese characters
    df['text_jp'] = df['text'].str.replace(r'[^\u3040-\u30FF\u4E00-\u9FFF\u3000-\u303F]', '', regex=True)

    # Filter out null, empty, or too long texts (Janome bug protection)
    valid_text_mask = df['text'].apply(lambda x: isinstance(x, str) and 0 < len(x) < 4000)
    rejected = df[~valid_text_mask]
    rejected.to_csv('outputs/rejected_texts.csv', index=False)

    df = df[valid_text_mask].copy()

    return df

if __name__ == "__main__":
    df = preprocess_data()
