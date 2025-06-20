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

    # Remove any characters that are not Japanese scripts (hiragana, katakana, kanji, punctuation)
    df2['text'] = df2['text'].str.replace(r'[^\u3040-\u30FF\u4E00-\u9FFF\u3000-\u303F]', '', regex=True)

    new_rows = []
    # Iterate over each row to split long texts into smaller chunks of 1000 characters max
    for i, row in df2.iterrows():
        text = row['text']
        level = row['level']
        # Split text into chunks of length 1000 characters
        chunks = [text[j:j+1000] for j in range(0, len(text), 1000)]
        # Create a new row for each chunk with the same level
        for chunk in chunks:
            new_rows.append({'level': level, 'text': chunk})

    # Create a new DataFrame from the split chunks
    df2_split = pd.DataFrame(new_rows)

    # Concatenate the original df with the split OCR data
    df = pd.concat([df, df2_split], ignore_index=True)

    # Clean text: remove carriage returns
    df['text'] = df['text'].str.replace('\r', '', regex=True)
    # Replace multiple newlines with a single newline
    df['text'] = df['text'].str.replace('\n+', '\n', regex=True)
    # Strip whitespace from start and end of text
    df['text'] = df['text'].str.strip()
    # Create a new column 'text_jp' containing only Japanese characters and punctuation
    df['text_jp'] = df['text'].str.replace(r'[^\u3040-\u30FF\u4E00-\u9FFF\u3000-\u303F]', '', regex=True)

    # Optional filtering: drop rows with very long text to prevent tokenization errors
    df = df[df['text'].apply(lambda x: isinstance(x, str) and len(x) < 5000)]

    return df

if __name__ == "__main__":
    # Run preprocessing if script is executed directly
    df = preprocess_data()
