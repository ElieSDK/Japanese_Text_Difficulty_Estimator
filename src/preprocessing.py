import pandas as pd

def preprocess_data():
    # Load existing JLPT reading exercises CSV
    df = pd.read_csv('jlpt_reading_exercises_n1_to_n5.csv', encoding='utf-8')
    # Load OCR extracted JLPT dataset CSV
    df2 = pd.read_csv('jlpt_dataset_from_pdfs.csv', encoding='utf-8-sig')

    # Remove any characters that are not Japanese scripts (hiragana, katakana, kanji, punctuation)
    df2['text'] = df2['text'].str.replace(r'[^\u3040-\u30FF\u4E00-\u9FFF\u3000-\u303F]', '', regex=True)

    new_rows = []    
    # Loop over both 'text' and 'level' columns at the same time
    for text, level in zip(df2['text'], df2['level']):
        chunks = [text[j:j+1000] for j in range(0, len(text), 1000)] # Split the text into chunks of 1000 characters each
        for chunk in chunks: # Loop through each text chunk
            new_rows.append({'level': level, 'text': chunk})  # Add each chunk as a dictionary to the new_rows list, keeping the original level
            
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

    return df

if __name__ == "__main__":
    # Run preprocessing if script is executed directly
    df = preprocess_data()
