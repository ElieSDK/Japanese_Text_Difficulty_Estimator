import pandas as pd

def preprocess_data():
    df = pd.read_csv('jlpt_reading_exercises_n1_to_n5.csv', encoding='utf-8')
    df2 = pd.read_csv('jlpt_dataset_from_pdfs.csv', encoding='utf-8-sig')
    df2['text'] = df2['text'].str.replace(r'[^\u3040-\u30FF\u4E00-\u9FFF\u3000-\u303F]', '', regex=True)

    new_rows = []    
    for text, level in zip(df2['text'], df2['level']):
        chunks = [text[j:j+1000] for j in range(0, len(text), 1000)] # Split the text into chunks of 1000 characters each
        for chunk in chunks: # Loop through each text chunk
            new_rows.append({'level': level, 'text': chunk})
    
    df2_split = pd.DataFrame(new_rows)
    df = pd.concat([df, df2_split], ignore_index=True)
    df['text'] = df['text'].str.replace('\r', '', regex=True)
    df['text'] = df['text'].str.replace('\n+', '\n', regex=True)
    df['text'] = df['text'].str.strip()
    df['text_jp'] = df['text'].str.replace(r'[^\u3040-\u30FF\u4E00-\u9FFF\u3000-\u303F]', '', regex=True)
    return df

if __name__ == "__main__":
    df = preprocess_data()
