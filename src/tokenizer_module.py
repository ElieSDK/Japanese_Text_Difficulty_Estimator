import streamlit as st
import pandas as pd
import re
from janome.tokenizer import Tokenizer

def clean_text(text):
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

def tokenize_japanese(text, max_chunk=1000):
    """
    Tokenize un texte japonais en découpant en chunks pour éviter bugs Janome,
    instancie Tokenizer à chaque appel pour éviter problèmes dans Streamlit.
    """
    if not isinstance(text, str):
        return []
    text = clean_text(text.strip())
    if text == "":
        return []

    tokens = []
    tokenizer = Tokenizer()  # Instanciation locale ici
    for i in range(0, len(text), max_chunk):
        chunk = text[i:i+max_chunk]
        try:
            tokens.extend([token.surface for token in tokenizer.tokenize(chunk)])
        except Exception as e:
            st.warning(f"[Tokenization error on chunk] {e} — chunk preview: {chunk[:30]}...")
            continue
    return tokens

def apply_tokenization(df):
    df = df.copy()
    df['text'] = df['text'].fillna('').astype(str)
    df['tokens'] = df['text'].apply(tokenize_japanese)
    return df

# Exemple minimal Streamlit
def main():
    st.title("Test Tokenization Janome")

    uploaded_file = st.file_uploader("Upload CSV with a 'text' column")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data before tokenization:", df.head())

        df_tokenized = apply_tokenization(df)
        st.write("Data after tokenization:", df_tokenized.head())

if __name__ == "__main__":
    main()
