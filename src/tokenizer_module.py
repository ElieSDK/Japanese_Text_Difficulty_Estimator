import re
import pandas as pd
from janome.tokenizer import Tokenizer
import streamlit as st

def clean_text(text):
    """Supprime caractères non imprimables et contrôle."""
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

def tokenize_japanese(text, max_chunk=1000):
    """
    Tokenize un texte japonais en morceaux pour éviter les erreurs Janome.
    Instancie un nouveau tokenizer à chaque appel.
    """
    if not isinstance(text, str):
        return []
    text = clean_text(text.strip())
    if not text:
        return []

    tokens = []
    tokenizer = Tokenizer()  # Création locale pour éviter état partagé
    for i in range(0, len(text), max_chunk):
        chunk = text[i:i+max_chunk]
        try:
            tokens.extend([token.surface for token in tokenizer.tokenize(chunk)])
        except Exception as e:
            st.warning(f"Erreur tokenization chunk: {e} (preview: {chunk[:30]})")
            continue
    return tokens

def apply_tokenization(df):
    df = df.copy()
    df['text'] = df['text'].fillna('').astype(str)
    df['tokens'] = df['text'].apply(tokenize_japanese)
    return df

def main():
    st.title("Test tokenization japonaise")
    uploaded_file = st.file_uploader("Fichier CSV avec colonne 'text'")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Avant tokenization :", df.head())
        df_tokenized = apply_tokenization(df)
        st.write("Après tokenization :", df_tokenized.head())

if __name__ == "__main__":
    main()
