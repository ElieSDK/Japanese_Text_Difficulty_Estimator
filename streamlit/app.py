import streamlit as st
import pickle
import re
from scipy.sparse import hstack, csr_matrix
from pathlib import Path


BASE_PATH = Path(__file__).parent

pipeline_path = BASE_PATH / 'logreg_pipeline.pkl'
vectorizer_path = BASE_PATH / "vectorizer.pkl"

with open(pipeline_path, 'rb') as f:
    pipeline = pickle.load(f)
    
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# === Preprocessing utilities ===

# Clean input text by removing non-Japanese characters and keeping basic punctuation
def clean_text(text):
    text = re.sub(r'[^\u3040-\u30FF\u4E00-\u9FFF\u3000-\u303Fー。、！？\s]', '', text)
    return text.strip()

# Initialize Janome tokenizer
from janome.tokenizer import Tokenizer
tokenizer = Tokenizer()

# Tokenize Japanese text into surface forms
def tokenize_japanese(text):
    return [token.surface for token in tokenizer.tokenize(text)]

# Remove invalid or noisy tokens, keeping only Japanese words
def clean_tokens(tokens):
    return [token.strip().replace('\n', '') for token in tokens if re.fullmatch(r'[ぁ-んァ-ン一-龯ー]+', token)]

# Compute kanji ratio in the text
def count_script_ratio(text):
    kanji = re.findall(r'[\u4e00-\u9FFF]', text)
    hira = re.findall(r'[\u3040-\u309F]', text)
    kata = re.findall(r'[\u30A0-\u30FF]', text)
    total = len(kanji) + len(hira) + len(kata)
    return len(kanji) / total if total else 0

# Count number of unique kanji characters
def count_unique_kanji(text):
    return len(set(re.findall(r'[\u4e00-\u9FFF]', text)))

# Count the number of katakana words (usually loanwords or emphasis)
def count_katakana_words(text):
    return len(re.findall(r'[ァ-ンー]{2,}', text))

# Count part-of-speech tags using Janome
def count_pos(text):
    pos_counts = {}
    for token in tokenizer.tokenize(text):
        pos = token.part_of_speech.split(',')[0]
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    return pos_counts

# === Streamlit UI ===

# Configure Streamlit page
st.set_page_config(page_title="Japanese Text Difficulty Estimator", layout="centered")
st.title("Japanese Text Difficulty Estimator")

# User input area
user_input = st.text_area("Enter a Japanese text (reading, sentence, etc.)", height=200)

# Button to trigger prediction
if st.button("Guess the level"):
    if not user_input.strip():
        st.warning("Please enter a Japanese text.")
    else:
        # Preprocess the input
        cleaned = clean_text(user_input)
        tokens = clean_tokens(tokenize_japanese(cleaned))
        joined = ' '.join(tokens)

        # Extract numerical linguistic features
        features = {
            "tokens_nb": len(tokens),
            "kanji_count": len(re.findall(r'[\u4e00-\u9FFF]', cleaned)),
            "kanji_ratio": count_script_ratio(cleaned),
            "unique_kanji_count": count_unique_kanji(cleaned),
            "katakana_word_count": count_katakana_words(cleaned),
        }

        # Map Japanese POS labels to English feature names
        pos_map = {
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
        }

        # Count POS and update feature dictionary
        pos_counts = count_pos(cleaned)
        for jp, en in pos_map.items():
            features[en] = pos_counts.get(jp, 0)

        # Ensure consistent feature order as during model training
        feature_order = [
            'tokens_nb', 'kanji_count', 'kanji_ratio', 'unique_kanji_count', 'katakana_word_count',
            'noun', 'verb', 'adjective', 'adverb', 'particle', 'auxiliary_verb',
            'adnominal_adjective', 'interjection', 'conjunction', 'prefix', 'symbol'
        ]

        # Prepare final input by stacking text and numeric features
        X_text = vectorizer.transform([joined])
        X_num = csr_matrix([[features.get(f, 0) for f in feature_order]])
        X_final = hstack([X_text, X_num])

        # Predict JLPT level and probability distribution
        pred = pipeline.predict(X_final)[0]
        proba = pipeline.predict_proba(X_final)[0]
        st.success(f"Predicted JLPT Level: **{pred}**")

        # Display probabilities per level
        st.subheader("Probabilities for each level:")
        for level, p in zip(pipeline.classes_, proba):
            st.write(f"**{level}** : {p:.2%}")

# Footer with GitHub link
st.markdown("---")
st.markdown("⚙️ [View source code on GitHub](https://github.com/ElieSDK/Japanese_Text_Difficulty_Estimator)")
