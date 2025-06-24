import streamlit as st
import pickle
import re
from scipy.sparse import hstack, csr_matrix

# === Load models and vectorizer ===

# Load the trained pipeline (includes scaling and logistic regression)
with open("logreg_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Load the TF-IDF vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# === Preprocessing utilities ===

# Keep only Japanese characters and some punctuation
def clean_text(text):
    text = re.sub(r'[^\u3040-\u30FF\u4E00-\u9FFF\u3000-\u303Fー。、！？\s]', '', text)
    return text.strip()

# Initialize Janome tokenizer
from janome.tokenizer import Tokenizer
tokenizer = Tokenizer()

# Tokenize Japanese text into surface forms
def tokenize_japanese(text):
    return [token.surface for token in tokenizer.tokenize(text)]

# Remove noise from tokens and keep only valid Japanese words
def clean_tokens(tokens):
    return [token.strip().replace('\n', '') for token in tokens if re.fullmatch(r'[ぁ-んァ-ン一-龯ー]+', token)]

# Compute the ratio of kanji characters in the text
def count_script_ratio(text):
    kanji = re.findall(r'[\u4e00-\u9FFF]', text)
    hira = re.findall(r'[\u3040-\u309F]', text)
    kata = re.findall(r'[\u30A0-\u30FF]', text)
    total = len(kanji) + len(hira) + len(kata)
    return len(kanji) / total if total else 0

# Count unique kanji characters
def count_unique_kanji(text):
    return len(set(re.findall(r'[\u4e00-\u9FFF]', text)))

# Count words written in katakana (typically loanwords)
def count_katakana_words(text):
    return len(re.findall(r'[ァ-ンー]{2,}', text))

# Count parts of speech using Janome
def count_pos(text):
    pos_counts = {}
    for token in tokenizer.tokenize(text):
        pos = token.part_of_speech.split(',')[0]
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    return pos_counts

# === Streamlit UI ===

st.set_page_config(page_title="Japanese Text Difficulty Estimator", layout="centered")
st.title("Japanese Text Difficulty Estimator")

# Input area for user to enter Japanese text
user_input = st.text_area("Enter a Japanese text (reading, sentence, etc.)", height=200)

# Button to trigger prediction
if st.button("Guess the level"):
    if not user_input.strip():
        st.warning("Please enter a Japanese text.")
    else:
        # Preprocess input
        cleaned = clean_text(user_input)
        tokens = clean_tokens(tokenize_japanese(cleaned))
        joined = ' '.join(tokens)

        # Extract handcrafted linguistic features
        features = {
            "tokens_nb": len(tokens),
            "kanji_count": len(re.findall(r'[\u4e00-\u9FFF]', cleaned)),
            "kanji_ratio": count_script_ratio(cleaned),
            "unique_kanji_count": count_unique_kanji(cleaned),
            "katakana_word_count": count_katakana_words(cleaned),
        }

        # Map Japanese POS tags to English feature names
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

        # Count POS and add to feature dict
        pos_counts = count_pos(cleaned)
        for jp, en in pos_map.items():
            features[en] = pos_counts.get(jp, 0)

        # Reorder features to match training order
        feature_order = [
            'tokens_nb', 'kanji_count', 'kanji_ratio', 'unique_kanji_count', 'katakana_word_count',
            'noun', 'verb', 'adjective', 'adverb', 'particle', 'auxiliary_verb',
            'adnominal_adjective', 'interjection', 'conjunction', 'prefix', 'symbol'
        ]

        # Vectorize text and stack with numeric features
        X_text = vectorizer.transform([joined])
        X_num = csr_matrix([[features.get(f, 0) for f in feature_order]])
        X_final = hstack([X_text, X_num])

        # Make prediction
        pred = pipeline.predict(X_final)[0]
        proba = pipeline.predict_proba(X_final)[0]

        # Display results
        st.success(f"Predicted JLPT Level: **{pred}**")

        st.subheader("Probabilities for each level:")
        for level, p in zip(pipeline.classes_, proba):
            st.write(f"**{level}** : {p:.2%}")
