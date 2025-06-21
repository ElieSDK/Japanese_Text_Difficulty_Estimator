import streamlit as st
import pickle
import re
from scipy.sparse import hstack, csr_matrix
from janome.tokenizer import Tokenizer

# === Load models and tokenizer ===

# Initialize the Janome tokenizer once
tokenizer = Tokenizer()

# Load the trained pipeline (scaler + logistic regression)
with open("streamlit/logreg_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Load the TF-IDF vectorizer (not part of the pipeline)
with open("streamlit/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# === Preprocessing functions ===

# Remove unwanted characters, keep Japanese scripts and common punctuation
def clean_text(text):
    text = re.sub(r'[^\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9faf\uff66-\uff9fー。、！？a-zA-Z0-9\s]', '', text)
    return text.strip()

# Keep only Japanese characters
def keep_japanese(text):
    return ''.join(re.findall(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9faf\uff66-\uff9fー]', text))

# Count the number of Kanji characters
def count_kanji(text):
    return sum(1 for char in text if '\u4e00' <= char <= '\u9faf')

# Calculate the ratio of Kanji characters over total characters
def count_script_ratio(text):
    total = len(text)
    kanji = count_kanji(text)
    return kanji / total if total > 0 else 0

# Tokenize Japanese text using Janome
def tokenize_japanese(text):
    return [token.surface for token in tokenizer.tokenize(text)]

# Remove short tokens (length <= 1)
def clean_tokens(tokens):
    return [token for token in tokens if len(token) > 1]

# Count part-of-speech tags in the text
def pos_count_from_text(text):
    pos_counts = {}
    for token in tokenizer.tokenize(text):
        pos = token.part_of_speech.split(',')[0]
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    return pos_counts

# === Streamlit App UI ===

# Configure the page
st.set_page_config(page_title="Japanese Text Difficulty Estimator", layout="centered")
st.title("Japanese Text Difficulty Estimator")

# Text input from the user
user_input = st.text_area("Enter a Japanese text (reading, sentence, etc.)", height=200)

# When the button is clicked
if st.button("Guess the level"):
    if not user_input.strip():
        st.warning("Please enter a Japanese text.")
    else:
        # Preprocess and tokenize text
        cleaned = clean_text(user_input)
        only_japanese = keep_japanese(cleaned)
        tokens = clean_tokens(tokenize_japanese(only_japanese))
        joined = ' '.join(tokens)

        # Build the feature dictionary
        features = {
            "tokens_nb": len(tokens),
            "kanji_count": count_kanji(only_japanese),
            "kanji_ratio": count_script_ratio(only_japanese),
            "unique_kanji_count": len(set(re.findall(r'[\u4e00-\u9faf]', only_japanese))),
            "katakana_word_count": len(re.findall(r'[ァ-ンー]{2,}', only_japanese)),
        }

        # Add POS counts
        pos_counts = pos_count_from_text(only_japanese)
        pos_list = ['名詞', '動詞', '形容詞', '副詞', '助詞', '助動詞', '連体詞', '感動詞', '接続詞', '接頭詞', '記号']
        for pos in pos_list:
            features[pos] = pos_counts.get(pos, 0)

        # Maintain the feature order expected by the model
        feature_order = ['tokens_nb', 'kanji_count', 'kanji_ratio', 'unique_kanji_count', 'katakana_word_count'] + pos_list

        # Transform text and numeric features
        X_text = vectorizer.transform([joined])
        X_num = csr_matrix([[features.get(col, 0) for col in feature_order]])
        X_final = hstack([X_text, X_num])

        # Predict JLPT level
        pred = pipeline.predict(X_final)[0]
        proba = pipeline.predict_proba(X_final)[0]

        # Show prediction result
        st.success(f"Predicted JLPT Level: **{pred}**")

        # Show probabilities for all levels
        classes = pipeline.classes_
        proba_dict = dict(zip(classes, proba))

        st.subheader("Probabilities for each level:")
        for jlpt_level in sorted(proba_dict.keys()):
            st.write(f"**{jlpt_level}** : {proba_dict[jlpt_level]:.2%}")
            
# Footer (always visible)
st.markdown("---")
st.markdown("🔗 [View source code on GitHub](https://github.com/ElieSDK/Japanese_Text_Difficulty_Estimator)")
