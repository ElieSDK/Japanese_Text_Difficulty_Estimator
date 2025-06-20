import streamlit as st
import pickle
import re
from scipy.sparse import hstack, csr_matrix
import MeCab

# Load the full pipeline (vectorizer, scaler, model) from a pickle file
with open("logreg_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Function to clean text by removing unwanted characters except Japanese scripts and common punctuation
def clean_text(text):
    text = re.sub(r'[^\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9faf\uff66-\uff9fー。、！？a-zA-Z0-9\s]', '', text)
    return text.strip()

# Function to keep only Japanese characters from the text
def keep_japanese(text):
    return ''.join(re.findall(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9faf\uff66-\uff9fー]', text))

# Count the number of Kanji characters in the text
def count_kanji(text):
    return sum(1 for char in text if '\u4e00' <= char <= '\u9faf')

# Calculate the ratio of Kanji characters over total characters in the text
def count_script_ratio(text):
    total = len(text)
    kanji = count_kanji(text)
    return kanji / total if total > 0 else 0

# Tokenize the Japanese text into words using MeCab tokenizer in "wakati" mode (word segmentation)
def tokenize_japanese(text):
    mecab = MeCab.Tagger("-Owakati")
    return mecab.parse(text).strip().split()

# Filter out tokens that are too short (length <= 1)
def clean_tokens(tokens):
    return [token for token in tokens if len(token) > 1]

# Count parts-of-speech (POS) tags in the text using MeCab
def pos_count_from_text(text):
    tagger = MeCab.Tagger()
    tagger.parse('')  # Initialize MeCab parser
    node = tagger.parseToNode(text)
    pos_counts = {}
    while node:
        features = node.feature.split(',')
        if features:
            pos = features[0]  # Get POS tag
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        node = node.next
    return pos_counts

# Streamlit UI setup: set page title and layout
st.set_page_config(page_title="Japanese Text Difficulty Estimator", layout="centered")
st.title("Japanese Text Difficulty Estimator")

# Text input area for the user to enter Japanese text
user_input = st.text_area("Enter a Japanese text (reading, sentence, etc.)", height=200)

# When the user clicks the button to predict JLPT level
if st.button("Guess the level"):
    if not user_input.strip():
        # Show warning if input is empty
        st.warning("Please enter a Japanese text.")
    else:
        # Preprocess input text
        cleaned = clean_text(user_input)
        only_japanese = keep_japanese(cleaned)
        tokens = clean_tokens(tokenize_japanese(only_japanese))
        joined = ' '.join(tokens)  # Join tokens into a single string separated by spaces

        # Build numerical features dictionary in the same order as during model training
        features = {
            "tokens_nb": len(tokens),
            "kanji_count": count_kanji(only_japanese),
            "kanji_ratio": count_script_ratio(only_japanese),
            "unique_kanji_count": len(set(re.findall(r'[\u4e00-\u9faf]', only_japanese))),
            "katakana_word_count": len(re.findall(r'[ァ-ンー]{2,}', only_japanese)),
        }
        # Get POS counts from the text
        pos_counts = pos_count_from_text(only_japanese)
        pos_list = ['名詞', '動詞', '形容詞', '副詞', '助詞', '助動詞', '連体詞', '感動詞', '接続詞', '接頭詞', '記号']
        for pos in pos_list:
            features[pos] = pos_counts.get(pos, 0)

        # Define the order of features used in the model
        feature_order = ['tokens_nb', 'kanji_count', 'kanji_ratio', 'unique_kanji_count', 'katakana_word_count'] + pos_list

        # Load the TF-IDF vectorizer separately from file (not included in pipeline)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        # Transform the input text into TF-IDF vector
        X_text = vectorizer.transform([joined])
        # Create a sparse matrix for numerical features
        X_num = csr_matrix([[features.get(col, 0) for col in feature_order]])
        # Combine text and numerical features horizontally
        X_final = hstack([X_text, X_num])

        # Predict JLPT level using the loaded pipeline (scaler + logistic regression)
        pred = pipeline.predict(X_final)[0]
        # Predict probabilities for each JLPT level
        proba = pipeline.predict_proba(X_final)[0]

        # Display the predicted JLPT level to the user
        st.success(f"Predicted JLPT Level: **{pred}**")

        # Map classes to their probabilities
        classes = pipeline.classes_
        proba_dict = dict(zip(classes, proba))

        # Show prediction probabilities for all levels
        st.subheader("Probabilities for each level:")
        for jlpt_level in sorted(proba_dict.keys()):
            st.write(f"**{jlpt_level}** : {proba_dict[jlpt_level]:.2%}")
