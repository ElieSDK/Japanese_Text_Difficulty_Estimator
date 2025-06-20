import os
import sys
import joblib
import streamlit as st
from pathlib import Path

@st.cache_resource
def load_model():
    try:
        current_dir = Path(__file__).parent
    except NameError:
        current_dir = Path(sys.argv[0]).parent.resolve()
    model_path = current_dir / "logreg_pipeline.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    return model

model = load_model()

# === Streamlit UI ===
st.title("JLPT Level Predictor")

user_input = st.text_area("Enter Japanese text to analyze:")

if st.button("Predict Level"):
    if user_input.strip():
        try:
            prediction = model.predict([user_input])[0]
            st.success(f"Predicted JLPT Level: N{prediction}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("Please enter some text.")
