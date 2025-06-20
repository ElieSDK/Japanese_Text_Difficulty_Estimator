import streamlit as st
import joblib

# Load the trained pipeline (e.g., TF-IDF + Logistic Regression)
@st.cache_resource
def load_model():
    return joblib.load("logreg_pipeline.pkl")

model = load_model()

st.title("JLPT Level Predictor")

text = st.text_area("Enter Japanese text below:")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        try:
            prediction = model.predict([text])[0]
            st.success(f"Predicted JLPT Level: {prediction}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
