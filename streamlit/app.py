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
