import os
import sys
import joblib
import streamlit as st
from pathlib import Path

@st.cache_resource
def load_model():
    # Get the path of the running script
    try:
        current_dir = Path(__file__).parent
    except NameError:
        # Fallback if __file__ is not defined (e.g. in Streamlit)
        current_dir = Path(sys.argv[0]).parent.resolve()
    
    model_path = current_dir / "logreg_pipeline.pkl"
    return joblib.load(model_path)

model = load_model()
