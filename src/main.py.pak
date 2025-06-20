from scraper import main as scraper_main
from ocr import main as ocr_main
from preprocessing import preprocess_data
from tokenizer_module import apply_tokenization

from features import extract_features
from vectorize import vectorize_text
from train import train_model

def main():
    # Run the web scraper to gather raw data
    scraper_main()
    # Run OCR to extract text from scanned documents or images
    ocr_main()
    # Preprocess the raw data (cleaning, formatting, etc.)
    df = preprocess_data()
    # Apply tokenization on the text data to split it into tokens
    df = apply_tokenization(df)
    # Extract linguistic and statistical features from the tokenized data
    df = extract_features(df)
    # Convert text and features into numerical vectors and get target labels
    X, y = vectorize_text(df)
    # Train the machine learning model using the feature vectors and labels
    train_model(X, y)
    
if __name__ == "__main__":
    main()
