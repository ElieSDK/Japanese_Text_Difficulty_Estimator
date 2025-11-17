from scraper import main as scraper_main
from ocr import main as ocr_main
from preprocessing import preprocess_data
from tokenizer_module import apply_tokenization

from features import extract_features
from vectorize import vectorize_text
from train import train_model

def main():
    scraper_main()
    ocr_main()
    df = preprocess_data()
    df = apply_tokenization(df)
    df = extract_features(df)
    X, y = vectorize_text(df)
    train_model(X, y)
    
if __name__ == "__main__":
    main()

