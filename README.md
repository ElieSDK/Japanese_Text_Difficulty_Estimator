# 🇯🇵 JLPT Level Classifier

This project processes and classifies Japanese reading texts by JLPT level (N5 to N1) using OCR, NLP, and machine learning.
A minimal Streamlit app is deployed online here for easy text-level prediction:
https://jptextdifficultyestimator.streamlit.app/

---

## Project Structure

```
jlpt-classifier/
├── data/                # Input data (PDFs used for OCR)
├── outputs/             # Generated datasets (CSV) and trained model files (PKL)
├── src/                 # Core Python scripts for scraping, OCR, preprocessing, and training
├── streamlit/           # Streamlit app for deployment (app.py, .pkl models, requirements)
├── app.py               # Full internal pipeline (scraping + OCR + training + prediction)
├── requirements.txt     # Full project dependencies
└── README.md            # Project documentation and usage instructions
```

---

## How to Run

1. Clone the repo or download the code.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the pipeline:

```bash
cd src
python main.py
```
4. To launch the Streamlit app locally:

```bash
cd streamlit
streamlit run app.py
```

---

## Pipeline Overview

1. **Scraping** JLPT reading materials from websites.
2. **OCR** of JLPT practice PDFs.
3. **Preprocessing** and cleaning of Japanese texts.
4. **Tokenization** using Janome or Mecab.
5. **Feature engineering**: counts of kanji, POS tags, etc.
6. **Vectorization** using TF-IDF + numeric features.
7. **Model training**: Logistic Regression.
8. **Prediction** exposed via the Streamlit web app.

---

## Dependencies

- `pandas`, `numpy`
- `mecab` (Japanese tokenizer)
- `pdf2image`, `pytesseract` (OCR)
- `scikit-learn`
- `selenium`

See [`requirements.txt`](./requirements.txt) for details.

---

## Output Files

- `vectorizer.pkl`: TF-IDF vectorizer
- `logreg_pipeline.pkl`: trained classifier
- `jlpt_dataset_from_pdfs.csv`: processed dataset
- `jlpt_reading_exercises_n1_to_n5.csv`: processed dataset

---

## Notes

- Works on Windows with [Poppler](http://blog.alivate.com.au/poppler-windows/) and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
- Requires ChromeDriver for scraping.
