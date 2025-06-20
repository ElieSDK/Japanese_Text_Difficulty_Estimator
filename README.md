# 🇯🇵 JLPT Level Classifier

This project processes and classifies Japanese reading texts by JLPT level (N5 to N1) using OCR, NLP, and machine learning.

---

## Project Structure

```
jlpt-classifier/
├── data/                # Input data (PDFs and CSV)
├── outputs/             # Generated datasets and models
├── src/                 # Python source files
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
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

---

## Pipeline Overview

1. **Scraping** JLPT reading materials from websites.
2. **OCR** of JLPT practice PDFs.
3. **Preprocessing** and cleaning of Japanese texts.
4. **Tokenization** using Janome.
5. **Feature engineering**: counts of kanji, POS tags, etc.
6. **Vectorization** using TF-IDF + numeric features.
7. **Model training**: Logistic Regression.

---

## Dependencies

- `pandas`, `numpy`
- `janome` (Japanese tokenizer)
- `pdf2image`, `pytesseract` (OCR)
- `scikit-learn`
- `selenium`

See [`requirements.txt`](./requirements.txt) for details.

---

## Output Files

- `vectorizer.pkl`: TF-IDF vectorizer
- `logreg_pipeline.pkl`: trained classifier
- `jlpt_dataset_from_pdfs.csv`: processed dataset

---

## Notes

- Works on Windows with [Poppler](http://blog.alivate.com.au/poppler-windows/) and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
- Requires ChromeDriver for scraping.
