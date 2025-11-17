import os
import pandas as pd

from pdf2image import convert_from_path
import pytesseract
from config import POPPLER_PATH, TESSERACT_CMD

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def ocr_pdf(pdf_path, poppler_path=POPPLER_PATH):
    try:
        images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    except Exception as e:  # Catch errors such as file not found or conversion errors
        print(f"Problem with the file {pdf_path} : {e}")
        return ""  # Return empty string on error
    
    text = ""
    for image in images: 
        page_text = pytesseract.image_to_string(image, lang='jpn')
        text += page_text + "\n" 
    return text

def main():
    ocr_data = []
    pdf_levels = ["N5", "N4", "N3", "N2", "N1"]
    for level in pdf_levels:
        filename = f"{level}.pdf"
        if not os.path.exists(filename):
            print(f"The file {filename} was not found.")
            continue

        print(f"OCR in progress for : {filename}")
        text = ocr_pdf(filename)
        ocr_data.append({"text": text.strip(), "level": level})

    df = pd.DataFrame(ocr_data)
    df.to_csv("jlpt_dataset_from_pdfs.csv", index=False, encoding="utf-8-sig")
    print("CSV file generated : jlpt_dataset_from_pdfs.csv")

if __name__ == "__main__":
    main()
