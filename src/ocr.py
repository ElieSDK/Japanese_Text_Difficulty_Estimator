import os
import pandas as pd

from pdf2image import convert_from_path
import pytesseract
from config import POPPLER_PATH, TESSERACT_CMD

# Set the tesseract executable path from config
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def ocr_pdf(pdf_path, poppler_path=POPPLER_PATH):
    try:
        # Convert PDF pages to images at 300 dpi using poppler
        images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    except Exception as e:  # Catch errors such as file not found or conversion errors
        print(f"Problem with the file {pdf_path} : {e}")
        return ""  # Return empty string on error
    
    text = ""
    # Loop over each image (i.e., each page of the PDF)
    for image in images: 
        # Use Tesseract to extract Japanese text from the image
        page_text = pytesseract.image_to_string(image, lang='jpn')
        # Add the extracted text to the total, with a newline between pages
        text += page_text + "\n" 
    return text

def main():
    ocr_data = []
    # List of JLPT levels corresponding to PDF filenames
    pdf_levels = ["N5", "N4", "N3", "N2", "N1"]

    # Process each PDF file by JLPT level
    for level in pdf_levels:
        filename = f"{level}.pdf"
        if not os.path.exists(filename):
            print(f"The file {filename} was not found.")
            continue

        print(f"OCR in progress for : {filename}")
        # Extract text from PDF using OCR
        text = ocr_pdf(filename)
        # Store the text along with its JLPT level
        ocr_data.append({"text": text.strip(), "level": level})

    # Convert list of dicts to DataFrame and save as CSV
    df = pd.DataFrame(ocr_data)
    df.to_csv("jlpt_dataset_from_pdfs.csv", index=False, encoding="utf-8-sig")
    print("CSV file generated : jlpt_dataset_from_pdfs.csv")

if __name__ == "__main__":
    main()
