import os
import re
from typing import Optional
import pdfplumber
import docx

def extract_pdf_text(path: str) -> str:
    """
    Extracts text from a PDF file using pdfplumber.
    """
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text

def extract_docx_text(path: str) -> str:
    """
    Extracts text from a DOCX file using python-docx.
    """
    text = ""
    try:
        doc = docx.Document(path)
        for para in doc.paragraphs:
            text += para.text + " "
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
    return text

def clean_text(text: str) -> str:
    """
    Refines raw text into clean natural language for NLP/Embeddings.
    Removes URLs, long numbers, and unusual characters while preserving readability.
    """
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Replace newlines and tabs with spaces
    text = re.sub(r'[\t\n\r]', ' ', text)

    # 3. Remove URLs (http, https, www)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 4. Remove email addresses (often messy for embeddings)
    text = re.sub(r'\S+@\S+', '', text)

    # 5. Remove phone numbers and sequences of 7+ digits
    # Matches common phone patterns and long digit strings
    text = re.sub(r'\+?\d[\d\-\(\) ]{7,}\d', '', text)
    text = re.sub(r'\b\d{7,}\b', '', text)

    # 6. Remove special characters except basic punctuation (. , -)
    # This keeps sentence structure and hyphens (like "hand-on")
    text = re.sub(r'[^a-z0-9\s\.\,\-]', ' ', text)

    # 7. Normalize whitespace (single spaces only)
    text = re.sub(r'\s+', ' ', text).strip()

    # 8. Final polish: Ensure spaces after punctuation for readability
    text = re.sub(r'([.,])([^\s])', r'\1 \2', text)
    
    return text

def parse_resume(path: str) -> str:
    """
    High-level entry point to parse and clean a resume.
    """
    if not os.path.exists(path):
        return f"Error: File not found at {path}"
    
    ext = os.path.splitext(path)[1].lower()
    
    raw_text = ""
    if ext == ".pdf":
        raw_text = extract_pdf_text(path)
    elif ext == ".docx":
        raw_text = extract_docx_text(path)
    else:
        return f"Error: Unsupported file format {ext}"
    
    if not raw_text.strip():
        return "Error: No text extracted from file."
        
    return clean_text(raw_text)

if __name__ == "__main__":
    # Test block with the local file path
    print(parse_resume(r"E:\resume matcher\resume (4).pdf"))
