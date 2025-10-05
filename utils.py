import os
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
import re

def extract_text_from_pdf(pdf_path):
    try:
        text = extract_pdf_text(pdf_path)
        # Clean the extracted text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with single space
        text = text.strip()
        return text
    except Exception as e:
        print(f"Error extracting PDF {pdf_path}: {e}")
        return ""

def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
        text = re.sub(r'\s+', ' ', text)  # Clean the text
        text = text.strip()
        return text
    except Exception as e:
        print(f"Error extracting DOCX {docx_path}: {e}")
        return ""

def extract_resume_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    print(f"DEBUG: Extracting from {file_path}, type: {ext}")
    
    if ext == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif ext in ['.docx', '.doc']:
        text = extract_text_from_docx(file_path)
    elif ext == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()
        except:
            text = ""
    else:
        print(f"Unsupported file type: {file_path}")
        text = ""
    
    print(f"DEBUG: Extracted {len(text)} characters from {file_path}")
    if len(text) < 50:  # If text is too short, there might be an issue
        print(f"DEBUG: WARNING - Very short text extracted from {file_path}")
    
    return text