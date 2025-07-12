import re
import os
import glob
from typing import List
import PyPDF2

import spacy
from nltk.corpus import stopwords

# Ensure NLTK stopwords are downloaded
try:
    _ = stopwords.words('english')
except LookupError:
    import nltk
    nltk.download('stopwords')

nlp = spacy.load('en_core_web_sm')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Lowercase, remove non-alphanumeric, strip, remove stopwords, lemmatize."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in STOPWORDS and not token.is_space]
    return " ".join(tokens)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
            return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def load_documents(raw_dir: str) -> List[str]:
    """Load all .txt and .pdf files from a directory as a list of strings."""
    docs = []
    
    # Load text files
    txt_files = glob.glob(os.path.join(raw_dir, '*.txt'))
    for file in txt_files:
        with open(file, 'r', encoding='utf-8') as f:
            docs.append(f.read())
    
    # Load PDF files
    pdf_files = glob.glob(os.path.join(raw_dir, '*.pdf'))
    for file in pdf_files:
        text = extract_text_from_pdf(file)
        if text:
            docs.append(text)
    
    return docs
