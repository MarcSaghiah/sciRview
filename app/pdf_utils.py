"""
Module for downloading a PDF from a URL and extracting its text (PyMuPDF).
"""
import os
import requests
import fitz  # PyMuPDF
from typing import Optional, List

def download_pdf(url: str, dest_folder: str = "data/pdfs") -> Optional[str]:
    """
    Download a PDF from a URL and save it to dest_folder.
    Returns the file path, or None on failure.
    """
    os.makedirs(dest_folder, exist_ok=True)
    filename = url.split("/")[-1]
    if not filename.endswith(".pdf"):
        filename += ".pdf"
    local_filename = os.path.join(dest_folder, filename)
    try:
        r = requests.get(url, stream=True, timeout=20)
        if r.status_code == 200 and r.headers.get('content-type','').startswith('application/pdf'):
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return local_filename
        else:
            return None
    except Exception:
        return None

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """
    Split text into chunks of chunk_size words with overlap.
    Returns a list of strings.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def extract_and_chunk_pdf(url: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """
    Download a PDF, extract its text, and return chunks.
    Returns an empty list if the download or extraction fails.
    """
    path = download_pdf(url)
    if not path:
        return []
    text = extract_text_from_pdf(path)
    if not text.strip():
        return []
    return chunk_text(text, chunk_size, overlap)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF using PyMuPDF.
    """
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception:
        return ""
    return text
