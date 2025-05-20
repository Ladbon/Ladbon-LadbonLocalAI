import os
from typing import Optional

def process_document(file_path: str) -> str:
    """Extract text from a document file"""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return _extract_pdf(file_path)
    elif ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']:
        return _extract_text(file_path)
    elif ext in ['.docx', '.doc']:
        return _extract_word(file_path)
    else:
        return f"Unsupported file format: {ext}"

def _extract_pdf(file_path: str) -> str:
    """Extract text from PDF files"""
    try:
        import PyPDF2
        text = []
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text.append(page.extract_text())
        return "\n".join(text)
    except ImportError:
        return "PyPDF2 library not found. Please install it with 'pip install PyPDF2'."
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

def _extract_text(file_path: str) -> str:
    """Extract text from plain text files"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

def _extract_word(file_path: str) -> str:
    """Extract text from Word documents"""
    try:
        import docx
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except ImportError:
        return "python-docx library not found. Please install it with 'pip install python-docx'."
    except Exception as e:
        return f"Error extracting Word document text: {str(e)}"