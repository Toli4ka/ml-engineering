from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class PDFPage:
    page_number: int
    text: str


def load_pdf(pdf_path: str) -> List[PDFPage]:
    """
    Load a PDF file and extract text page by page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of PDFPage objects.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    reader = PdfReader(str(path))
    pages: List[PDFPage] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        pages.append(PDFPage(page_number=page_number, text=text))

    if not any(page.text for page in pages):
        raise ValueError("No extractable text found in the PDF.")

    return pages