from __future__ import annotations

import fitz  # PyMuPDF


def split_pdf_bytes(pdf_bytes: bytes, batch_size: int = 200) -> list[bytes]:
    """Split a PDF into batches of at most `batch_size` pages.

    Returns a list of PDF byte strings. A single-batch document returns a
    list with one element containing the original pages.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = len(doc)
    
    if total <= batch_size:
        doc.close()
        return [pdf_bytes]

    batches: list[bytes] = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total) - 1  # fitz slice is inclusive
        
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=start, to_page=end)
        batches.append(new_doc.tobytes())
        new_doc.close()

    doc.close()
    return batches
