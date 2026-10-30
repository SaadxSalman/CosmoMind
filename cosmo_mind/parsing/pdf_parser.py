"""
cosmo_mind.parsing.pdf_parser
=============================

Multi-modal parser for research PDFs built on PyMuPDF (``fitz``).

For each page it extracts:
* the running OCR-free text (preserving reading order),
* rasterized figures / plots with their bounding boxes and the surrounding
  "caption" text (the nearest paragraph),
* embedded LaTeX-style equations (delegated to :mod:`latex_extractor`),
* celestial-object & coordinate references (delegated to
  :mod:`entity_extractor`).

The returned :class:`Document` is *not* yet chunked — chunking happens in
:mod:`cosmo_mind.indexing.ingest` so the same strategy applies to every
document type.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import fitz  # PyMuPDF

from cosmo_mind.utils.logging import get_logger
from cosmo_mind.parsing.document import DocType, Document, Figure
from cosmo_mind.parsing.latex_extractor import extract_equations
from cosmo_mind.parsing.entity_extractor import extract_objects

log = get_logger("cosmo_mind.parsing.pdf")


def _safe_metadata(doc: fitz.Document) -> dict:
    md = doc.metadata or {}
    return {
        "title": md.get("title") or "",
        "author": md.get("author") or "",
        "subject": md.get("subject") or "",
        "keywords": md.get("keywords") or "",
        "creator": md.get("creator") or "",
        "producer": md.get("producer") or "",
    }


def _extract_figures(doc: fitz.Document) -> list[Figure]:
    """Pull image blocks out of a PDF as :class:`Figure` objects."""
    figures: list[Figure] = []
    for page_no, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        # lines/words grouped so we can read captions
        words = page.get_text("words")
        for blk in blocks:
            if blk["type"] != 0:  # 0 == text; 1 == image
                continue
            if blk.get("image"):
                bbox = blk["bbox"]
                page_words = [
                    w for w in words
                    if w[3] >= bbox[1] - 20 and w[1] <= bbox[3] + 20
                ]
                # caption = words directly below the image
                below = [
                    w[4] for w in page_words
                    if w[1] >= bbox[3] and w[4]
                ]
                above = [w[4] for w in page_words if w[3] <= bbox[1] and w[4]]
                caption = " ".join(below[:40] if below else above[:40])
                figures.append(
                    Figure(
                        caption=caption.strip(),
                        page=page_no + 1,
                        bbox=tuple(bbox),
                        ocr_text=blk.get("text", ""),
                    )
                )
    return figures


def parse_pdf(path: str | Path, title: str | None = None) -> Document:
    """Parse a PDF file into a :class:`Document`."""
    path = Path(path)
    doc = fitz.open(str(path))
    md = _safe_metadata(doc)
    page_count = doc.page_count

    full_parts: list[str] = []
    figures: list[Figure] = []
    equations = []
    objects = []
    full_text = ""
    try:
        for page_no, page in enumerate(doc):
            text = page.get_text() or ""
            full_parts.append(text)
        full_text = "\n".join(full_parts)
        figures = _extract_figures(doc)
        equations = extract_equations(full_text)
        objects = extract_objects(full_text)
    finally:
        doc.close()

    doc_id = f"pdf:{hashlib.md5(str(path).encode()).hexdigest()[:12]}"

    # Heuristic abstract: first block between 'Abstract' and '1.' / 'Introduction'
    abs_match = _find_abstract(full_text)
    title_val = title or md.get("title") or path.stem
    return Document(
        doc_id=doc_id,
        source=str(path),
        title=title_val,
        doc_type=DocType.ARXIV_PAPER,
        authors=[a.strip() for a in md.get("author", "").split(",") if a.strip()],
        abstract=abs_match,
        full_text=full_text,
        celestial_objects=objects,
        equations=equations,
        figures=figures,
        metadata={**md, "n_pages": page_count},
    )


def _find_abstract(text: str) -> str:
    import re

    m = re.search(r"(?:Abstract|A\b)\s*[-:]?\s*\n(.*)", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    body = m.group(1)
    cut = re.search(r"\n\s*(1\.|Introduction|I\. )", body)
    return (body[: cut.start()] if cut else body)[:2000].strip()


__all__ = ["parse_pdf", parse_pdf.__doc__]
