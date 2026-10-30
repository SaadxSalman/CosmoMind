"""
cosmo_mind.indexing.chunker
===========================

Deterministic, sentence-aware chunking of an ingested :class:`Document` into
:class:`Chunk` objects.  Chunking is the bridge between multi-modal parsing
and indexing: each chunk becomes one vector-store point and a node in the
knowledge graph.
"""

from __future__ import annotations

import re

from cosmo_mind.utils.logging import get_logger
from cosmo_mind.parsing.document import Chunk, DocType, Document
from cosmo_mind.parsing.entity_extractor import extract_objects
from cosmo_mind.parsing.latex_extractor import extract_equations

log = get_logger("cosmo_mind.indexing.chunker")

_SENT_SPLIT = re.compile(r"(?<![0-9])([.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        for piece in _SENT_SPLIT.split(para):
            piece = piece.strip(" .")
            if piece:
                out.append(piece)
    return out


def chunk_document(
    doc: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    min_chunk: int = 50,
) -> list[Chunk]:
    """Chunk a document's text into roughly ``chunk_size``-token slices.

    Sentences are grouped greedily so that no chunk breaks a sentence, an
    ``overlap`` of the last paragraph is carried into the next chunk, and
    every chunk re-derives its own celestial objects + equations so it is
    self-describing.
    """
    text = doc.all_text()
    sentences = _split_sentences(text)
    chunks: list[Chunk] = []
    span = 0
    buf: list[str] = []
    buf_len = 0
    overlap_text = ""

    def _flush(start_offset: int) -> int:
        nonlocal span
        if not buf:
            return start_offset
        content = overlap_text + " " + " ".join(buf) if overlap_text else " ".join(buf)
        content = content.strip()
        if len(content) < min_chunk and chunks:
            # too small to stand alone; merge into previous chunk
            prev = chunks[-1]
            prev.text = prev.text + " " + content
            return start_offset
        end_offset = start_offset + len(content)
        chunk = Chunk(
            chunk_id=f"{doc.doc_id}:c{len(chunks)}",
            doc_id=doc.doc_id,
            span=(start_offset, end_offset),
            source_type=doc.doc_type,
            text=content,
            celestial_objects=extract_objects(content),
            equations=extract_equations(content),
            metadata={
                "title": doc.title,
                "source_type": doc.doc_type.value,
                "page": doc.metadata.get("n_pages"),
            },
        )
        chunks.append(chunk)
        return end_offset

    for sent in sentences:
        if buf_len + len(sent) > chunk_size and buf:
            span = _flush(span)
            overlap_text = " ".join(buf[-2:]) if len(buf) >= 2 else buf[-1]
            buf = []
            buf_len = 0
        buf.append(sent)
        buf_len += len(sent) + 1
    if buf:
        _flush(span)

    log.debug(f"chunked {doc.doc_id} into {len(chunks)} chunks")
    return chunks


__all__ = ["chunk_document"]
