"""
cosmo_mind.indexing.ingest
==========================

The ingestion pipeline ties the multi-modal parsers, chunker, embedder, vector
store and knowledge graph together into a single ``ingest`` call.

Typical usage::

    from cosmo_mind.indexing.ingest import IngestionPipeline
    pipe = IngestionPipeline()
    pipe.ingest_pdf("paper.pdf")
    pipe.ingest_fits("obs.fits")
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from cosmo_mind.config import get_settings
from cosmo_mind.indexing.embeddings import Embedder, get_embedder
from cosmo_mind.indexing.knowledge_graph import KnowledgeGraph, get_knowledge_graph
from cosmo_mind.indexing.chunker import chunk_document
from cosmo_mind.indexing.vector_store import VectorStore, get_vector_store
from cosmo_mind.parsing.document import DocType, Document
from cosmo_mind.parsing.entity_extractor import extract_objects
from cosmo_mind.parsing.latex_extractor import extract_equations
from cosmo_mind.parsing.pdf_parser import parse_pdf
from cosmo_mind.parsing.fits_parser import parse_fits
from cosmo_mind.utils.logging import get_logger

log = get_logger("cosmo_mind.indexing.ingest")


class IngestionPipeline:
    """End-to-end ingest: parse -> chunk -> embed -> vector store + graph."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.vs = vector_store or get_vector_store()
        self.kg = knowledge_graph or get_knowledge_graph()
        self.embedder = embedder or get_embedder()
        self._dim: int | None = None

    # -- low-level ----------------------------------------------------------
    def _ensure_collection(self, dim: int) -> None:
        if self._dim != dim:
            self.vs.create_collection(dim)
            self._dim = dim

    def ingest_document(self, doc: Document) -> dict:
        """Ingest a single parsed :class:`Document`."""
        # enrich entities/equations at document level if missing
        text = doc.all_text()
        if not doc.celestial_objects:
            doc.celestial_objects = extract_objects(text)
        if not doc.equations:
            doc.equations = extract_equations(text)

        chunks = chunk_document(doc)
        if not chunks:
            log.warning(f"no chunks produced for {doc.doc_id}; skipping")
            return {"doc_id": doc.doc_id, "n_chunks": 0}
        vectors = self.embedder.embed([c.text for c in chunks])
        self._ensure_collection(len(vectors[0]))
        self.vs.add_chunks(chunks, vectors)

        # knowledge graph
        self.kg.upsert_document(doc)
        for c in chunks:
            self.kg.upsert_chunk(c.chunk_id, c.doc_id, c.text)
            for obj in c.celestial_objects:
                self.kg.link(f"obj:{obj.name}", f"chunk:{c.chunk_id}", "MENTIONS")
        return {"doc_id": doc.doc_id, "n_chunks": len(chunks), "n_vectors": self.vs.count()}

    def ingest_text(
        self, text: str, source: str = "inline", doc_type: DocType = DocType.GENERAL_TEXT
    ) -> dict:
        from cosmo_mind.parsing.document import Document
        import hashlib

        doc = Document(
            doc_id=f"text:{hashlib.md5(text.encode()).hexdigest()[:12]}",
            source=source,
            title=source,
            doc_type=doc_type,
            full_text=text,
        )
        return self.ingest_document(doc)

    # -- high-level convenience --------------------------------------------
    def ingest_pdf(self, path: str | Path) -> dict:
        doc = parse_pdf(path)
        return self.ingest_document(doc)

    def ingest_fits(self, path: str | Path) -> dict:
        doc = parse_fits(path)
        return self.ingest_document(doc)

    def ingest_many_pdfs(self, paths: Sequence[str | Path]) -> list[dict]:
        return [self.ingest_pdf(p) for p in paths]

    def close(self) -> None:
        self.vs.close()
        self.kg.close()


# -- module-level convenience (uses configured singleton backends) ----------
_PIPELINE: "IngestionPipeline | None" = None


def get_pipeline() -> IngestionPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = IngestionPipeline()
    return _PIPELINE


__all__ = ["IngestionPipeline", "get_pipeline"]
