"""
cosmo_mind.indexing.vector_store
================================

Vector-store abstraction over dense embeddings.  Two backends:

``QdrantVectorStore``     — remote Qdrant server (PROD).
``InMemoryVectorStore``   — NumPy-backed cosine store (LOCAL), with pickle
                            persistence.  No external service required.
"""

from __future__ import annotations

import pickle
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cosmo_mind.config import get_settings
from cosmo_mind.utils.logging import get_logger

log = get_logger("cosmo_mind.indexing.vector")

DEFAULT_COLLECTION = "cosmomind_chunks"


def _chunk_payload(chunk: Any) -> dict:
    """Serialise a :class:`Chunk` into a JSON-friendly payload dict."""
    objs = [
        {"name": o.name, "type": o.object_type, "catalog": o.catalog_id}
        for o in getattr(chunk, "celestial_objects", [])
    ]
    eqs = [e.latex for e in getattr(chunk, "equations", [])]
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source_type": chunk.source_type.value,
        "text": (chunk.text or "")[:4000],
        "celestial_objects": objs,
        "equation_latex": eqs,
        "metadata": chunk.metadata or {},
    }


class VectorStore:
    """Abstract vector-store interface."""

    collection: str = DEFAULT_COLLECTION

    def create_collection(self, dim: int) -> None: ...
    def add_chunks(self, chunks: Sequence[Any], vectors: Sequence[Sequence[float]]) -> None: ...
    def search(self, vector: Sequence[float], top_k: int = 8) -> list[dict]: ...
    def count(self) -> int: ...
    def delete_collection(self) -> None: ...
    def close(self) -> None: ...


class InMemoryVectorStore(VectorStore):
    """NumPy-backed cosine-similarity store with optional pickle persistence."""

    def __init__(self, collection: str = DEFAULT_COLLECTION,
                 persist_path: str | Path | None = None) -> None:
        self.collection = collection
        self._vectors: dict[str, np.ndarray] = {}
        self._payloads: dict[str, dict] = {}
        self._persist = Path(persist_path) if persist_path else None
        if self._persist and self._persist.exists():
            with self._persist.open("rb") as fh:
                self._vectors, self._payloads = pickle.load(fh)

    def create_collection(self, dim: int) -> None:  # dim unused in-memory
        return None

    def add_chunks(self, chunks, vectors) -> None:
        for chunk, vec in zip(chunks, vectors):
            v = np.asarray(vec, dtype=np.float32)
            nrm = float(np.linalg.norm(v))
            v = v / nrm if nrm > 0 else v
            self._vectors[chunk.chunk_id] = v
            self._payloads[chunk.chunk_id] = _chunk_payload(chunk)

    def search(self, vector, top_k=8) -> list[dict]:
        q = np.asarray(vector, dtype=np.float32)
        nrm = float(np.linalg.norm(q))
        if nrm == 0 or not self._vectors:
            return []
        q = q / nrm
        ids = list(self._vectors.keys())
        mat = np.stack([self._vectors[i] for i in ids])
        sims = mat @ q  # cosine = dot
        order = np.argsort(sims)[::-1][:top_k]
        out = []
        for idx in order:
            cid = ids[idx]
            out.append({"vector_score": float(sims[idx]), **self._payloads[cid]})
        return out

    def count(self) -> int:
        return len(self._vectors)

    def delete_collection(self) -> None:
        self._vectors.clear()
        self._payloads.clear()

    def close(self) -> None:
        if self._persist and self._vectors:
            self._persist.parent.mkdir(parents=True, exist_ok=True)
            with self._persist.open("wb") as fh:
                pickle.dump((self._vectors, self._payloads), fh)


import uuid


class QdrantVectorStore(VectorStore):
    """Real Qdrant-backed vector store (PROD)."""

    def __init__(self, url: str | None = None, api_key: str | None = None,
                 collection: str = DEFAULT_COLLECTION, prefer_grpc: bool = False) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        self._qmodels = models
        self.collection = collection
        self._client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
        try:
            info = self._client.get_collection(collection)
            self._dim = info.config.params.size
        except Exception:
            self._dim = None

    def create_collection(self, dim: int) -> None:
        self._client.recreate_collection(
            collection_name=self.collection,
            vectors_config={
                "default": self._qmodels.VectorParams(
                    size=dim, distance=self._qmodels.Distance.COSINE
                )
            },
        )
        self._dim = dim

    def add_chunks(self, chunks, vectors) -> None:
        points = []
        for chunk, vec in zip(chunks, vectors):
            uid = uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)
            points.append(
                self._qmodels.PointStruct(
                    id=str(uid), vector=list(vec), payload=_chunk_payload(chunk)
                )
            )
        self._client.upsert(self.collection, points, wait=True)

    def search(self, vector, top_k=8) -> list[dict]:
        hits = self._client.search(
            collection_name=self.collection,
            query_vector=("default", list(vector)),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        out = []
        for h in hits:
            out.append({"vector_score": h.score, **(h.payload or {})})
        return out

    def count(self) -> int:
        try:
            coll, _ = self._client.scroll(
                collection_name=self.collection,
                with_payload=False,
                with_vectors=False,
                limit=1,
            )
            return len(coll)
        except Exception:
            return 0

    def delete_collection(self) -> None:
        try:
            self._client.delete_collection(self.collection)
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


_LOCAL_VS: "InMemoryVectorStore | None" = None


def get_vector_store(collection: str = DEFAULT_COLLECTION) -> "VectorStore":
    """Factory: Qdrant in PROD mode, else the (cached) in-memory store.

    The LOCAL in-memory store is a process-wide singleton so ingestion, the
    retriever and the agent share one collection rather than each loading an
    empty disk pickle.  Cross-process state persists to vectors.pkl on close().
    """
    global _LOCAL_VS
    s = get_settings()
    if s.is_prod and s.qdrant_url:
        try:
            store = QdrantVectorStore(
                url=s.qdrant_url,
                api_key=s.qdrant_api_key,
                collection=collection,
                prefer_grpc=s.qdrant_prefer_grpc,
            )
            store._client.get_collections()
            log.info("Connected to Qdrant vector store.")
            return store
        except Exception as e:  # pragma: no cover
            log.warning(f"Qdrant unavailable ({e}); falling back to in-memory store.")
    if _LOCAL_VS is None:
        _LOCAL_VS = InMemoryVectorStore(
            collection=collection,
            persist_path=Path(get_settings().cosmo_vector_cache) / "vectors.pkl",
        )
    return _LOCAL_VS


def reset_local_vector_store() -> None:
    global _LOCAL_VS
    _LOCAL_VS = None


__all__ = [
    "VectorStore",
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "get_vector_store",
]