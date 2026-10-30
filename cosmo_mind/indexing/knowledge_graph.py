"""
cosmo_mind.indexing.knowledge_graph
===================================

Abstract :class:`KnowledgeGraph` interface plus a dependency-free
:class:`NetworkxKnowledgeGraph` used automatically in ``LOCAL`` mode (and as
a resilient fallback when Neo4j is unreachable).

The knowledge graph is the *symbolic* half of CosmoMind's dual-layer
GraphRAG.  It stores celestial entities and their relationships to documents,
equations, observations and to each other — enabling traversal queries such
as "which papers observe the object that hosts exoplanet HD 209458 b?".
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import networkx as nx

from cosmo_mind.parsing.document import CelestialObject, Document
from cosmo_mind.utils.logging import get_logger

log = get_logger("cosmo_mind.indexing.kg")


class KnowledgeGraph:
    """Abstract knowledge-graph interface."""

    def upsert_document(self, doc: Document) -> None: ...
    def upsert_chunk(self, chunk_id: str, doc_id: str, text: str) -> None: ...
    def add_entity(self, obj: CelestialObject) -> None: ...
    def add_equation(self, latex: str, chunk_id: str) -> None: ...
    def link(self, src: str, dst: str, rel: str, **props: Any) -> None: ...
    def search_objects(self, query: str, top_k: int = 10) -> list[dict]: ...
    def neighbors(self, node: str, rel: str | None = None, top_k: int = 10) -> list[dict]: ...
    def related_documents(self, object_name: str, top_k: int = 10) -> list[dict]: ...
    def serialize(self) -> nx.MultiDiGraph: ...
    def close(self) -> None: ...


class NetworkxKnowledgeGraph(KnowledgeGraph):
    """In-memory :mod:`networkx` implementation (LOCAL fallback)."""

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._g = nx.MultiDiGraph()
        self._persist = Path(persist_path) if persist_path else None
        if self._persist and self._persist.exists():
            with self._persist.open("rb") as fh:
                self._g = pickle.load(fh)

    def upsert_document(self, doc: Document) -> None:
        nid = f"doc:{doc.doc_id}"
        self._g.add_node(
            nid, node_type="Document", doc_id=doc.doc_id, title=doc.title,
            source=doc.source, doc_type=doc.doc_type.value, year=doc.year,
            telescope=doc.metadata.get("telescop") or doc.metadata.get("telescope"),
        )
        for obj in doc.celestial_objects:
            self.add_entity(obj)
            self.link(f"obj:{obj.name}", nid, "OBSERVED_IN")
        for eq in doc.equations:
            self.add_equation(eq.latex, f"doc:{doc.doc_id}")

    def upsert_chunk(self, chunk_id: str, doc_id: str, text: str) -> None:
        nid = f"chunk:{chunk_id}"
        self._g.add_node(nid, node_type="Chunk", chunk_id=chunk_id, doc_id=doc_id, text=text)
        self.link(f"doc:{doc_id}", nid, "HAS_CHUNK")

    def add_entity(self, obj: CelestialObject) -> None:
        nid = f"obj:{obj.name}"
        data: dict[str, Any] = {
            "node_type": "CelestialObject", "name": obj.name,
            "object_type": obj.object_type,
        }
        if obj.catalog_id:
            data["catalog_id"] = obj.catalog_id
        if obj.spectral_type:
            data["spectral_type"] = obj.spectral_type
        if obj.coord is not None:
            data["ra"] = float(obj.coord.ra.deg)
            data["dec"] = float(obj.coord.dec.deg)
        self._g.add_node(nid, **data)

    def add_equation(self, latex: str, chunk_id: str) -> None:
        nid = f"eq:{abs(hash(latex))}"
        self._g.add_node(nid, node_type="Equation", latex=latex, chunk_id=chunk_id)
        self.link(nid, f"chunk:{chunk_id}", "IN_CHUNK")

    def link(self, src: str, dst: str, rel: str, **props: Any) -> None:
        self._g.add_edge(src, dst, key=rel, relation=rel, **props)

    def search_objects(self, query: str, top_k: int = 10) -> list[dict]:
        q = query.lower()
        matches = []
        for _, data in self._g.nodes(data=True):
            if data.get("node_type") != "CelestialObject":
                continue
            name = (data.get("name", "") or "").lower()
            cat = (data.get("catalog_id", "") or "").lower()
            if q in name or q in cat:
                matches.append((1.0, data))
        matches.sort(key=lambda x: -x[0])
        return [d for _, d in matches[:top_k]]

    def neighbors(self, node: str, rel: str | None = None, top_k: int = 10) -> list[dict]:
        n = self._g.nodes()
        nid = node if node in n else f"obj:{node}" if f"obj:{node}" in n else f"doc:{node}"
        out: list[dict] = []
        for _, nb, edata in self._g.out_edges(nid, data=True):
            if rel and edata.get("relation") != rel:
                continue
            out.append({"node": nb, **self._g.nodes[nb]})
            if len(out) >= top_k:
                break
        return out

    def related_documents(self, object_name: str, top_k: int = 10) -> list[dict]:
        nid = f"obj:{object_name}"
        if nid not in self._g:
            return []
        docs = []
        for _, nb, _ in self._g.out_edges(nid, data=True):
            nd = self._g.nodes[nb]
            if nb.startswith("doc:") and nd.get("node_type") == "Document":
                docs.append(nd)
        return docs[:top_k]

    def serialize(self) -> nx.MultiDiGraph:
        return self._g

    def close(self) -> None:
        if self._persist:
            self._persist.parent.mkdir(parents=True, exist_ok=True)
            with self._persist.open("wb") as fh:
                pickle.dump(self._g, fh)


from cosmo_mind.config import get_settings  # noqa: E402


_LOCAL_KG: "NetworkxKnowledgeGraph | None" = None


def get_knowledge_graph() -> KnowledgeGraph:
    """Factory: Neo4j (PROD) or a cached NetworkX graph (LOCAL / fallback).

    The LOCAL graph is a process-wide singleton so ingestion and retrieval
    share the same in-memory topology rather than each loading an empty
    disk pickle.  Cross-process state persists to kg.pkl on close().
    """
    global _LOCAL_KG
    from cosmo_mind.indexing.neo4j_graph import Neo4jKnowledgeGraph

    s = get_settings()
    if s.is_prod:
        try:
            kg = Neo4jKnowledgeGraph(
                uri=s.neo4j_uri, user=s.neo4j_user, password=s.neo4j_password
            )
            kg._health_check()
            log.info("Connected to Neo4j knowledge graph.")
            return kg
        except Exception as e:  # pragma: no cover
            log.warning(f"Neo4j unavailable ({e}); falling back to NetworkX graph.")
    if _LOCAL_KG is None:
        _LOCAL_KG = NetworkxKnowledgeGraph(
            persist_path=Path(get_settings().cosmo_kg_cache) / "kg.pkl"
        )
    return _LOCAL_KG


def reset_local_knowledge_graph() -> None:
    global _LOCAL_KG
    _LOCAL_KG = None


__all__ = ["KnowledgeGraph", "NetworkxKnowledgeGraph", "get_knowledge_graph"]
