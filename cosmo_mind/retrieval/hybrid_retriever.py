"""
cosmo_mind.retrieval.hybrid_retriever
=====================================

Dual-layer GraphRAG retrieval.

Combines:

1. **Vector (semantic) search** — embed the query and fetch the nearest
   chunk vectors from the vector store.
2. **Graph (symbolic) traversal** — extract celestial objects from the query,
   look them up in the knowledge graph, and find related documents /
   neighboring objects.

The two rankings are fused with **Reciprocal Rank Fusion (RRF)** so that a
chunk which is both semantically close *and* graph-connected gets a boost —
this is what lets CosmoMind ground a free-text question like *"light curves
of HD 189733 b"* in both the literature and the JWST/FITS observations.
"""

from __future__ import annotations

import math
from typing import Any

from cosmo_mind.indexing.embeddings import Embedder, get_embedder
from cosmo_mind.indexing.knowledge_graph import KnowledgeGraph, get_knowledge_graph
from cosmo_mind.indexing.vector_store import VectorStore, get_vector_store
from cosmo_mind.parsing.entity_extractor import extract_objects
from cosmo_mind.utils.logging import get_logger

log = get_logger("cosmo_mind.retrieval.hybrid")

RRF_K = 60.0  # RRF constant


def _rrf(rank: int, k: float = RRF_K) -> float:
    return 1.0 / (k + max(rank, 1))


class HybridRetriever:
    """Fuses vector + knowledge-graph retrieval."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.vs = vector_store or get_vector_store()
        self.kg = knowledge_graph or get_knowledge_graph()
        self.embedder = embedder or get_embedder()

    # -- public API ---------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """Return ``top_k`` fused, ranked results for *query*."""
        qvec = self.embedder.embed([query])[0]
        vec_hits = self.vs.search(qvec, top_k=top_k * 3)

        # graph side
        objects = extract_objects(query)
        graph_docs: dict[str, dict] = {}  # doc_id -> meta
        graph_objs: list[dict] = []
        for obj in objects:
            rel_docs = self.kg.related_documents(obj.name, top_k=top_k)
            for d in rel_docs:
                did = d.get("doc_id") or d.get("node_id", "")
                if did and did not in graph_docs:
                    graph_docs[did] = d
                    graph_docs[did]["_graph_rank"] = len(graph_docs)
            neighbors = self.kg.neighbors(obj.name, rel="OBSERVED_IN", top_k=top_k)
            graph_objs.extend(neighbors)
            # also broaden the graph with objects related to this object
            for nb in neighbors:
                nb_objs = self.kg.search_objects(query, top_k=top_k)
                graph_objs.extend(nb_objs)

        # fuse with RRF
        fused: dict[str, dict] = {}
        for rank, hit in enumerate(vec_hits, start=1):
            cid = hit.get("chunk_id")
            did = hit.get("doc_id")
            score = _rrf(rank) + math.log1p(hit.get("vector_score", 0.0))
            if did in graph_docs:
                score += _rrf(graph_docs[did]["_graph_rank"]) + 0.5
            entry = {
                "chunk_id": cid,
                "doc_id": did,
                "score": round(score, 5),
                "vector_score": hit.get("vector_score", 0.0),
                "graph_relevant": bool(did in graph_docs),
                "text": hit.get("text", ""),
                "celestial_objects": hit.get("celestial_objects", []),
                "equation_latex": hit.get("equation_latex", []),
            }
            fused[cid] = entry

        results = sorted(fused.values(), key=lambda x: -x["score"])[:top_k]
        # attach graph context (related objects & docs) to each result
        ctx = {"related_objects": [o.get("name") for o in graph_objs[:5]],
               "related_documents": [d.get("doc_id") or d.get("title") for d in graph_docs.values()][:5]}
        for r in results:
            r["graph_context"] = ctx
        return results

    def retrieve_raw(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """Vector-only retrieval (no graph fusion) — handy for comparison."""
        qvec = self.embedder.embed([query])[0]
        return self.vs.search(qvec, top_k=top_k)


def get_retriever() -> HybridRetriever:
    return HybridRetriever()


__all__ = ["HybridRetriever", get_retriever, RRF_K]
