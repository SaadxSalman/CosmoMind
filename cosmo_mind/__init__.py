"""
cosmo_mind
==========

Agentic Multi-Modal GraphRAG for Astrophysical Discovery.

CosmoMind is a research co-pilot that bridges massive literature
repositories (arXiv / ADS), observational data logs (FITS), and live
astronomical databases (SIMBAD, VizieR, NASA Exoplanet Archive) using a
dual-layer Knowledge Graph augmented by a vector store.

The package is designed to be **fully functional offline** in ``LOCAL`` mode
(TF-IDF embeddings, in-memory vector store, NetworkX knowledge graph, and a
deterministic LLM stub) while switching seamlessly to production cloud
backends (OpenAI, Qdrant, Neo4j, live astroquery) when configured via
``.env`` with ``COSMO_MODE=PROD``.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
