"""
cosmo_mind.indexing.embeddings
==============================

Pluggable densification of text into fixed-length vectors for the vector
store.  Two strategies are shipped:

``OpenAIEmbedder``
    Calls the OpenAI embeddings API (``text-embedding-3-small`` by default).
    Requires ``OPENAI_API_KEY``.  This is the high-fidelity PROD backend.

``LocalHashingEmbedder``
    A dependency-free feature-hashing embedder that L2-normalises sparse
    hashed word-count vectors.  Deterministic and offline — used automatically
    in ``LOCAL`` mode so retrieval/search works with zero configuration.

Both expose the same interface (``.embed(texts)`` and ``.dimension``).
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

from cosmo_mind.utils.logging import get_logger
from cosmo_mind.config import get_settings

log = get_logger("cosmo_mind.indexing.embeddings")

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_+\-.]*")


class Embedder:
    """Abstract embedding interface."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    @property
    def dimension(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError


class LocalHashingEmbedder(Embedder):
    """Fast, deterministic, dependency-free feature-hashing embedder."""

    def __init__(self, dimension: int = 512) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_word(self, word: str) -> int:
        h = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "little") % self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self._dim, dtype=np.float32)
            for tok in _WORD.findall(text.lower()):
                vec[self._hash_word(tok)] += 1.0
            # sub-linear tf (suppress log(0) for unseen buckets)
            with np.errstate(divide="ignore"):
                vec = np.where(vec > 0, 1.0 + np.log(vec), 0.0)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            out.append(vec.tolist())
        return out


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings backend (PROD)."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model = model or get_settings().openai_embedding_model
        self._api_key = api_key or get_settings().openai_api_key
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY required for OpenAIEmbedder")
        from openai import OpenAI

        self._client = OpenAI(api_key=self._api_key)

    @property
    def dimension(self) -> int:
        return 1536  # text-embedding-3-small

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        results = self._client.embeddings.create(
            model=self._model, input=texts
        ).data
        return [r.embedding for r in results]


def get_embedder() -> Embedder:
    """Factory selecting the embedder based on runtime settings."""
    s = get_settings()
    if s.use_openai and s.openai_api_key:
        try:
            return OpenAIEmbedder(model=s.openai_embedding_model, api_key=s.openai_api_key)
        except Exception as e:  # pragma: no cover
            log.warning(f"OpenAI embedder unavailable ({e}); falling back to local hashing.")
    return LocalHashingEmbedder()


__all__ = ["Embedder", "LocalHashingEmbedder", "OpenAIEmbedder", "get_embedder"]
