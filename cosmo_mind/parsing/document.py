"""
cosmo_mind.parsing.document
===========================

Canonical data models shared across parsing, indexing, retrieval and
agentic execution.  Everything in CosmoMind speaks in terms of these
pydantic models so the pipeline stays type-safe and serializable.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from pydantic import BaseModel, ConfigDict, Field


class DocType(str, Enum):
    """Enumerates the kinds of documents CosmoMind can ingest."""

    ARXIV_PAPER = "arxiv_paper"
    ADS_PAPER = "ads_paper"
    FITS_FILE = "fits_file"
    OBSERVATION_LOG = "observation_log"
    CODE_SNIPPET = "code_snippet"
    GENERAL_TEXT = "general_text"


class CelestialObject(BaseModel):
    """A named astronomical entity referenced in the corpus."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    object_type: str = Field(
        default="unknown",
        description="star | exoplanet | galaxy | supernova | gw_event | ...",
    )
    catalog_id: str | None = None  # e.g. HD 189733, Gaia DR3 1234, 2MASS J...
    aliases: list[str] = Field(default_factory=list)
    coord: SkyCoord | None = None  # astropy SkyCoord (RA/Dec, any frame)
    spectral_type: str | None = None
    distance_pc: Quantity | None = None
    source_refs: list[str] = Field(default_factory=list)


class Equation(BaseModel):
    """A mathematical expression extracted from a document."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    latex: str = Field(description="Original LaTeX source of the equation.")
    rendered: str | None = None  # unicode render if available
    context: str = ""  # surrounding text / section heading


class Figure(BaseModel):
    """A plot / chart image extracted from a PDF."""

    caption: str = ""
    page: int = 0
    bbox: tuple[float, float, float, float] | None = None
    ocr_text: str = ""


class Chunk(BaseModel):
    """A unit of retrievable content (a slice of a Document)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_id: str
    doc_id: str
    span: tuple[int, int] = (0, 0)  # char offsets into the source document
    source_type: DocType = DocType.GENERAL_TEXT
    text: str = ""
    equations: list[Equation] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    celestial_objects: list[CelestialObject] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector: list[float] | None = None  # populated at indexing time


class Document(BaseModel):
    """A top-level ingested document with its parsed components."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    doc_id: str
    source: str  # path / URL / arxiv id
    title: str = ""
    doc_type: DocType = DocType.GENERAL_TEXT
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    full_text: str = ""
    celestial_objects: list[CelestialObject] = Field(default_factory=list)
    equations: list[Equation] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    fits_headers: dict[str, Any] = Field(
        default_factory=dict,
        description="FITS header cards keyed by extension / HDU index.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunks: list[Chunk] = Field(default_factory=list)

    def all_text(self) -> str:
        """Concatenate all textual content (abstract + body)."""
        return f"{self.title}\n{self.abstract}\n{self.full_text}".strip()


__all__ = [
    "DocType",
    "CelestialObject",
    "Equation",
    "Figure",
    "Chunk",
    "Document",
]
