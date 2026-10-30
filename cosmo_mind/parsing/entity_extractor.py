"""
cosmo_mind.parsing.entity_extractor
====================================

Regex-driven extraction of astronomical entities from raw text, followed by
lightweight normalization with ``astropy.coordinates``.

Recognized entities
-------------------
* Catalog numbers: ``HD``, ``HIP``, ``TYC``, ``Gaia DR3``, ``2MASS``,
  ``WASP``, ``Kepler``, ``KOI``, ``TOI``.
* Exoplanet designations: ``HD 209458 b``, ``Kepler-442b``, ``TRAPPIST-1b``,
  ``TOI-700b``, ``55 Cnc b``.
* Stellar objects: ``GJ 1214``, ``Alpha Cen A``, proper names.
* Gravitational-wave events: ``GW150914``, ``GW170817``.
* Supernovae: ``SN 2014J``.
* Coordinates: sexagesimal RA/Dec ``12h34m56s +01d02m03s`` or decimal
  ``12.345 +1.234``.
* Spectral types: ``G2V``, ``M3``, ``K5III``.
"""

from __future__ import annotations

import re
from typing import Iterator

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import Angle

from cosmo_mind.utils.logging import get_logger
from cosmo_mind.parsing.document import CelestialObject

log = get_logger("cosmo_mind.parsing.entity")

# --- compiled patterns -------------------------------------------------------
_PATTERNS: list[tuple[str, str, str]] = [
    # (object_type, catalog, regex)
    ("exoplanet", "HD", r"\bHD\s*(\d{1,6})\s*([b-z])\b"),
    ("exoplanet", "Kepler", r"\bKepler-(\d{1,5})\s*([b-z])?\b"),
    ("exoplanet", "KOI", r"\bKOI-(\d{1,5})\s*([b-z])?\b"),
    ("exoplanet", "TOI", r"\bTOI-(\d{1,5})\s*([b-z])?\b"),
    ("exoplanet", "TRAPPIST", r"\b(TRAPPIST-1|UCF-101|LHS)\s*([b-z])\b"),
    ("exoplanet", "WASP", r"\bWASP-(\d{1,5})\s*([b-z])?\b"),
    ("exoplanet", "55 Cnc", r"\b55\s+Cnc\s*([b-z])\b"),
    ("star", "HD", r"\bHD\s*(\d{1,6})\b"),
    ("star", "HIP", r"\bHIP\s*(\d{1,6})\b"),
    ("star", "TYC", r"\bTYC\s*(\d+-\d+-\d+)\b"),
    ("star", "Gaia", r"\bGaia\s*DR\d\s*(\d{1,10})\b"),
    ("star", "2MASS", r"\b2MASS\s*([JHKS]?[0-9]\d{2,5}[+-][0-9]\d{2,5})\b"),
    ("star", "GJ", r"\bGJ\s*(\d{1,5})\b"),
    ("star", "proper-name", r"\b(Alpha Centauri|Betelgeuse|Rigel|Proxima\s+Centauri|Sirius|Achernar|Polaris|Procyon|Arcturus)\b"),
    ("gw_event", "GW", r"\b(GW\d{6}_\d{2}|GW\d{6})\b"),
    ("supernova", "SN", r"\bSN\s*\/?\s*(\d{4}[A-Za-z]?)\b"),
]

# sexagesimal coordinate pair (RA then Dec), e.g. "12 34 56.7 +01 02 03"
_COORD_SEX = re.compile(
    r"\b([0-9]{1,2}[h\s][0-5]?\d[m\s][0-5]?\d(?:\.\d+)?[sS]?)\s*"
    r"([+-][0-9]{1,2}[d\sd\s][0-5]?\d[m\s][0-5]?\d(?:\.\d+)?)"
)
# decimal degree pair, e.g. "12.3456 -30.1234"
_COORD_DEC = re.compile(r"\b([+-]?\d{1,3}\.\d{1,4})\s+([+-]?\d{1,3}\.\d{1,4})\b")

_SPECTRAL = re.compile(
    r"\b([OBAFGKM])([0-9](?:\d)?)([IeV?]{1,3})\b", re.IGNORECASE
)


def _to_skycoord(ra: str, dec: str) -> SkyCoord | None:
    """Best-effort conversion of two coordinate strings to a SkyCoord."""
    try:
        return SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")
    except Exception:
        pass
    try:
        # maybe both are already decimal degrees
        fra = Angle(ra, unit=u.hourangle)
        fdec = Angle(dec, unit=u.deg)
        return SkyCoord(ra=fra, dec=fdec, frame="icrs")
    except Exception:
        return None


def extract_objects(text: str) -> list[CelestialObject]:
    """Extract a de-duplicated list of ``CelestialObject`` references."""
    found: dict[str, CelestialObject] = {}

    for obj_type, catalog, pattern in _PATTERNS:
        for m in re.finditer(pattern, text):
            groups = [g for g in m.groups() if g is not None]
            name = m.group(0).strip()
            catalog_id = f"{catalog} {groups[0]}" if groups else name
            if name in found:
                continue
            found[name] = CelestialObject(
                name=name,
                object_type=obj_type,
                catalog_id=catalog_id,
                source_refs=[],
            )

    # coordinates
    for m in _COORD_SEX.finditer(text):
        sc = _to_skycoord(m.group(1), m.group(2))
        if sc is not None:
            label = f"coord:{m.group(0).strip()[:24]}"
            if label not in found:
                found[label] = CelestialObject(
                    name=label,
                    object_type="coordinate",
                    catalog_id=f"RA={sc.ra.to_string(unit=u.hour, sep=':', pad=True)} "
                    f"Dec={sc.dec.to_string(unit=u.deg, sep=':', pad=True)}",
                    coord=sc,
                )

    for m in _COORD_DEC.finditer(text):
        if m.group(0).strip() in found:
            continue
        try:
            sc = SkyCoord(
                ra=float(m.group(1)) * u.deg,
                dec=float(m.group(2)) * u.deg,
                frame="icrs",
            )
        except Exception:
            continue
        label = f"coord:{m.group(1)},{m.group(2)}"
        if label not in found:
            found[label] = CelestialObject(
                name=label, object_type="coordinate", catalog_id=label, coord=sc
            )

    return list(found.values())


def extract_spectral_types(text: str) -> list[str]:
    """Return a de-duplicated list of spectral-type strings found in text."""
    return sorted({m.group(0) for m in _SPECTRAL.finditer(text)})


__all__ = ["extract_objects", "extract_spectral_types"]
