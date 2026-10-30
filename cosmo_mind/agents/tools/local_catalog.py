"""
cosmo_mind.agents.tools.local_catalog
=====================================

A small, curated reference catalogue of well-known astronomical objects.

This is **not** a stub for correctness — it is a graceful-degradation
fallback used *only* when the live astroquery services (SIMBAD / VizieR /
NASA Exoplanet) are unreachable (e.g. a firewalled or offline environment).
Live calls always take precedence; this local catalogue simply ensures the
CosmoMind agent still returns meaningful data in LOCAL/DEMO mode.

Each entry is tagged with the fields CosmoMind surfaces to the user.
"""

from __future__ import annotations

from typing import Any

# object_name (lower-cased) -> record.  Aliases are expanded in lookup().
_LOCAL_CATALOG: dict[str, dict[str, Any]] = {
    "hd 209458 b": {
        "main_id": "HD 209458 b",
        "object_type": "exoplanet",
        "spectral_type": "G0V",
        "host": "HD 209458",
        "ra": 20.25884,
        "dec": 15.20072,
        "period_days": 3.524744,
        "mass_mjup": 0.690,
        "radius_rjup": 1.38,
        "distance_pc": 49.0,
        "discovery": "Charbonneau et al. 2002",
        "description": (
            "HD 209458 b is a hot Jupiter discovered by the transit method. "
            "It was the first extra-solar planet detected by optical transit "
            "photometry. Its inflated radius and 3.5-day orbit make it a "
            "benchmark for atmospheric characterisation with JWST."
        ),
    },
    "hd 209458": {
        "main_id": "HD 209458",
        "object_type": "star",
        "spectral_type": "G0V",
        "ra": 20.25884,
        "dec": 15.20072,
        "distance_pc": 49.0,
        "vmag": 7.6,
        "description": "G0V main-sequence star hosting the hot Jupiter HD 209458 b.",
    },
    "kepler-442 b": {
        "main_id": "Kepler-442 b",
        "object_type": "exoplanet",
        "spectral_type": "K3V",
        "host": "Kepler-442",
        "period_days": 112.34,
        "radius_rearth": 1.34,
        "incident_flux": 73.0,
        "eq_temp_k": 251,
        "distance_pc": 1200.0,
        "description": (
            "Kepler-442 b is a near-Earth-sized exoplanet in the habitable zone "
            "of a K3V star, one of the closest analogues to the Earth announced "
            "by Kepler."
        ),
    },
    "trappist-1": {
        "main_id": "TRAPPIST-1",
        "object_type": "star",
        "spectral_type": "M8V",
        "ra": 23.16136,
        "dec": -5.04328,
        "distance_pc": 12.1,
        "n_planets": 7,
        "age_gyr": 7.6,
        "description": (
            "Ultra-cool M8V dwarf hosting seven Earth-sized planets near a "
            "resonant chain; three of them lie in the habitable zone."
        ),
        "aliases": ["UCDEm J2309-0503", "2MASS J2309-0503"],
    },
    "gw170817": {
        "main_id": "GW170817",
        "object_type": "gw_event",
        "ra": 13.37016,
        "dec": -30.81625,
        "distance_mpc": 40.0,
        "redshift": 0.0097,
        "host_galaxy": "NGC 4993",
        "description": (
            "Binary neutron-star merger observed by LIGO/Virgo and localised "
            "to NGC 4993. Associated kilonova AT 2017gfo confirmed heavy-element "
            "r-process nucleosynthesis."
        ),
    },
    "betelgeuse": {
        "main_id": "Betelgeuse",
        "object_type": "star",
        "spectral_type": "M1-2 Iab",
        "ra": 8.80268,
        "dec": 7.41096,
        "distance_pc": 222.0,
        "description": "Red supergiant, nearest massive star; a Type II-P supernova progenitor candidate.",
    },
    "sirius": {
        "main_id": "Sirius",
        "object_type": "star",
        "spectral_type": "A1V",
        "ra": 6.7525,
        "dec": -16.7161,
        "distance_pc": 2.64,
        "description": "Brightest star in Earth's night sky, a white-dwarf + main-sequence binary.",
    },
    "rigel": {
        "main_id": "Rigel",
        "object_type": "star",
        "spectral_type": "B8Ia",
        "ra": 5.2423,
        "dec": -8.2017,
        "distance_pc": 265.0,
        "description": "Blue supergiant, the brightest star in Orion, an evolved massive star.",
    },
    "proxima centauri": {
        "main_id": "Proxima Centauri",
        "object_type": "star",
        "spectral_type": "M5.5Ve",
        "ra": 14.2610,
        "dec": -62.6762,
        "distance_pc": 1.30,
        "description": "Nearest known star to the Sun, an ultra-cool red dwarf with a confirmed rocky planet (Proxima b).",
    },
}

def _norm(s: str) -> str:
    return s.strip().lower().replace(" ", "").replace("-", "")


# Map both the raw lower-cased name AND its normalised form to the canonical key.
_ALIASES: dict[str, str] = {}
for _key, _rec in _LOCAL_CATALOG.items():
    for _al in [_key] + list(_rec.get("aliases", [])):
        _aln = _al.lower()
        _ALIASES[_aln] = _key
        _ALIASES[_norm(_aln)] = _key


def lookup(name: str) -> dict[str, Any] | None:
    """Best-effort local lookup by exact, normalised, or alias name."""
    if not name:
        return None
    canon = _ALIASES.get(_norm(name))
    if canon is None:
        canon = _ALIASES.get(name.strip().lower())
    if canon is None:
        # fuzzy: alias is a substring of the query or vice-versa
        raw = name.strip().lower()
        for al, key in _ALIASES.items():
            if al and (al in raw or raw in al):
                canon = key
                break
    if canon is None:
        return None
    return dict(_LOCAL_CATALOG[canon])


def describe(name: str) -> str | None:
    rec = lookup(name)
    if rec is None:
        return None
    fields = ["main_id", "object_type", "spectral_type", "ra", "dec",
              "period_days", "mass_mjup", "radius_rjup", "distance_pc", "distance_pc"]
    summary = []
    for f in fields:
        v = rec.get(f)
        if v is not None and f not in ("distance_pc",):
            summary.append(f"{f}={v}")
    if rec.get("description"):
        summary.insert(0, rec["description"])
    return " | ".join(summary)


__all__ = ["lookup", "describe"]
