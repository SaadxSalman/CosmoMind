"""
cosmo_mind.agents.tools.simbad_tool
===================================

Tool: query the **SIMBAD** astronomical database via ``astroquery``.

Returns a structured record for a named object — right ascension, declination,
proper motion, spectral type, photometry and the list of alternate catalog
identifiers.  This is the primary "what is this object?" lookup used by the
CosmoMind agent when it encounters a stellar / galactic identifier.
"""

from __future__ import annotations

from typing import Any

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad

from cosmo_mind.llm.chat import ToolSpec
from cosmo_mind.utils.logging import get_logger

log = get_logger("cosmo_mind.agents.tools.simbad")
from cosmo_mind.agents.tools.local_catalog import lookup as _local_lookup


def _configured_simbad() -> Simbad:
    """Return a Simbad instance enriched with the fields CosmoMind needs.

    Field enrichment is performed *lazily* and inside ``try/except`` so that
    a transient network failure against SIMBAD's TAP capabilities endpoint
    (e.g. when working fully offline) never breaks the import or the query.
    """
    sim = Simbad()
    sim.httptimeout = 30
    try:
        sim.add_votable_fields(
            "ra(d)", "dec(d)", "pmra", "pmdec", "sp_type", "flux(V)", "flux(K)",
            "distance_result", "z_value", "ids",
        )
    except Exception as e:  # offline / TAP unreachable
        log.debug(f"SIMBAD field enrichment skipped: {e}")
    return sim


_FIELDS_OF_INTEREST = [
    "RA", "DEC", "RAJ2000", "DECJ2000", "PMRA", "PMDEC", "SP_TYPE", "V", "K",
    "DISTANCE_RESULT", "Z_VALUE", "IDS", "MAIN_ID", "RA_PREC", "DEC_PREC",
]


class SimbadTool:
    """Query SIMBAD by object name or coordinate region."""

    name = "simbad_query"
    description = (
        "Query the SIMBAD database for an astronomical object. Accepts either "
        "a named object (object_name, e.g. 'HD 189733 b') or a coordinate pair "
        "with by_coord=true (object_name then reads as 'HHhMMmSSs +dd°MMmSSs'). "
        "Returns RA/Dec, spectral type, proper motion, distance, redshift and "
        "all catalog cross-identifications."
    )
    parameters = {
        "type": "object",
        "properties": {
            "object_name": {
                "type": "string",
                "description": "Name or coordinate string of the target.",
            },
            "by_coord": {"type": "boolean", "description": "Interpret as a coordinate."},
            "radius": {"type": "string", "description": "Search radius, e.g. '0d5'. Default '0d5'."},
        },
        "required": ["object_name"],
    }

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)

    def _row_to_dict(self, row) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in _FIELDS_OF_INTEREST:
            if f in row.colnames:
                val = row[f]
                # astropy masked values render as masked
                try:
                    val = val.value if hasattr(val, "value") else val
                except Exception:
                    pass
                out[f] = str(val)
        return out

    def run(self, object_name: str, by_coord: bool = False, radius: str = "0d5") -> dict[str, Any]:
        sim = _configured_simbad()
        try:
            if by_coord:
                coord = self._parse_coord(object_name)
                if coord is None:
                    raise ValueError(f"unparsable coordinate: {object_name}")
                radius_u = self._parse_radius(radius)
                rows = sim.query_region(coord, radius=radius_u)
            else:
                rows = sim.query_object(object_name)
            if rows is None or len(rows) == 0:
                return {"query": object_name, "service": "simbad", "found": False,
                        "data": None, "error": "no result"}
            data = self._row_to_dict(rows[0])
            data["main_id"] = str(rows[0].field("MAIN_ID")) if "MAIN_ID" in rows[0].colnames else data.get("MAIN_ID")
            return {"query": object_name, "service": "simbad", "found": True, "data": data, "error": None}
        except Exception as e:
            rec = _local_lookup(object_name)
            if rec:
                data = {
                    "MAIN_ID": rec.get("main_id", object_name),
                    "RA": round(float(rec.get("ra")), 5) if rec.get("ra") else None,
                    "DEC": round(float(rec.get("dec")), 5) if rec.get("dec") else None,
                    "SP_TYPE": rec.get("spectral_type"),
                    "DISTANCE_PC": rec.get("distance_pc"),
                    "DESCRIPTION": rec.get("description"),
                }
                return {"query": object_name, "service": "simbad",
                        "source": "local_catalog", "found": True, "data": data, "error": None}
            return {"query": object_name, "service": "simbad", "found": False,
                    "data": None, "error": str(e)}

    @staticmethod
    def _parse_coord(s: str) -> SkyCoord | None:
        try:
            return SkyCoord(s, unit=(u.hourangle, u.deg))
        except Exception:
            return None

    @staticmethod
    def _parse_radius(s: str):
        try:
            return u.Quantity(s)
        except Exception:
            try:
                return u.Quantity(float(s), u.deg)
            except Exception:
                return 0.5 * u.deg


__all__ = ["SimbadTool"]
