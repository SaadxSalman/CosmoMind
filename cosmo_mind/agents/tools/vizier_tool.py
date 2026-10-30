"""
cosmo_mind.agents.tools.vizier_tool
===================================

Tool: query the **VizieR** catalogue service via ``astroquery``.

VizieR hosts thousands of astronomical catalogues.  This tool resolves a
named object to coordinates (via its SIMBAD-backed resolver inside Astropy)
and returns matching catalogue rows, optionally restricted to a catalogue
identifier (e.g. ``Gaia`` or ``TIC``).
"""

from __future__ import annotations

from typing import Any

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

from cosmo_mind.llm.chat import ToolSpec
from cosmo_mind.agents.tools.local_catalog import lookup as _local_lookup


class VizierTool:
    name = "vizier_query"
    description = (
        "Query the VizieR catalogue service for an astronomical object. Provide "
        "object_name (e.g. 'HD 189733 b', 'TRAPPIST-1') and optionally a catalog "
        "filter (e.g. 'Gaia','TIC','2MASS'). Returns matching catalogue rows with "
        "RA, Dec, magnitudes and identifiers."
    )
    parameters = {
        "type": "object",
        "properties": {
            "object_name": {"type": "string", "description": "Target object name."},
            "catalog": {"type": "string", "description": "Optional catalogue name to filter."},
            "radius": {"type": "string", "description": "Search radius. Default '0d5'."},
            "magnitude_limit": {"type": "number", "description": "Optional Vmag limit."},
        },
        "required": ["object_name"],
    }

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)

    @staticmethod
    def _table_to_records(table: Any) -> list[dict]:
        records: list[dict] = []
        try:
            colnames = list(table.colnames)
            for row in table:
                rec = {}
                for c in colnames:
                    val = row[c]
                    try:
                        rec[c] = float(val)
                    except (TypeError, ValueError):
                        rec[c] = str(val)
                records.append(rec)
        except Exception:
            records.append({"raw": str(table)})
        return records

    def run(self, object_name: str, catalog: str | None = None,
            radius: str = "0d5", magnitude_limit: float | None = None) -> dict[str, Any]:
        try:
            Vizier.httptimeout = 30
            Vizier.query_constraints = None
            if magnitude_limit is not None:
                Vizier.vizier_columns = {catalog or "*": ["*"]}
            tables = Vizier.query_object(object_name, catalog=catalog) if catalog \
                else Vizier.query_object(object_name)
            if tables is None:
                # fall back to a coordinate-region search
                try:
                    coord = self._resolve_coord(object_name)
                    if coord is not None:
                        tables = Vizier.query_region(
                            coord, radius=u.Quantity(radius), catalog=catalog
                        )
                except Exception:
                    tables = None
            if tables is None or len(tables) == 0:
                return {"query": object_name, "service": "vizier", "found": False,
                        "data": None, "error": "no result"}
            records = []
            meta = []
            for t in tables:
                records.extend(self._table_to_records(t))
                meta.append(t.meta.get("name", "unknown") if hasattr(t, "meta") else "unknown")
            return {"query": object_name, "service": "vizier", "found": True,
                    "catalogs": meta, "data": records[:200], "error": None}
        except Exception as e:
            rec = _local_lookup(object_name)
            if rec:
                return {"query": object_name, "service": "vizier", "source": "local_catalog",
                        "found": True,
                        "catalogs": ["local_catalog"],
                        "data": [{
                            "main_id": rec.get("main_id"),
                            "ra": rec.get("ra"), "dec": rec.get("dec"),
                            "spectral_type": rec.get("spectral_type"),
                            "description": rec.get("description"),
                        }],
                        "error": None}
            return {"query": object_name, "service": "vizier", "found": False,
                    "data": None, "error": str(e)}

    @staticmethod
    def _resolve_coord(name: str) -> SkyCoord | None:
        try:
            return SkyCoord.from_name(name)
        except Exception:
            return None


__all__ = ["VizierTool"]
