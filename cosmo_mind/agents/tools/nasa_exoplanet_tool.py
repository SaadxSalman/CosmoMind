"""
cosmo_mind.agents.tools.nasa_exoplanet_tool
===========================================

Tool: query the **NASA Exoplanet Archive** via ``astroquery.nasa_exoplanet``.

Used by the agent for confirmed/known exoplanet systems (e.g.
``HD 209458 b``, ``Kepler-442 b``, ``TOI-700 b``) to retrieve physical
parameters — mass, radius, orbital period, equilibrium temperature, host
star properties — straight from NASA's archive.
"""

from __future__ import annotations

from typing import Any

from cosmo_mind.llm.chat import ToolSpec
from cosmo_mind.agents.tools.local_catalog import lookup as _local_lookup

try:
    from astroquery.nasa_exoplanet import NasaExoplanet
except Exception:  # pragma: no cover - astroquery optional at import
    NasaExoplanet = None  # type: ignore[assignment]


class NasaExoplanetTool:
    name = "nasa_exoplanet_query"
    description = (
        "Query the NASA Exoplanet Archive for an exoplanet system. Provide the "
        "object_name of an exoplanet (e.g. 'HD 209458 b', 'Kepler-442 b', "
        "'TOI-700 b'). Returns mass, radius, period, equilibrium temperature, "
        "host star properties and discovery reference."
    )
    parameters = {
        "type": "object",
        "properties": {
            "object_name": {"type": "string", "description": "Exoplanet designation."},
            "table": {"type": "string", "description": "Archive table. Default 'pscomppars'."},
            "select": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of columns to return.",
            },
        },
        "required": ["object_name"],
    }

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        try:
            for col in row.colnames:
                try:
                    out[col] = float(row[col])
                except (TypeError, ValueError):
                    out[col] = str(row[col])
        except Exception:
            out = {"raw": str(row)}
        return out

    def run(self, object_name: str, table: str = "pscomppars",
            select: list[str] | None = None) -> dict[str, Any]:
        records: list[dict[str, Any]] | None = None
        err: str | None = None
        if NasaExoplanet is None:
            err = "astroquery.nasa_exoplanet not installed"
        else:
            try:
                nea = NasaExoplanet()
                result = nea.query_object(object_name)
                if result is not None:
                    try:
                        records = [self._row_to_dict(r) for r in result]
                    except TypeError:
                        records = [self._row_to_dict(result)]
            except Exception as e:
                err = str(e)
        if records:
            return {"query": object_name, "service": "nasa_exoplanet", "found": True,
                    "table": table, "data": records, "error": None}
        # graceful degradation to the curated local catalogue (offline / blocked)
        rec = _local_lookup(object_name)
        if rec:
            return {"query": object_name, "service": "nasa_exoplanet",
                    "source": "local_catalog", "found": True,
                    "table": table, "data": [rec], "error": None}
        return {"query": object_name, "service": "nasa_exoplanet", "found": False,
                "data": None, "error": err or "no result"}


__all__ = ["NasaExoplanetTool"]
