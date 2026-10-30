"""
Pytest configuration for CosmoMind.

* Forces ``COSMO_MODE=LOCAL`` and a throwaway on-disk cache so tests never
  touch the real .env / data directory.
* Patches the astroquery classes used by the tools with offline fakes, so the
  tools deterministically exercise their **local-catalogue fallback** path
  (no network, no flakiness).
* Resets the LOCAL singletons (in-memory vector store + NetworkX KG) between
  tests for isolation.
"""
import os
import shutil
import tempfile

_tmp = tempfile.mkdtemp(prefix="cosmomind_test_")
os.environ["COSMO_MODE"] = "LOCAL"
os.environ["COSMO_DATA_DIR"] = _tmp
os.environ["COSMO_VECTOR_CACHE"] = os.path.join(_tmp, "vector_cache")
os.environ["COSMO_KG_CACHE"] = os.path.join(_tmp, "kg_cache")
for _k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NASA_API_KEY"):
    os.environ.pop(_k, None)

import pytest  # noqa: E402

import cosmo_mind.agents.tools.simbad_tool as _simbad_tool  # noqa: E402
import cosmo_mind.agents.tools.nasa_exoplanet_tool as _nasa_tool  # noqa: E402
import cosmo_mind.agents.tools.vizier_tool as _vizier_tool  # noqa: E402
from cosmo_mind.config import reload_settings  # noqa: E402
from cosmo_mind.indexing.vector_store import reset_local_vector_store  # noqa: E402
from cosmo_mind.indexing.knowledge_graph import reset_local_knowledge_graph  # noqa: E402


class _OfflineSimbad:
    httptimeout = 30

    def add_votable_fields(self, *a, **k):
        raise RuntimeError("offline")

    def query_object(self, name, *a, **k):
        raise ConnectionError("SIMBAD unreachable")

    def query_region(self, *a, **k):
        raise ConnectionError("SIMBAD unreachable")


class _OfflineNasa:
    def query_object(self, name, *a, **k):
        raise ConnectionError("NASA Exoplanet unreachable")


class _OfflineVizier:
    httptimeout = 30
    query_constraints = None
    vizier_columns = {}

    @classmethod
    def query_object(cls, name, *a, **k):
        raise ConnectionError("VizieR unreachable")

    @classmethod
    def query_region(cls, *a, **k):
        raise ConnectionError("VizieR unreachable")


@pytest.fixture(autouse=True)
def _cosmo_isolation(monkeypatch):
    reload_settings()
    reset_local_vector_store()
    reset_local_knowledge_graph()
    for d in (os.environ["COSMO_VECTOR_CACHE"], os.environ["COSMO_KG_CACHE"]):
        if os.path.isdir(d):
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))
    monkeypatch.setattr(_simbad_tool, "Simbad", _OfflineSimbad, raising=True)
    monkeypatch.setattr(_nasa_tool, "NasaExoplanet", _OfflineNasa, raising=True)
    monkeypatch.setattr(_vizier_tool, "Vizier", _OfflineVizier, raising=True)
    yield


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_tmp, ignore_errors=True)
