from cosmo_mind.agents.tools.nasa_exoplanet_tool import NasaExoplanetTool
from cosmo_mind.agents.tools.vizier_tool import VizierTool


def test_nasa_local_catalog_fallback_kepler():
    r = NasaExoplanetTool().run("Kepler-442 b")
    assert r["found"] is True
    assert r["source"] == "local_catalog"
    assert r["data"][0]["period_days"] == 112.34
    assert r["data"][0]["main_id"] == "Kepler-442 b"


def test_nasa_local_catalog_fallback_hd():
    r = NasaExoplanetTool().run("HD 209458 b")
    assert r["found"] is True
    assert r["data"][0]["mass_mjup"] == 0.69


def test_nasa_unknown_object():
    r = NasaExoplanetTool().run("Nonexistent Planet 9999")
    assert r["found"] is False
    assert r["error"]


def test_vizier_local_catalog_fallback():
    r = VizierTool().run("TRAPPIST-1")
    assert r["found"] is True
    assert r["source"] == "local_catalog"
    assert r["data"][0]["main_id"] == "TRAPPIST-1"


def test_vizier_unknown_object():
    r = VizierTool().run("Nonexistent Object 999999")
    assert r["found"] is False
