from cosmo_mind.agents.tools.local_catalog import lookup, describe


def test_lookup_normalized_variants():
    for name in [
        "HD 209458 b", "HD209458b", "hd 209458 b", "HD-209458-b",
        "Kepler-442b", "Kepler 442 b", "Kepler-442 b",
        "TRAPPIST-1", "TRAPPIST 1", "TRAPPIST-1b",
        "Betelgeuse", "GW170817", "HD 209458", "Proxima Centauri", "Rigel",
    ]:
        rec = lookup(name)
        assert rec is not None, f"lookup failed for {name!r}"
        assert "main_id" in rec and "object_type" in rec


def test_lookup_distinguishes_star_and_planet():
    assert lookup("HD 209458 b")["object_type"] == "exoplanet"
    assert lookup("HD 209458")["object_type"] == "star"
    assert lookup("TRAPPIST-1")["object_type"] == "star"
    assert lookup("Kepler-442b")["main_id"] == "Kepler-442 b"


def test_lookup_unknown_returns_none():
    assert lookup("Nonexistent Object 999999") is None
    assert lookup("") is None


def test_describe_returns_summary():
    d = describe("Betelgeuse")
    assert d is not None and "Betelgeuse" in d
