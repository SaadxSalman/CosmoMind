from cosmo_mind.codegen.astropy_generator import (
    AstropyCodeGenerator,
    extract_object,
    classify_intent,
)


def test_extract_object():
    assert extract_object("Plot the light curve of Kepler-442b") == "Kepler-442b"
    assert extract_object("show me the spectrum of Betelgeuse") == "Betelgeuse"
    assert extract_object("orbit of TRAPPIST-1") == "TRAPPIST-1"
    assert extract_object("what is HD 209458 b") == "HD 209458 b"


def test_classify_intent():
    assert classify_intent("plot the light curve of X") == "lightcurve"
    assert classify_intent("show me a spectrum") == "spectrum"
    assert classify_intent("the orbital period") == "orbit"
    assert classify_intent("what is HD 209458 b") == "lookup"


def test_lightcurve_codegen():
    res = AstropyCodeGenerator().generate("Plot the light curve of Kepler-442b")
    assert res.intent == "lightcurve"
    assert res.valid is True
    assert "Kepler-442b" in res.code
    assert "matplotlib" in res.code


def test_spectrum_codegen():
    res = AstropyCodeGenerator().generate("Show me the spectrum of Betelgeuse")
    assert res.intent == "spectrum"
    assert res.valid is True
    assert "Betelgeuse" in res.code
    assert "Simbad" in res.code


def test_orbit_codegen():
    res = AstropyCodeGenerator().generate("orbit of TRAPPIST-1")
    assert res.intent == "orbit"
    assert res.valid is True
    assert "TRAPPIST-1" in res.code


def test_lookup_codegen():
    res = AstropyCodeGenerator().generate("what is HD 209458 b")
    assert res.intent == "lookup"
    assert res.valid is True
    assert "HD 209458 b" in res.code
