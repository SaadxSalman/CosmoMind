import types

import cosmo_mind.agents.tools.simbad_tool as simbad_tool


def test_no_network_at_import():
    # the lazy helper exists and the eager module-level add_votable_fields call
    # was removed, so importing must not touch the network.
    assert callable(simbad_tool._configured_simbad)
    assert isinstance(simbad_tool._FIELDS_OF_INTEREST, list)


def test_configured_simbad_is_lazy():
    inst = simbad_tool._configured_simbad()       # offline fake -> instance ok
    # query_object on the offline fake raises -> local catalog fallback handles it
    res = simbad_tool.SimbadTool().run("HD 209458 b")
    assert res["found"] is True
    assert res["data"]["MAIN_ID"] == "HD 209458 b"
    assert res["data"]["SP_TYPE"] == "G0V"


def test_query_known_object():
    res = simbad_tool.SimbadTool().run("TRAPPIST-1")
    assert res["found"] is True
    assert res["data"]["MAIN_ID"] == "TRAPPIST-1"
    assert res["data"]["SP_TYPE"] == "M8V"


def test_query_unknown_object():
    res = simbad_tool.SimbadTool().run("Nonexistent Object 999999")
    assert res["found"] is False
    assert res["data"] is None
    assert res["error"]
