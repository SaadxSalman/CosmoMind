from cosmo_mind.agents.cosmo_agent import CosmoAgent, AgentAnswer
from cosmo_mind.main import _seed_corpus


def _agent_with_corpus():
    from cosmo_mind.indexing.ingest import IngestionPipeline
    pipe = IngestionPipeline()
    _seed_corpus(pipe, quiet=True)
    agent = CosmoAgent()
    return agent, pipe


def test_agent_object_lookup_returns_catalog_data():
    agent, _ = _agent_with_corpus()
    ans = agent.run("Tell me about HD 209458 b")
    assert isinstance(ans, AgentAnswer)
    assert "HD 209458 b" in ans.text
    assert any("nasa_exoplanet_query" in t["tool"] for t in ans.tool_calls)
    assert len(ans.citations) >= 1
    assert "HD 209458" in ans.citations[0]["text"]


def test_agent_simbad_route_for_star():
    agent, _ = _agent_with_corpus()
    ans = agent.run("What is GW170817?")
    assert "GW170817" in ans.text
    assert any("simbad_query" in t["tool"] for t in ans.tool_calls)


def test_agent_code_generation():
    agent, _ = _agent_with_corpus()
    ans = agent.run("Plot the light curve of Kepler-442b")
    assert "```python" in ans.text
    assert ans.code_valid is True
    assert ans.code_intent == "lightcurve"
    assert "Kepler-442b" in ans.code


def test_agent_uses_local_context_for_general_query():
    agent, _ = _agent_with_corpus()
    ans = agent.run("Tell me about TRAPPIST-1")
    assert "TRAPPIST-1" in ans.text
    assert len(ans.citations) >= 1
