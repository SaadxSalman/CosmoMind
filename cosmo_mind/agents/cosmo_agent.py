"""
cosmo_mind.agents.cosmo_agent
=============================

The CosmoMind **agentic orchestrator** — a tool-augmented reasoner that
turns a natural-language astronomy question into a grounded answer.

Loop (max ``max_turns``):
  1. retrieve the most relevant local corpus chunks (hybrid vector + KG).
  2. ask the LLM to act, given the tools it can call.
  3. execute returned tool calls (dedup-guarded so the loop always
     terminates) and feed their results back as ``tool`` messages.
  4. when no more tool calls are needed, synthesize the final answer.

Runnable offline: with no LLM key the :class:`LocalStubLLM` routes queries
to the real astroquery-backed tools, and the agent's own ``_synthesize`` step
turns the collected evidence into a coherent answer.
"""

from __future__ import annotations

import inspect
import json
import textwrap
from dataclasses import dataclass, field
from typing import Any

from cosmo_mind.agents.tools.code_generator import CodeGeneratorTool
from cosmo_mind.agents.tools.nasa_exoplanet_tool import NasaExoplanetTool
from cosmo_mind.agents.tools.simbad_tool import SimbadTool
from cosmo_mind.agents.tools.vizier_tool import VizierTool
from cosmo_mind.llm.chat import ChatMessage, LLMBackend, LocalStubLLM, LLMResult, get_llm
from cosmo_mind.retrieval.hybrid_retriever import HybridRetriever, get_retriever
from cosmo_mind.utils.logging import get_logger

log = get_logger("cosmo_mind.agent")


@dataclass
class AgentAnswer:
    text: str
    citations: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    code: str | None = None
    code_valid: bool = False
    code_intent: str | None = None


class CosmoAgent:
    def __init__(
        self,
        llm: LLMBackend | None = None,
        retriever: HybridRetriever | None = None,
        code_generator=None,
        max_turns: int = 4,
    ) -> None:
        self.llm = llm or get_llm()
        self.retriever = retriever or get_retriever()
        self.max_turns = max_turns
        self.code_generator = code_generator
        self.tools = [
            SimbadTool(),
            VizierTool(),
            NasaExoplanetTool(),
            CodeGeneratorTool(generator=code_generator),
        ]
        self._tool_map = {t.name: t for t in self.tools}

    def run(self, query: str) -> AgentAnswer:
        log.info(f"CosmoAgent query: {query[:80]!r}")
        context = self._gather_context(query)
        specs = [t.spec() for t in self.tools]
        system = self._system_prompt(context)
        messages: list[ChatMessage] = [system, ChatMessage("user", query)]

        seen: set[tuple[str, str]] = set()
        tool_log: list[dict] = []
        final_text: str | None = None
        code_info: dict | None = None

        for _ in range(self.max_turns):
            res: LLMResult = self.llm.chat(messages, tools=specs)
            if res.has_tool_calls:
                made_new = False
                messages.append(
                    ChatMessage("assistant", content=res.text, tool_calls=res.tool_calls)
                )
                for tc in res.tool_calls:
                    key = (tc.name, _canonical_args(tc.arguments))
                    if key in seen:
                        messages.append(
                            ChatMessage(
                                "tool", tool_call_id=tc.id, name=tc.name,
                                content=json.dumps({"note": "result already provided above"}),
                            )
                        )
                        continue
                    seen.add(key)
                    made_new = True
                    result = self.dispatch(tc.name, tc.arguments)
                    tool_log.append({"tool": tc.name, "args": tc.arguments, "result": result})
                    if tc.name == "code_generate":
                        code_info = result
                    messages.append(
                        ChatMessage(
                            "tool", tool_call_id=tc.id, name=tc.name,
                            content=_stringify_result(tc.name, result),
                        )
                    )
                if not made_new:
                    final_text = self._synthesize(query, context, tool_log, code_info)
                    break
                continue
            final_text = res.text or ""
            break

        if final_text is None:
            final_text = self._synthesize(query, context, tool_log, code_info)

        # In LOCAL mode the stub LLM's free-text answers are canned fallbacks;
        # always recompose a context- and evidence-grounded answer instead.
        if isinstance(self.llm, LocalStubLLM):
            final_text = self._synthesize(query, context, tool_log, code_info)

        citations = [
            {"chunk_id": c.get("chunk_id"),
             "score": round(c.get("vector_score", 0.0), 4),
             "text": (c.get("text") or "")[:300]}
            for c in context[:3]
        ]
        return AgentAnswer(
            text=final_text.strip(),
            citations=citations,
            tool_calls=tool_log,
            code=code_info["code"] if code_info else None,
            code_valid=code_info["valid"] if code_info else False,
            code_intent=code_info.get("intent") if code_info else None,
        )

    def dispatch(self, name: str, args: dict) -> dict:
        tool = self._tool_map.get(name)
        if tool is None:
            return {"error": f"unknown tool {name}"}
        try:
            sig = inspect.signature(tool.run)
            filtered = {k: v for k, v in args.items() if k in sig.parameters}
            return tool.run(**filtered)  # type: ignore[arg-type]
        except Exception as e:
            log.exception(f"tool {name} failed")
            return {"tool": name, "error": str(e), "found": False}

    def _gather_context(self, query: str) -> list[dict]:
        try:
            return self.retriever.retrieve(query, top_k=4) or []
        except Exception as e:
            log.warning(f"retrieval failed: {e}")
            return []

    def _system_prompt(self, context: list[dict]) -> ChatMessage:
        ctx_text = "\n---\n".join([c.get("text", "") for c in context][:4])
        prompt = textwrap.dedent(
            f"""
            You are CosmoMind, an autonomous research co-pilot for
            astrophysics. Answer the user's question by (a) consulting the
            retrieved literature context below and (b) calling available tools
            - SIMBAD / VizieR / NASA Exoplanet lookups and astropy code
            generation - whenever the question mentions a celestial object or
            asks for a plot / script. Cite which literature chunks informed
            your answer.

            Retrieved local context:
            {ctx_text or '(no local corpus chunks retrieved)'}
            """
        ).strip()
        return ChatMessage("system", content=prompt)

    def _synthesize(
        self, query: str, context: list[dict], tool_log: list[dict], code_info: dict | None
    ) -> str:
        """Build a final answer from context + executed tool results.

        Used both as the final LLM answer pass-through and as the fully
        deterministic LOCAL-mode answer when the stub LLM cannot resolve the
        query itself.
        """
        parts: list[str] = []
        if context:
            quotes = []
            for c in context[:3]:
                txt = (c.get("text") or "").strip()
                if txt:
                    quotes.append(f"- {txt[:180]}")
            parts.append("Based on the retrieved literature:\n" + "\n".join(quotes))
        for entry in tool_log:
            tool, result = entry["tool"], entry["result"]
            if "simbad_query" in tool:
                if result.get("found"):
                    d = result.get("data") or {}
                    parts.append(
                        f"SIMBAD -> {result.get('query')}: {d.get('MAIN_ID')} "
                        f"RA={d.get('RA')} Dec={d.get('DEC')} SpType={d.get('SP_TYPE')}."
                    )
                else:
                    parts.append(
                        f"SIMBAD lookup for '{result.get('query')}' was unavailable "
                        f"({result.get('error')})."
                    )
            elif "nasa_exoplanet_query" in tool:
                if result.get("found") and result.get("data"):
                    d = result["data"][0]
                    pick = {k: d.get(k) for k in
                            ("pl_orbper", "pl_rade", "pl_bmassj", "st_teff",
                             "period_days", "mass_mjup", "radius_rjup",
                             "eq_temp_k", "incident_flux", "spectral_type",
                             "distance_pc", "main_id")}
                    pick = {k: v for k, v in pick.items() if v is not None}
                    parts.append(f"NASA Exoplanet Archive -> {result.get('query')}: {pick}")
                else:
                    parts.append("NASA Exoplanet lookup unavailable.")
            elif "vizier_query" in tool:
                if result.get("found"):
                    n = len(result.get("data") or [])
                    parts.append(f"VizieR -> {result.get('query')}: returned {n} rows.")
                else:
                    parts.append("VizieR lookup unavailable.")
            elif "code_generate" in tool:
                if result.get("valid"):
                    parts.append(
                        f"Here is verified Astropy code (intent={result.get('intent')}):\n"
                        f"```python\n{result.get('code','')}\n```"
                    )
                else:
                    parts.append(f"Generated code had issues: {result.get('errors')}")
        if not parts:
            parts.append(
                "I could not resolve that query. Try a known identifier such as "
                "'HD 209458 b', 'Kepler-442 b', or a coordinate."
            )
        return "\n\n".join(parts)


def _canonical_args(args: dict) -> str:
    return json.dumps(args, sort_keys=True, default=str)


def _stringify_result(name: str, result: dict) -> str:
    try:
        return json.dumps(result, default=str, ensure_ascii=False)[:1500]
    except Exception:
        return str(result)[:1500]


__all__ = ["AgentAnswer", "CosmoAgent"]