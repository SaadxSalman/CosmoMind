"""
cosmo_mind.llm.chat
===================

Unified, framework-light LLM abstraction with a **deterministic offline
stub**, so the entire agentic loop is runnable with zero configuration in
``LOCAL`` mode.

Public surface
--------------
* :class:`ChatMessage`     — a single message in a conversation.
* :class:`ToolCall`        — a requested tool invocation.
* :class:`LLMResult`       — what the LLM returns (text + optional tool calls).
* :class:`ToolSpec`        — static description of a tool (name + JSON schema).
* :class:`LLMBackend`      — abstract interface.
* :class:`OpenAIChat`      — real OpenAI chat/completions (PROD).
* :class:`AnthropicChat`   — real Anthropic (PROD, optional).
* :class:`LocalStubLLM`    — deterministic rule-based stub (LOCAL / tests).
* :func:`get_llm`          — factory driven by settings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from cosmo_mind.config import get_settings
from cosmo_mind.utils.logging import get_logger

log = get_logger("cosmo_mind.llm")


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = "call-0"


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_openai(self) -> dict:
        msg: dict = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg


@dataclass
class ToolSpec:
    """Static description of a tool the LLM can call."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic(self) -> dict:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class LLMResult:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMBackend:
    """Abstract interface every LLM backend implements."""

    model: str = "stub"

    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        raise NotImplementedError


class OpenAIChat(LLMBackend):
    """Real OpenAI chat/completions backend (PROD)."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from openai import OpenAI

        self._model = model or get_settings().openai_model
        self._client = OpenAI(api_key=api_key or get_settings().openai_api_key)

    def chat(self, messages, tools=None) -> "LLMResult":  # type: ignore[override]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_openai() for m in messages],
            "max_tokens": get_settings().openai_max_tokens,
            "temperature": get_settings().openai_temperature,
        }
        if tools:
            kwargs["tools"] = [t.to_openai() for t in tools]
            kwargs["tool_choice"] = "auto"
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        return LLMResult(text=msg.content, tool_calls=_openai_tool_calls(msg))


def _openai_tool_calls(msg) -> list[ToolCall]:
    tcs: list[ToolCall] = []
    if getattr(msg, "tool_calls", None):
        for i, tc in enumerate(msg.tool_calls):
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {"raw": tc.function.arguments or {}}
            tcs.append(
                ToolCall(name=tc.function.name, arguments=args, id=tc.id or f"call-{i}")
            )
    return tcs


class AnthropicChat(LLMBackend):
    """Real Anthropic backend (PROD, optional)."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        import anthropic

        self._model = model or get_settings().anthropic_model
        self._client = anthropic.Anthropic(
            api_key=api_key or get_settings().anthropic_api_key
        )

    def chat(self, messages, tools=None) -> "LLMResult":  # type: ignore[override]
        am: list[dict] = []
        for m in messages:
            if m.role == "system":
                continue
            content: list | str = m.content or ""
            if m.role == "assistant" and m.tool_calls:
                content = [{"type": "text", "text": m.content or ""}] + [
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                    for tc in m.tool_calls
                ]
            am.append(
                {"role": m.role if m.role != "system" else "user", "content": content}
            )
        kwargs: dict[str, Any] = {"model": self._model, "messages": am, "max_tokens": 4000}
        if tools:
            kwargs["tools"] = [t.to_anthropic() for t in tools]
        resp = self._client.messages.create(**kwargs)
        tc_map: dict[str, ToolCall] = {}
        text_parts: list[str] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tc_map[block.id] = ToolCall(
                    name=block.name, arguments=dict(block.input), id=block.id
                )
        return LLMResult(text="".join(text_parts) or None, tool_calls=list(tc_map.values()))


_EXO_RE = re.compile(
    r"\b(?:HD\s*\d+\s*[b-z]|Kepler-\d+\s*[b-z]?|KOI-\d+|TOI-\d+\s*[b-z]?|"
    r"WASP-\d+\s*[b-z]?)\b",
    re.I,
)
# broad object recognition for the LOCAL stub (stars, transients, coords)
_OBJ_RE = re.compile(
    r"\b(?:HD\s*\d+\s*[b-z]?|Kepler-\d+\s*[b-z]?|KOI-\d+|TOI-\d+\s*[b-z]?|"
    r"WASP-\d+\s*[b-z]?|TRAPPIST-1(?:\s*[b-z])?|GJ\s*\d+|Wolf\s*\d+|"
    r"Proxima\s+Centauri|Alpha\s+Centauri|Betelgeuse|Rigel|Sirius|Procyon|"
    r"Arcturus|Aldebaran|Vega|Capella|Polaris|GW\d{6,}|Cygnus\s+X-1)\b",
    re.I,
)
_HD_RE = re.compile(r"\bHD\s*(\d{1,6})\b", re.I)
_COORD_RE = re.compile(
    r"([0-9]{1,2}h[0-9]{2}m[0-9.]+s)\s*([+-][0-9]+d[0-9]{2}m[0-9.]+s)", re.I
)


class LocalStubLLM(LLMBackend):
    """
    Deterministic LLM fallback that pattern-matches astronomy queries to the
    correct tool call and otherwise returns a canned synthesis.  It genuinely
    exercises the SIMBAD / VizieR / NASA / code-generation tool paths so the
    whole pipeline produces real results in LOCAL mode.
    """

    model = "local-stub"

    def __init__(self) -> None:
        self._call_counter = 0

    def _next_id(self) -> str:
        self._call_counter += 1
        return f"call-{self._call_counter}"

    @staticmethod
    def _last_user(messages: list["ChatMessage"]) -> str:
        for m in reversed(messages):
            if m.role == "user" and m.content:
                return m.content
        return ""

    def chat(self, messages, tools=None) -> "LLMResult":  # type: ignore[override]
        query = self._last_user(messages)
        available = {t.name for t in (tools or [])}

        # code-generation intent (checked before lookup so "plot X" produces a
        # script rather than an endless object-lookup loop)
        if "code_generate" in available and re.search(
            r"plot|light.?curve|spectrum|code|astropy|script|orbit", query, re.I
        ):
            obj = _OBJ_RE.search(query) or _EXO_RE.search(query) or _HD_RE.search(query)
            args = {"request": query.strip()}
            if obj:
                args["object_name"] = obj.group(0).strip()
            return LLMResult(
                text=None,
                tool_calls=[ToolCall("code_generate", args, self._next_id())],
            )

        # object-lookup intent (exoplanet via NASA, otherwise SIMBAD)
        obj_match = _EXO_RE.search(query) or _OBJ_RE.search(query) or _HD_RE.search(query)
        if "simbad_query" in available and obj_match:
            obj_name = obj_match.group(0).strip()
            if "nasa_exoplanet_query" in available and _EXO_RE.search(query):
                return LLMResult(
                    text=None,
                    tool_calls=[
                        ToolCall(
                            "nasa_exoplanet_query",
                            {"object_name": obj_name},
                            self._next_id(),
                        )
                    ],
                )
            return LLMResult(
                text=None,
                tool_calls=[
                    ToolCall("simbad_query", {"object_name": obj_name}, self._next_id())
                ],
            )

        # coordinate lookup intent
        if "simbad_query" in available and _COORD_RE.search(query):
            coord = _COORD_RE.search(query).group(0).strip()
            return LLMResult(
                text=None,
                tool_calls=[
                    ToolCall(
                        "simbad_query",
                        {"object_name": coord, "by_coord": True},
                        self._next_id(),
                    )
                ],
            )

        # fallback synthesis
        return LLMResult(text=self._synthesize(query))

    @staticmethod
    def _synthesize(query: str) -> str:
        base = (
            "CosmoMind (LOCAL stub LLM, offline mode): I could not resolve "
            f"'{query.strip()[:80]}' to a tool-using intent. "
        )
        hints = [
            "To query a catalog object, mention an identifier such as "
            "'HD 209458 b', 'Kepler-442 b', or a coordinate.",
            "To generate Astropy/astroquery code, include a verb like "
            "'plot', 'light curve', or 'script'.",
            "For full reasoning, set OPENAI_API_KEY in .env and use COSMO_MODE=PROD.",
        ]
        return base + " ".join(hints)


def get_llm() -> LLMBackend:
    """Factory selecting the LLM backend based on settings."""
    s = get_settings()
    if s.is_prod and s.use_openai:
        return OpenAIChat(model=s.openai_model, api_key=s.openai_api_key)
    if s.is_prod and s.use_anthropic:
        return AnthropicChat(model=s.anthropic_model, api_key=s.anthropic_api_key)
    return LocalStubLLM()


__all__ = [
    "ChatMessage",
    "ToolCall",
    "ToolSpec",
    "LLMResult",
    "LLMBackend",
    "OpenAIChat",
    "AnthropicChat",
    "LocalStubLLM",
    "get_llm",
]
