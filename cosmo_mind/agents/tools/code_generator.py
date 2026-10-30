"""
cosmo_mind.agents.tools.code_generator
======================================

Agent tool that wraps the Astropy code generator.  When invoked the agent
can hand a researcher a fully validated Python script that fetches live
catalog data and plots / analyses it with astropy + astroquery.
"""

from __future__ import annotations

from cosmo_mind.codegen.astropy_generator import AstropyCodeGenerator, GenerationResult
from cosmo_mind.llm.chat import ToolSpec


class CodeGeneratorTool:
    name = "code_generate"
    description = (
        "Generate verified, runnable Python code using astropy, astroquery and "
        "numpy/matplotlib to plot light curves, analyse spectra, or compute "
        "orbital parameters. Provide a `request` describing the goal (optionally "
        "with an `object_name` and `context`). Returns the source code, an "
        "explanation and whether it passed static validation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "request": {"type": "string", "description": "Natural-language goal, e.g. 'plot light curve of HD 209458 b'."},
            "object_name": {"type": "string", "description": "Target object name, if known."},
            "context": {"type": "string", "description": "Optional retrieved context to seed the script."},
        },
        "required": ["request"],
    }

    def __init__(self, generator: AstropyCodeGenerator | None = None) -> None:
        self.generator = generator or AstropyCodeGenerator()

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)

    def run(self, request: str, object_name: str | None = None, context: str | None = None) -> dict:
        if object_name and object_name not in request:
            request = f"{request} ({object_name})".strip()
        result: GenerationResult = self.generator.generate(request, context=context)
        return {
            "code": result.code,
            "explanation": result.explanation,
            "intent": result.intent,
            "valid": result.valid,
            "errors": result.errors,
            "object_name": object_name or self._extract_object(request),
            "language": "python",
        }

    @staticmethod
    def _extract_object(request: str) -> str:
        from cosmo_mind.codegen.astropy_generator import extract_object
        return extract_object(request)


__all__ = ["CodeGeneratorTool"]
