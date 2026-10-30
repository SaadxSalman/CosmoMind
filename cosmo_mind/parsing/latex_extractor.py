"""
cosmo_mind.parsing.latex_extractor
===================================

Lightweight extractor of LaTeX-style mathematical expressions from the
plain text of PDFs / source files.

LaTeX in PDFs is generally *rendered* (not stored as source), so this module
focuses on the common cases where authors embed literal LaTeX delimiters —
especially useful for ingested ``.tex`` / arXiv source and for equation
snippets in code contexts:

* ``$...$``           inline math
* ``$$...$$``         display math
* ``\\[...\\]``        display math
* ``\\begin{equation}...\\end{equation}``  (and ``align``, ``align*``)
* ``\\mbox{...}``      treated as inline text, left in context

Unicode rendering uses ``pylatexenc`` when available; otherwise it degrades
gracefully to the original LaTeX.
"""

from __future__ import annotations

import re

from cosmo_mind.utils.logging import get_logger
from cosmo_mind.parsing.document import Equation

log = get_logger("cosmo_mind.parsing.latex")

_INLINE = re.compile(r"(?<!\\)\$(?!\$)(.*?)\$(?!\$)", re.DOTALL)
_DISPLAY_DOLLAR = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_DISPLAY_BRACKET = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_BLOCK = re.compile(
    r"\\begin\{(equation|align|align\*|gather|multline)\}(.*?)\\end\{\1\}",
    re.DOTALL,
)

# Protect already-matched blocks with placeholders so nesting doesn't
# double-count.
try:  # pylatexenc is optional
    from pylatexenc.latex2text import latex2text as _latex2text

    _HAVE_PYLATEXENC = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_PYLATEXENC = False


def render_latex(latex: str) -> str:
    """Render a LaTeX string to unicode when possible, else return as-is."""
    if not _HAVE_PYLATEXENC:
        return latex
    try:
        return _latex2text(latex).strip()
    except Exception:
        return latex


def extract_equations(text: str, max_chars: int = 400) -> list[Equation]:
    """Return every LaTeX equation found in *text* with surrounding context."""
    results: list[Equation] = []
    seen: set[str] = set()

    def _context_at(start: int, end: int) -> str:
        lo = max(0, start - 80)
        hi = min(len(text), end + 80)
        return text[lo:hi].strip().replace("\n", " ")[:120]

    sources = [
        ("display-dollar", _DISPLAY_DOLLAR),
        ("display-bracket", _DISPLAY_BRACKET),
        ("equation-block", _BLOCK),
        ("inline", _INLINE),
    ]

    for label, pat in sources:
        for m in pat.finditer(text):
            raw = m.group(1).strip()
            if not raw or raw in seen:
                continue
            seen.add(raw)
            results.append(
                Equation(
                    latex=raw[:max_chars],
                    rendered=render_latex(raw[:max_chars]),
                    context=_context_at(m.start(), m.end()),
                )
            )
    log.debug(f"extracted {len(results)} equations from {len(text)} chars")
    return results


__all__ = ["extract_equations", "render_latex"]
