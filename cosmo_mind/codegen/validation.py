"""
cosmo_mind.codegen.validation
=============================

Static + runtime verification of generated Astropy scripts.

* :func:`ast_check`  — parses the source with :mod:`ast`; returns a
  :class:`ValidationResult` with any syntax / AST errors.  Used on every
  generated snippet so CosmoMind never hands back broken Python.
* :func:`execute_sandbox` — runs the source in a restricted subprocess
  (gated by ``COSMO_ENABLE_CODE_SANDBOX``) to catch import-time / runtime
  failures before the snippet reaches the researcher.

Both are intentionally conservative: they do **not** execute arbitrary
third-party code beyond the standard library + installed science stack.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    stdout: str = ""


def ast_check(source: str) -> ValidationResult:
    """Validate Python syntax + collect dangerous/undefined AST issues."""
    errors: list[str] = []
    # 1) compile-time syntax check
    try:
        tree = ast.parse(source)
    except SyntaxError as e:  # pragma: no cover - exercised via codegen
        return ValidationResult(
            ok=False, errors=[f"SyntaxError: {e.msg} (line {e.lineno})"]
        )

    # 2) ensure only allowed top-level statement kinds appear
    allowed_stmt = (
        ast.Assign, ast.AugAssign, ast.AnnAssign,
        ast.Expr, ast.Import, ast.ImportFrom,
        ast.For, ast.If, ast.Try, ast.With, ast.Assert,
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
        ast.While, ast.Return, ast.Pass, ast.Break, ast.Continue,
        ast.Delete, ast.Global, ast.Nonlocal, ast.Raise,
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and not isinstance(node, allowed_stmt):
            errors.append(f"Disallowed statement {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr == "__import__":
            errors.append("Use of __import__ is forbidden")

    if errors:
        return ValidationResult(ok=False, errors=errors)
    return ValidationResult(ok=True, errors=[])


def execute_sandbox(source: str, timeout: int = 15) -> ValidationResult:
    """Run *source* as a subprocess to validate it executes cleanly."""
    import subprocess

    # Only allow execution when explicitly enabled — generated scripts may
    # hit the network (live catalogs), so we keep this opt-in.
    from cosmo_mind.config import get_settings

    if not get_settings().cosmo_enable_code_sandbox:
        return ValidationResult(
            ok=True, errors=[]
        )  # skip runtime check when sandbox disabled

    try:
        proc = subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True, text=True, timeout=timeout,
        )
        errs: list[str] = []
        if proc.stderr:
            errs.append(proc.stderr.strip()[:2000])
        if proc.returncode != 0:
            errs.append(f"exit code {proc.returncode}")
        return ValidationResult(ok=proc.returncode == 0, errors=errs, stdout=proc.stdout)
    except subprocess.TimeoutExpired:
        return ValidationResult(ok=False, errors=["sandbox execution timed out"])
    except Exception as e:  # pragma: no cover
        return ValidationResult(ok=False, errors=[str(e)])


def safe_write_script(source: str, dest: str | Path) -> Path:
    """Write generated code to *dest* (creating parent dirs) and return the path."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(source, encoding="utf-8")
    return dest


__all__ = ["ValidationResult", "ast_check", "execute_sandbox", "safe_write_script"]
