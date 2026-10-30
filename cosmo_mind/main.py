"""
cosmo_mind.main
===============

Command-line interface for CosmoMind.

Subcommands: check | seed | ingest | query | demo | serve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cosmo_mind.agents.cosmo_agent import CosmoAgent
from cosmo_mind.config import reload_settings
from cosmo_mind.indexing.ingest import IngestionPipeline
from cosmo_mind.parsing.document import Document
from cosmo_mind.utils.logging import get_logger

log = get_logger("cosmo_mind.main")

SAMPLE_QUERIES = [
    "Tell me about HD 209458 b",
    "Plot the light curve of Kepler-442b",
    "What instruments observed TRAPPIST-1?",
    "Show me the spectrum of Betelgeuse",
    "What is GW170817?",
    "Explain the Hubble tension",
]


def _seed_corpus(pipeline: IngestionPipeline | None = None, quiet: bool = False) -> IngestionPipeline:
    from cosmo_mind.data import all_documents

    pipe = pipeline or IngestionPipeline()
    n = 0
    for entry in all_documents():
        doc = Document(
            doc_id=entry["source"],
            source=entry["source"],
            title=entry["title"],
            doc_type=entry["doc_type"],
            full_text=entry["text"],
        )
        pipe.ingest_document(doc)
        n += 1
    if not quiet:
        print(f"Seeded {n} sample documents ({pipe.vs.count()} chunks indexed).")
    return pipe


def _format_answer(ans) -> str:
    sep = "=" * 72
    out = [sep, "CosmoMind answer", sep, ans.text, ""]
    if ans.citations:
        out.append("Citations (retrieved literature):")
        for c in ans.citations:
            out.append(f"  - [{c['score']:.3f}] {c['chunk_id']}")
            out.append(f"    {c['text']}")
    if ans.tool_calls:
        out.append("\nTool calls performed:")
        for tc in ans.tool_calls:
            out.append(f"  * {tc['tool']}({json.dumps(tc['args'], default=str)[:120]})")
    if ans.code and "```python" not in ans.text:
        mark = "```python" if ans.code_valid else "```python (validation issues)"
        out.append(f"\nGenerated code {mark}\n{ans.code}\n```")
    return "\n".join(out)


def cmd_check(args) -> int:
    from cosmo_mind.config import get_settings
    from cosmo_mind.llm.chat import get_llm
    from cosmo_mind.indexing.embeddings import get_embedder
    from cosmo_mind.indexing.vector_store import get_vector_store
    from cosmo_mind.indexing.knowledge_graph import get_knowledge_graph

    s = get_settings()
    print(json.dumps(s.summary(), indent=2))
    print("LLM backend   :", get_llm().model)
    print("Embedder      :", type(get_embedder()).__name__)
    print("Vector store  :", type(get_vector_store()).__name__)
    print("Knowledge graph:", type(get_knowledge_graph()).__name__)
    get_vector_store().close(); get_knowledge_graph().close()
    return 0


def cmd_seed(args) -> int:
    pipe = _seed_corpus()
    pipe.close()
    return 0


def cmd_ingest(args) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 2
    pipe = IngestionPipeline()
    if path.suffix.lower() == ".pdf":
        res = pipe.ingest_pdf(path)
    elif path.suffix.lower() in (".fits", ".fit", ".fts"):
        res = pipe.ingest_fits(path)
    else:
        pipe.ingest_text(
            path.read_text(encoding="utf-8", errors="replace"), source=str(path)
        )
        res = {"doc_id": str(path)}
    pipe.close()
    print(f"Ingested {path.name}: {res}")
    return 0


def cmd_query(args) -> int:
    pipe = IngestionPipeline()
    if not pipe.vs.count():
        _seed_corpus(pipe, quiet=True)
    agent = CosmoAgent()
    ans = agent.run(args.query)
    print(_format_answer(ans))
    if args.json:
        print(json.dumps({"text": ans.text, "citations": ans.citations,
                          "tool_calls": ans.tool_calls, "code": ans.code,
                          "code_valid": ans.code_valid}, default=str, indent=2))
    pipe.close()
    return 0


def cmd_demo(args) -> int:
    pipe = IngestionPipeline()
    if not pipe.vs.count():
        _seed_corpus(pipe, quiet=True)
    agent = CosmoAgent()
    for q in SAMPLE_QUERIES:
        print("\n" + "=" * 72)
        print("Q:", q)
        print("=" * 72)
        ans = agent.run(q)
        print(ans.text)
        if ans.code:
            print("\n[Generated code valid=" + str(ans.code_valid) + "]")
        if ans.tool_calls:
            print("[Tools:", ", ".join(t["tool"] for t in ans.tool_calls) + "]")
    pipe.close()
    return 0


def cmd_serve(args) -> int:
    from cosmo_mind.api.server import run_server

    return run_server(host=args.host, port=args.port, seed=not args.no_seed)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cosmomind",
        description="CosmoMind: Agentic Multi-Modal GraphRAG for Astrophysical Discovery.",
    )
    p.add_argument("-V", "--version", action="version", version="cosmomind 0.1.0")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("check", help="show runtime configuration & backends")
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("seed", help="ingest the bundled sample corpus")
    sp.set_defaults(func=cmd_seed)

    sp = sub.add_parser("ingest", help="ingest a PDF or FITS file")
    sp.add_argument("path", help="path to a .pdf/.fits file")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("query", help="ask CosmoMind a question")
    sp.add_argument("query", help="the question to ask")
    sp.add_argument("-k", "--top-k", type=int, default=6)
    sp.add_argument("--json", action="store_true", help="also print machine JSON")
    sp.set_defaults(func=cmd_query)

    sp = sub.add_parser("demo", help="seed corpus and run sample queries")
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser("serve", help="start the HTTP API + chat UI")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--no-seed", action="store_true", help="do not auto-seed corpus")
    sp.set_defaults(func=cmd_serve)

    return p


def app(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    reload_settings()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(app())