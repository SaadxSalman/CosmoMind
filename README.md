# CosmoMind

**Agentic Multi-Modal GraphRAG for Astrophysical Discovery.**

CosmoMind is an AI research co-pilot that answers natural‑language astronomy
questions by reasoning over **three** connected layers at once:

1. **your local literature / observation corpus** (PDFs, FITS logs, arXiv
   abstracts) — indexed with a **dual‑layer GraphRAG** store (dense vector
   search *fused* with a celestial‑entity Knowledge Graph via Reciprocal Rank
   Fusion);
2. **live astronomical databases** — SIMBAD, VizieR, and the NASA Exoplanet
   Archive, queried on demand through an autonomous tool‑calling agent; and
3. **executable Astropy code** — generated, syntax‑verified (and optionally
   sandbox‑executed) scripts that plot light curves, fetch spectra, or compute
   orbital parameters.

It runs **fully offline in `LOCAL` mode** (deterministic LLM stub, hashing
embedder, in‑memory vector store + NetworkX graph, and a curated reference
catalogue used for graceful degradation) and switches transparently to
production cloud backends (OpenAI, Qdrant, Neo4j, live astroquery) when
configured via `.env`.

> ⭐ If a catalog service is unreachable, CosmoMind falls back to a curated
> local reference catalogue and still returns a grounded, cited answer — i
> never crashes on a network error.

---

## Architecture

```
            ┌──────────────────────────────────────────────────┐
 user query │  CosmoAgent  (tool‑calling loop, max_turns)     │
────────────► LLM (OpenAI / Anthropic / LocalStub)            │
            │  ┌──────────────┐ ┌──────────┐ ┌──────────────┐ │
            │  │ SimbadTool   │ │ Vizier   │ │ NasaExoplanet│ │
            │  │ code_gen     │ │ Tool     │ │ Tool          │ │
            │  └──────┬───────┘ └────┬─────┘ └──────┬───────┘ │
            └────────┴──────────────┴───────────────┴────────┘
              │                       │            live catalog
              │                       │
   ┌──────────▼───────────┐   ┌───────▼──────────────────────┐
   │ HybridRetriever      │   │ AstropyCodeGenerator        │
   │  RRF: vector + KG   │   │  ast_check + sandbox exec   │
   └────┬──────────────┬──┘   └──────────────────────────────┘
        │   vectors    │                     result
        ▼              ▼
  Qdrant /            NetworkX /
  InMemoryVector    KnowledgeGraph
     Store           (celestial objs)
        ▲
        │  chunks
   IngestionPipeline  ← PDF / FITS / TEXT parsers
```

### Key components
| Layer | LOCAL (offline) | PROD |
|-------|-----------------|------|
| LLM chat | `LocalStubLLM` (deterministic routing) | OpenAI / Anthropic |
| Embedder | `LocalHashingEmbedder` (feature hashing) | OpenAI embeddings |
| Vector store | `InMemoryVectorStore` (pickle‑persisted) | Qdrant |
| Knowledge graph | `NetworkxKnowledgeGraph` (pickle‑persisted) | Neo4j |
| Catalogs | curated local catalogue *fallback* (live call first) | SIMBAD / VizieR / NASA live |

---

## Install

```bash
git clone https://github.com/SaadxSalman/CosmoMind.git
cd CosmoMind
pip install -e .            # runtime deps
pip install -e ".[dev]"     # + pytest, httpx, etc. for testing
```

Runtime deps (see `requirements.txt`): `astropy`, `astroquery`, `numpy`,
`networkx`, `qdrant-client`, `neo4j`, `pydantic-settings`, `PyMuPDF`,
`tiktoken`, `llama-index-core`, `openai`.

---

## Configuration (`.env`)

Copy the template:

```bash
cp .env.example .env   # or just edit .env
```

| Variable | Description | Default |
|----------|-------------|---------|
| `COSMO_MODE` | `LOCAL` (offline) or `PROD` (cloud) | `LOCAL` |
| `OPENAI_API_KEY` | OpenAI key (chat + embeddings) | — |
| `ANTHROPIC_API_KEY` | Optional Anthropic key | — |
| `QDRANT_URL` / `QDRANT_API_KEY` | Qdrant server | `localhost:6333` |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | Neo4j graph | `localhost:7687` |
| `COSMO_DATA_DIR` / `COSMO_VECTOR_CACHE` / `COSMO_KG_CACHE` | Local cache dirs | `./data/...` |
| `COSMO_ENABLE_CODE_SANDBOX` | Execute generated scripts to validate runtime | `false` |
| `NASA_API_KEY` | Optional NASA Exoplanet API key | — |

In `LOCAL` mode no keys are required and **no external network calls block the
agent** — catalog tools degrade to the local reference catalogue.

---

## Usage

### Command line

```bash
cosmomind check            # show runtime config & backends
cosmomind seed             # ingest the bundled sample astrophysics corpus
cosmomind ingest paper.pdf # ingest a PDF or FITS file
cosmomind query "Tell me about HD 209458 b"
cosmomind demo             # seed corpus, then run 6 example queries
cosmomind serve            # start the HTTP API + chat UI (http://localhost:8000)
```

> `python -m cosmo_mind` runs the same CLI.

### As a library

```python
from cosmo_mind.agents.cosmo_agent import CosmoAgent
from cosmo_mind.indexing.ingest import IngestionPipeline
from cosmo_mind.main import _seed_corpus

pipe = IngestionPipeline()
_seed_corpus(pipe, quiet=True)          # load the sample corpus

agent = CosmoAgent()
ans = agent.run("Plot the light curve of Kepler-442b")
print(ans.text)        # grounded answer citing retrieved literature
print(ans.code)        # verified Astropy script
print(ans.code_valid)  # True if it passed ast + sandbox validation
```

### HTTP API + chat UI

```bash
cosmomind serve --port 8000
```

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| `GET`  | `/`        | — | chat UI (HTML) |
| `GET`  | `/health`  | — | `{"status":"ok","mode":"LOCAL"}` |
| `POST` | `/query`   | `{"question":"..."}` | `{text, citations, tool_calls, code, code_valid, intent}` |
| `POST` | `/ingest`  | `{"path":"/abs/paper.pdf"}` | ingest a PDF/FITS/TXT |

```bash
curl.exe -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the mass of HD 209458 b?"}'
```

---

## What the agent does

Given a question, `CosmoAgent.run`:

1. **retrieves** the top‑$k$ chunks from your local corpus using the hybrid
   retrievers (RRF‑fused vector + Knowledge‑Graph traversal over detected
   celestial objects);
2. **calls tools** the LLM requests — `simbad_query`, `vizier_query`,
   `nasa_exoplanet_query`, and `code_generate` — dedup‑guarded so the loop
   always terminates;
3. **generates & validates** Astropy/`astroquery`/`matplotlib` scripts
   (static `ast` check + optional sandbox execution); and
4. **synthesizes** a final, cited answer.

Live catalog calls always take precedence; when a service is unreachable the
tool transparently returns data from the **curated local reference catalogue**
(`cosmo_mind/agents/tools/local_catalog.py`) tagged `source="local_catalog"`,
so the agent still produces a meaningful answer offline.

---

## Development & testing

```bash
pip install -e ".[dev]"
pytest            # offline test suite (astroquery faked in conftest)
```

Tests force `COSMO_MODE=LOCAL` with a throwaway data dir and patch the
astroquery classes with offline fakes, so they are deterministic and network‑free.

### Project layout

```
cosmo_mind/
├── config.py                       # pydantic-settings (.env)
├── llm/chat.py                     # OpenAI/Anthropic/LocalStub + ToolSpec
├── parsing/                      # document, entity & equation extractors; PDF/FITS parsers
├── indexing/                     # chunker, embeddings, vector store, KG, ingest
├── retrieval/hybrid_retriever.py # RRF vector + KG fusion
├── codegen/                      # astropy_generator + validation (ast + sandbox)
├── agents/
│   ├── cosmo_agent.py            # tool-calling orchestrator
│   └── tools/                    # simbad / vizier / nasa / code_generator / local_catalog
├── api/server.py                 # stdlib HTTP API + chat UI
└── data/sample_corpus.py         # bundled sample astrophysics corpus
tests/                            # offline test suite
```

---

## License

MIT.
