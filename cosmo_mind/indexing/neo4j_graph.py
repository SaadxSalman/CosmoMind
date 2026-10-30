"""
cosmo_mind.indexing.neo4j_graph
===============================

Production Neo4j backing for the :class:`KnowledgeGraph` interface.

Schema:
  (:CelestialObject {node_id,name,object_type,catalog_id,ra,dec,spectral_type})
      -[:OBSERVED_IN]-> (:Document)
      -[:HOSTS]-> (:CelestialObject)
  (:Document {node_id,doc_id,title,source,doc_type,year,telescope})
      -[:CITES]-> (:Document)
      -[:HAS_CHUNK]-> (:Chunk)
  (:Chunk {node_id,chunk_id,doc_id,text}) -[:MENTIONS]-> (:CelestialObject)
  (:Equation {node_id,latex}) -[:IN_CHUNK]-> (:Chunk)
"""

from __future__ import annotations

from typing import Any

import networkx as nx
from neo4j import GraphDatabase

from cosmo_mind.indexing.knowledge_graph import KnowledgeGraph
from cosmo_mind.parsing.document import CelestialObject, Document
from cosmo_mind.utils.logging import get_logger

log = get_logger("cosmo_mind.indexing.neo4j")


class Neo4jKnowledgeGraph(KnowledgeGraph):
    def __init__(self, uri: str, user: str, password: str, db: str = "neo4j") -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._db = db
        self._init_schema()

    def _init_schema(self) -> None:
        schema = [
            "CREATE INDEX doc_id FOR (d:Document) ON (d.doc_id)",
            "CREATE INDEX obj_name FOR (o:CelestialObject) ON (o.name)",
            "CREATE INDEX chunk_id FOR (c:Chunk) ON (c.chunk_id)",
            "CREATE INDEX node_id FOR () [ (n) ON (n.node_id) ]",
        ]
        with self._driver.session(database=self._db) as s:
            for stmt in schema:
                try:
                    s.run(stmt)
                except Exception as e:  # index already exists etc.
                    log.debug(f"schema stmt skipped: {e}")

    def _health_check(self) -> None:
        with self._driver.session(database=self._db) as s:
            s.run("RETURN 1").consume()

    def _run(self, query: str, params: dict | None = None, read: bool = False):
        with self._driver.session(database=self._db) as s:
            fn = s.execute_read if read else s.execute_write
            return fn(lambda tx: list(tx.run(query, params or {})))

    def upsert_document(self, doc: Document) -> None:
        self._run(
            "MERGE (d:Document {doc_id:$doc_id}) "
            "SET d.node_id=$nid, d.title=$title, d.source=$source, "
            "d.doc_type=$dt, d.year=$year, d.telescope=$tel",
            {
                "nid": f"doc:{doc.doc_id}", "doc_id": doc.doc_id, "title": doc.title,
                "source": doc.source, "dt": doc.doc_type.value, "year": doc.year,
                "tel": doc.metadata.get("telescop") or doc.metadata.get("telescope"),
            },
        )
        for obj in doc.celestial_objects:
            self.add_entity(obj)
            self.link(f"obj:{obj.name}", f"doc:{doc.doc_id}", "OBSERVED_IN")
        for eq in doc.equations:
            self.add_equation(eq.latex, f"doc:{doc.doc_id}")

    def upsert_chunk(self, chunk_id: str, doc_id: str, text: str) -> None:
        self._run(
            "MERGE (c:Chunk {chunk_id:$cid}) SET c.node_id=$nid, c.doc_id=$did, c.text=$text",
            {"nid": f"chunk:{chunk_id}", "cid": chunk_id, "did": doc_id, "text": text},
        )
        self._run(
            "MATCH (a {node_id:$a}) MATCH (b {node_id:$b}) MERGE (a)-[:HAS_CHUNK]->(b)",
            {"a": f"doc:{doc_id}", "b": f"chunk:{chunk_id}"},
        )

    def add_entity(self, obj: CelestialObject) -> None:
        params = {"nid": f"obj:{obj.name}", "name": obj.name, "otype": obj.object_type}
        if obj.catalog_id:
            params["catalog_id"] = obj.catalog_id
        if obj.spectral_type:
            params["spectral_type"] = obj.spectral_type
        if obj.coord is not None:
            params["ra"] = float(obj.coord.ra.deg)
            params["dec"] = float(obj.coord.dec.deg)
        self._run(
            "MERGE (o:CelestialObject {name:$name}) "
            "SET o.node_id=$nid, o.object_type=$otype, o.catalog_id=$catalog_id, "
            "o.spectral_type=$spectral_type, o.ra=$ra, o.dec=$dec",
            params,
        )

    def add_equation(self, latex: str, chunk_id: str) -> None:
        nid = f"eq:{abs(hash(latex))}"
        self._run(
            "MERGE (e:Equation {latex:$latex}) SET e.node_id=$nid",
            {"nid": nid, "latex": latex},
        )
        self.link(nid, chunk_id, "IN_CHUNK")

    def link(self, src: str, dst: str, rel: str, **props: Any) -> None:
        self._run(
            "MATCH (a {node_id:$a}) MATCH (b {node_id:$b}) "
            "MERGE (a)-[r:REL {type:$rel}]->(b) SET r += $props",
            {"a": src, "b": dst, "rel": rel, "props": props},
        )

    def search_objects(self, query: str, top_k: int = 10) -> list[dict]:
        rows = self._run(
            "MATCH (o:CelestialObject) "
            "WHERE toLower(o.name) CONTAINS $q OR toLower(o.catalog_id) CONTAINS $q "
            "RETURN o{.*} LIMIT $limit",
            {"q": query.lower(), "limit": top_k},
            read=True,
        )
        return [r["o"] for r in rows]

    def neighbors(self, node: str, rel: str | None = None, top_k: int = 10) -> list[dict]:
        nid = node if node.startswith(("doc:", "obj:", "chunk:", "eq:")) else f"obj:{node}"
        if rel is None:
            q = ("MATCH (n {node_id:$nid})-[r]->(m) RETURN m{.*}, r.type AS rel LIMIT $limit")
            params = {"nid": nid, "limit": top_k}
        else:
            q = ("MATCH (n {node_id:$nid})-[r]->(m) WHERE r.type=$rel "
                 "RETURN m{.*}, r.type AS rel LIMIT $limit")
            params = {"nid": nid, "rel": rel, "limit": top_k}
        return [r["m"] for r in self._run(q, params, read=True)]

    def related_documents(self, object_name: str, top_k: int = 10) -> list[dict]:
        rows = self._run(
            "MATCH (o:CelestialObject {name:$name})-[:OBSERVED_IN]->(d:Document) "
            "RETURN d{.*} LIMIT $limit",
            {"name": object_name, "limit": top_k},
            read=True,
        )
        return [r["d"] for r in rows]

    def serialize(self) -> nx.MultiDiGraph:
        rows = self._run(
            "MATCH (a)-[r]->(b) "
            "RETURN a{node_id,.name,.object_type} AS a, b{node_id,.name,.object_type} AS b, "
            "r.type AS rel",
            read=True,
        )
        g = nx.MultiDiGraph()
        for r in rows:
            a, b, rel = r["a"], r["b"], r["rel"] or "REL"
            g.add_node(a["node_id"], **(a or {}))
            g.add_node(b["node_id"], **(b or {}))
            g.add_edge(a["node_id"], b["node_id"], relation=rel)
        return g

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:
            pass


__all__ = ["Neo4jKnowledgeGraph"]