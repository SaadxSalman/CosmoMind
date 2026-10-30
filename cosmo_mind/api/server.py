"""
cosmo_mind.api.server
=====================

A tiny, dependency-free HTTP API + single-page chat UI for CosmoMind built on
the Python standard library (``http.server``).  No FastAPI / Flask required.

Endpoints
---------
GET  /            -> the chat UI (static HTML, fetches /query)
POST /query       -> {"question": "..."} -> JSON answer
POST /ingest     -> {"path": "..."} -> ingest a PDF/FITS file
GET  /health     -> liveness probe
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from cosmo_mind.agents.cosmo_agent import CosmoAgent, AgentAnswer
from cosmo_mind.config import get_settings, reload_settings
from cosmo_mind.indexing.ingest import IngestionPipeline
from cosmo_mind.main import _seed_corpus
from cosmo_mind.utils.logging import get_logger

log = get_logger("cosmo_mind.api.server")

_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CosmoMind Chat</title>
<style>
  body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0b0f15;color:#e6e9f2}
  .wrap{max-width:820px;margin:0 auto;padding:18px}
  h1{font-size:20px;margin:0 0 2px}
  .sub{color:#9aa3b8;font-size:13px;margin-bottom:16px}
  #messages{overflow-y:auto;padding:14px 0;min-height:60vh}
  .msg{padding:10px 14px;border-radius:12px;margin:8px 0;line-height:1.45}
  .user{background:#2a2f3a;margin-left:18%}
  .bot{background:#1e222c;border:1px solid #2a2f3a;margin-right:18%}
  .meta{color:#9aa3b8;font-size:11px;margin-top:6px}
  pre{background:#0d1117;border:1px solid #2a2f3a;border-radius:8px;padding:10px;
      overflow:auto;font-size:12.5px;color:#cdd9e5}
  .cite{background:#16203a;border-left:3px solid #4da6ff;padding:6px 10px;border-radius:6px;
      font-size:12.5px;margin:6px 0;color:#c7d6ee}
  #q{width:100%;box-sizing:border-box;background:#1e222c;color:#e6e9f2;border:1px solid #3a4052;
      border-radius:10px;padding:10px 12px;font-size:14px;outline:none}
  #q::placeholder{color:#7b8295}
  button{background:#4da6ff;color:#071120;border:none;border-radius:10px;padding:10px 16px;
      font-weight:600;cursor:pointer;margin-left:8px}
  button:disabled{opacity:.6;cursor:not-allowed}
</style></head><body>
<div class="wrap">
  <h1>CosmoMind</h1>
  <div class="sub">Agentic Multi-Modal GraphRAG for Astrophysical Discovery.</div>
  <div id="messages"></div>
  <div style="position:sticky;bottom:0;background:#0b0f15;padding-top:8px;">
    <input id="q" placeholder="Ask about an object (e.g. HD 209458 b), a plot, or the Hubble tension...)" autocomplete="off">
    <button id="send" disabled>Send</button>
  </div>
</div>
<script>
const $ = (id)=>document.getElementById(id);
function esc(s){return String(s).replace(/[&<>"']/g, c=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function addMsg(cls, html){const d=document.createElement('div');d.className='msg '+cls;d.innerHTML=html;$('#messages').appendChild(d);
  $('#messages').scrollTop=$('#messages').scrollHeight;}
async function ask(){
  const q=$.value.trim(); if(!q||sending)return; sending=true; $('#send').disabled=true;
  addMsg('user', esc(q));
  const t0=performance.now();
  addMsg('bot','<i>thinking...</i>');
  try{
    const r=await fetch('/query',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({question:q})});
    const d=await r.json();
    const last=$('#messages').lastChild;
    let h=esc(d.text)+'<pre>'+esc(d.code||'')+'</pre>';
    (d.citations||[]).forEach(c=>{h+='<div class="cite">['+c.score.toFixed(3)+'] '+esc(c.chunk_id)+'<br>'+esc(c.text)+'</div>';});
    if(d.tool_calls&&d.tool_calls.length){h+='<div class="meta">tools: '
      +esc(d.tool_calls.map(t=>t.tool).join(', '))+'</div>';}
    h+='<div class="meta">'+(d.code_valid?'valid code':'')+'  ·  '+(performance.now()-t0).toFixed(0)+'ms</div>';
    last.innerHTML=h;
  }catch(e){$('#messages').lastChild.innerHTML='<span style=color:#ff6b6b>error: '+esc(e)+'</span>';}
  $.value='';sending=false;$('#send').disabled=false;
}
let sending=false;
$('#q').addEventListener('keyup',e=>{if(e.key==='Enter')ask();});
$('#send').addEventListener('click',ask);
</script></body></html>
"""


class _State:
    """Process-wide shared agent + pipeline (built on first request)."""

    lock = threading.Lock()
    agent: CosmoAgent | None = None
    pipeline: IngestionPipeline | None = None


def _ensure_state(seed: bool = True) -> None:
    if _State.agent is not None:
        return
    with _State.lock:
        if _State.agent is not None:
            return
        pipe = IngestionPipeline()
        if seed and not pipe.vs.count():
            _seed_corpus(pipe, quiet=True)
        _State.pipeline = pipe
        _State.agent = CosmoAgent()


class Handler(BaseHTTPRequestHandler):
    server_version = "CosmoMind/0.1.0"

    def log_message(self, fmt, *args):  # silence default logging
        return

    def _send(self, code: int, payload: dict | str, content_type: str = "application/json") -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {"status": "ok", "mode": get_settings().cosmo_mode})
            return
        if self.path in ("", "/"):
            self._send(200, _HTML, content_type="text/html; charset=utf-8")
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except Exception:
            self._send(400, {"error": "invalid JSON"})
            return

        if self.path == "/query":
            _ensure_state(seed=True)
            q = (body.get("question") or "").strip()
            if not q:
                self._send(400, {"error": "missing 'question'"})
                return
            ans: AgentAnswer = _State.agent.run(q)
            self._send(200, {
                "text": ans.text, "citations": ans.citations,
                "tool_calls": ans.tool_calls, "code": ans.code,
                "code_valid": ans.code_valid, "intent": ans.code_intent,
            })
            return

        if self.path == "/ingest":
            _ensure_state(seed=False)
            p = body.get("path")
            if not p:
                self._send(400, {"error": "missing 'path'"})
                return
            try:
                _State.pipeline.ingest_text(Path(p).read_text(), source=p) if Path(p).suffix.lower() == ".txt" \
                    else _State.pipeline.ingest_fits(p) if Path(p).suffix.lower() in (".fits",".fit",".fts") \
                    else _State.pipeline.ingest_pdf(p)
                self._send(200, {"status": "ingested", "path": p})
            except Exception as e:
                self._send(500, {"error": str(e)})
            return
        self._send(404, {"error": "not found"})


def run_server(host: str = "127.0.0.1", port: int = 8000, seed: bool = True) -> int:
    reload_settings()
    s = get_settings()
    log.info(f"CosmoMind API starting on {host}:{port} ({s.cosmo_mode} mode)")
    _ensure_state(seed=seed)
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"CosmoMind API + Chat UI running at http://{host}:{port}/  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        httpd.server_close()
    return 0
