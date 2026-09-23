"""An OpenAI shaped /v1/embeddings route in front of the shared embedder.

Letta computes its archival embeddings inside its own server process, so the
only way to put it on the same weights as the rest of the table is to hand it
an endpoint. Ollama also serves nomic and is the obvious answer, but it is a
second install and a separately quantised build of the model, and then the
results table is comparing two embedders as well as five memory systems. This
serves the exact weights every other adapter uses.

http.server rather than a web framework on purpose: one route does not justify
a dependency. One request at a time is enough, because Letta embeds a single
passage per insert and the forward pass is the slow part either way.

The prefixes are not applied here. Callers hand over text that is already
prefixed, the way the Letta adapter does, and a second prefix would corrupt it.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import Config
from .embeddings import LocalEmbedder

MAX_BODY = 8 * 1024 * 1024  # a batch of passages, not a file upload


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    embedder: LocalEmbedder
    model_name: str

    def log_message(self, *args) -> None:
        """Quiet. One line per passage on stderr would bury the run's output."""

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        # Some clients probe the model list before their first embed.
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [
                {"id": self.model_name, "object": "model", "owned_by": "membench"}
            ]})
        else:
            self._send(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self) -> None:
        if not self.path.rstrip("/").endswith("/embeddings"):
            self._send(404, {"error": {"message": f"no route {self.path}"}})
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._send(413, {"error": {"message": "body too large"}})
            return

        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            self._send(400, {"error": {"message": f"body is not json: {e}"}})
            return

        raw = payload.get("input")
        texts = [raw] if isinstance(raw, str) else list(raw or [])
        if not all(isinstance(t, str) for t in texts):
            # token-id input is in the OpenAI schema and nothing here sends it
            self._send(400, {"error": {"message": "input must be a string or a list of strings"}})
            return

        try:
            vectors = self.embedder.embed_documents(texts) if texts else []
        except Exception as e:  # a failed forward pass is a 500, not a crashed server
            self._send(500, {"error": {"message": f"{type(e).__name__}: {e}"}})
            return

        self._send(200, {
            "object": "list",
            "model": payload.get("model") or self.model_name,
            "data": [
                {"object": "embedding", "index": i, "embedding": v}
                for i, v in enumerate(vectors)
            ],
            # clients read these even when nothing is being billed
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        })


class EmbedServer:
    """Runs on a background thread for as long as the adapter needs it.

    Port 0 means the OS picks one that is free, so two runs on the same box do
    not collide and nothing has to be configured.
    """

    def __init__(self, cfg: Config, host: str = "127.0.0.1", port: int = 0):
        self.cfg = cfg
        self._host, self._port = host, port
        self._srv: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if not self._srv:
            raise RuntimeError("the embedding server is not running")
        host, port = self._srv.server_address[:2]
        return f"http://{host}:{port}/v1"

    def start(self) -> str:
        embedder = LocalEmbedder(self.cfg, prefixes=False)
        embedder.warmup()  # load the weights here, not inside the first request

        handler = type("Handler", (_Handler,), {
            "embedder": embedder,
            "model_name": self.cfg.embed_model.split("/")[-1],
        })
        self._srv = ThreadingHTTPServer((self._host, self._port), handler)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()
        return self.base_url

    def stop(self) -> None:
        if self._srv:
            self._srv.shutdown()
            self._srv.server_close()
            self._srv = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
