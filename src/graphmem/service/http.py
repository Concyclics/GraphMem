"""Bounded internal HTTP adapter for the persistent Memory SDK."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import threading
from urllib.parse import unquote, urlsplit

from .memory import MemoryConflict, MemoryNotFound, MemoryService, ModelUnavailable


LOG = logging.getLogger(__name__)


class MemoryHTTPServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True

    def __init__(self, address, handler, *, max_requests: int = 16):
        self._slots = threading.BoundedSemaphore(max_requests)
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            try:
                request.settimeout(2)
                payload = b'{"error":"busy"}\n'
                request.sendall(b"HTTP/1.0 429 Too Many Requests\r\n"
                                b"Content-Type: application/json\r\n"
                                b"Content-Length: " + str(len(payload)).encode() +
                                b"\r\nConnection: close\r\n\r\n" + payload)
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def make_server(service: MemoryService, host="127.0.0.1", port=8090, *,
                api_key: str = "", max_requests: int = 16) -> MemoryHTTPServer:
    if host not in {"127.0.0.1", "localhost"} and not api_key:
        raise ValueError("binding outside loopback requires GRAPHMEM_API_KEY")
    if type(max_requests) is not int or max_requests < 1:
        raise ValueError("max_requests must be positive")

    class Handler(BaseHTTPRequestHandler):
        server_version = "GraphMem/5.81"

        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def _send(self, status, payload):
            def encode(value):
                if is_dataclass(value):
                    return asdict(value)
                if isinstance(value, (set, frozenset)):
                    return sorted(value)
                raise TypeError(type(value).__name__)
            body = (json.dumps(payload, ensure_ascii=False, default=encode,
                               allow_nan=False) + "\n").encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def _dispatch(self):
            url = urlsplit(self.path)
            if self.command == "GET" and url.path == "/healthz":
                self._send(200, service.health())
                return
            if api_key and not hmac.compare_digest(
                    self.headers.get("Authorization", "").encode(), ("Bearer " + api_key).encode()):
                self._send(401, {"error": "unauthorized"})
                return
            tenant = self.headers.get("X-Tenant-ID", "")
            if not tenant:
                raise ValueError("X-Tenant-ID is required")
            parts = url.path.strip("/").split("/")
            if url.query or len(parts) not in {3, 4} or parts[:2] != ["v1", "memories"]:
                self._send(404, {"error": "not_found"})
                return
            memory_id = unquote(parts[2])
            if self.command == "GET" and len(parts) == 3:
                self._send(200, service.status(memory_id, tenant_id=tenant))
                return
            if self.command != "POST" or len(parts) != 4:
                self._send(404, {"error": "not_found"})
                return
            action = parts[3]
            if action not in {"messages", "retrieve", "prepare", "answer", "rebuild"}:
                self._send(404, {"error": "not_found"})
                return
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
                raise ValueError("exactly one Content-Length and no Transfer-Encoding are required")
            length = int(lengths[0])
            if not 0 < length <= 1024 * 1024:
                raise ValueError("request body must be between 1 B and 1 MiB")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Content-Type must be application/json")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("incomplete request body")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            allowed = ({"messages", "request_id", "session_id"} if action == "messages"
                       else set() if action == "rebuild"
                       else {"query"} if action == "retrieve" else {"query", "question_date"})
            if set(payload) - allowed:
                raise ValueError("unknown request fields")
            if action == "messages":
                result = service.add(memory_id, payload.get("messages"), tenant_id=tenant,
                                     session_id=payload.get("session_id"), request_id=payload.get("request_id"))
            elif action == "rebuild":
                result = service.rebuild(memory_id, tenant_id=tenant)
            else:
                options = {"question_date": payload.get("question_date")} if action in {"prepare", "answer"} else {}
                result = getattr(service, action)(memory_id, payload.get("query"), tenant_id=tenant, **options)
                if action == "prepare":
                    result = result.to_record()
                    result["memory_id"] = memory_id
                elif action == "answer":
                    result = asdict(result)
                    result["memory_id"] = memory_id
            self._send(200, result)

        def _handle(self):
            try:
                self._dispatch()
            except MemoryNotFound:
                self._send(404, {"error": "memory_not_found"})
            except MemoryConflict as error:
                self._send(409, {"error": "memory_conflict", "detail": str(error)})
            except (ValueError, TypeError, UnicodeError) as error:
                self._send(400, {"error": "invalid_request", "detail": str(error)})
            except ModelUnavailable as error:
                self._send(503, {"error": "model_unavailable", "detail": str(error)})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True
            except Exception:
                # Exception messages from upstream SDKs may contain credentials
                # or source text. Neither is returned to clients or access logs.
                LOG.error("memory operation failed")
                self._send(500, {"error": "internal_error"})

        do_GET = _handle
        do_POST = _handle

        def log_message(self, format_, *args):
            return  # The embedding product owns access logging and request IDs.

    return MemoryHTTPServer((host, port), Handler, max_requests=max_requests)
