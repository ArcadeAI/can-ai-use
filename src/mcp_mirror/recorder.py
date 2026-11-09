"""Serve a fixture through a recording proxy and log the wire, both directions.

The harness instrument. A harness is measured on the wire only (see
``plan/harness_surface.yaml``): the recorder sits between the client under test and a
fixture, forwards every JSON-RPC message unchanged, and appends each one to a JSONL
capture. Harness cells in ``cells.py`` are functions of that capture.

    mcp-mirror record "python fixtures/results_server.py" --out capture.jsonl
    mcp-mirror record "python fixtures/results_server.py" --transport http --port 8765

The fixture is always launched as a stdio subprocess. The client reaches the recorder
over stdio (the recorder *is* the server command the harness launches) or over
streamable HTTP at ``http://<host>:<port>/mcp``, where the recorder owns the HTTP
session and each HTTP session gets its own fixture process. The recorder parses JSON
but imports no MCP SDK, so it records either SDK line, and any revision, as sent.

Capture shape
-------------

One JSON object per line. Every record carries ``type``, ``seq`` (0-based, strictly
increasing), ``t`` (UTC ISO-8601 wall clock, microseconds) and ``elapsed`` (seconds
since the capture opened, monotonic). The other fields depend on ``type``.

``capture`` (always the first line)
    ``format`` (``"mcp-mirror.wire/1"``), ``recorder`` (``"mcp-mirror <version>"``),
    ``fixture`` (the command string), ``transport`` (``"stdio"`` or ``"http"``),
    ``url`` (HTTP only, else null).

``message`` (one per JSON-RPC message; a batch array yields one record per element)
    ``session``
        ``"stdio"``; over HTTP the ``Mcp-Session-Id`` the recorder issued, or
        ``"stateless"`` for requests that carry none and are not ``initialize``.
    ``direction``
        ``"client_to_server"`` or ``"server_to_client"``.
    ``origin``
        ``"client"``, ``"server"``, or ``"recorder"`` for a message the recorder
        injected itself (``--ping``). The client's answer to an injected message is
        recorded with origin ``"client"`` and is not forwarded to the fixture.
    ``kind``
        ``"request"``, ``"notification"``, ``"response"``, ``"error"`` (an error
        response) or ``"invalid"`` (not a JSON-RPC object; ``raw`` holds the text).
    ``method``
        The method; for a response or error, the method of the request it answers
        (null if that request was never seen).
    ``id``
        The JSON-RPC id, or null.
    ``revision``
        The protocol revision in force for this message: the ``_meta``
        ``io.modelcontextprotocol/protocolVersion`` a 2026-07-28 request carries, else
        the revision negotiated so far on the session, else null.
    ``batch``
        True when the message arrived inside a JSON array.
    ``message``
        The parsed message exactly as received.
    ``http`` (client-to-server over HTTP only)
        ``{"method": "POST", "headers": {...}}`` with the request's ``accept``,
        ``content-type``, ``last-event-id`` and every ``mcp-*`` header. Nothing else
        is kept, so credentials never reach a capture.
    ``route`` (server-to-client over HTTP only)
        Where the recorder delivered it: ``"post"`` (the SSE response of the POST
        whose request it answers or relates to), ``"get"`` (the session's standalone
        SSE stream), or ``"backlog"`` (held until the next stream opens).

``negotiation`` (whenever the session's revision is established or changes)
    ``session``, ``via`` and:

    * ``via: "initialize"``, written when the initialize response arrives:
      ``requested_revision`` (``params.protocolVersion``), ``negotiated_revision``
      (``result.protocolVersion``, null on an error response), ``client_capabilities``,
      ``client_info``, ``server_capabilities``, ``server_info``, ``error``.
    * ``via: "server/discover"``, written when the discover response arrives:
      ``requested_revision`` (the probe's ``_meta`` revision), ``supported_versions``,
      ``client_capabilities`` and ``client_info`` from the probe's ``_meta``,
      ``server_capabilities``, ``error``. ``negotiated_revision`` is null: at
      2026-07-28 the client picks, and says so on its next request.
    * ``via: "request-meta"``, written on the first client request (other than
      ``server/discover``) whose ``_meta`` names a revision different from the one in
      force: ``negotiated_revision``, ``client_capabilities``, ``client_info`` from that
      request's ``_meta``, and ``method``/``id`` of the request.

``http`` (HTTP events that carry no JSON-RPC message)
    ``session``, ``event`` (``"get_opened"``, ``"get_closed"``, ``"get_rejected"``,
    ``"delete"``, ``"unknown_session"``, ``"parse_error"``, ``"fixture_exited"``, the
    last written once when the fixture's stdout closes and again, with status 502, for
    each later POST to that session), ``status`` where one was returned, ``headers`` as
    for ``message.http``.

``end`` (last line, written on a clean shutdown)
    ``messages`` (count of message records) and ``sessions`` (count of sessions).

Limits, on purpose: over HTTP the recorder never rewrites ids, so two concurrent
sessionless clients that reuse ids share one fixture and can collide; a harness
session mounts one recorder per fixture. SSE events carry no ids, so a dropped stream
is not resumable.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
from anyio.abc import Process
from anyio.streams.buffered import BufferedByteReceiveStream

from . import __version__

FORMAT = "mcp-mirror.wire/1"
CLIENT_TO_SERVER = "client_to_server"
SERVER_TO_CLIENT = "server_to_client"
PING_ID = "mcp-mirror-ping-1"
STATELESS = "stateless"
MCP_PATH = "/mcp"

PROTOCOL_VERSION_META = "io.modelcontextprotocol/protocolVersion"
CLIENT_CAPABILITIES_META = "io.modelcontextprotocol/clientCapabilities"
CLIENT_INFO_META = "io.modelcontextprotocol/clientInfo"
SUBSCRIPTION_ID_META = "io.modelcontextprotocol/subscriptionId"

_KEPT_HEADERS = ("accept", "content-type", "last-event-id")
_MAX_LINE = 64 * 1024 * 1024


class RecorderError(RuntimeError):
    """The recorder cannot start: bad fixture command or unusable output path."""


def default_capture_path(label: str, server: str, transport: str) -> Path:
    """``measurements/wire/<label>/<UTC stamp>-<fixture stem>-<transport>.jsonl``."""

    parts = shlex.split(server)
    script = next((part for part in reversed(parts) if part.endswith(".py")), parts[0] if parts else "fixture")
    stem = Path(script).stem or "fixture"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("measurements") / "wire" / label / f"{stamp}-{stem}-{transport}.jsonl"


def read_capture(path: Path | str) -> list[dict]:
    """Every record of a capture, in order."""

    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def messages(records: Iterable[dict], method: str | None = None) -> Iterator[dict]:
    """The ``message`` records of a capture, optionally only those for ``method``."""

    for record in records:
        if record.get("type") == "message" and (method is None or record.get("method") == method):
            yield record


class Capture:
    """Append-only JSONL writer. Each record is flushed as it is written, so a capture
    is readable while the session is live and survives the recorder being killed.

    The file is truncated once and then opened for appending, so every record lands at
    the file's current end. A process that rewrites the capture while the recorder is
    still open (the harness runner scrubs home paths out of it at teardown) can shorten
    it, but a later record can never land past the new end under a stale offset and
    leave a run of NUL bytes."""

    def __init__(self, path: Path, *, fixture: str, transport: str, url: str | None = None) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
            self._handle = path.open("a", encoding="utf-8")
        except OSError as exc:
            raise RecorderError(f"cannot write capture {path}: {exc}") from exc
        self.path = path
        self._lock = threading.Lock()
        self._seq = 0
        self._start = time.monotonic()
        self.message_count = 0
        self.sessions: set[str] = set()
        self.write({"type": "capture", "format": FORMAT, "recorder": f"mcp-mirror {__version__}",
                    "fixture": fixture, "transport": transport, "url": url})

    def write(self, record: dict) -> None:
        with self._lock:
            if self._handle.closed:
                return
            stamped = {
                "type": record["type"],
                "seq": self._seq,
                "t": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "elapsed": round(time.monotonic() - self._start, 6),
                **{key: value for key, value in record.items() if key != "type"},
            }
            self._seq += 1
            if record["type"] == "message":
                self.message_count += 1
            self._handle.write(json.dumps(stamped, ensure_ascii=False, separators=(",", ":")) + "\n")
            self._handle.flush()

    def close(self) -> None:
        if self._handle.closed:
            return
        self.write({"type": "end", "messages": self.message_count, "sessions": len(self.sessions)})
        with self._lock:
            self._handle.close()


def _id_key(value: Any) -> str:
    # 1 and "1" are different JSON-RPC ids.
    return json.dumps(value, sort_keys=True)


def _meta(message: dict) -> dict:
    params = message.get("params")
    meta = params.get("_meta") if isinstance(params, dict) else None
    return meta if isinstance(meta, dict) else {}


def _kind(message: Any) -> str:
    if not isinstance(message, dict):
        return "invalid"
    if "method" in message:
        return "request" if "id" in message else "notification"
    if "error" in message:
        return "error"
    if "result" in message:
        return "response"
    return "invalid"


class Session:
    """One MCP session's bookkeeping: which request each response answers, and which
    revision is in force. Transport-independent; the transports call :meth:`observe`
    for every message and act on what it returns."""

    def __init__(self, capture: Capture, key: str, *, ping: bool = False) -> None:
        self.capture = capture
        self.key = key
        self.revision: str | None = None
        self._ping = ping
        self._ping_sent = False
        self._pending: dict[tuple[str, str], dict] = {}
        capture.sessions.add(key)

    def observe(self, direction: str, payload: Any, *, origin: str, extra: dict | None = None) -> list[Any]:
        """Record ``payload`` (a parsed message or batch) and return its messages."""

        batch = isinstance(payload, list)
        items = payload if batch else [payload]
        for item in items:
            self._observe_one(direction, item, origin=origin, batch=batch, extra=extra or {})
        return items

    def observe_raw(self, direction: str, line: bytes, *, origin: str, extra: dict | None = None) -> list[Any]:
        try:
            payload = json.loads(line)
        except ValueError:
            self.capture.write({"type": "message", "session": self.key, "direction": direction,
                                "origin": origin, "kind": "invalid", "method": None, "id": None,
                                "revision": self.revision, "batch": False,
                                "raw": line.decode("utf-8", "replace"), **(extra or {})})
            return []
        return self.observe(direction, payload, origin=origin, extra=extra)

    def _observe_one(self, direction: str, message: Any, *, origin: str, batch: bool, extra: dict) -> None:
        kind = _kind(message)
        method = message.get("method") if isinstance(message, dict) else None
        message_id = message.get("id") if isinstance(message, dict) else None
        request: dict | None = None
        negotiation: dict | None = None
        revision = self.revision

        if kind == "request":
            self._pending[(direction, _id_key(message_id))] = message
            requested = _meta(message).get(PROTOCOL_VERSION_META)
            if requested:
                revision = requested
            if (direction == CLIENT_TO_SERVER and method not in ("initialize", "server/discover")
                    and requested and requested != self.revision):
                self.revision = requested
                meta = _meta(message)
                negotiation = {"via": "request-meta", "negotiated_revision": requested,
                               "client_capabilities": meta.get(CLIENT_CAPABILITIES_META),
                               "client_info": meta.get(CLIENT_INFO_META), "method": method, "id": message_id}
        elif kind in ("response", "error"):
            answering = SERVER_TO_CLIENT if direction == CLIENT_TO_SERVER else CLIENT_TO_SERVER
            request = self._pending.pop((answering, _id_key(message_id)), None)
            method = request.get("method") if request else None
            if direction == SERVER_TO_CLIENT and method in ("initialize", "server/discover"):
                negotiation = self._handshake(method, request, message)
                if negotiation.get("negotiated_revision"):
                    self.revision = negotiation["negotiated_revision"]
                    revision = self.revision

        record = {"type": "message", "session": self.key, "direction": direction, "origin": origin,
                  "kind": kind, "method": method, "id": message_id, "revision": revision,
                  "batch": batch, "message": message, **extra}
        self.capture.write(record)
        if negotiation is not None:
            self.capture.write({"type": "negotiation", "session": self.key, **negotiation})

    @staticmethod
    def _handshake(method: str, request: dict, response: dict) -> dict:
        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        result = response.get("result") if isinstance(response.get("result"), dict) else {}
        error = response.get("error")
        if method == "initialize":
            return {"via": "initialize", "requested_revision": params.get("protocolVersion"),
                    "negotiated_revision": result.get("protocolVersion"),
                    "client_capabilities": params.get("capabilities"), "client_info": params.get("clientInfo"),
                    "server_capabilities": result.get("capabilities"), "server_info": result.get("serverInfo"),
                    "error": error}
        meta = _meta(request)
        return {"via": "server/discover", "requested_revision": meta.get(PROTOCOL_VERSION_META),
                "negotiated_revision": None,
                "supported_versions": result.get("supportedVersions", result.get("supported_versions")),
                "client_capabilities": meta.get(CLIENT_CAPABILITIES_META), "client_info": meta.get(CLIENT_INFO_META),
                "server_capabilities": result.get("capabilities"), "error": error}

    @staticmethod
    def is_for_recorder(message: Any) -> bool:
        """A client's answer to something the recorder injected; never forwarded."""

        return (isinstance(message, dict) and "method" not in message and message.get("id") == PING_ID)

    def take_ping(self, message: Any) -> dict | None:
        """The ping to inject once the client has sent ``notifications/initialized``."""

        if (self._ping and not self._ping_sent and isinstance(message, dict)
                and message.get("method") == "notifications/initialized"):
            self._ping_sent = True
            return {"jsonrpc": "2.0", "id": PING_ID, "method": "ping"}
        return None


def _encode(message: Any) -> bytes:
    return json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"


async def _spawn(argv: list[str]) -> Process:
    try:
        return await anyio.open_process(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)
    except OSError as exc:
        raise RecorderError(f"cannot start fixture {shlex.join(argv)!r}: {exc}") from exc


async def _lines(process: Process) -> AsyncIterator[bytes]:
    assert process.stdout is not None
    buffered = BufferedByteReceiveStream(process.stdout)
    while True:
        try:
            line = await buffered.receive_until(b"\n", _MAX_LINE)
        except (anyio.EndOfStream, anyio.IncompleteRead, anyio.ClosedResourceError, anyio.BrokenResourceError):
            return
        line = line.rstrip(b"\r")
        if line.strip():
            yield line


async def _stop(process: Process) -> None:
    if process.stdin is not None:
        with anyio.CancelScope(shield=True):
            try:
                await process.stdin.aclose()
            except (anyio.BrokenResourceError, anyio.ClosedResourceError, OSError):
                pass
    with anyio.CancelScope(shield=True):
        with anyio.move_on_after(2):
            await process.wait()
        if process.returncode is None:
            process.terminate()
            with anyio.move_on_after(2):
                await process.wait()
        if process.returncode is None:
            process.kill()
            await process.wait()


async def _send(process: Process, data: bytes) -> None:
    assert process.stdin is not None
    try:
        await process.stdin.send(data)
    except (anyio.BrokenResourceError, anyio.ClosedResourceError, OSError):
        pass


# --- stdio -----------------------------------------------------------------------------


async def _serve_stdio(argv: list[str], capture: Capture, *, ping: bool) -> None:
    session = Session(capture, "stdio", ping=ping)
    process = await _spawn(argv)
    stdout_lock = anyio.Lock()
    out = sys.stdout.buffer

    async def to_client(data: bytes) -> None:
        async with stdout_lock:
            await anyio.to_thread.run_sync(lambda: (out.write(data), out.flush()), abandon_on_cancel=True)

    async def from_client() -> None:
        stdin = sys.stdin.buffer
        while True:
            line = await anyio.to_thread.run_sync(stdin.readline, abandon_on_cancel=True)
            if not line:
                return
            line = line.rstrip(b"\r\n")
            if not line.strip():
                continue
            for message in session.observe_raw(CLIENT_TO_SERVER, line, origin="client"):
                injected = session.take_ping(message)
                if not Session.is_for_recorder(message):
                    # A batch is forwarded one message per line; the SDK stdio readers
                    # parse single messages only.
                    await _send(process, _encode(message))
                if injected is not None:
                    session.observe(SERVER_TO_CLIENT, injected, origin="recorder")
                    await to_client(_encode(injected))

    async def from_server(scope: anyio.CancelScope) -> None:
        async for line in _lines(process):
            session.observe_raw(SERVER_TO_CLIENT, line, origin="server")
            await to_client(line + b"\n")
        scope.cancel()

    async def on_signal(scope: anyio.CancelScope) -> None:
        with anyio.open_signal_receiver(signal.SIGTERM) as signals:
            async for _ in signals:
                scope.cancel()
                return

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(from_server, tg.cancel_scope)
            if os.name != "nt":
                tg.start_soon(on_signal, tg.cancel_scope)
            await from_client()
            # The client closed stdin: let the fixture see EOF and drain its output.
            if process.stdin is not None:
                await process.stdin.aclose()
            with anyio.move_on_after(3):
                await process.wait()
            tg.cancel_scope.cancel()
    finally:
        await _stop(process)


# --- streamable HTTP -------------------------------------------------------------------


class _Stream:
    """One SSE response body: a POST's (closed once its requests are answered) or the
    session's standalone GET stream."""

    def __init__(self, kind: str, pending: set[str]) -> None:
        self.kind = kind
        self.pending = pending
        self.progress_tokens: set[str] = set()
        self.send, self.receive = anyio.create_memory_object_stream[bytes](math.inf)

    def deliver(self, message: Any) -> None:
        try:
            self.send.send_nowait(b"event: message\ndata: " + _encode(message).rstrip(b"\n") + b"\n\n")
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            pass

    def close(self) -> None:
        self.send.close()


class _HttpSession(Session):
    def __init__(self, capture: Capture, key: str, argv: list[str], task_group, *, ping: bool) -> None:
        super().__init__(capture, key, ping=ping)
        self.argv = argv
        self.task_group = task_group
        self.process: Process | None = None
        self.posts: dict[str, _Stream] = {}
        self.get_stream: _Stream | None = None
        self.backlog: list[Any] = []
        self.closed = False
        self.exited = False

    async def start(self) -> None:
        self.process = await _spawn(self.argv)
        self.task_group.start_soon(self._pump)

    async def _pump(self) -> None:
        assert self.process is not None
        async for line in _lines(self.process):
            try:
                payload = json.loads(line)
            except ValueError:
                self.observe_raw(SERVER_TO_CLIENT, line, origin="server")
                continue
            for message in payload if isinstance(payload, list) else [payload]:
                self._route(message, origin="server")
        self.exited = True
        if not self.closed:
            self.capture.write({"type": "http", "session": self.key, "event": "fixture_exited",
                                "status": None, "headers": {}})
            self._close_streams()

    def _route(self, message: Any, *, origin: str) -> None:
        stream, route = self._stream_for(message)
        self.observe(SERVER_TO_CLIENT, message, origin=origin, extra={"route": route})
        if stream is None:
            self.backlog.append(message)
            return
        stream.deliver(message)
        if isinstance(message, dict) and "method" not in message and stream.kind == "post":
            key = _id_key(message.get("id"))
            stream.pending.discard(key)
            self.posts.pop(key, None)
            if not stream.pending:
                stream.close()

    def _stream_for(self, message: Any) -> tuple[_Stream | None, str]:
        if not isinstance(message, dict):
            return self._fallback()
        if "method" not in message:
            stream = self.posts.get(_id_key(message.get("id")))
            return (stream, "post") if stream is not None else self._fallback()
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        subscription = _meta(message).get(SUBSCRIPTION_ID_META)
        if subscription is not None and _id_key(subscription) in self.posts:
            return self.posts[_id_key(subscription)], "post"
        token = params.get("progressToken")
        if token is not None:
            for stream in self.posts.values():
                if _id_key(token) in stream.progress_tokens:
                    return stream, "post"
        return self._fallback()

    def _fallback(self) -> tuple[_Stream | None, str]:
        if self.get_stream is not None:
            return self.get_stream, "get"
        if self.posts:
            return list(self.posts.values())[-1], "post"
        return None, "backlog"

    def open_post(self, requests: list[dict]) -> _Stream:
        stream = _Stream("post", {_id_key(request.get("id")) for request in requests})
        for request in requests:
            token = _meta(request).get("progressToken")
            if token is not None:
                stream.progress_tokens.add(_id_key(token))
            self.posts[_id_key(request.get("id"))] = stream
        self._flush_backlog(stream)
        return stream

    def open_get(self) -> _Stream:
        self.get_stream = _Stream("get", set())
        self._flush_backlog(self.get_stream)
        return self.get_stream

    def _flush_backlog(self, stream: _Stream) -> None:
        held, self.backlog = self.backlog, []
        for message in held:
            stream.deliver(message)

    def forget(self, stream: _Stream) -> None:
        for key in [key for key, value in self.posts.items() if value is stream]:
            del self.posts[key]
        if self.get_stream is stream:
            self.get_stream = None

    def _close_streams(self) -> None:
        for stream in set(self.posts.values()):
            stream.close()
        self.posts.clear()
        if self.get_stream is not None:
            self.get_stream.close()
            self.get_stream = None

    async def forward(self, message: Any) -> None:
        if self.process is not None:
            await _send(self.process, _encode(message))

    async def close(self) -> None:
        self.closed = True
        self._close_streams()
        if self.process is not None:
            await _stop(self.process)


def _kept_headers(headers) -> dict[str, str]:
    return {name: value for name, value in headers.items()
            if name.lower() in _KEPT_HEADERS or name.lower().startswith("mcp-")}


def _build_app(argv: list[str], capture: Capture, task_group, *, ping: bool):
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response, StreamingResponse
    from starlette.routing import Route

    sessions: dict[str, _HttpSession] = {}

    def http_event(session: str | None, event: str, status: int | None, headers: dict) -> None:
        capture.write({"type": "http", "session": session, "event": event, "status": status, "headers": headers})

    async def new_session(key: str) -> _HttpSession:
        session = _HttpSession(capture, key, argv, task_group, ping=ping)
        await session.start()
        sessions[key] = session
        return session

    def sse(session: _HttpSession, stream: _Stream, on_close: Callable[[], None] | None = None):
        async def body():
            try:
                async with stream.receive:
                    async for chunk in stream.receive:
                        yield chunk
            finally:
                session.forget(stream)
                if on_close is not None:
                    on_close()

        headers = {"cache-control": "no-cache"}
        if session.key != STATELESS:
            headers["mcp-session-id"] = session.key
        return StreamingResponse(body(), media_type="text/event-stream", headers=headers)

    async def post(request: Request) -> Response:
        headers = _kept_headers(request.headers)
        session_id = request.headers.get("mcp-session-id")
        body = await request.body()
        try:
            payload = json.loads(body)
        except ValueError:
            http_event(session_id, "parse_error", 400, headers)
            return JSONResponse({"jsonrpc": "2.0", "id": None,
                                 "error": {"code": -32700, "message": "Parse error"}}, status_code=400)
        items = payload if isinstance(payload, list) else [payload]

        if session_id:
            session = sessions.get(session_id)
            if session is None:
                http_event(session_id, "unknown_session", 404, headers)
                return Response(status_code=404)
        elif any(isinstance(item, dict) and item.get("method") == "initialize" for item in items):
            session = await new_session(uuid.uuid4().hex)
        else:
            session = sessions.get(STATELESS) or await new_session(STATELESS)
        if session.exited:
            # Nothing would ever answer; say so instead of holding the stream open.
            http_event(session.key, "fixture_exited", 502, headers)
            return Response(status_code=502)

        session.observe(CLIENT_TO_SERVER, payload, origin="client",
                        extra={"http": {"method": "POST", "headers": headers}})
        requests = [item for item in items if _kind(item) == "request"]
        stream = session.open_post(requests) if requests else None
        for item in items:
            injected = session.take_ping(item)
            if not Session.is_for_recorder(item):
                await session.forward(item)
            if injected is not None:
                session._route(injected, origin="recorder")
        if stream is None:
            reply = {} if session.key == STATELESS else {"mcp-session-id": session.key}
            return Response(status_code=202, headers=reply)
        return sse(session, stream)

    async def get(request: Request) -> Response:
        headers = _kept_headers(request.headers)
        session_id = request.headers.get("mcp-session-id")
        session = sessions.get(session_id) if session_id else None
        if session is None:
            status = 404 if session_id else 405
            http_event(session_id, "get_rejected", status, headers)
            return Response(status_code=status)
        if session.get_stream is not None:
            http_event(session_id, "get_rejected", 409, headers)
            return Response(status_code=409)
        stream = session.open_get()
        http_event(session_id, "get_opened", 200, headers)
        return sse(session, stream, on_close=lambda: http_event(session_id, "get_closed", None, {}))

    async def delete(request: Request) -> Response:
        headers = _kept_headers(request.headers)
        session_id = request.headers.get("mcp-session-id")
        session = sessions.pop(session_id, None) if session_id else None
        if session is None:
            http_event(session_id, "unknown_session", 404, headers)
            return Response(status_code=404)
        http_event(session_id, "delete", 200, headers)
        await session.close()
        return Response(status_code=200)

    async def endpoint(request: Request) -> Response:
        handler = {"POST": post, "GET": get, "DELETE": delete}.get(request.method)
        if handler is None:
            return Response(status_code=405)
        return await handler(request)

    app = Starlette(routes=[Route(MCP_PATH, endpoint, methods=["GET", "POST", "DELETE"])])
    return app, sessions


async def _serve_http(argv: list[str], capture: Capture, sock: socket.socket, *, ping: bool,
                      serving: threading.Event) -> None:
    import uvicorn

    async with anyio.create_task_group() as tg:
        app, sessions = _build_app(argv, capture, tg, ping=ping)
        # h11, not "auto": with httptools installed, uvicorn hands the app an empty body for a
        # POST that asks to upgrade to h2c, which the JDK HttpClient sends on every request.
        config = uvicorn.Config(app, log_level="warning", lifespan="off", timeout_graceful_shutdown=1,
                                http="h11")
        server = uvicorn.Server(config)
        try:
            serving.set()
            await server.serve(sockets=[sock])
        finally:
            for session in list(sessions.values()):
                await session.close()
            tg.cancel_scope.cancel()


# --- entry point -----------------------------------------------------------------------


def _fixture_argv(server: str) -> list[str]:
    if server.startswith(("http://", "https://")):
        raise RecorderError("the recorder serves a fixture it launches itself; give a command, not a URL")
    argv = shlex.split(server)
    if not argv:
        raise RecorderError("empty fixture command")
    return argv


def run(server: str, *, transport: str = "stdio", out: Path, host: str = "127.0.0.1", port: int = 0,
        ping: bool = False, announce: Callable[[str], None] | None = None) -> Path:
    """Serve ``server`` until the client disconnects (stdio) or the process is stopped
    (HTTP), writing the capture to ``out``. Returns the capture path."""

    argv = _fixture_argv(server)
    announce = announce or (lambda text: print(text, file=sys.stderr, flush=True))
    if transport == "stdio":
        capture = Capture(out, fixture=server, transport="stdio")
        announce(f"mcp-mirror record: capture {out}")
        try:
            anyio.run(lambda: _serve_stdio(argv, capture, ping=ping))
        except KeyboardInterrupt:
            pass
        finally:
            capture.close()
        return out
    if transport != "http":
        raise RecorderError(f"unknown transport {transport!r}; use stdio or http")

    sock = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as exc:
        raise RecorderError(f"cannot listen on {host}:{port}: {exc}") from exc
    sock.listen(128)
    bound = sock.getsockname()[1]
    url = f"http://{host}:{bound}{MCP_PATH}"
    capture = Capture(out, fixture=server, transport="http", url=url)
    announce(f"mcp-mirror record: capture {out}")
    announce(f"mcp-mirror record: listening on {url}")
    # uvicorn traps SIGINT/SIGTERM while serving, then restores the previous handlers
    # and re-raises the signal. With the default SIGTERM handler that re-raise kills the
    # process before the fixtures are stopped and the ``end`` record is written, so the
    # handler it restores is this one: interrupt before serving starts, no-op after.
    serving = threading.Event()

    def on_signal(signum, _frame) -> None:
        if not serving.is_set():
            raise KeyboardInterrupt

    previous = {signum: signal.signal(signum, on_signal) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        anyio.run(lambda: _serve_http(argv, capture, sock, ping=ping, serving=serving))
    except KeyboardInterrupt:
        pass
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        capture.close()
        sock.close()
    return out
