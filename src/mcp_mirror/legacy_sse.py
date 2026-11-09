"""Serve a stdio fixture over the deprecated HTTP+SSE transport, recording the wire.

The 2024-11-05 transport the 2025-03-26 revision replaced with Streamable HTTP: the
client opens ``GET /sse``, the server's first event names a message endpoint, and every
client message is a ``POST`` to that endpoint answered ``202 Accepted``. Everything the
server says, answers included, arrives on the one SSE stream.

    python -m mcp_mirror.legacy_sse "python fixtures/results_server.py" --out capture.jsonl

Each ``GET /sse`` gets its own fixture process. The capture uses the recorder's format
(``mcp-mirror.wire/1``) with ``transport: "sse"``, so the cell rules read it the same
way they read a Streamable HTTP capture. ``POST /sse`` is answered 405, which is how a
client that tries Streamable HTTP first (Mastra) learns to fall back.
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import threading
import sys
import uuid
from pathlib import Path

import anyio

from .recorder import (
    CLIENT_TO_SERVER,
    SERVER_TO_CLIENT,
    Capture,
    RecorderError,
    Session,
    _encode,
    _fixture_argv,
    _kept_headers,
    _lines,
    _send,
    _spawn,
    _stop,
)

SSE_PATH = "/sse"
MESSAGES_PATH = "/messages"


def _build_app(argv: list[str], capture: Capture, task_group):
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response, StreamingResponse
    from starlette.routing import Route

    sessions: dict[str, tuple[Session, object]] = {}

    def http_event(session: str | None, event: str, status: int | None, headers: dict) -> None:
        capture.write({"type": "http", "session": session, "event": event, "status": status, "headers": headers})

    async def open_stream(request: Request) -> Response:
        headers = _kept_headers(request.headers)
        if request.method != "GET":
            http_event(None, "post_to_sse", 405, headers)
            return Response(status_code=405)
        key = uuid.uuid4().hex
        session = Session(capture, key)
        process = await _spawn(argv)
        send, receive = anyio.create_memory_object_stream[bytes](float("inf"))
        sessions[key] = (session, process)
        http_event(key, "get_opened", 200, headers)

        async def pump() -> None:
            async for line in _lines(process):
                for message in session.observe_raw(SERVER_TO_CLIENT, line, origin="server",
                                                   extra={"route": "get"}):
                    try:
                        send.send_nowait(b"event: message\ndata: " + _encode(message).rstrip(b"\n") + b"\n\n")
                    except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                        return
            http_event(key, "fixture_exited", None, {})
            send.close()

        task_group.start_soon(pump)

        async def body():
            yield f"event: endpoint\ndata: {MESSAGES_PATH}?session_id={key}\n\n".encode()
            try:
                async with receive:
                    async for chunk in receive:
                        yield chunk
            finally:
                sessions.pop(key, None)
                http_event(key, "get_closed", None, {})
                await _stop(process)

        return StreamingResponse(body(), media_type="text/event-stream", headers={"cache-control": "no-cache"})

    async def post_message(request: Request) -> Response:
        headers = _kept_headers(request.headers)
        key = request.query_params.get("session_id") or ""
        entry = sessions.get(key)
        if entry is None:
            http_event(key or None, "unknown_session", 404, headers)
            return Response(status_code=404)
        session, process = entry
        try:
            payload = json.loads(await request.body())
        except ValueError:
            http_event(key, "parse_error", 400, headers)
            return Response(status_code=400)
        for message in session.observe(CLIENT_TO_SERVER, payload, origin="client",
                                       extra={"http": {"method": "POST", "headers": headers}}):
            await _send(process, _encode(message))
        return Response("Accepted", status_code=202)

    routes = [
        Route(SSE_PATH, open_stream, methods=["GET", "POST"]),
        Route(MESSAGES_PATH, post_message, methods=["POST"]),
    ]
    return Starlette(routes=routes), sessions


async def _serve(argv: list[str], capture: Capture, sock: socket.socket, serving: threading.Event) -> None:
    import uvicorn

    async with anyio.create_task_group() as tg:
        app, sessions = _build_app(argv, capture, tg)
        # h11, not "auto": with httptools installed, uvicorn hands the app an empty body for a
        # POST that asks to upgrade to h2c, which the JDK HttpClient sends on every request.
        config = uvicorn.Config(app, log_level="warning", lifespan="off", timeout_graceful_shutdown=1,
                                http="h11")
        try:
            serving.set()
            await uvicorn.Server(config).serve(sockets=[sock])
        finally:
            for _session, process in list(sessions.values()):
                await _stop(process)  # type: ignore[arg-type]
            tg.cancel_scope.cancel()


def run(server: str, *, out: Path, host: str = "127.0.0.1", port: int = 0) -> Path:
    """Serve ``server`` until the process is stopped, writing the capture to ``out``."""

    argv = _fixture_argv(server)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as exc:
        raise RecorderError(f"cannot listen on {host}:{port}: {exc}") from exc
    sock.listen(128)
    url = f"http://{host}:{sock.getsockname()[1]}{SSE_PATH}"
    capture = Capture(out, fixture=server, transport="sse", url=url)
    print(f"mcp-mirror legacy-sse: capture {out}", file=sys.stderr, flush=True)
    print(f"mcp-mirror legacy-sse: listening on {url}", file=sys.stderr, flush=True)
    # As in the recorder: uvicorn re-raises the signal it trapped once it has shut
    # down, so the handler it restores must not kill the process before the fixtures
    # are stopped and the capture's ``end`` record is written.
    serving = threading.Event()

    def on_signal(signum, _frame) -> None:
        if not serving.is_set():
            raise KeyboardInterrupt

    previous = {signum: signal.signal(signum, on_signal) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        anyio.run(lambda: _serve(argv, capture, sock, serving))
    except KeyboardInterrupt:
        pass
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        capture.close()
        sock.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("server", help="fixture command, launched over stdio per SSE connection")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    arguments = parser.parse_args()
    try:
        run(arguments.server, out=arguments.out, host=arguments.host, port=arguments.port)
    except RecorderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
