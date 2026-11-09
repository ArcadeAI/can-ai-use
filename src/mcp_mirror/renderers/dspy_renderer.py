"""DSPy renderer (DESIGN.md section 8).

DSPy's MCP support is one converter, ``dspy.Tool.from_mcp_tool(session, tool)``, which
turns one MCP tool into a ``dspy.Tool`` over an MCP client the caller already holds.
DSPy opens no connection of its own: the session is the caller's. The renderer is that
caller. It opens an ``mcp.Client`` (the MCP SDK's high-level client, which the
``from_mcp_tool`` docstring names) over stdio, Streamable HTTP or HTTP+SSE with the
SDK's defaults, lists the tools on it, and hands each one to ``from_mcp_tool``.

The capture is what each converted ``dspy.Tool`` hands a model. Its
``format_as_litellm_function_call()`` builds the ``{"type": "function", ...}`` dict
DSPy's adapters pass an LM as ``tools=`` under native function calling. That dict
holds the tool's ``args`` (the input schema's properties) and a ``required`` list DSPy
recomputes as every property without a ``default``. ``dspy.ReAct`` instead prints
``str(tool)`` into its instructions, which carries the properties and no required list.

Tool results have three boundaries. ``direct`` is what ``Tool.acall`` returns with the
default ``result_mode="text"``. ``react`` is the observation a ``dspy.ReAct`` run puts in
front of the model on the next turn, driven by DSPy's ``DummyLM``. ``structured`` is
``Tool.acall`` under the documented non-default ``result_mode="structured"``.
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import asynccontextmanager
from importlib import util
from typing import Any

from ..models import RendererEvidence, ResultBoundary, ResultRep, ToolRep
from ..normalize import normalize_framework_result, normalize_function_tool
from ..source import ServerHandle
from . import RenderError
from ._common import dist_version

ID = "dspy"
# mcp.client.client is the 2.x SDK's high-level Client; the 1.x line has no such module.
REQUIRES = ("dspy", "mcp.client.client")

_TIMEOUT_SECONDS = 30
_SESSION_NOTE = "on the caller's own mcp.Client, which dspy.Tool.from_mcp_tool is handed and never configures"
# ChatAdapter prints each trajectory field under a "[[ ## name ## ]]" header.
_OBSERVATION = re.compile(r"\[\[ ## observation_0 ## \]\]\n(.*?)(?=\n\n\[\[ ## |\n\nRespond with |\Z)", re.S)


def available() -> bool:
    try:
        return all(util.find_spec(mod) is not None for mod in REQUIRES)
    except ModuleNotFoundError:
        return False


def get_renderer() -> "DSPyRenderer":
    return DSPyRenderer()


class DSPyRenderer:
    id = ID

    def versions(self) -> dict:
        return {"framework": dist_version("dspy"), "adapter": dist_version("dspy")}

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "dspy.Tool.from_mcp_tool(session, tool) over a caller-supplied mcp.Client; "
                "dspy.Tool.format_as_litellm_function_call()"
            ),
            capture_object=(
                "the {'type': 'function', 'function': {name, description, parameters}} dict "
                "dspy.Tool.format_as_litellm_function_call returns, which dspy adapters pass "
                "the LM as tools= under native function calling"
            ),
            capture_stage="provider_format",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "mcp.Client.protocol_version of the caller's session; DSPy opens no "
                "connection and negotiates nothing itself, so this is the MCP SDK default "
                "(mode='auto') the caller chose"
            ),
            limitation=(
                "No LM was called and no provider request was serialized. dspy.ReAct "
                "prints str(tool) into its instructions instead of this dict."
            ),
        )

    # ------------------------------------------------------------------
    # Connections: the caller's session
    # ------------------------------------------------------------------

    def _client(self, handle: ServerHandle, **options: Any):
        from mcp import Client

        options.setdefault("read_timeout_seconds", _TIMEOUT_SECONDS)
        # auth: an httpx2.Auth the conformance suite supplies, as a caller would.
        auth = options.pop("auth", None)
        if handle.transport == "sse":
            from mcp.client.sse import sse_client

            return Client(sse_client(handle.url, headers=dict(handle.headers) or None, auth=auth), **options)
        if handle.transport == "http":
            if not handle.headers and auth is None:
                return Client(handle.url, **options)
            from mcp.client.streamable_http import streamable_http_client
            from mcp.shared._httpx_utils import create_mcp_http_client

            http_client = create_mcp_http_client(headers=dict(handle.headers) or None, auth=auth)
            return Client(streamable_http_client(handle.url, http_client=http_client), **options)
        from mcp import StdioServerParameters

        return Client(StdioServerParameters(command=handle.command, args=list(handle.args)), **options)

    @asynccontextmanager
    async def _connected(self, handle: ServerHandle, **options: Any):
        """An entered mcp.Client, with the version it negotiated recorded."""

        async with self._client(handle, **options) as client:
            version = getattr(client, "protocol_version", None)
            if not version:
                raise RenderError("dspy renderer could not read the negotiated MCP protocol version")
            self._negotiated_mcp_spec_version = str(version)
            yield client

    @staticmethod
    async def _tools(client: Any, result_mode: str = "text") -> dict[str, Any]:
        """Every listed MCP tool, converted by dspy.Tool.from_mcp_tool.

        DSPy has no listing of its own; its documented pattern calls the caller's
        ``session.list_tools()`` once, which returns only the first page. The caller
        here follows ``nextCursor`` on its session, so pagination is the session's.
        """

        import dspy

        tools: list = []
        cursor = None
        while True:
            listed = await client.list_tools(cursor=cursor)
            tools.extend(listed.tools)
            cursor = listed.next_cursor
            if not cursor:
                break
        return {tool.name: dspy.Tool.from_mcp_tool(client, tool, result_mode=result_mode) for tool in tools}

    @staticmethod
    def _tool(tools: dict[str, Any], name: str) -> Any:
        if name not in tools:
            raise RenderError(f"dspy renderer found no tool named {name!r} (has: {sorted(tools)})")
        return tools[name]

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def render(self, server: ServerHandle) -> list[ToolRep]:
        try:
            return asyncio.run(self._render_async(server))
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"dspy renderer failed: {exc}") from exc

    async def _render_async(self, server: ServerHandle) -> list[ToolRep]:
        async with self._connected(server) as client:
            tools = await self._tools(client)
        # from_mcp_tool passes dspy.Tool only the name, the description and the input
        # schema's properties. The MCP tool's annotations, title, icons and outputSchema
        # are kept on no Tool field, so nothing is passed as retained metadata.
        return [normalize_function_tool(tool.format_as_litellm_function_call(), origin=self.id) for tool in tools.values()]

    # ------------------------------------------------------------------
    # Tool results
    # ------------------------------------------------------------------

    DEFAULT_RESULT_BOUNDARY = "react"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Tool return value",
                capture_api="dspy.Tool.acall on the Tool from_mcp_tool built (result_mode='text', the default)",
                capture_object=(
                    "what acall returns: a str, a list of str, or the non-text MCP content "
                    "objects, or the RuntimeError it raises for an isError result"
                ),
                agent_path=False,
            ),
            ResultBoundary(
                id="react",
                label="ReAct observation to the model",
                capture_api="dspy.ReAct(...).acall with the Tool; the messages of the next LM call",
                capture_object="the observation_0 field ChatAdapter prints into the next turn's prompt",
                agent_path=True,
                limitation="The LM is dspy.utils.dummies.DummyLM; no provider LM serialized the prompt.",
            ),
            ResultBoundary(
                id="structured",
                label="Tool return value, result_mode='structured'",
                capture_api="dspy.Tool.from_mcp_tool(..., result_mode='structured') then Tool.acall",
                capture_object="what acall returns: the structuredContent when the result has one, else as result_mode='text'",
                agent_path=False,
                limitation="A documented non-default option of from_mcp_tool.",
            ),
        ]

    def render_result(
        self,
        server: ServerHandle,
        tool: str,
        arguments: dict[str, Any],
        boundary: str | None = None,
    ) -> ResultRep:
        boundary = boundary or self.DEFAULT_RESULT_BOUNDARY
        runners = {
            "direct": lambda: self._direct(server, tool, arguments, "text"),
            "structured": lambda: self._direct(server, tool, arguments, "structured"),
            "react": lambda: self._react(server, tool, arguments),
        }
        if boundary not in runners:
            raise RenderError(f"dspy renderer has no result boundary {boundary!r}")
        try:
            return asyncio.run(runners[boundary]())
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001 - the failure mode is the observation
            return self._raised(tool, exc)

    def _raised(self, tool: str, exc: BaseException) -> ResultRep:
        # _convert_mcp_tool_result raises RuntimeError("Failed to call a MCP tool: ...")
        # for an isError result; that exception is the only error signal acall gives.
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            is_error=True,
            text=f"{type(exc).__name__}: {exc}",
            error_channel="exception",
            raw=repr(exc)[:2000],
        )

    async def _direct(self, server: ServerHandle, tool: str, arguments: dict[str, Any], result_mode: str) -> ResultRep:
        async with self._connected(server) as client:
            function = self._tool(await self._tools(client, result_mode), tool)
            try:
                value = await function.acall(**arguments)
            except Exception as exc:  # noqa: BLE001
                return self._raised(tool, exc)
        return self._value_rep(tool, value)

    def _value_rep(self, tool: str, value: Any) -> ResultRep:
        """What acall returned, as the caller holds it.

        With one text block acall returns that string; with several, a list of strings.
        Non-text blocks come back as the MCP content objects only when there is no text
        at all. In ``structured`` mode a result with structuredContent returns it alone.
        """

        if isinstance(value, str):
            return normalize_framework_result(tool=tool, origin=self.id, blocks=[{"type": "text", "text": value}], raw=value[:2000])
        if isinstance(value, list):
            blocks = [{"type": "text", "text": item} if isinstance(item, str) else _content(item) for item in value]
            return normalize_framework_result(tool=tool, origin=self.id, blocks=blocks, raw=_raw(value))
        # structured mode: the structuredContent value itself, with no content blocks.
        return normalize_framework_result(tool=tool, origin=self.id, structured_content=value, raw=_raw(value))

    async def _react(self, server: ServerHandle, tool: str, arguments: dict[str, Any]) -> ResultRep:
        import dspy
        from dspy.utils.dummies import DummyLM

        lm = DummyLM(
            [
                {"next_thought": "call the tool", "next_tool_name": tool, "next_tool_args": arguments},
                {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
                {"reasoning": "done", "answer": "done"},
            ]
        )
        async with self._connected(server) as client:
            function = self._tool(await self._tools(client), tool)
            react = dspy.ReAct("question -> answer", tools=[function], max_iters=2)
            with dspy.context(lm=lm):
                prediction = await react.acall(question="Call the tool once, then finish.")
        if len(lm.history) < 2:
            raise RenderError("dspy ReAct run never made a second LM call")
        prompt = "\n\n".join(str(m.get("content")) for m in lm.history[1]["messages"] if m.get("role") == "user")
        match = _OBSERVATION.search(prompt)
        if not match:
            raise RenderError("dspy ReAct prompt carried no observation_0 field")
        text = match.group(1)
        observation = prediction.trajectory.get("observation_0")
        # ReAct catches a raising tool and stores "Execution error in <name>: <traceback>"
        # as the observation; the model gets that text and no error flag.
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=[{"type": "text", "text": text}],
            text=text,
            raw=json.dumps({"observation_0": _raw(observation)})[:4000],
        )

    # ------------------------------------------------------------------
    # Resources and prompts
    # ------------------------------------------------------------------

    def probe_surface(self, server: ServerHandle) -> dict:
        """DSPy converts MCP tools and nothing else; that is the finding."""
        from .. import probe as P

        return P.unsupported(
            "dspy's MCP support is dspy.Tool.from_mcp_tool, which converts one tool; it has "
            "no resource or prompt API, and resources and prompts stay on the caller's MCP "
            "session, which DSPy never reads"
        )

    # ------------------------------------------------------------------
    # Sampling, elicitation and roots
    # ------------------------------------------------------------------

    def probe_client_features(self, server: ServerHandle) -> dict:
        """Hooks on the caller's mcp.Client, exercised through a dspy.Tool call.

        DSPy takes no sampling, elicitation or roots handler. The session it calls
        through is the caller's, so the handlers are registered there, on the MCP SDK's
        ``mcp.Client``, and every tool is called through ``dspy.Tool.acall``. That is
        recorded as the ``underlying`` route: the client library DSPy is handed.
        """
        return asyncio.run(self._client_features_async(server))

    async def _client_features_async(self, server: ServerHandle) -> dict:
        import mcp.types as types

        from .. import client_probe as C

        log = C.HandlerLog()
        sdk = C.sdk_callbacks(log)

        async def call(tool: str, options: dict | None = None, before=None) -> ResultRep:
            async with self._connected(server, **(options or {})) as client:
                if before is not None:
                    await before(client)
                function = self._tool(await self._tools(client), tool)
                try:
                    return self._value_rep(tool, await function.acall())
                except Exception as exc:  # noqa: BLE001
                    return self._raised(tool, exc)

        sampling = {"sampling_callback": sdk["sampling_callback"]}
        sampling_tools = {
            **sampling,
            "sampling_capabilities": types.SamplingCapability(tools=types.SamplingToolsCapability()),
        }
        elicit = {"elicitation_callback": sdk["elicitation_callback"]}
        roots = {"list_roots_callback": sdk["list_roots_callback"]}

        async def changed_roots(client: Any) -> None:
            await client.send_roots_list_changed()

        hooks = {
            "sampling": ("ask_sampling", "mcp.Client(sampling_callback=...)", sampling, None),
            "sampling.tools": (
                "ask_sampling_tools", "mcp.Client(sampling_callback=..., sampling_capabilities=...)", sampling_tools, None,
            ),
            "elicitation.form": ("ask_elicitation_form", "mcp.Client(elicitation_callback=...)", elicit, None),
            "elicitation.url": ("ask_elicitation_url", "mcp.Client(elicitation_callback=...)", elicit, None),
            "roots.list": ("list_client_roots", "mcp.Client(list_roots_callback=...)", roots, None),
            "roots.list_changed": (
                "roots_changes_seen", "mcp.Client(list_roots_callback=...).send_roots_list_changed()", roots, changed_roots,
            ),
        }
        out: dict = {}
        for feature, (tool, hook, options, before) in hooks.items():
            out[feature] = await C.attempt_async(
                feature, f"{hook} {_SESSION_NOTE}", "underlying",
                lambda tool=tool, options=options, before=before: call(tool, options, before), log,
            )
        try:
            rep = await call(C.CAPABILITIES_TOOL, {**sampling_tools, **elicit, **roots})
            out["capabilities"] = {"text": C.rep_text(rep)[0]}
        except Exception as exc:  # noqa: BLE001
            out["capabilities"] = {"text": f"{type(exc).__name__}: {exc}"}
        return out

    # ------------------------------------------------------------------
    # Server utilities
    # ------------------------------------------------------------------

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through dspy.Tool and the caller's session.

        The converted Tool calls ``session.call_tool(name, arguments=kwargs)`` and
        nothing else, so it passes no progress callback. Log messages, logging/setLevel,
        completion and re-listing are reachable only on the caller's ``mcp.Client``,
        reported with ``route: "session"``.
        """
        return asyncio.run(self._utilities_async(servers))

    async def _utilities_async(self, servers: dict[str, ServerHandle]) -> dict:
        import mcp.types as types

        from .. import probe as P

        out: dict = {}

        async def call(feature: str, tool: str, arguments: dict | None = None, **options: Any) -> Any:
            async with self._connected(servers[feature], **options) as client:
                return await self._tool(await self._tools(client), tool).acall(**(arguments or {}))

        if "tools.list" in servers:
            async def _names():
                async with self._connected(servers["tools.list"]) as client:
                    return {"route": "session", "api": "mcp.Client.list_tools, each tool through dspy.Tool.from_mcp_tool",
                            "names": list(await self._tools(client))}

            out["tools.list"] = await P.attempt_async(_names)
        if "progress" in servers:
            answer = await P.attempt_async(lambda: _wrap(call("progress", "slow_with_progress")))
            out["progress"] = {
                "ok": False,
                "unsupported": "the Tool from_mcp_tool builds calls session.call_tool(name, arguments=kwargs) with no progress_callback, and takes none",
                "text": answer.get("text"),
                "call": answer,
            }
        if "logging" in servers:
            async def _logging():
                messages: list[dict] = []

                async def on_log(params) -> None:  # noqa: ANN001
                    messages.append({"level": str(getattr(params, "level", None)), "data": getattr(params, "data", None)})

                text = await call("logging", "emit_logs", logging_callback=on_log, log_level="debug")
                return {"route": "session", "api": f"mcp.Client(logging_callback=..., log_level='debug') {_SESSION_NOTE}",
                        "messages": messages, "text": _text(text)}

            out["logging"] = await P.attempt_async(_logging)
        if "set_level" in servers:
            async def _set_level():
                async with self._connected(servers["set_level"]) as client:
                    await client.set_logging_level(P.SET_LEVEL)
                    text = await self._tool(await self._tools(client), "emit_logs").acall()
                    return {"route": "session", "api": f"mcp.Client.set_logging_level {_SESSION_NOTE}",
                            "level": P.SET_LEVEL, "text": _text(text)}

            out["set_level"] = await P.attempt_async(_set_level)
        if "completion" in servers:
            async def _completion():
                async with self._connected(servers["completion"]) as client:
                    name, argument, value = P.COMPLETION_PROMPT
                    prompt = await client.complete(
                        types.PromptReference(type="ref/prompt", name=name), {"name": argument, "value": value}
                    )
                    uri, argument, value = P.COMPLETION_RESOURCE
                    resource = await client.complete(
                        types.ResourceTemplateReference(type="ref/resource", uri=uri), {"name": argument, "value": value}
                    )
                    return {"route": "session", "api": f"mcp.Client.complete {_SESSION_NOTE}",
                            "prompt": list(prompt.completion.values), "resource": list(resource.completion.values)}

            out["completion"] = await P.attempt_async(_completion)
        if "cancellation" in servers:
            async def _cancellation():
                async with self._connected(servers["cancellation"]) as client:
                    function = self._tool(await self._tools(client), "cancellable_wait")
                    loop = asyncio.get_running_loop()
                    started = loop.time()
                    task = asyncio.create_task(function.acall(seconds=P.CANCEL_WAIT_S))
                    await asyncio.sleep(P.CANCEL_AFTER_S)
                    task.cancel()
                    try:
                        outcome = f"returned: {_text(await task)}"
                    except asyncio.CancelledError:
                        outcome = "cancelled"
                    except Exception as exc:  # noqa: BLE001
                        outcome = f"threw: {type(exc).__name__}: {exc}"
                    abandoned = loop.time() - started
                    await asyncio.sleep(0.3)
                    return {"route": "adapter", "mechanism": "asyncio task.cancel() around dspy.Tool.acall",
                            "abandoned_after_s": round(abandoned, 3), "outcome": outcome}

            out["cancellation"] = await P.attempt_async(_cancellation)
        if "tasks" in servers:
            out["tasks"] = await P.attempt_async(lambda: _wrap(call("tasks", "task_capable"), route="adapter"))
        if "caching" in servers:
            async def _caching():
                async with self._connected(servers["caching"]) as client:
                    marks = []
                    for step in ("first", "within_ttl", "after_ttl"):
                        if step == "after_ttl":
                            await asyncio.sleep(P.CACHE_PAUSE_S)
                        started = P.wall_clock()
                        # DSPy has no relist; a caller lists on its session and converts again.
                        count = len(await self._tools(client))
                        marks.append({"call": step, "started": started, "finished": P.wall_clock(), "count": count})
                    return {"route": "session", "api": "mcp.Client.list_tools, then dspy.Tool.from_mcp_tool", "marks": marks}

            out["caching"] = await P.attempt_async(_caching)
        return out

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes against fixtures/notify_server.py.

        DSPy has no notification hook and no relist: a caller that wants the new tools
        lists them again on its session and converts each with ``from_mcp_tool``, which
        is what the probe does. DSPy converts tools only, so resources and prompts have
        no listing, and subscriptions go through the caller's session.
        """
        return asyncio.run(self._notify_async(server))

    async def _notify_async(self, server: ServerHandle) -> dict:
        import anyio

        from .. import notify as N

        recorder = N.Recorder()
        async with self._connected(server) as client, anyio.create_task_group() as group:
            negotiated = self._negotiated_mcp_spec_version
            subscriptions = N.SessionSubscriptions(client.session, recorder, negotiated, group)

            async def tools():
                return list(await self._tools(client))

            async def call(name):
                return await self._tool(await self._tools(client), name).acall()

            out = await N.run(
                N.Adapter(
                    call=call,
                    recorder=recorder,
                    hook=None,
                    lists={"tools": tools, "resources": None, "prompts": None},
                    no_list_reason="dspy.Tool.from_mcp_tool converts tools only; DSPy has no resource or prompt API",
                    subscribe=subscriptions.subscribe,
                    unsubscribe=subscriptions.unsubscribe,
                    subscribe_route="session",
                    subscribe_api=f"the caller's mcp.Client.session: {subscriptions.api}",
                    listen=subscriptions.listen_all,
                    listen_route="session",
                    listen_api="mcp.client.subscriptions.listen(the caller's mcp.Client.session, ...)",
                    negotiated=negotiated,
                )
            )
            group.cancel_scope.cancel()
        return out


def _content(item: Any) -> dict:
    """An MCP content object acall handed back, in wire form."""

    dump = getattr(item, "model_dump", None)
    if callable(dump):
        return dump(mode="json", by_alias=True, exclude_none=True)
    return {"type": type(item).__name__, "text": str(item)}


def _text(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(item if isinstance(item, str) else json.dumps(_content(item)) for item in value)
    return json.dumps(value, default=str)


def _raw(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps([item if isinstance(item, str) else _content(item) for item in value], default=str)[:2000]
    if isinstance(value, str):
        return value[:2000]
    return json.dumps(value, default=str)[:2000]


async def _wrap(awaitable, route: str | None = None) -> dict:
    text = _text(await awaitable)
    return {"route": route, "text": text} if route else {"text": text}
