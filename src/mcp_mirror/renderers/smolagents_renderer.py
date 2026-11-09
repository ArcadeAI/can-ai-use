"""smolagents renderer (DESIGN.md section 8).

Connects through ``smolagents.MCPClient``, the client smolagents documents for MCP.
It is a thin first-party wrapper over the separately maintained ``mcpadapt`` package:
``MCPClient`` builds an ``mcpadapt.core.MCPAdapt`` with
``mcpadapt.smolagents_adapter.SmolAgentsAdapter``, which opens one ``ClientSession`` on
a background event loop and turns each MCP tool into a ``smolagents.Tool`` subclass
(``MCPAdaptTool``) with a name, a description and an ``inputs`` dict built from the
input schema's ``properties``. ``structured_output`` is left at its default (False).

The capture is the ``tools`` list a real ``smolagents.ToolCallingAgent`` run builds for
its model: ``Model._prepare_completion_kwargs`` turns each Tool into
``get_tool_json_schema(tool)``, the function-tool dict every API model class sends.
smolagents ships no offline model, so the model is a scripted ``smolagents.models.Model``
subclass that calls ``_prepare_completion_kwargs`` the way ``OpenAIModel.generate``
does, records the result and never sends a request.

Tool results have two boundaries. Calling the Tool returns whatever
``MCPAdaptTool.forward`` builds from the CallToolResult; the agent run then puts that
value's ``str()`` into the observation message the next model turn receives.
"""

from __future__ import annotations

import json
import warnings
from contextlib import contextmanager
from functools import lru_cache
from importlib import util
from typing import Any

from ..models import RendererEvidence, ResultBoundary, ResultRep, ToolRep
from ..normalize import normalize_framework_result, normalize_function_tool
from ..source import ServerHandle
from . import RenderError
from ._common import dist_version

ID = "smolagents"
REQUIRES = ("smolagents", "mcpadapt")

_CALL_ID = "mcp-mirror-call"
_FINAL_ANSWER = "final_answer"


def available() -> bool:
    try:
        return all(util.find_spec(mod) is not None for mod in REQUIRES)
    except ModuleNotFoundError:
        return False


def get_renderer() -> "SmolagentsRenderer":
    return SmolagentsRenderer()


@lru_cache(maxsize=1)
def _scripted_model_class():
    """A ``Model`` that answers from a script instead of a provider.

    ``generate`` builds the completion kwargs exactly as ``OpenAIModel.generate`` does
    and hands them to ``responses(turn, completion_kwargs)``, which returns the
    ``ChatMessage`` for that turn. Nothing is serialized or sent.
    """

    from smolagents.models import Model

    class ScriptedModel(Model):
        def __init__(self, responses, **kwargs):
            super().__init__(model_id="mcp-mirror-scripted", **kwargs)
            self.responses = responses
            self.turns: list[dict] = []

        def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
            completion_kwargs = self._prepare_completion_kwargs(
                messages=messages,
                stop_sequences=stop_sequences,
                response_format=response_format,
                tools_to_call_from=tools_to_call_from,
                model=self.model_id,
                custom_role_conversions=None,
                convert_images_to_image_urls=True,
                **kwargs,
            )
            self.turns.append(completion_kwargs)
            return self.responses(len(self.turns))

    return ScriptedModel


def _tool_call(name: str, arguments: dict[str, Any]):
    from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole

    return ChatMessage(
        role=MessageRole.ASSISTANT,
        content=None,
        tool_calls=[
            ChatMessageToolCall(
                id=_CALL_ID,
                type="function",
                function=ChatMessageToolCallFunction(name=name, arguments=arguments),
            )
        ],
    )


class SmolagentsRenderer:
    id = ID

    def versions(self) -> dict:
        return {"framework": dist_version("smolagents"), "adapter": dist_version("mcpadapt")}

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api=(
                "smolagents.MCPClient.get_tools handed to smolagents.ToolCallingAgent; "
                "the tools Model._prepare_completion_kwargs builds on the first turn"
            ),
            capture_object=(
                "function-tool dicts smolagents.models.get_tool_json_schema builds from "
                "each mcpadapt MCPAdaptTool"
            ),
            capture_stage="provider_format",
            provider_request_captured=False,
            negotiated_mcp_spec_version=getattr(self, "_negotiated_mcp_spec_version", None),
            protocol_version_evidence=(
                "protocolVersion instrumented from ClientSession.initialize inside the "
                "mcpadapt MCPAdapt that MCPClient builds; the adapter does not expose it"
            ),
            limitation=(
                "The model is a scripted smolagents.models.Model subclass; no provider "
                "client serialized a request. CodeAgent, smolagents' other agent type, "
                "renders tools as Python signatures in its prompt instead and was not "
                "captured."
            ),
        )

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    @staticmethod
    def _params(handle: ServerHandle) -> Any:
        if handle.transport in {"http", "sse"}:
            params: dict[str, Any] = {
                "url": handle.url,
                "transport": "sse" if handle.transport == "sse" else "streamable-http",
            }
            if handle.headers:
                params["headers"] = dict(handle.headers)
            return params
        from mcp import StdioServerParameters

        return StdioServerParameters(command=handle.command, args=list(handle.args))

    @contextmanager
    def _client(self, handle: ServerHandle):
        """A connected MCPClient, with the version its ClientSession negotiated recorded."""

        from unittest.mock import patch

        from mcp.client.session import ClientSession
        from smolagents import MCPClient

        versions: list[str] = []
        original_initialize = ClientSession.initialize

        async def capture_initialize(session, *args, **kwargs):
            result = await original_initialize(session, *args, **kwargs)
            versions.append(str(result.protocolVersion))
            return result

        with patch.object(ClientSession, "initialize", capture_initialize), warnings.catch_warnings():
            # Leaving structured_output unset is the default; it only warns that the
            # default may change.
            warnings.simplefilter("ignore", FutureWarning)
            client = MCPClient(self._params(handle))
        try:
            if len(set(versions)) != 1:
                raise RenderError(
                    f"smolagents renderer could not establish one negotiated MCP protocol version (observed: {versions})"
                )
            self._negotiated_mcp_spec_version = versions[0]
            yield client
        finally:
            client.disconnect()

    @staticmethod
    def _tool(client: Any, name: str) -> Any:
        tools = {tool.name: tool for tool in client.get_tools()}
        if name not in tools:
            raise RenderError(f"smolagents renderer found no tool named {name!r} (has: {sorted(tools)})")
        return tools[name]

    def _run_agent(self, client: Any, script) -> list[dict]:
        """One ``ToolCallingAgent.run`` with the scripted model; returns each turn's kwargs."""

        from smolagents import ToolCallingAgent

        model = _scripted_model_class()(responses=script)
        agent = ToolCallingAgent(tools=client.get_tools(), model=model, max_steps=3, verbosity_level=-1)
        agent.run("go")
        return model.turns

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def render(self, server: ServerHandle) -> list[ToolRep]:
        try:
            with self._client(server) as client:
                turns = self._run_agent(client, lambda _turn: _tool_call(_FINAL_ANSWER, {"answer": "done"}))
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderError(f"smolagents renderer failed: {exc}") from exc
        if not turns:
            raise RenderError("smolagents agent run never called the model")
        # ToolCallingAgent always adds its own final_answer tool; it is not from MCP.
        # MCPAdaptTool keeps the tool's name, description and input properties only:
        # annotations, title, icons and the outputSchema (with the default
        # structured_output=False) are not kept anywhere on the Tool.
        return [
            normalize_function_tool(tool, origin=self.id)
            for tool in turns[0].get("tools", [])
            if tool["function"]["name"] != _FINAL_ANSWER
        ]

    # ------------------------------------------------------------------
    # Tool results
    # ------------------------------------------------------------------

    DEFAULT_RESULT_BOUNDARY = "agent"

    def result_boundaries(self) -> list[ResultBoundary]:
        return [
            ResultBoundary(
                id="direct",
                label="Tool return value",
                capture_api="calling the smolagents.Tool (mcpadapt MCPAdaptTool) MCPClient.get_tools returns",
                capture_object="the value MCPAdaptTool.forward builds from the CallToolResult",
                agent_path=False,
            ),
            ResultBoundary(
                id="agent",
                label="Observation to the model",
                capture_api="smolagents.ToolCallingAgent.run, the messages of the next model turn",
                capture_object="the observation message (role tool-response, sent as user) the next turn's completion kwargs carry",
                agent_path=True,
                limitation=(
                    "The model is a scripted smolagents.models.Model subclass; no "
                    "provider client serialized the message."
                ),
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
        if boundary not in {"direct", "agent"}:
            raise RenderError(f"smolagents renderer has no result boundary {boundary!r}")
        runner = self._direct if boundary == "direct" else self._agent
        try:
            return runner(server, tool, arguments)
        except RenderError:
            raise
        except Exception as exc:  # noqa: BLE001 - the failure mode is the observation
            return normalize_framework_result(
                tool=tool,
                origin=self.id,
                is_error=True,
                text=f"{type(exc).__name__}: {exc}",
                error_channel="exception",
                raw=repr(exc)[:2000],
            )

    def _direct(self, server: ServerHandle, tool: str, arguments: dict[str, Any]) -> ResultRep:
        with self._client(server) as client:
            returned = self._tool(client, tool)(**arguments)
        return _returned_rep(tool, self.id, returned)

    def _agent(self, server: ServerHandle, tool: str, arguments: dict[str, Any]) -> ResultRep:
        def script(turn: int):
            if turn == 1:
                return _tool_call(tool, arguments)
            return _tool_call(_FINAL_ANSWER, {"answer": "done"})

        with self._client(server) as client:
            self._tool(client, tool)
            turns = self._run_agent(client, script)
        if len(turns) < 2:
            raise RenderError("smolagents agent run never returned an observation to the model")
        messages = turns[1]["messages"]
        # ActionStep.to_messages writes the call as a tool-call message and then either
        # "Observation:\n<str(output)>" or, when the tool raised, an "Error:\n..." user
        # message; both arrive as role "user" after tool_role_conversions.
        call = _last_call(messages)
        if call is not None:
            replies = messages[messages.index(call) + 1 :]
        else:
            # When smolagents rejects the arguments before calling the tool (its schema
            # marks every input required), no call message is written and the "Error:"
            # text is merged into the trailing user message. What the model newly sees
            # is that message minus the text it already held on turn 1.
            replies = _appended(turns[0]["messages"], messages)
        if not replies:
            raise RenderError("smolagents agent run never returned an observation to the model")
        blocks: list[dict] = []
        for message in replies:
            content = message.get("content")
            for part in content if isinstance(content, list) else [{"type": "text", "text": content}]:
                if part.get("type") == "text":
                    blocks.append({"type": "text", "text": part.get("text")})
                else:
                    blocks.append({"type": part.get("type"), "data": "<omitted>"})
        text = "\n".join(b["text"] for b in blocks if b.get("type") == "text" and b.get("text"))
        # The observation is plain text with no error field. A tool that raised reaches
        # the model as an "Error:" message, which is the only failure signal it gets.
        failed = any(isinstance(b.get("text"), str) and b["text"].startswith("Error:") for b in blocks)
        return normalize_framework_result(
            tool=tool,
            origin=self.id,
            blocks=blocks,
            is_error=True if failed else None,
            text=text,
            error_channel="message" if failed else None,
            raw=json.dumps(replies, default=str)[:4000],
        )

    # ------------------------------------------------------------------
    # Resources and prompts
    # ------------------------------------------------------------------

    def probe_surface(self, server: ServerHandle) -> dict:
        """MCPClient exposes tools and nothing else; that is the finding."""
        from .. import probe as P

        return P.unsupported(
            "smolagents.MCPClient exposes only get_tools(); mcpadapt keeps its ClientSession private"
        )

    # ------------------------------------------------------------------
    # Sampling, elicitation and roots
    # ------------------------------------------------------------------

    def probe_client_features(self, server: ServerHandle) -> dict:
        """MCPClient takes server parameters, adapter_kwargs (connect and session
        timeouts) and structured_output only: mcpadapt opens its ClientSession with no
        sampling, elicitation or roots callback, and there is no option to pass one."""
        from .. import client_probe as C

        return C.without_hooks(lambda tool: self.render_result(server, tool, {}, "direct"))

    # ------------------------------------------------------------------
    # Server utilities
    # ------------------------------------------------------------------

    def probe_utilities(self, servers: dict[str, ServerHandle]) -> dict:
        """Server utilities through MCPClient, which hands out Tools and nothing else.

        Each tool is still called where the fixture needs a call to show anything, so
        the wire records what the adapter sends (no progress token, no log level).
        """
        from .. import probe as P

        out: dict = {}

        def call(feature: str, tool: str, arguments: dict | None = None) -> str | None:
            with self._client(servers[feature]) as client:
                return P.result_text(self._tool(client, tool)(**(arguments or {})))

        if "tools.list" in servers:
            def _names():
                with self._client(servers["tools.list"]) as client:
                    return {"route": "adapter", "api": "MCPClient.get_tools", "names": [t.name for t in client.get_tools()]}

            out["tools.list"] = P.attempt(_names)
        for feature, tool, reason in (
            ("progress", "slow_with_progress", "MCPClient takes no progress callback and mcpadapt sends no progressToken"),
            ("logging", "emit_logs", "mcpadapt builds its ClientSession without a logging_callback"),
        ):
            if feature in servers:
                answer = P.attempt(lambda feature=feature, tool=tool: {"text": call(feature, tool)})
                out[feature] = {"ok": False, "unsupported": reason, "text": answer.get("text"), "call": answer}
        for feature, reason in (
            ("set_level", "MCPClient has no logging/setLevel API; the session lives on the private MCPClient._adapter"),
            ("completion", "MCPClient has no completion API; the session lives on the private MCPClient._adapter"),
            ("cancellation", "a smolagents Tool call blocks on the adapter loop's future with no timeout or cancel route"),
            ("caching", "MCPClient.get_tools returns the list built at connect and never relists, so there is no caching decision to observe"),
        ):
            if feature in servers:
                out[feature] = {"ok": False, "unsupported": reason}
        if "tasks" in servers:
            out["tasks"] = P.attempt(lambda: {"route": "adapter", "text": call("tasks", "task_capable")})
        return out

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def probe_notifications(self, server: ServerHandle) -> dict:
        """Catalogue changes against fixtures/notify_server.py, through MCPClient.

        MCPClient takes no message handler and exposes only ``get_tools()``, which
        returns the tools adapted at connect. The probe relists through it, which is
        what an agent is handed, and calls the fixture's tools through the Tools.
        """
        import anyio

        from .. import notify as N

        with self._client(server) as client:
            recorder = N.Recorder()

            async def tools():
                return [t.name for t in client.get_tools()]

            async def call(name):
                tool = self._tool(client, name)
                return await anyio.to_thread.run_sync(lambda: tool())

            return anyio.run(
                N.run,
                N.Adapter(
                    call=call,
                    recorder=recorder,
                    hook=None,
                    lists={"tools": tools, "resources": None, "prompts": None},
                    no_list_reason="smolagents.MCPClient exposes only get_tools(); no resource or prompt API",
                    no_subscribe_reason="smolagents.MCPClient exposes only get_tools(); no subscribe API",
                    negotiated=self._negotiated_mcp_spec_version,
                ),
            )


def _last_call(messages: list[dict]) -> dict | None:
    """The last message that carries the scripted tool call."""

    calls = [m for m in messages if _CALL_ID in json.dumps(m.get("content"), default=str)]
    return calls[-1] if calls else None


def _appended(before: list[dict], after: list[dict]) -> list[dict]:
    """The text the trailing user message gained between two turns, as one message."""

    def text(message: dict) -> str:
        content = message.get("content")
        if isinstance(content, list):
            return "\n".join(p.get("text") or "" for p in content if p.get("type") == "text")
        return content or ""

    if not before or not after:
        return []
    old, new = text(before[-1]), text(after[-1])
    if not new.startswith(old) or new == old:
        return []
    return [{"role": after[-1].get("role"), "content": new[len(old) :].lstrip("\n")}]


def _returned_rep(tool: str, origin: str, returned: Any) -> ResultRep:
    """What ``MCPAdaptTool.forward`` returned, classified by what the caller can tell.

    ``forward`` reads only the first content block: text comes back as a ``str``, an
    image as a ``PIL.Image``, audio as a torchaudio tensor. ``isError`` is never read.
    """

    if isinstance(returned, str):
        return normalize_framework_result(tool=tool, origin=origin, blocks=[{"type": "text", "text": returned}], raw=returned[:2000])
    if isinstance(returned, (dict, list)):
        return normalize_framework_result(
            tool=tool, origin=origin, structured_content=returned, text=json.dumps(returned, default=str), raw=returned
        )
    if type(returned).__module__.startswith("PIL"):
        mime = f"image/{(getattr(returned, 'format', None) or 'png').lower()}"
        return normalize_framework_result(
            tool=tool, origin=origin, blocks=[{"type": "image", "mimeType": mime}], text="", raw=repr(returned)[:2000]
        )
    if type(returned).__module__.startswith("torch"):
        return normalize_framework_result(tool=tool, origin=origin, blocks=[{"type": "audio"}], text="", raw=repr(returned)[:2000])
    return normalize_framework_result(tool=tool, origin=origin, text=str(returned), raw=repr(returned)[:2000])
