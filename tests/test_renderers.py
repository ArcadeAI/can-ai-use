"""Integration tests: each installed renderer vs the tricky fixture (DESIGN.md section 14).

These spawn ``fixtures/definitions_server.py`` (which ``tricky_server.py`` aliases) over stdio and drive each framework's real
adapter. Renderers whose framework is not installed are skipped automatically, so the
suite is green whether you installed ``[langchain]``, ``[all]``, or nothing.
"""

from __future__ import annotations

import json
import os
import shlex
import tempfile
import subprocess
import shutil
import sys
from pathlib import Path

import pytest

from mcp_mirror.cli import _render_isolated
from mcp_mirror.diff import diff_reps
from mcp_mirror.models import Dimension, FrameworkRun, Report
from mcp_mirror.renderers import available_renderers
from mcp_mirror.scorecard import build_scorecard
from mcp_mirror.source import load_source, parse_server

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "validator"))
from validate import validate_all  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "definitions_server.py"


def _published_rows(snapshot: Path) -> dict[str, dict]:
    """Capability rows as the site publishes them: overlays merged with derived cells."""
    _root, capabilities, errors = validate_all(snapshot)
    assert errors == [], errors
    return {capability["id"]: capability for capability in capabilities}
FRAMEWORK_DATA = (
    Path(__file__).resolve().parents[1] / "data" / "frameworks.json"
)
RENDERERS = available_renderers()

EXPECTED_BOUNDARIES = {
    "autogen": {
        "capture_api": (
            "autogen_ext.tools.mcp.McpWorkbench.list_tools, as called by "
            "autogen_agentchat.agents.AssistantAgent(workbench=...) for one turn"
        ),
        "capture_object": (
            "autogen_core.tools.ToolSchema in the tools argument AssistantAgent "
            "passes to ChatCompletionClient.create"
        ),
        "capture_stage": "framework_tool_definition",
    },
    "langchain": {
        "capture_api": "langchain_core.utils.function_calling.convert_to_openai_tool",
        "capture_object": "OpenAI-compatible function-tool dictionary",
        "capture_stage": "provider_format",
    },
    "pydantic_ai": {
        "capture_api": "pydantic_ai.mcp.MCPToolset.get_tools",
        "capture_object": "pydantic_ai.tools.ToolDefinition",
        "capture_stage": "framework_tool_definition",
    },
    "crewai": {
        "capture_api": "crewai_tools.MCPServerAdapter",
        "capture_object": "CrewAI BaseTool name, description, and args_schema",
        "capture_stage": "framework_tool_definition",
    },
    "openai_agents": {
        "capture_api": "agents.mcp.util.MCPUtil.to_function_tool",
        "capture_object": "agents.tool.FunctionTool",
        "capture_stage": "framework_tool_definition",
    },
    "google_adk": {
        "capture_api": (
            "google.adk.tools.mcp_tool.McpToolset.get_tools, then "
            "McpTool.process_llm_request inside an LlmAgent run"
        ),
        "capture_object": (
            "google.genai.types.FunctionDeclaration in the LlmRequest.config.tools "
            "handed to BaseLlm.generate_content_async"
        ),
        "capture_stage": "provider_format",
    },
    "ms_agent_framework": {
        "capture_api": (
            "agent_framework.MCPStdioTool / MCPStreamableHTTPTool.load_tools, then "
            "FunctionTool.to_json_schema_spec inside an Agent run"
        ),
        "capture_object": (
            "agent_framework.FunctionTool in the options['tools'] an Agent hands "
            "its chat client"
        ),
        "capture_stage": "framework_tool_definition",
    },
    "strands": {
        "capture_api": (
            "strands.tools.mcp.MCPClient.load_tools, then "
            "ToolRegistry.get_all_tool_specs inside a strands.Agent run"
        ),
        "capture_object": "strands.types.tools.ToolSpec in the tool_specs handed to Model.stream",
        "capture_stage": "framework_tool_definition",
    },
    "llamaindex": {
        "capture_api": (
            "llama_index.tools.mcp.McpToolSpec.to_tool_list_async, then "
            "ToolMetadata.to_openai_tool(skip_length_check=True)"
        ),
        "capture_object": (
            "OpenAI-compatible function-tool dict built from the FunctionTool's "
            "ToolMetadata (the call llama-index-llms-openai makes)"
        ),
        "capture_stage": "provider_format",
    },
    "mastra": {
        "capture_api": "@mastra/mcp MCPClient.listTools",
        "capture_object": "Mastra Tool action returned by listTools",
        "capture_stage": "framework_tool_definition",
    },
    "vercel_ai": {
        "capture_api": "@ai-sdk/mcp createMCPClient().tools",
        "capture_object": "AI SDK dynamicTool returned by MCPClient.tools",
        "capture_stage": "framework_tool_definition",
    },
    "openai_agents_js": {
        "capture_api": (
            "@openai/agents Runner.run with Agent({ mcpServers: [MCPServerStdio | "
            "MCPServerStreamableHttp | MCPServerSSE] }), tools loaded by getAllMcpTools"
        ),
        "capture_object": "SerializedFunctionTool in the ModelRequest.tools handed to Model.getResponse",
        "capture_stage": "framework_tool_definition",
    },
    "genkit_js": {
        "capture_api": (
            "genkit ai.generate({ tools: await createMcpHost(...).getActiveTools(ai) }) "
            "(@genkit-ai/mcp GenkitMcpHost)"
        ),
        "capture_object": "ToolDefinition in the GenerateRequest.tools handed to the model action",
        "capture_stage": "framework_tool_definition",
    },
    "google_adk_js": {
        "capture_api": (
            "@google/adk InMemoryRunner.runAsync with LlmAgent({ tools: [MCPToolset] }), "
            "tools loaded by MCPToolset.getTools (@google/adk/tools/mcp)"
        ),
        "capture_object": (
            "FunctionDeclaration in the LlmRequest.config.tools handed to "
            "BaseLlm.generateContentAsync"
        ),
        "capture_stage": "provider_format",
    },
    "agno": {
        "capture_api": (
            "agno.tools.mcp.MCPTools(protocol_mode='auto') handed to agno.agent.Agent; "
            "the tools argument of the model's ainvoke on the first turn"
        ),
        "capture_object": (
            "function-tool dicts Model._format_tools builds from each Function: "
            "{'type': 'function', 'function': Function.to_dict()}"
        ),
        "capture_stage": "provider_format",
    },
    "haystack": {
        "capture_api": (
            "haystack_integrations.tools.mcp.MCPToolset handed to "
            "haystack.components.agents.Agent; the tools argument of the chat "
            "generator's run on the first turn"
        ),
        "capture_object": (
            "haystack.tools.Tool.tool_spec ({name, description, parameters}) for each "
            "Tool MCPToolset.warm_up builds"
        ),
        "capture_stage": "framework_tool_definition",
    },
    "dspy": {
        "capture_api": (
            "dspy.Tool.from_mcp_tool(session, tool) over a caller-supplied mcp.Client; "
            "dspy.Tool.format_as_litellm_function_call()"
        ),
        "capture_object": (
            "the {'type': 'function', 'function': {name, description, parameters}} dict "
            "dspy.Tool.format_as_litellm_function_call returns, which dspy adapters pass "
            "the LM as tools= under native function calling"
        ),
        "capture_stage": "provider_format",
    },
    "beeai": {
        "capture_api": (
            "beeai_framework.tools.mcp.MCPTool.from_client handed to "
            "beeai_framework.agents.requirement.RequirementAgent; the tools "
            "LiteLLMChatModel._transform_input passes litellm.acompletion on the first turn"
        ),
        "capture_object": (
            "function-tool dicts {'type': 'function', 'function': {name, description, "
            "parameters, strict}} built from each MCPTool's input_schema, with the "
            "ChatModel default use_strict_tool_schema=True"
        ),
        "capture_stage": "provider_format",
    },
    "semantic_kernel": {
        "capture_api": (
            "semantic_kernel.connectors.mcp.MCPStdioPlugin / MCPStreamableHttpPlugin / "
            "MCPSsePlugin added to a Kernel, then OpenAIChatCompletion."
            "get_chat_message_content with FunctionChoiceBehavior.Auto()"
        ),
        "capture_object": (
            "the tools array of the serialized chat.completions request body the "
            "AsyncOpenAI client sends (intercepted by an httpx.MockTransport)"
        ),
        "capture_stage": "provider_request",
    },
    "pi": {
        "capture_api": "@earendil-works/pi-mcp McpClient.listTools over StdioTransport | StreamableHttpTransport",
        "capture_object": "Tool (the MCP tools/list entry pi-mcp returns unchanged)",
        "capture_stage": "framework_tool_definition",
    },
    "crewai_native": {
        "capture_api": (
            "crewai.Agent(mcps=[MCPServerStdio|MCPServerHTTP|MCPServerSSE]).get_mcp_tools "
            "(crewai.mcp.tool_resolver.MCPToolResolver), then "
            "crewai.utilities.agent_utils.convert_tools_to_openai_schema"
        ),
        "capture_object": (
            "function-tool dicts convert_tools_to_openai_schema builds from each "
            "crewai.tools.mcp_native_tool.MCPNativeTool"
        ),
        "capture_stage": "provider_format",
    },
    "smolagents": {
        "capture_api": (
            "smolagents.MCPClient.get_tools handed to smolagents.ToolCallingAgent; "
            "the tools Model._prepare_completion_kwargs builds on the first turn"
        ),
        "capture_object": (
            "function-tool dicts smolagents.models.get_tool_json_schema builds from "
            "each mcpadapt MCPAdaptTool"
        ),
        "capture_stage": "provider_format",
    },
    "langchain_js": {
        "capture_api": "@langchain/core/utils/function_calling convertToOpenAITool",
        "capture_object": (
            "OpenAI-compatible function-tool object built from the "
            "DynamicStructuredTool MCPAdapter.listTools returns"
        ),
        "capture_stage": "provider_format",
    },
    "spring_ai": {
        "capture_api": (
            "spring-ai-starter-mcp-client autoconfiguration (McpClientAutoConfiguration, "
            "McpToolCallbackAutoConfiguration): SyncMcpToolCallbackProvider.getToolCallbacks "
            "over the McpSyncClient bean, resolved by DefaultToolCallingManager.resolveToolDefinitions"
        ),
        "capture_object": "org.springframework.ai.tool.definition.ToolDefinition (name, description, inputSchema)",
        "capture_stage": "framework_tool_definition",
    },
    "ms_agent_framework_dotnet": {
        "capture_api": (
            "Microsoft.Agents.AI IChatClient.AsAIAgent(tools: McpClient.ListToolsAsync()) run by "
            "AIAgent.RunAsync; a scripted IChatClient.GetResponseAsync receives the ChatOptions"
        ),
        "capture_object": (
            "Microsoft.Extensions.AI.AIFunctionDeclaration in ChatOptions.Tools "
            "(Name, Description, JsonSchema, ReturnJsonSchema), built by McpClientTool"
        ),
        "capture_stage": "framework_tool_definition",
    },
    "copilot_sdk": {
        "capture_api": "CopilotRequestHandler.sendRequest (CopilotClientOptions.requestHandler)",
        "capture_object": (
            "the function entries named src-<tool> in the tools array of the "
            "chat/completions request body the Copilot runtime sent the session's "
            "BYOK provider"
        ),
        "capture_stage": "provider_request",
    },
    "google_adk_java": {
        "capture_api": (
            "com.google.adk LlmAgent.builder().tools(McpToolset) run by InMemoryRunner.runAsync; "
            "a scripted BaseLlm.generateContent receives the LlmRequest"
        ),
        "capture_object": (
            "com.google.genai.types.FunctionDeclaration in LlmRequest.config().tools() "
            "(name, description, parametersJsonSchema, responseJsonSchema)"
        ),
        "capture_stage": "provider_format",
    },
    "langchain4j": {
        "capture_api": (
            "AiServices with McpToolProvider over DefaultMcpClient: the "
            "ChatRequest.toolSpecifications() AiServices hands ChatModel.doChat"
        ),
        "capture_object": (
            "dev.langchain4j.agent.tool.ToolSpecification (name, description, parameters "
            "as JsonObjectSchema, serialized by JsonSchemaElementUtils.toMap)"
        ),
        "capture_stage": "framework_tool_definition",
    },
    "google_adk_go": {
        "capture_api": (
            "google.golang.org/adk/v2 llmagent.New(llmagent.Config{Toolsets: mcptoolset.New(...)}) "
            "run by runner.NewInMemory Runner.Run; a scripted model.LLM.GenerateContent receives "
            "the LLMRequest"
        ),
        "capture_object": (
            "genai.FunctionDeclaration in LLMRequest.Config.Tools "
            "(name, description, parametersJsonSchema, responseJsonSchema)"
        ),
        "capture_stage": "provider_format",
    },
    "koog": {
        "capture_api": (
            "ai.koog AIAgent(toolRegistry = McpToolRegistryProvider.fromClient(...)) run with "
            "singleRunStrategy; OpenAILLMClient posts to a local fake /v1/chat/completions"
        ),
        "capture_object": (
            "the tools array of the chat-completions request body Koog's OpenAILLMClient sends "
            "({type: function, function: {name, description, parameters}})"
        ),
        "capture_stage": "provider_request",
    },
}

PUBLISHED_CODE_OVERRIDES = {
    "autogen": {
        # McpWorkbench rebuilds each schema from properties, required and
        # additionalProperties (defaulting it to False) and keeps nothing else of the tool.
        "additional-properties": "a",
        "json-schema-dialect": "a",
        "destructive-hint": "n",
        "idempotent-hint": "n",
        "open-world-hint": "n",
        "read-only-hint": "n",
        "title-annotation": "n",
        "tool-icons": "n",
        "tool-output-schema": "n",
        # Results: the workbench keeps text and images only and raises on audio and
        # resource links; the agent reads ToolResult.to_text(); mcp_server_tools'
        # run_json hands back the MCP blocks. Every content kind depends on the route.
        "tool-text-content": "a",
        "tool-image-content": "a",
        "tool-audio-content": "a",
        "embedded-resource-content": "a",
        "resource-link-content": "a",
        "content-annotations": "a",
        "tool-execution-errors": "a",
        "structured-content": "n",
        # No message handler; every list refetches.
        "tools-list-changed": "a",
        "resources-list-changed": "a",
        "prompts-list-changed": "a",
        "resource-subscriptions": "n",
        # Client features: sampling only, through model_client.
        "sampling-tools": "n",
        "elicitation-form-mode": "n",
        "elicitation-url-mode": "n",
        "roots-list": "n",
        "roots-list-changed": "n",
        # Server utilities: one tools/list page, no callbacks, a CancellationToken that
        # raises instead of cancelling.
        "pagination-cursor": "n",
        "progress-notifications": "n",
        "logging-notifications": "n",
        "logging-set-level": "n",
        "completion-complete": "n",
        "cancellation": "n",
        "tasks-experimental": "n",
        # No auth hook on StreamableHttpServerParams.
        "authorization-resource-metadata": "n",
    },
    "crewai": {
        # F2-notifications: fixtures/notify_server.py
        "tools-list-changed": "n",
        "resources-list-changed": "n",
        "prompts-list-changed": "n",
        "resource-subscriptions": "n",
        "tool-icons": "n",
        "tool-output-schema": "n",
        "additional-properties": "a",
        "description-fidelity": "a",
        "format": "a",
        "null-acceptance": "a",
        "one-of-any-of": "a",
        "property-descriptions": "a",
        "destructive-hint": "n",
        "idempotent-hint": "n",
        "open-world-hint": "n",
        "read-only-hint": "n",
        "title-annotation": "n",
        # Tool results: text is stringified; a lone media or resource block survives only
        # as the stringified content model, and beside other kinds it is dropped.
        "tool-text-content": "a",
        "tool-image-content": "a",
        "tool-audio-content": "a",
        "embedded-resource-content": "a",
        "resource-link-content": "a",
        "content-annotations": "n",
        "structured-content": "n",
        "tool-execution-errors": "n",
        # Base protocol: a draft-07 tuple schema stops the adapter loading any tool, and
        # a 2020-12 schema loses prefixItems, dependentRequired and $schema.
        "json-schema-dialect": "n",
        # No resource or prompt API on MCPServerAdapter.
        "prompt-arguments": "n",
        "prompts-get": "n",
        "prompts-list": "n",
        "resource-binary-contents": "n",
        "resource-templates": "n",
        "resources-list": "n",
        "resources-read": "n",
        # Client features: no hook of any kind, so every request is refused.
        "sampling-create-message": "n",
        "sampling-tools": "n",
        "elicitation-form-mode": "n",
        "elicitation-url-mode": "n",
        "roots-list": "n",
        "roots-list-changed": "n",
    },
    "langchain": {
        # F2-notifications: fixtures/notify_server.py
        "prompts-list-changed": "n",
        "resource-subscriptions": "a",
        "tool-icons": "n",
        "tool-output-schema": "n",
        "destructive-hint": "a",
        "idempotent-hint": "a",
        "open-world-hint": "a",
        "read-only-hint": "a",
        "title-annotation": "a",
        # Audio conversion raises and aborts the call. Measured one kind per call,
        # the other kinds are observed: images survive, an embedded resource loses its
        # uri, a resource link becomes a `file` block, and annotations are dropped.
        "tool-audio-content": "n",
        "embedded-resource-content": "n",
        "resource-link-content": "a",
        "content-annotations": "n",
        "structured-content": "n",
        # Boundary-dependent: gone through a direct invocation, present as
        # ToolMessage.status == "error" through the invocation an agent makes.
        "tool-execution-errors": "a",
        # No listing API for prompts or templates; resources arrive as Blobs
        # without their names.
        "prompt-arguments": "n",
        "prompts-list": "n",
        "resource-templates": "n",
        "resources-list": "a",
        # Client features: Callbacks.on_elicitation is the adapter's own hook; sampling
        # and roots answer only through session_kwargs forwarded to the MCP SDK.
        "sampling-create-message": "a",
        "sampling-tools": "a",
        "roots-list": "a",
        "roots-list-changed": "n",
    },
    "openai-agents": {
        # F2-notifications: fixtures/notify_server.py
        "resource-subscriptions": "a",
        "tool-icons": "n",
        "tool-output-schema": "n",
        "destructive-hint": "n",
        "idempotent-hint": "n",
        "open-world-hint": "n",
        "read-only-hint": "n",
        "title-annotation": "a",
        # Text and images survive; everything else is serialized into text.
        "tool-audio-content": "a",
        "embedded-resource-content": "a",
        "resource-link-content": "a",
        "content-annotations": "n",
        "structured-content": "n",
        "tool-execution-errors": "n",
        # Client features: only message_handler reaches the ClientSession.
        "sampling-create-message": "n",
        "sampling-tools": "n",
        "elicitation-form-mode": "n",
        "elicitation-url-mode": "n",
        "roots-list": "n",
        "roots-list-changed": "n",
    },
    "pydantic-ai": {
        # F2-notifications: fixtures/notify_server.py
        "resource-subscriptions": "a",
        "tool-icons": "n",
        # Kept as ToolDefinition.return_schema, withheld from the model by default.
        "tool-output-schema": "a",
        "destructive-hint": "a",
        "idempotent-hint": "a",
        "open-world-hint": "a",
        "read-only-hint": "a",
        "title-annotation": "a",
        # Native media survives; resources are reduced to bare strings.
        "embedded-resource-content": "n",
        "resource-link-content": "a",
        "content-annotations": "n",
        # structuredContent is returned only for an all-text result, and then in
        # place of the text: each of the two rows is kept on one call and lost on
        # the other.
        "structured-content": "a",
        "tool-text-content": "a",
        "tool-execution-errors": "a",
        # A text resource comes back as a bare string without uri or mimeType.
        "resources-read": "a",
        # Client features: declaring sampling.tools and changing roots both go
        # through the wrapped fastmcp Client, not MCPToolset.
        "sampling-tools": "a",
        "roots-list-changed": "a",
    },
    "google_adk": {
        # F2-notifications: McpToolset takes no message handler; get_tools and
        # list_resources relist every call, there is no prompt API, and a raw-session
        # subscription's update reaches no caller.
        "tools-list-changed": "a",
        "resources-list-changed": "a",
        "prompts-list-changed": "n",
        "resource-subscriptions": "n",
        # The model-bound declaration fences the description in untrusted markers.
        "description-fidelity": "a",
        # McpTool keeps the whole MCP tool as raw_mcp_tool; the FunctionDeclaration
        # carries no annotations or icons.
        "destructive-hint": "a",
        "idempotent-hint": "a",
        "open-world-hint": "a",
        "read-only-hint": "a",
        "title-annotation": "a",
        "tool-icons": "a",
        # The tool returns the dumped CallToolResult; the model reads it as JSON in
        # FunctionResponse.response, so every content kind is boundary-dependent.
        "tool-text-content": "a",
        "tool-image-content": "a",
        "tool-audio-content": "a",
        "embedded-resource-content": "a",
        "resource-link-content": "a",
        "content-annotations": "a",
        # McpToolset has no template or prompt methods.
        "resource-templates": "n",
        "prompts-list": "n",
        "prompt-arguments": "n",
        "prompts-get": "n",
        # response_json_schema keeps every keyword but fences the description.
        "tool-output-schema": "a",
        # Client features: sampling and elicitation are McpToolset parameters; there is
        # no roots hook.
        "roots-list": "n",
        "roots-list-changed": "n",
        # Server utilities: get_tools reads one tools/list page, there is no logging hook,
        # and the level and completion are reachable only on a raw MCPSessionManager
        # session.
        "pagination-cursor": "n",
        "logging-notifications": "n",
        "logging-set-level": "a",
        "completion-complete": "a",
        "tasks-experimental": "n",
    },
    "strands": {
        # on_tools_changed covers tools only; resources and prompts relist from the server.
        "resources-list-changed": "a",
        "prompts-list-changed": "a",
        "resource-subscriptions": "n",
        # Icons stay on MCPAgentTool.mcp_tool; the ToolSpec has none.
        "tool-icons": "a",
        # map_mcp_content_to_tool_result_content keeps text and images only: annotations,
        # the embedded resource's uri and mimeType, audio and resource links are dropped.
        "content-annotations": "n",
        "embedded-resource-content": "n",
        "resource-link-content": "n",
        "tool-audio-content": "n",
        # MCPClient takes elicitation_callback only; no sampling or roots hook.
        "sampling-create-message": "n",
        "sampling-tools": "n",
        "roots-list": "n",
        "roots-list-changed": "n",
        # No log hook, setLevel or completion API; the ClientSession is private.
        "logging-notifications": "n",
        "logging-set-level": "n",
        "completion-complete": "n",
        # Tasks are opt-in through tasks_config; the defaults declare none.
        "tasks-experimental": "n",
    },
    "mastra": {
        # F2-notifications: fixtures/notify_server.py
        "tools-list-changed": "a",
        "resources-list-changed": "a",
        "prompts-list-changed": "a",
        "resource-subscriptions": "n",
        "tool-icons": "n",
        # Client features: inputRequests answers elicitation only; sampling and roots
        # are documented as unsupported.
        "sampling-create-message": "n",
        "sampling-tools": "n",
        "roots-list": "n",
        "roots-list-changed": "n",
        # @mastra/mcp 2.2.0 builds only a Streamable HTTP transport; a legacy /sse server
        # answers its POST with 405 and it never falls back to GET /sse.
        "transport-http-sse": "n",
        # 2.x keeps the server's annotations on the tool action's `mcp` metadata
        # and mirrors the title onto `title`; neither is in the captured definition.
        "destructive-hint": "a",
        "idempotent-hint": "a",
        "open-world-hint": "a",
        "read-only-hint": "a",
        "title-annotation": "a",
        "tool-name": "a",
        # The only adapter that keeps structuredContent, and it keeps it by returning
        # the typed value in place of the whole result. Without structuredContent the
        # result passes through untouched, so every content kind is conditional.
        "structured-content": "y",
        "tool-text-content": "a",
        "tool-image-content": "a",
        "tool-audio-content": "a",
        "embedded-resource-content": "a",
        "resource-link-content": "a",
        "content-annotations": "a",
        "tool-execution-errors": "a",
    },
    "vercel_ai": {
        # toolsFromDefinitions closes every root object with additionalProperties: false,
        # drops icons, and keeps annotations and the title on the tool's app-side
        # `metadata` and `title`, outside the provider-bound fields.
        "additional-properties": "a",
        "tool-icons": "n",
        "destructive-hint": "a",
        "idempotent-hint": "a",
        "open-world-hint": "a",
        "read-only-hint": "a",
        "title-annotation": "a",
        # In the default schemas: "automatic" mode, tools() builds each tool from the
        # inputSchema only, so the server's outputSchema is dropped.
        "tool-output-schema": "n",
        # execute resolves with the whole CallToolResult; toModelOutput, which builds
        # what the model reads, keeps only text and image blocks, so every other
        # result row depends on which boundary is read.
        "structured-content": "a",
        "tool-audio-content": "a",
        "embedded-resource-content": "a",
        "resource-link-content": "a",
        "content-annotations": "a",
        "tool-execution-errors": "a",
        # Conformance suite 0.1.16: no Last-Event-ID on reconnect, authorization
        # server metadata also requested at the root path in metadata-var3, and one
        # pre-initialize SSE GET on a second connection counted as missing its session.
        "sse-resumability": "a",
        "authorization-resource-metadata": "a",
        "session-identifier": "a",
        # Client features: elicitation is the one server request MCPClient answers, and
        # its schema requires requestedSchema, so a URL-mode request is refused before
        # the handler runs. Sampling and roots are answered -32601.
        "sampling-create-message": "n",
        "sampling-tools": "n",
        "elicitation-url-mode": "n",
        "roots-list": "n",
        "roots-list-changed": "n",
        # Server utilities: no progress, logging or setLevel API, no notifications/cancelled
        # on abort, no tasks capability. tools() pages through every cursor.
        "cancellation": "n",
        "logging-notifications": "n",
        "logging-set-level": "n",
        "progress-notifications": "n",
        "tasks-experimental": "n",
        # Notifications reach only onUncaughtError; every list call refetches, and there
        # is no subscribe API.
        "tools-list-changed": "a",
        "resources-list-changed": "a",
        "prompts-list-changed": "a",
        "resource-subscriptions": "n",
    },
    "ms_agent_framework": {
        # FunctionTool carries name, description and inputSchema only.
        "destructive-hint": "n",
        "idempotent-hint": "n",
        "open-world-hint": "n",
        "read-only-hint": "n",
        "title-annotation": "n",
        "tool-icons": "n",
        "tool-output-schema": "n",
        # Results: structured_first hands the model the structuredContent JSON in place
        # of the content blocks, so every kind beside it is lost.
        "structured-content": "n",
        "content-annotations": "n",
        "embedded-resource-content": "n",
        "tool-image-content": "a",
        "tool-audio-content": "a",
        "resource-link-content": "a",
        "tool-execution-errors": "a",
        # No resource API; prompts only as FunctionTools and a joined string.
        "resources-list": "n",
        "resources-read": "n",
        "resource-templates": "n",
        "resource-binary-contents": "n",
        "resources-list-changed": "n",
        "resource-subscriptions": "a",
        "prompts-list": "a",
        "prompts-get": "a",
        # Client features: sampling through client=; nothing for elicitation or roots.
        "elicitation-form-mode": "n",
        "elicitation-url-mode": "n",
        "roots-list": "n",
        "roots-list-changed": "n",
        # Utilities.
        "progress-notifications": "a",
        "completion-complete": "a",
        "cancellation": "n",
        "tasks-experimental": "n",
        # No HTTP+SSE client.
        "transport-http-sse": "n",
    },
    "openai_agents_js": {
        # mcpToFunctionTool builds the FunctionTool from name, description and
        # inputSchema only; nothing else is kept on it or beside it.
        "destructive-hint": "n",
        "idempotent-hint": "n",
        "open-world-hint": "n",
        "read-only-hint": "n",
        "title-annotation": "n",
        "tool-icons": "n",
        "tool-output-schema": "n",
        # Tool results: MCPServer.callTool keeps every block, isError and
        # structuredContent; the function_call_result the model receives keeps a lone
        # text or image block, stringifies every other kind (and any mixed result) into
        # JSON text, and has no error or structured field.
        "content-annotations": "a",
        "embedded-resource-content": "a",
        "resource-link-content": "a",
        "structured-content": "a",
        "tool-audio-content": "a",
        "tool-execution-errors": "a",
        "tool-image-content": "a",
        # No notification hook: list changes show only on the next listing, and there
        # is no subscribe or prompts API.
        "tools-list-changed": "a",
        "resources-list-changed": "a",
        "resource-subscriptions": "n",
        "prompt-arguments": "n",
        "prompts-get": "n",
        "prompts-list": "n",
        "prompts-list-changed": "n",
        # Client features: no sampling, elicitation or roots handler of any kind.
        "elicitation-form-mode": "n",
        "elicitation-url-mode": "n",
        "roots-list": "n",
        "roots-list-changed": "n",
        "sampling-create-message": "n",
        "sampling-tools": "n",
        # Server utilities: the MCP client sits in a protected field, and no hook
        # receives progress or log notifications.
        "completion-complete": "n",
        "logging-notifications": "n",
        "logging-set-level": "n",
        "tasks-experimental": "n",
        "progress-notifications": "a",
        # Unmeasured for every framework.
        "authorization-oauth": "u",
    },
    "semantic_kernel": {
        # Functions are named `<plugin>-<tool>` in the request.
        "tool-name": "a",
        # load_tools keeps name, description and inputSchema only.
        "destructive-hint": "n",
        "idempotent-hint": "n",
        "open-world-hint": "n",
        "read-only-hint": "n",
        "title-annotation": "n",
        "tool-icons": "n",
        "tool-output-schema": "n",
        # parameters are rebuilt from properties/required; $defs and $schema are lost.
        "json-schema-dialect": "a",
        # The model reads str() of SK's content objects; a resource link raises and
        # fails the call; structuredContent, isError and annotations are dropped.
        "tool-text-content": "a",
        "tool-image-content": "a",
        "tool-audio-content": "a",
        "embedded-resource-content": "a",
        "resource-link-content": "n",
        "content-annotations": "n",
        "structured-content": "n",
        "tool-execution-errors": "n",
        # No resource API; prompts are KernelFunctions with every argument required.
        "resources-list": "n",
        "resources-read": "n",
        "resource-binary-contents": "n",
        "resource-templates": "n",
        "prompts-list": "a",
        "prompt-arguments": "a",
        # The default message_handler reloads inline and deadlocks until the timeout.
        "tools-list-changed": "a",
        "prompts-list-changed": "a",
        "resources-list-changed": "n",
        "resource-subscriptions": "a",
        # Client features: sampling only, and without tools.
        "sampling-tools": "n",
        "elicitation-form-mode": "n",
        "elicitation-url-mode": "n",
        "roots-list": "n",
        "roots-list-changed": "n",
        # Server utilities.
        "pagination-cursor": "n",
        "progress-notifications": "n",
        "cancellation": "n",
        "completion-complete": "a",
        "tasks-experimental": "n",
    },
}

# Semantic Kernel names each function <plugin>-<tool>; the renderer's plugin is "mcp".
SK_PREFIX = "mcp-"
# MCPToolResolver names a stdio tool f"{command}_{'_'.join(args)}_{tool}", sanitized.
CREWAI_NATIVE_PREFIX = "python_fixtures_definitions_server_py_"

PUBLISHED_CODE_OVERRIDES["llamaindex"] = {
    # McpToolSpec rebuilds each inputSchema as a Pydantic model: nested objects
    # collapse, formats and bounds drop, optional fields gain null.
    "arrays-of-objects": "a",
    "nested-objects": "a",
    "required": "a",
    "format": "n",
    "numeric-range": "n",
    "null-acceptance": "a",
    "one-of-any-of": "a",
    "property-descriptions": "a",
    # FunctionTool keeps name, description and fn_schema; nothing else survives.
    "destructive-hint": "n",
    "idempotent-hint": "n",
    "open-world-hint": "n",
    "read-only-hint": "n",
    "title-annotation": "n",
    "tool-icons": "n",
    "tool-output-schema": "n",
    # Every result reaches the model as repr(CallToolResult) in one text block.
    "tool-image-content": "a",
    "tool-audio-content": "a",
    "embedded-resource-content": "a",
    "resource-link-content": "a",
    "content-annotations": "n",
    "structured-content": "n",
    "tool-execution-errors": "n",
    # Either dialect tool (a boolean subschema, a tuple items list) fails the whole listing.
    "json-schema-dialect": "n",
    # fetch_tools ignores nextCursor.
    "pagination-cursor": "n",
    # BasicMCPClient: sampling_callback only, a session per call, no message handler.
    "completion-complete": "n",
    "logging-set-level": "n",
    "logging-notifications": "n",
    "tasks-experimental": "n",
    "elicitation-form-mode": "n",
    "elicitation-url-mode": "n",
    "roots-list": "n",
    "roots-list-changed": "n",
    "sampling-tools": "n",
    "tools-list-changed": "n",
    "resources-list-changed": "n",
    "prompts-list-changed": "n",
    "resource-subscriptions": "n",
}

PUBLISHED_CODE_OVERRIDES["pi"] = {
    # McpClient.callTool keeps the whole CallToolResult; toLlmContent, the model
    # conversion, keeps text and images, drops annotations, isError and (beside
    # content) structuredContent, and turns other blocks into placeholder text.
    "content-annotations": "a",
    "embedded-resource-content": "a",
    "resource-link-content": "a",
    "structured-content": "a",
    "tool-audio-content": "a",
    "tool-execution-errors": "a",
    # Only the generic McpClient.request() reaches these.
    "completion-complete": "a",
    "logging-set-level": "a",
    "resource-subscriptions": "a",
    # No prompts method.
    "prompt-arguments": "n",
    "prompts-get": "n",
    "prompts-list": "n",
    "prompts-list-changed": "n",
    # The roots option is the only client-feature hook, without listChanged.
    "elicitation-form-mode": "n",
    "elicitation-url-mode": "n",
    "roots-list-changed": "n",
    "sampling-create-message": "n",
    "sampling-tools": "n",
    "tasks-experimental": "n",
    # No HTTP+SSE transport.
    "transport-http-sse": "n",
}

PUBLISHED_CODE_OVERRIDES["google_adk_js"] = {
    # toGeminiSchema keeps type, description, enum, items, properties, required and anyOf.
    "default-value": "n",
    "format": "n",
    "numeric-range": "n",
    "one-of-any-of": "a",
    "json-schema-dialect": "n",
    "tool-output-schema": "a",
    # MCPTool's FunctionDeclaration carries name, description and the schemas only.
    "destructive-hint": "n",
    "idempotent-hint": "n",
    "open-world-hint": "n",
    "read-only-hint": "n",
    "title-annotation": "n",
    "tool-icons": "n",
    # The model reads the whole CallToolResult as one functionResponse JSON object.
    "tool-text-content": "a",
    "tool-image-content": "a",
    "tool-audio-content": "a",
    "embedded-resource-content": "a",
    "resource-link-content": "a",
    "content-annotations": "a",
    # MCPToolset: no prompts, no templates, one session per operation, no client hooks.
    "prompts-list": "n",
    "prompts-get": "n",
    "prompt-arguments": "n",
    "prompts-list-changed": "n",
    "resource-templates": "n",
    "tools-list-changed": "n",
    "resources-list-changed": "n",
    "resource-subscriptions": "a",
    "sampling-create-message": "n",
    "sampling-tools": "n",
    "elicitation-form-mode": "n",
    "elicitation-url-mode": "n",
    "roots-list": "n",
    "roots-list-changed": "n",
    # getTools ignores nextCursor; the rest only on a raw MCPSessionManager session.
    "pagination-cursor": "n",
    "progress-notifications": "a",
    "logging-notifications": "a",
    "logging-set-level": "a",
    "completion-complete": "a",
    "tasks-experimental": "n",
    # No HTTP+SSE variant. sse-retry reconnects with Last-Event-ID, so
    # sse-resumability is y and needs no override.
    "transport-http-sse": "n",
}

PUBLISHED_CODE_OVERRIDES["agno"] = {
    # build_tools registers Function(name, description, parameters=inputSchema); the
    # schema passes through and nothing else of the tool does.
    "destructive-hint": "n",
    "idempotent-hint": "n",
    "open-world-hint": "n",
    "read-only-hint": "n",
    "title-annotation": "n",
    "tool-icons": "n",
    "tool-output-schema": "n",
    # Results: isError becomes an "Error from MCP tool" text with no flag, structured
    # content stays on the direct ToolResult's metadata, resource blocks become text.
    "tool-execution-errors": "n",
    "structured-content": "a",
    "embedded-resource-content": "a",
    "resource-link-content": "n",
    "content-annotations": "n",
    # MCPTools has tools only, and builds its fastmcp Client with no handlers or auth.
    "resources-list": "n",
    "resources-read": "n",
    "resource-templates": "n",
    "resource-binary-contents": "n",
    "prompts-list": "n",
    "prompts-get": "n",
    "prompt-arguments": "n",
    "resources-list-changed": "n",
    "prompts-list-changed": "n",
    "resource-subscriptions": "n",
    "tools-list-changed": "a",
    "sampling-create-message": "n",
    "sampling-tools": "n",
    "elicitation-form-mode": "n",
    "elicitation-url-mode": "n",
    "roots-list": "n",
    "roots-list-changed": "n",
    "authorization-resource-metadata": "n",
    # Server utilities: completion and setLevel only on MCPTools.session.
    "progress-notifications": "a",
    "logging-notifications": "n",
    "logging-set-level": "a",
    "completion-complete": "a",
    "tasks-experimental": "n",
}

PUBLISHED_CODE_OVERRIDES["haystack"] = {
    # MCPToolset builds Tool(name, description, parameters=inputSchema); the schema
    # passes through and nothing else of the tool does.
    "destructive-hint": "n",
    "idempotent-hint": "n",
    "open-world-hint": "n",
    "read-only-hint": "n",
    "title-annotation": "n",
    "tool-icons": "n",
    "tool-output-schema": "n",
    # haystack.tools.Tool rejects a schema that is not valid 2020-12, failing the listing.
    "json-schema-dialect": "a",
    # Results: the direct return is the CallToolResult JSON; the Agent hands the model
    # that JSON as one text, and isError raises instead of returning.
    "tool-execution-errors": "a",
    "structured-content": "a",
    "content-annotations": "a",
    "tool-image-content": "a",
    "tool-audio-content": "a",
    "embedded-resource-content": "a",
    "resource-link-content": "a",
    # MCPToolset has tools only, and its ClientSession has no callbacks or auth.
    "resources-list": "n",
    "resources-read": "n",
    "resource-templates": "n",
    "resource-binary-contents": "n",
    "prompts-list": "n",
    "prompts-get": "n",
    "prompt-arguments": "n",
    "resources-list-changed": "n",
    "prompts-list-changed": "n",
    "resource-subscriptions": "n",
    "tools-list-changed": "n",
    "sampling-create-message": "n",
    "sampling-tools": "n",
    "elicitation-form-mode": "n",
    "elicitation-url-mode": "n",
    "roots-list": "n",
    "roots-list-changed": "n",
    "authorization-resource-metadata": "n",
    # Server utilities: one list_tools call, no handlers, a private session.
    "pagination-cursor": "n",
    "progress-notifications": "n",
    "logging-notifications": "n",
    "logging-set-level": "n",
    "completion-complete": "n",
    "tasks-experimental": "n",
}

PUBLISHED_CODE_OVERRIDES["smolagents"] = {
    # mcpadapt keeps name, description and inputSchema.properties (refs inlined);
    # get_tool_json_schema marks every property required and flattens anyOf to types.
    "required": "a",
    "one-of-any-of": "a",
    "json-schema-dialect": "a",
    "destructive-hint": "n",
    "idempotent-hint": "n",
    "open-world-hint": "n",
    "read-only-hint": "n",
    "title-annotation": "n",
    "tool-icons": "n",
    "tool-output-schema": "n",
    # Results: MCPAdaptTool.forward reads content[0] only and never isError; an image
    # reaches the caller as a PIL.Image but the agent stores it and tells the model so.
    "tool-execution-errors": "n",
    "structured-content": "n",
    "tool-image-content": "a",
    "tool-audio-content": "n",
    "embedded-resource-content": "n",
    "resource-link-content": "n",
    "content-annotations": "n",
    # MCPClient exposes get_tools() only, and mcpadapt opens its session with no callbacks.
    "resources-list": "n",
    "resources-read": "n",
    "resource-templates": "n",
    "resource-binary-contents": "n",
    "prompts-list": "n",
    "prompts-get": "n",
    "prompt-arguments": "n",
    "tools-list-changed": "n",
    "resources-list-changed": "n",
    "prompts-list-changed": "n",
    "resource-subscriptions": "n",
    "sampling-create-message": "n",
    "sampling-tools": "n",
    "elicitation-form-mode": "n",
    "elicitation-url-mode": "n",
    "roots-list": "n",
    "roots-list-changed": "n",
    "pagination-cursor": "n",
    "progress-notifications": "n",
    "logging-notifications": "n",
    "logging-set-level": "n",
    "completion-complete": "n",
    "cancellation": "n",
    "tasks-experimental": "n",
}

PUBLISHED_CODE_OVERRIDES["dspy"] = {
    # from_mcp_tool keeps name, description and the inputSchema properties; the litellm
    # dict recomputes required from defaults and drops $schema and top-level keywords.
    "destructive-hint": "n",
    "idempotent-hint": "n",
    "open-world-hint": "n",
    "read-only-hint": "n",
    "title-annotation": "n",
    "tool-icons": "n",
    "tool-output-schema": "n",
    "required": "a",
    "json-schema-dialect": "a",
    # Results: isError raises (direct) and becomes a traceback observation (ReAct);
    # media comes back as content objects, printed as reprs in ReAct, dropped beside text.
    "tool-execution-errors": "a",
    "structured-content": "n",
    "content-annotations": "n",
    "tool-image-content": "a",
    "tool-audio-content": "a",
    "embedded-resource-content": "a",
    "resource-link-content": "a",
    # Tools only: resources and prompts stay on the caller's session.
    "resources-list": "n",
    "resources-read": "n",
    "resource-templates": "n",
    "resource-binary-contents": "n",
    "prompts-list": "n",
    "prompts-get": "n",
    "prompt-arguments": "n",
    "resources-list-changed": "n",
    "prompts-list-changed": "n",
    "resource-subscriptions": "n",
    "tools-list-changed": "a",
    # Handlers, logging, completion and paging work only on the caller's mcp.Client.
    "sampling-create-message": "a",
    "sampling-tools": "a",
    "elicitation-form-mode": "a",
    "elicitation-url-mode": "a",
    "roots-list": "a",
    "roots-list-changed": "a",
    "pagination-cursor": "a",
    "progress-notifications": "n",
    "logging-notifications": "a",
    "logging-set-level": "a",
    "completion-complete": "a",
    "tasks-experimental": "n",
    # Unmeasured for every framework.
    "authorization-oauth": "u",
}

PUBLISHED_CODE_OVERRIDES["langchain_js"] = {
    # MCPAdapter prefixes every tool `src_`; annotations stay on the tool's app-side
    # metadata, which convertToOpenAITool does not read; icons and outputSchema are not
    # carried at all.
    "tool-name": "a",
    "destructive-hint": "a",
    "idempotent-hint": "a",
    "open-world-hint": "a",
    "read-only-hint": "a",
    "title-annotation": "a",
    "tool-icons": "n",
    "tool-output-schema": "n",
    # Results are content_and_artifact: structuredContent, embedded resources and
    # annotations reach only the app-side artifact; a resource link becomes a `file`
    # block; an error is a status:"error" ToolMessage in a ToolNode and a thrown
    # ToolException on a direct invoke.
    "content-annotations": "n",
    "embedded-resource-content": "n",
    "structured-content": "n",
    "resource-link-content": "a",
    "tool-execution-errors": "a",
    # No prompts API; subscriptions are fixed at connect with no unsubscribe.
    "prompt-arguments": "n",
    "prompts-get": "n",
    "prompts-list": "n",
    "prompts-list-changed": "n",
    "resource-subscriptions": "a",
    # Client features: elicitation only, no sampling or roots handler.
    "roots-list": "n",
    "roots-list-changed": "n",
    "sampling-create-message": "n",
    "sampling-tools": "n",
    # Server utilities: completion only through getClient(), no tasks.
    "completion-complete": "a",
    "tasks-experimental": "n",
    # The SDK client reconnects without Last-Event-ID.
    "sse-resumability": "a",
}

PUBLISHED_CODE_OVERRIDES["spring_ai"] = {
    # ToolDefinition holds name, description and inputSchema only.
    "destructive-hint": "n",
    "idempotent-hint": "n",
    "open-world-hint": "n",
    "read-only-hint": "n",
    "title-annotation": "n",
    "tool-icons": "n",
    "tool-output-schema": "n",
    # Every result arrives as one string, the JSON of the content list without each
    # block's type; structuredContent is dropped. An isError result is flagless text on
    # the agent path and a ToolExecutionException from a direct call.
    "tool-text-content": "a",
    "tool-image-content": "a",
    "tool-audio-content": "a",
    "embedded-resource-content": "a",
    "resource-link-content": "a",
    "content-annotations": "n",
    "structured-content": "n",
    "tool-execution-errors": "a",
    # MCP Java SDK 2.0.0 types LoggingMessageNotification.data as a String, so structured
    # log data fails to deserialize before @McpLogging sees it.
    "logging-notifications": "n",
    # Reachable only on the McpSyncClient bean or through an McpClientCustomizer.
    "logging-set-level": "a",
    "completion-complete": "a",
    "resource-subscriptions": "a",
    "elicitation-url-mode": "a",
    "roots-list": "a",
    "roots-list-changed": "a",
    # CreateMessageRequest has no tools field; no task-augmented call; no
    # notifications/cancelled.
    "sampling-tools": "n",
    "tasks-experimental": "n",
    "cancellation": "n",
    # Conformance: no OAuth hook; SSE reconnects ignore retry and Last-Event-ID.
    "authorization-resource-metadata": "n",
    "sse-resumability": "a",
}

PUBLISHED_CODE_OVERRIDES["google_adk_java"] = {
    # The FunctionDeclaration carries name, description and the two schemas; McpTool
    # keeps the annotations beside it, never the icons.
    "destructive-hint": "a",
    "idempotent-hint": "a",
    "open-world-hint": "a",
    "read-only-hint": "a",
    "title-annotation": "a",
    "tool-icons": "n",
    # AbstractMcpTool.wrapCallResult keeps text blocks as {"text_output": [...]} and turns
    # a result with no text into an {"error": ...} map; structuredContent is dropped.
    "tool-text-content": "a",
    "tool-image-content": "n",
    "tool-audio-content": "n",
    "embedded-resource-content": "n",
    "resource-link-content": "n",
    "content-annotations": "n",
    "structured-content": "n",
    # MCP Java SDK 2.0.0 types LoggingMessageNotification.data as a String, so structured
    # log data fails to deserialize before McpServerLogConsumer sees it.
    "logging-notifications": "n",
    # Reachable only through McpTool.getMcpSession().
    "logging-set-level": "a",
    "completion-complete": "a",
    # McpSessionManager sets no consumer or client capability beyond logging.
    "progress-notifications": "n",
    "resource-subscriptions": "n",
    "sampling-create-message": "n",
    "sampling-tools": "n",
    "elicitation-form-mode": "n",
    "elicitation-url-mode": "n",
    "roots-list": "n",
    "roots-list-changed": "n",
    "tools-list-changed": "a",
    "resources-list-changed": "a",
    "prompts-list-changed": "a",
    "cancellation": "n",
    "tasks-experimental": "n",
    # Conformance: no OAuth hook; SSE reconnects ignore retry and Last-Event-ID.
    "authorization-resource-metadata": "n",
    "sse-resumability": "a",
}

PUBLISHED_CODE_OVERRIDES["langchain4j"] = {
    # McpToolProvider maps inputSchema into JsonSchemaElement, which has no default,
    # minimum/maximum, format, oneOf or dialect keywords. Annotations, icons and
    # outputSchema go to ToolSpecification.metadata(), which the ChatModel never reads.
    "default-value": "n",
    "numeric-range": "n",
    "format": "n",
    "one-of-any-of": "a",
    "json-schema-dialect": "a",
    "destructive-hint": "a",
    "idempotent-hint": "a",
    "open-world-hint": "a",
    "read-only-hint": "a",
    "title-annotation": "a",
    "tool-icons": "a",
    "tool-output-schema": "a",
    # The result converter accepts text blocks only and throws on any other kind; with
    # structuredContent present it returns that value and its JSON in place of the blocks.
    # An isError result is flagged for the agent and a ToolExecutionException when direct.
    "tool-text-content": "a",
    "tool-image-content": "n",
    "tool-audio-content": "n",
    "embedded-resource-content": "n",
    "resource-link-content": "n",
    "content-annotations": "n",
    "structured-content": "a",
    "tool-execution-errors": "a",
    # McpPrompt has no title.
    "prompts-list": "a",
    # No sampling or elicitation handler, and the requests are never answered.
    "sampling-create-message": "n",
    "sampling-tools": "n",
    "elicitation-form-mode": "n",
    "elicitation-url-mode": "n",
    # No setLevel or completion method, no tasks capability.
    "logging-set-level": "n",
    "completion-complete": "n",
    "tasks-experimental": "n",
    # McpOperationHandler drops requests whose id is not a number, so a string-id ping
    # is never answered.
    "ping": "n",
    # No HTTP+SSE transport, no OAuth hook, no GET reconnect after a closed SSE stream.
    "transport-http-sse": "n",
    "authorization-resource-metadata": "n",
    "sse-resumability": "n",
}

PUBLISHED_CODE_OVERRIDES["genkit_js"] = {
    # createDynamicTool keeps name, description and inputSchema (verbatim); outputSchema
    # becomes z.any() and the annotations, title and icons go.
    "tool-output-schema": "a",
    "destructive-hint": "n",
    "idempotent-hint": "n",
    "open-world-hint": "n",
    "read-only-hint": "n",
    "title-annotation": "n",
    "tool-icons": "n",
    # processResult: structuredContent wins over every block, all-text results are
    # joined into one string, a lone block is passed through.
    "tool-text-content": "a",
    "tool-image-content": "a",
    "tool-audio-content": "a",
    "embedded-resource-content": "a",
    "resource-link-content": "a",
    "content-annotations": "n",
    # Resources and prompts lose mimeType, uri on read and title.
    "resources-list": "a",
    "resource-templates": "a",
    "resources-read": "a",
    "prompts-list": "a",
    # No notification handlers: re-listing is the only refresh, and templates/list
    # failing empties the resource list.
    "tools-list-changed": "a",
    "prompts-list-changed": "a",
    "resources-list-changed": "n",
    "resource-subscriptions": "a",
    # Only on the SDK Client from GenkitMcpClient.mcpClient.
    "progress-notifications": "a",
    "logging-notifications": "a",
    "logging-set-level": "a",
    "completion-complete": "a",
    # callTool gets no signal; the Client declares only roots.
    "cancellation": "n",
    "tasks-experimental": "n",
    "sampling-create-message": "n",
    "sampling-tools": "n",
    "elicitation-form-mode": "n",
    "elicitation-url-mode": "n",
    # stdio and Streamable HTTP only.
    "transport-http-sse": "n",
}

PUBLISHED_CODE_OVERRIDES["beeai"] = {
    # MCPTool keeps name, description and inputSchema; the strict tool schema closes every
    # object and requires every property, and simplify_json_schema crashes on boolean items.
    "destructive-hint": "n",
    "idempotent-hint": "n",
    "open-world-hint": "n",
    "read-only-hint": "n",
    "title-annotation": "n",
    "tool-icons": "n",
    "tool-output-schema": "n",
    "additional-properties": "a",
    "required": "a",
    "json-schema-dialect": "n",
    # Results: isError raises (direct) and reaches the model through the error template
    # (agent); structuredContent replaces every content block; the model gets one JSON string.
    "tool-execution-errors": "a",
    "structured-content": "a",
    "tool-text-content": "a",
    "content-annotations": "a",
    "tool-image-content": "a",
    "tool-audio-content": "a",
    "embedded-resource-content": "a",
    "resource-link-content": "a",
    # Tools only.
    "resources-list": "n",
    "resources-read": "n",
    "resource-templates": "n",
    "resource-binary-contents": "n",
    "prompts-list": "n",
    "prompts-get": "n",
    "prompt-arguments": "n",
    "resources-list-changed": "n",
    "prompts-list-changed": "n",
    "resource-subscriptions": "n",
    "tools-list-changed": "a",
    # Handlers, logging and completion work only on a caller-built ClientSession.
    "sampling-create-message": "a",
    "sampling-tools": "a",
    "elicitation-form-mode": "a",
    "elicitation-url-mode": "a",
    "roots-list": "a",
    "roots-list-changed": "a",
    "logging-notifications": "a",
    "logging-set-level": "a",
    "completion-complete": "a",
    # from_session lists once; call_tool takes no progress callback; no cancel is sent.
    "pagination-cursor": "n",
    "progress-notifications": "n",
    "cancellation": "n",
    "tasks-experimental": "n",
    # Unmeasured for every framework.
    "authorization-oauth": "u",
}

PUBLISHED_CODE_OVERRIDES["copilot_sdk"] = {
    # The runtime prefixes every tool `src-`, moves format and numeric bounds into the
    # description text, and sends name, description and parameters only; title and
    # readOnlyHint survive in its own permission request.
    "tool-name": "a",
    "format": "n",
    "numeric-range": "n",
    "json-schema-dialect": "a",
    "read-only-hint": "a",
    "title-annotation": "a",
    "destructive-hint": "n",
    "idempotent-hint": "n",
    "open-world-hint": "n",
    "tool-icons": "n",
    "tool-output-schema": "n",
    # The provider's tool message is one string: audio and resource links leave it
    # empty, structuredContent is appended as JSON, annotations and the error flag are
    # gone. session.rpc.tools.execute returns the server's contents whole.
    "content-annotations": "a",
    "embedded-resource-content": "a",
    "resource-link-content": "a",
    "structured-content": "a",
    "tool-audio-content": "a",
    "tool-execution-errors": "a",
    # No subscribe, completion or setLevel API; log messages are not surfaced.
    # session.rpc.mcp.prompts (SDK 1.0.19) lists and gets prompts, so those are y.
    "resource-subscriptions": "n",
    "completion-complete": "n",
    "logging-notifications": "n",
    "logging-set-level": "n",
    "tasks-experimental": "n",
    # The runtime answers sampling as cancelled and roots as an empty list itself.
    "sampling-create-message": "a",
    "sampling-tools": "n",
    "roots-list": "a",
    "roots-list-changed": "n",
}

PUBLISHED_CODE_OVERRIDES["crewai-native"] = {
    # Names carry the launch command or URL; the strict OpenAI schema rewrites
    # required, additionalProperties, nullability, oneOf and some formats/descriptions.
    "tool-name": "a",
    "additional-properties": "a",
    "required": "a",
    "null-acceptance": "a",
    "one-of-any-of": "a",
    "format": "a",
    "property-descriptions": "a",
    "json-schema-dialect": "n",
    # MCPNativeTool keeps name, description and an args_schema only.
    "destructive-hint": "n",
    "idempotent-hint": "n",
    "open-world-hint": "n",
    "read-only-hint": "n",
    "title-annotation": "n",
    "tool-icons": "n",
    "tool-output-schema": "n",
    # Results: content[0] only, non-text blocks stringified; isError is a ToolFailure
    # at BaseTool.run and plain text in the agent's tool message.
    "tool-execution-errors": "a",
    "structured-content": "n",
    "content-annotations": "n",
    "tool-image-content": "a",
    "tool-audio-content": "a",
    "embedded-resource-content": "a",
    "resource-link-content": "a",
    # Prompts without title; no resource API.
    "prompts-list": "a",
    "resources-list": "n",
    "resources-read": "n",
    "resource-templates": "n",
    "resource-binary-contents": "n",
    "resources-list-changed": "n",
    "resource-subscriptions": "n",
    # ClientSession is built with no callbacks or message handler.
    "sampling-create-message": "n",
    "sampling-tools": "n",
    "elicitation-form-mode": "n",
    "elicitation-url-mode": "n",
    "roots-list": "n",
    "roots-list-changed": "n",
    "tools-list-changed": "n",
    "prompts-list-changed": "n",
    "logging-notifications": "n",
    "progress-notifications": "n",
    # Log level and completion only on the raw MCPClient.session.
    "logging-set-level": "a",
    "completion-complete": "a",
    "pagination-cursor": "n",
    "cancellation": "n",
    "tasks-experimental": "n",
    # Port-dependent names over HTTP and SSE; on Python 3.11 every Streamable HTTP
    # call fails closing its transport.
    "transport-streamable-http": "a",
    "transport-http-sse": "a",
    # Conformance: MCPServerHTTP takes static headers only, no OAuth.
    "authorization-resource-metadata": "n",
}

PUBLISHED_CODE_OVERRIDES["ms_agent_framework_dotnet"] = {
    # McpClientTool gives the model name, description and the two schemas; the MCP Tool
    # (annotations, title, icons) stays on its public ProtocolTool.
    "destructive-hint": "a",
    "idempotent-hint": "a",
    "open-world-hint": "a",
    "read-only-hint": "a",
    "title-annotation": "a",
    "tool-icons": "a",
    # A lone block becomes AIContent (TextContent drops annotations and a resource's uri);
    # a result with structuredContent or a resource_link comes back as CallToolResult JSON.
    "content-annotations": "a",
    "embedded-resource-content": "a",
    "resource-link-content": "a",
    "tool-image-content": "a",
    "tool-audio-content": "a",
    # The MCP C# SDK sends no notifications/cancelled and declares no tasks capability.
    "cancellation": "n",
    "tasks-experimental": "n",
    # Conformance: a DiscoverResult the client cannot parse ends sse-retry before it starts.
    "sse-resumability": "n",
}

PUBLISHED_CODE_OVERRIDES["google_adk_go"] = {
    # convertTool keeps only name, description and the two schemas of each *mcp.Tool.
    "destructive-hint": "n",
    "idempotent-hint": "n",
    "open-world-hint": "n",
    "read-only-hint": "n",
    "title-annotation": "n",
    "tool-icons": "n",
    # mcpTool.Run returns {"output": structuredContent} when there is one, else
    # {"output": formatMCPContent(...)}, which turns non-text blocks into placeholder
    # lines; an isError result becomes a Go error the agent sees as {"error": ...}.
    "tool-text-content": "a",
    "tool-image-content": "a",
    "tool-audio-content": "a",
    "embedded-resource-content": "a",
    "resource-link-content": "a",
    "content-annotations": "n",
    "tool-execution-errors": "a",
    # Client features work only through handlers on the caller's *mcp.Client
    # (mcptoolset.Config.Client).
    "sampling-create-message": "a",
    "sampling-tools": "a",
    "elicitation-form-mode": "a",
    "elicitation-url-mode": "a",
    "roots-list": "a",
    "roots-list-changed": "a",
    "logging-notifications": "a",
    # CallTool carries no progress token or task; the session stays private.
    "progress-notifications": "n",
    "tasks-experimental": "n",
    "logging-set-level": "n",
    "completion-complete": "n",
    "prompts-list": "n",
    "prompts-get": "n",
    "prompt-arguments": "n",
    "resources-list": "n",
    "resources-read": "n",
    "resource-templates": "n",
    "resource-binary-contents": "n",
    "resource-subscriptions": "n",
    "resources-list-changed": "n",
    "prompts-list-changed": "n",
    # Conformance: SSE reconnects ignore retry and send no Last-Event-ID.
    "sse-resumability": "a",
}

PUBLISHED_CODE_OVERRIDES["koog"] = {
    # McpToolRegistryProvider.fromClient logs and skips a tool DefaultMcpToolDescriptorParser
    # cannot parse: deliver_payload (typeless oneOf) and both dialect tools are left out,
    # and every keyword deliver_payload carries is lost with it.
    "tool-name": "a",
    "description-fidelity": "a",
    "property-descriptions": "a",
    "one-of-any-of": "n",
    "json-schema-dialect": "n",
    # ToolParameterType has no slot for enum, format, range or default; nested required is
    # kept in the ToolDescriptor and left out of OpenAILLMClient's request schema.
    "enum": "n",
    "format": "n",
    "default-value": "n",
    "numeric-range": "n",
    "required": "a",
    # McpTool keeps the Client, the ToolDescriptor and string metadata, not the SDK Tool.
    "destructive-hint": "n",
    "idempotent-hint": "n",
    "open-world-hint": "n",
    "read-only-hint": "n",
    "title-annotation": "n",
    "tool-icons": "n",
    "tool-output-schema": "n",
    # The tool message is McpTool.encodeResultToString: the CallToolResult as JSON text,
    # or "Error: <text>"; the direct CallToolResult is intact.
    "tool-text-content": "a",
    "tool-image-content": "a",
    "tool-audio-content": "a",
    "embedded-resource-content": "a",
    "resource-link-content": "a",
    "content-annotations": "a",
    # Handlers set on the Kotlin SDK Client before it is handed to fromClient.
    "sampling-create-message": "a",
    "elicitation-form-mode": "a",
    "elicitation-url-mode": "a",
    "roots-list": "a",
    "roots-list-changed": "a",
    "logging-notifications": "a",
    "logging-set-level": "a",
    "completion-complete": "a",
    "resource-subscriptions": "a",
    "resources-list-changed": "a",
    "prompts-list-changed": "a",
    # Kotlin SDK 0.11.1: no sampling tools, no tasks, no notifications/cancelled; fromClient
    # lists one page into a fixed registry and passes no progress callback.
    "sampling-tools": "n",
    "tasks-experimental": "n",
    "cancellation": "n",
    "pagination-cursor": "n",
    "tools-list-changed": "n",
    "progress-notifications": "n",
    # Conformance: no OAuth flow in Koog or the Kotlin SDK client.
    "authorization-resource-metadata": "n",
}

# Google ADK's markers around every server-supplied description on the model path.
ADK_FENCE_BEGIN = "<<<BEGIN_UNTRUSTED_TOOL_DESCRIPTION>>>"
ADK_FENCE_END = "<<<END_UNTRUSTED_TOOL_DESCRIPTION>>>"

# Server utilities (scripts/probe_utilities.py against fixtures/utilities_server.py). A
# utility reachable only on a raw SDK session the adapter hands out is "a".
_UTILITY_OVERRIDES = {
    "crewai": {
        "pagination-cursor": "n", "progress-notifications": "n", "logging-notifications": "n",
        "logging-set-level": "n", "completion-complete": "n", "cancellation": "n", "tasks-experimental": "n",
    },
    "langchain": {"logging-set-level": "a", "completion-complete": "a", "cancellation": "n", "tasks-experimental": "n"},
    "openai-agents": {
        "progress-notifications": "a", "logging-set-level": "a", "completion-complete": "a", "tasks-experimental": "n",
    },
    "pydantic-ai": {"completion-complete": "a", "tasks-experimental": "n"},
    "mastra": {"logging-set-level": "n", "completion-complete": "n", "tasks-experimental": "n"},
}
for _framework, _codes in _UTILITY_OVERRIDES.items():
    PUBLISHED_CODE_OVERRIDES.setdefault(_framework, {}).update(_codes)

DEEP_SCHEMA_CAPABILITIES = {
    "additional-properties",
    "default-value",
    "null-acceptance",
    "property-descriptions",
}


def _handle():
    command = shlex.join([sys.executable, str(FIXTURE)])
    return parse_server(command)


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture server missing")
def test_source_loader_reads_fixture():
    spec_version, source_reps = load_source(_handle())
    assert spec_version and spec_version != "unknown"
    by_name = {rep.name: rep for rep in source_reps}
    assert set(by_name) == {
        "search_records",
        "create_report",
        "deliver_payload",
        "delete_account",
        "explain_topic",
    }

    search = by_name["search_records"].params
    assert search["required"] == ["query"]
    assert search["properties"]["status"]["enum"] == [
        "active",
        "archived",
        "pending",
        "deleted",
    ]
    assert search["properties"]["owner_email"]["format"] == "email"
    assert search["properties"]["created_after"]["format"] == "date-time"
    assert search["properties"]["limit"]["minimum"] == 1
    assert search["properties"]["limit"]["maximum"] == 100

    report = by_name["create_report"].params
    assert report["properties"]["config"]["type"] == "object"
    assert report["properties"]["sections"]["items"]["type"] == "object"

    delivery = by_name["deliver_payload"].params
    assert len(delivery["properties"]["destination"]["anyOf"]) == 2
    assert len(delivery["properties"]["mode"]["oneOf"]) == 2

    assert by_name["explain_topic"].description.endswith(
        "SENTINEL_END_OF_LONG_DESCRIPTION."
    )
    delete = by_name["delete_account"]
    assert delete.annotations.get("destructiveHint") is True


@pytest.mark.skipif(not RENDERERS, reason="no framework renderers installed")
@pytest.mark.parametrize("renderer_id", sorted(RENDERERS))
def test_renderer_declares_exact_capture_boundary(renderer_id):
    evidence = type(RENDERERS[renderer_id])().evidence()
    expected = EXPECTED_BOUNDARIES[renderer_id]

    assert evidence.capture_api == expected["capture_api"]
    assert evidence.capture_object == expected["capture_object"]
    assert evidence.capture_stage == expected["capture_stage"]
    # Only a renderer that intercepts the serialized request body may claim it.
    assert evidence.provider_request_captured is (expected["capture_stage"] == "provider_request")
    assert evidence.negotiated_mcp_spec_version is None
    assert evidence.protocol_version_evidence


@pytest.mark.skipif(
    "langchain" not in RENDERERS,
    reason="langchain renderer not installed",
)
def test_isolated_worker_returns_renderer_evidence():
    versions, evidence, reps = _render_isolated(
        "langchain",
        _handle().spec,
        {},
    )

    assert versions["framework"] != "unknown"
    assert evidence.capture_api == EXPECTED_BOUNDARIES["langchain"]["capture_api"]
    assert evidence.provider_request_captured is False
    assert evidence.negotiated_mcp_spec_version == "2025-11-25"
    assert reps


PUBLISHED_IDS = {
    "autogen": "autogen",
    "langchain": "langchain",
    "pydantic_ai": "pydantic-ai",
    "crewai": "crewai",
    "crewai_native": "crewai-native",
    "openai_agents": "openai-agents",
    "google_adk": "google_adk",
    "llamaindex": "llamaindex",
    "ms_agent_framework": "ms_agent_framework",
    "strands": "strands",
    "mastra": "mastra",
    "vercel_ai": "vercel_ai",
    "openai_agents_js": "openai_agents_js",
    "google_adk_js": "google_adk_js",
    "semantic_kernel": "semantic_kernel",
    "pi": "pi",
    "agno": "agno",
    "haystack": "haystack",
    "smolagents": "smolagents",
    "dspy": "dspy",
    "spring_ai": "spring_ai",
    "langchain_js": "langchain_js",
    "genkit_js": "genkit_js",
    "beeai": "beeai",
    "copilot_sdk": "copilot_sdk",
    "google_adk_java": "google_adk_java",
    "langchain4j": "langchain4j",
    "ms_agent_framework_dotnet": "ms_agent_framework_dotnet",
    "google_adk_go": "google_adk_go",
    "koog": "koog",
}


def _manifest_pin(renderer_id: str, package: str) -> str:
    """The version of ``package`` that the managed runner for ``renderer_id`` installs.

    Published cells come from managed runs, so the manifest is what the site is making
    a claim about. A developer's local environment drifts freely and says nothing about
    whether the published number is right.
    """

    manifest = json.loads(
        (Path(__file__).resolve().parent.parent / "src" / "mcp_mirror" / "runner-manifests.json").read_text()
    )["runners"][renderer_id]
    for requirement in manifest["packages"]:
        # npm and NuGet pin with @, Python with ==, Maven as group:artifact:version, and
        # Go as module@vX.Y.Z.
        go = "/" in package and not package.startswith("@")
        nuget = requirement.startswith(f"{package}@")
        delimiter = "@" if package.startswith("@") or go or nuget else ":" if ":" in package else "=="
        name, separator, version = requirement.rpartition(delimiter)
        if name.partition("[")[0] == package and separator:
            return version.removeprefix("v") if go else version
    raise AssertionError(f"{renderer_id}: manifest installs no pin for {package}")


def test_published_framework_versions_match_the_managed_runner_pins():
    published = json.loads(FRAMEWORK_DATA.read_text())["agents"]

    for renderer_id, published_id in PUBLISHED_IDS.items():
        record = published[published_id]
        assert record["current_version"] == _manifest_pin(renderer_id, record["package"]), (
            f"{published_id}: the site publishes {record['current_version']} but the "
            f"managed runner measures {_manifest_pin(renderer_id, record['package'])}"
        )


@pytest.mark.skipif(not RENDERERS, reason="no framework renderers installed")
def test_published_boundaries_match_installed_renderers():
    published = json.loads(FRAMEWORK_DATA.read_text())["agents"]

    for renderer_id, renderer in RENDERERS.items():
        renderer.render(_handle())
        record = published[PUBLISHED_IDS[renderer_id]]
        assert record["capture_boundary"] == renderer.evidence().model_dump(
            mode="json",
            exclude={"limitation"},
        )


@pytest.mark.skipif(not RENDERERS, reason="no framework renderers installed")
@pytest.mark.parametrize("renderer_id", sorted(RENDERERS))
def test_published_result_boundaries_match_installed_renderers(renderer_id):
    """The site's result boundaries are the renderer's, not a prose description.

    The published capture boundary used to describe only where a tool *definition*
    was read, so a result cell could be measured somewhere the site never named.
    Pinning the published list to the renderer's own declaration is what stops that
    from happening again silently.
    """

    published = json.loads(FRAMEWORK_DATA.read_text())["agents"]
    renderer = RENDERERS[renderer_id]
    declared = [
        boundary.model_dump(mode="json", exclude_none=True)
        for boundary in renderer.result_boundaries()
    ]

    assert published[PUBLISHED_IDS[renderer_id]]["result_boundaries"] == declared
    assert sum(1 for boundary in declared if boundary["agent_path"]) == 1


@pytest.mark.skipif(not RENDERERS, reason="no framework renderers installed")
@pytest.mark.parametrize("renderer_id", sorted(RENDERERS))
def test_every_declared_result_boundary_can_actually_be_captured(renderer_id):
    """A declared boundary that cannot be driven is a claim, not a measurement."""

    renderer = RENDERERS[renderer_id]
    handle = _handle()
    tool = "src_delete_account" if renderer_id == "mastra" else "delete_account"

    for boundary in renderer.result_boundaries():
        rep = renderer.render_result(
            handle, tool.removeprefix("src_"), {"account_id": "a", "confirm": False}, boundary.id
        )
        assert rep.tool
        assert rep.origin == renderer.id


@pytest.mark.skipif(
    "langchain" not in RENDERERS,
    reason="langchain renderer not installed",
)
def test_langchain_result_boundaries_disagree_about_the_error_flag():
    """The measurement this row was corrected for, pinned as a regression test.

    ``langchain-mcp-adapters`` builds its tool with ``handle_tool_error``, so an
    ``isError`` result reaches a direct caller as ordinary content blocks and reaches
    an agent as a ``ToolMessage`` with ``status="error"``. Publishing only the first
    reported the flag as dropped. Both are asserted here so a future change that
    collapses them is visible rather than quietly re-breaking the cell.
    """

    renderer = RENDERERS["langchain"]
    handle = _handle()
    arguments = {"account_id": "a", "confirm": False}

    direct = renderer.render_result(handle, "delete_account", arguments, "direct")
    agent = renderer.render_result(handle, "delete_account", arguments, "agent")

    assert direct.is_error is None, "direct invocation still carries no failure signal"
    assert agent.is_error is True, "the agent boundary must expose the failure"
    assert "SENTINEL_TOOL_ERROR" in (agent.text or "")
    assert 'status=\'error\'' in str(agent.raw)


@pytest.mark.skipif(not RENDERERS, reason="no framework renderers installed")
def test_boundary_dependent_cells_name_an_agent_that_publishes_two_boundaries():
    """Only a framework with more than one result boundary can depend on one."""

    published = json.loads(FRAMEWORK_DATA.read_text())["agents"]
    capability_dir = FRAMEWORK_DATA.parent / "capabilities"

    for path in sorted(capability_dir.glob("*.json")):
        capability = json.loads(path.read_text())
        for framework_id in capability.get("boundary_dependent") or []:
            agent = published[framework_id]
            assert len(agent["result_boundaries"]) >= 2, (
                f"{capability['id']} / {framework_id}: marked boundary-dependent but "
                "only one result boundary is published"
            )
            cell = capability["stats"][framework_id][agent["current_version"]]
            assert cell.split()[0] == "a", (
                f"{capability['id']} / {framework_id}: a boundary-dependent cell must "
                f"be 'a', found {cell!r}"
            )


def test_published_current_cells_match_fixture_goldens():
    # Aliases have no renderer of their own; test_alias_visibility checks them against their source.
    published = {
        a: agent
        for a, agent in json.loads(FRAMEWORK_DATA.read_text())["agents"].items()
        if not agent.get("aliases_of")
    }

    for capability_id, capability in sorted(_published_rows(FRAMEWORK_DATA.parent).items()):
        for framework_id, agent in published.items():
            current_version = agent["current_version"]
            raw_code = capability["stats"][framework_id][current_version]
            expected = PUBLISHED_CODE_OVERRIDES.get(framework_id, {}).get(
                capability_id,
                "y",
            )
            assert raw_code.split()[0] == expected, (
                f"{capability_id} / {framework_id} {current_version}: "
                f"expected {expected!r}, found {raw_code!r}"
            )


def test_published_census_covers_every_measured_deep_schema_semantic():
    published = _published_rows(FRAMEWORK_DATA.parent)

    assert DEEP_SCHEMA_CAPABILITIES <= set(published)
    for capability_id in DEEP_SCHEMA_CAPABILITIES:
        capability = published[capability_id]
        # Deep-schema rows are definition-scan rows; the scan runs the definitions fixture
        # (tricky_server.py is its published alias).
        assert capability["measured"]["fixture"] in {"fixtures/tricky_server.py", "fixtures/definitions_server.py"}
        assert capability["question"]
        assert capability["docs_url"].startswith("https://json-schema.org/")


@pytest.mark.skipif(not RENDERERS, reason="no framework renderers installed")
@pytest.mark.parametrize("renderer_id", sorted(RENDERERS))
def test_renderer_fixture_contract(renderer_id, monkeypatch):
    handle = _handle()
    if renderer_id == "crewai_native":
        # Agent(mcps=[MCPServerStdio]) prefixes every name with the launch command and
        # its args, and an absolute path pushes the result past 64 characters, where
        # sanitize_tool_name swaps it for a hash (see the test below). Launch the
        # fixture the way the published measurements do, by a relative path.
        monkeypatch.chdir(FIXTURE.parents[1])
        monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}:{os.environ['PATH']}")
        handle = parse_server(f"python fixtures/{FIXTURE.name}")
    source_spec_version, source_reps = load_source(handle)
    rendered_reps = RENDERERS[renderer_id].render(handle)

    # Koog: McpToolRegistryProvider.fromClient catches the parser's Throwable and logs
    # it, so deliver_payload (mode is a oneOf of consts with no type) never registers.
    dropped = {"deliver_payload"} if renderer_id == "koog" else set()
    assert len(source_reps) == 5
    assert len(rendered_reps) == 5 - len(dropped)
    assert (
        RENDERERS[renderer_id].evidence().negotiated_mcp_spec_version
        == source_spec_version
    )
    source_by_name = {rep.name: rep for rep in source_reps}
    rendered_by_source_name = {
        (
            rep.name.removeprefix("src_")
            if renderer_id == "mastra"
            else rep.name.removeprefix(SK_PREFIX)
            if renderer_id == "semantic_kernel"
            else rep.name.removeprefix("src_")
            if renderer_id == "langchain_js"
            else rep.name.removeprefix("src-")
            if renderer_id == "copilot_sdk"
            else rep.name.removeprefix(CREWAI_NATIVE_PREFIX)
            if renderer_id == "crewai_native"
            else rep.name
        ): rep
        for rep in rendered_reps
    }
    assert set(rendered_by_source_name) == set(source_by_name) - dropped
    explain = rendered_by_source_name["explain_topic"].description
    if renderer_id == "google_adk":
        # The model-bound declaration is fenced; the source text sits inside the fence.
        assert explain.startswith(ADK_FENCE_BEGIN + "\n")
        assert explain.endswith("\n" + ADK_FENCE_END)
        explain = explain[len(ADK_FENCE_BEGIN) + 1 : -len(ADK_FENCE_END) - 1]
    assert explain.endswith("SENTINEL_END_OF_LONG_DESCRIPTION.")

    diffs = diff_reps(source_reps, rendered_reps)

    schema_diffs = [
        diff
        for diff in diffs
        if diff.dimension
        in {Dimension.PARAM_TYPE, Dimension.CONSTRAINT, Dimension.REQUIRED}
    ]
    if renderer_id == "crewai":
        expected_schema_diffs = {
            ("search_records", "params.additionalProperties", "transformative", Dimension.CONSTRAINT),
            ("search_records", "params.properties.status.null", "additive", Dimension.PARAM_TYPE),
            ("search_records", "params.properties.owner_email.null", "additive", Dimension.PARAM_TYPE),
            ("search_records", "params.properties.created_after.null", "additive", Dimension.PARAM_TYPE),
            ("search_records", "params.properties.limit.null", "additive", Dimension.PARAM_TYPE),
            ("create_report", "params.additionalProperties", "transformative", Dimension.CONSTRAINT),
            ("create_report", "params.properties.config.additionalProperties", "transformative", Dimension.CONSTRAINT),
            ("create_report", "params.properties.config.properties.include_summary.null", "additive", Dimension.PARAM_TYPE),
            ("create_report", "params.properties.sections.null", "additive", Dimension.PARAM_TYPE),
            ("create_report", "params.properties.sections.items.additionalProperties", "transformative", Dimension.CONSTRAINT),
            ("create_report", "params.properties.sections.items.properties.body.null", "additive", Dimension.PARAM_TYPE),
            ("deliver_payload", "params.additionalProperties", "transformative", Dimension.CONSTRAINT),
            ("deliver_payload", "params.properties.destination.anyOf[0].format", "lossy", Dimension.CONSTRAINT),
            ("deliver_payload", "params.properties.destination.anyOf[0].description", "lossy", Dimension.CONSTRAINT),
            ("deliver_payload", "params.properties.destination.anyOf[1].additionalProperties", "transformative", Dimension.CONSTRAINT),
            ("deliver_payload", "params.properties.destination.anyOf[1].description", "lossy", Dimension.CONSTRAINT),
            ("deliver_payload", "params.properties.mode.null", "additive", Dimension.PARAM_TYPE),
            ("delete_account", "params.additionalProperties", "transformative", Dimension.CONSTRAINT),
            ("explain_topic", "params.additionalProperties", "transformative", Dimension.CONSTRAINT),
            ("explain_topic", "params.properties.max_words.null", "additive", Dimension.PARAM_TYPE),
        }
        assert {
            (diff.tool, diff.path, diff.category, diff.dimension)
            for diff in schema_diffs
        } == expected_schema_diffs
    elif renderer_id in {"vercel_ai", "autogen"}:
        # Vercel AI: toolsFromDefinitions closes each root object and changes nothing
        # else. AutoGen: McpWorkbench rebuilds each schema from properties, required and
        # additionalProperties, defaulting the last to False when the server omits it.
        assert {
            (diff.tool, diff.path, diff.category, diff.dimension)
            for diff in schema_diffs
        } == {
            (name, "params.additionalProperties", "transformative", Dimension.CONSTRAINT)
            for name in source_by_name
        }
    elif renderer_id == "llamaindex":
        # McpToolSpec rebuilds each inputSchema as a Pydantic model and the model's
        # JSON schema is what the LLM gets: formats and numeric bounds are gone,
        # optional fields newly accept null, and a nested object's properties and
        # required list collapse to a free-form object.
        assert {
            (diff.tool, diff.path, diff.category, diff.dimension)
            for diff in schema_diffs
        } == {
            ("search_records", "params.properties.status.null", "additive", Dimension.PARAM_TYPE),
            ("search_records", "params.properties.owner_email.null", "additive", Dimension.PARAM_TYPE),
            ("search_records", "params.properties.owner_email.format", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.created_after.null", "additive", Dimension.PARAM_TYPE),
            ("search_records", "params.properties.created_after.format", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.limit.minimum", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.limit.maximum", "lossy", Dimension.CONSTRAINT),
            ("create_report", "params.properties.config.required", "lossy", Dimension.REQUIRED),
            ("create_report", "params.properties.sections.null", "additive", Dimension.PARAM_TYPE),
            ("create_report", "params.properties.sections.items.required", "lossy", Dimension.REQUIRED),
            ("deliver_payload", "params.properties.destination.anyOf[0].format", "lossy", Dimension.CONSTRAINT),
            ("deliver_payload", "params.properties.destination.anyOf[0].description", "lossy", Dimension.CONSTRAINT),
            ("deliver_payload", "params.properties.destination.anyOf[1].description", "lossy", Dimension.CONSTRAINT),
            ("deliver_payload", "params.properties.destination.anyOf[1].required", "lossy", Dimension.REQUIRED),
            ("deliver_payload", "params.properties.mode.null", "additive", Dimension.PARAM_TYPE),
            ("explain_topic", "params.properties.max_words.null", "additive", Dimension.PARAM_TYPE),
            ("explain_topic", "params.properties.max_words.minimum", "lossy", Dimension.CONSTRAINT),
        }
    elif renderer_id == "copilot_sdk":
        # The Copilot runtime moves format and numeric bounds out of the schema and into
        # the parameter description as text, e.g. `{minimum: 1, maximum: 100}`.
        assert {
            (diff.tool, diff.path, diff.category, diff.dimension)
            for diff in schema_diffs
        } == {
            ("search_records", "params.properties.owner_email.format", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.created_after.format", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.limit.minimum", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.limit.maximum", "lossy", Dimension.CONSTRAINT),
            ("deliver_payload", "params.properties.destination.anyOf[0].format", "lossy", Dimension.CONSTRAINT),
            ("explain_topic", "params.properties.max_words.minimum", "lossy", Dimension.CONSTRAINT),
        }
    elif renderer_id == "google_adk_js":
        # toGeminiSchema keeps type, description, enum, items, properties, required and
        # anyOf; formats, numeric bounds and defaults do not reach the model.
        assert {
            (diff.tool, diff.path, diff.category, diff.dimension)
            for diff in schema_diffs
        } == {
            ("search_records", "params.properties.owner_email.format", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.created_after.format", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.limit.minimum", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.limit.maximum", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.limit.default", "lossy", Dimension.CONSTRAINT),
            ("deliver_payload", "params.properties.destination.anyOf[0].format", "lossy", Dimension.CONSTRAINT),
            ("explain_topic", "params.properties.max_words.minimum", "lossy", Dimension.CONSTRAINT),
        }
    elif renderer_id == "crewai_native":
        # MCPToolResolver rebuilds each schema as a Pydantic model: optionals newly
        # accept null, formats move into the description, nested objects close; then
        # convert_tools_to_openai_schema marks the tool strict, so every property is
        # listed as required.
        assert {
            (diff.tool, diff.path, diff.category, diff.dimension)
            for diff in schema_diffs
        } == {
            ("search_records", "params.additionalProperties", "transformative", Dimension.CONSTRAINT),
            ("search_records", "params.properties.status.null", "additive", Dimension.PARAM_TYPE),
            ("search_records", "params.properties.owner_email.null", "additive", Dimension.PARAM_TYPE),
            ("search_records", "params.properties.owner_email.format", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.created_after.null", "additive", Dimension.PARAM_TYPE),
            ("search_records", "params.properties.limit.null", "additive", Dimension.PARAM_TYPE),
            ("search_records", "params.required", "transformative", Dimension.REQUIRED),
            ("create_report", "params.additionalProperties", "transformative", Dimension.CONSTRAINT),
            ("create_report", "params.properties.config.additionalProperties", "transformative", Dimension.CONSTRAINT),
            ("create_report", "params.properties.config.description", "lossy", Dimension.CONSTRAINT),
            ("create_report", "params.properties.config.properties.include_summary.null", "additive", Dimension.PARAM_TYPE),
            ("create_report", "params.properties.config.required", "transformative", Dimension.REQUIRED),
            ("create_report", "params.properties.sections.null", "additive", Dimension.PARAM_TYPE),
            ("create_report", "params.properties.sections.items.additionalProperties", "transformative", Dimension.CONSTRAINT),
            ("create_report", "params.properties.sections.items.properties.body.null", "additive", Dimension.PARAM_TYPE),
            ("create_report", "params.properties.sections.items.required", "transformative", Dimension.REQUIRED),
            ("create_report", "params.required", "transformative", Dimension.REQUIRED),
            ("deliver_payload", "params.additionalProperties", "transformative", Dimension.CONSTRAINT),
            ("deliver_payload", "params.properties.destination.anyOf[1].additionalProperties", "transformative", Dimension.CONSTRAINT),
            ("deliver_payload", "params.properties.destination.anyOf[1].description", "lossy", Dimension.CONSTRAINT),
            ("deliver_payload", "params.properties.destination.anyOf[0].format", "lossy", Dimension.CONSTRAINT),
            ("deliver_payload", "params.properties.destination.anyOf[0].description", "lossy", Dimension.CONSTRAINT),
            ("deliver_payload", "params.properties.mode.null", "additive", Dimension.PARAM_TYPE),
            ("deliver_payload", "params.required", "transformative", Dimension.REQUIRED),
            ("delete_account", "params.additionalProperties", "transformative", Dimension.CONSTRAINT),
            ("explain_topic", "params.additionalProperties", "transformative", Dimension.CONSTRAINT),
            ("explain_topic", "params.properties.max_words.null", "additive", Dimension.PARAM_TYPE),
            ("explain_topic", "params.required", "transformative", Dimension.REQUIRED),
        }
    elif renderer_id == "langchain4j":
        # McpToolProvider maps inputSchema into JsonSchemaElement, which keeps type,
        # description, enum, items, properties, required and anyOf; formats, numeric
        # bounds and defaults do not reach the model.
        assert {
            (diff.tool, diff.path, diff.category, diff.dimension)
            for diff in schema_diffs
        } == {
            ("search_records", "params.properties.owner_email.format", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.created_after.format", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.limit.minimum", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.limit.maximum", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.limit.default", "lossy", Dimension.CONSTRAINT),
            ("deliver_payload", "params.properties.destination.anyOf[0].format", "lossy", Dimension.CONSTRAINT),
            ("explain_topic", "params.properties.max_words.minimum", "lossy", Dimension.CONSTRAINT),
        }
    elif renderer_id == "koog":
        # DefaultMcpToolDescriptorParser keeps a type, a description and nested structure;
        # enum on a typed string, formats, bounds and defaults have no ToolParameterType
        # slot. The nested requiredProperties it keeps do not reach the OpenAI schema.
        assert {
            (diff.tool, diff.path, diff.category, diff.dimension)
            for diff in schema_diffs
        } == {
            ("search_records", "params.properties.status.enum", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.owner_email.format", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.created_after.format", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.limit.minimum", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.limit.maximum", "lossy", Dimension.CONSTRAINT),
            ("search_records", "params.properties.limit.default", "lossy", Dimension.CONSTRAINT),
            ("create_report", "params.properties.config.properties.format.enum", "lossy", Dimension.CONSTRAINT),
            ("create_report", "params.properties.config.required", "lossy", Dimension.REQUIRED),
            ("create_report", "params.properties.sections.items.required", "lossy", Dimension.REQUIRED),
            ("explain_topic", "params.properties.max_words.minimum", "lossy", Dimension.CONSTRAINT),
        }
    elif renderer_id == "smolagents":
        # get_tool_json_schema rebuilds `required` from every non-nullable input, so each
        # tool with an optional parameter gains it as required.
        assert {
            (diff.tool, diff.path, diff.category, diff.dimension)
            for diff in schema_diffs
        } == {
            (name, "params.required", "transformative", Dimension.REQUIRED)
            for name in ("search_records", "create_report", "deliver_payload", "explain_topic")
        }
    else:
        assert schema_diffs == []

    structure = [diff for diff in diffs if diff.dimension == Dimension.STRUCTURE]
    if renderer_id in {"crewai", "crewai_native", "google_adk_js", "langchain4j"}:
        # google_adk_js: toGeminiSchema has no oneOf, so the property arrives untyped.
        # langchain4j: JsonSchemaElement has no oneOf; the property becomes an empty object.
        assert [(diff.path, diff.category) for diff in structure] == [
            ("params.properties.mode.oneOf", "transformative")
        ]
    elif renderer_id == "smolagents":
        # anyOf becomes a list of its branches' types; the object branch's properties go.
        assert [(diff.tool, diff.path, diff.category) for diff in structure] == [
            ("deliver_payload", "params.properties.destination.anyOf", "transformative")
        ]
    elif renderer_id == "llamaindex":
        assert {(diff.tool, diff.path, diff.category) for diff in structure} == {
            ("create_report", "params.properties.config.properties", "transformative"),
            ("create_report", "params.properties.sections.items.properties", "transformative"),
            ("deliver_payload", "params.properties.destination.anyOf[1].properties", "transformative"),
            ("deliver_payload", "params.properties.mode.oneOf", "transformative"),
        }
    else:
        assert structure == []

    descriptions = [
        diff for diff in diffs if diff.dimension in {Dimension.DESCRIPTION, Dimension.INJECTION}
    ]
    if renderer_id in {"crewai", "google_adk"}:
        assert len(descriptions) == 5
        assert all(diff.category == "additive" for diff in descriptions)
        assert all(
            source_by_name[name].description in rendered_by_source_name[name].description
            for name in source_by_name
        )
    else:
        assert descriptions == []

    names = [diff for diff in diffs if diff.dimension == Dimension.NAME]
    if renderer_id in {"mastra", "semantic_kernel", "langchain_js", "copilot_sdk", "crewai_native"}:
        assert len(names) == 5
        assert all(diff.category == "transformative" for diff in names)
        prefix = {
            "mastra": "src_",
            "semantic_kernel": SK_PREFIX,
            "langchain_js": "src_",
            "copilot_sdk": "src-",
            "crewai_native": CREWAI_NATIVE_PREFIX,
        }[renderer_id]
        assert all(rep.name.startswith(prefix) for rep in rendered_reps)
    elif renderer_id == "koog":
        assert [(diff.tool, diff.path, diff.category) for diff in names] == [
            ("deliver_payload", "name", "lossy")
        ]
    else:
        assert names == []

    annotations = {
        diff.path.rsplit(".", 1)[-1]: diff
        for diff in diffs
        if diff.tool == "delete_account" and diff.dimension == Dimension.ANNOTATION
    }
    if renderer_id in {"strands", "pi"}:
        # Strands: MCPAgentTool passes the annotations through on the captured ToolSpec.
        # Pi: McpClient.listTools returns the tools/list entry unchanged.
        assert annotations == {}
        assert rendered_by_source_name["delete_account"].annotations == source_by_name["delete_account"].annotations
        return
    assert set(annotations) == {
        "title",
        "readOnlyHint",
        "destructiveHint",
        "idempotentHint",
        "openWorldHint",
    }
    behavior_hints = {
        key: annotations[key]
        for key in (
            "readOnlyHint",
            "destructiveHint",
            "idempotentHint",
            "openWorldHint",
        )
    }
    # Mastra 2.x keeps the server's annotations on the tool action's `mcp`
    # metadata; 1.x carried nothing, so its hints were destroyed. Google ADK (Java)
    # keeps them on McpTool.annotations(), and Agent Framework (.NET) on
    # McpClientTool.ProtocolTool.
    expected_hint_category = (
        "transformative"
        if renderer_id in {"langchain", "pydantic_ai", "mastra", "google_adk", "vercel_ai", "langchain_js", "google_adk_java", "langchain4j", "ms_agent_framework_dotnet"}
        else "lossy"
    )
    if renderer_id == "copilot_sdk":
        # The Copilot runtime's approval step keeps readOnlyHint (a read-only tool runs
        # unasked; any other asks with readOnly in the request); the rest are dropped.
        assert annotations["readOnlyHint"].category == "transformative"
        assert {
            annotations[key].category for key in ("destructiveHint", "idempotentHint", "openWorldHint")
        } == {"lossy"}
    else:
        assert {
            diff.category for diff in behavior_hints.values()
        } == {expected_hint_category}

    expected_title_category = (
        "transformative"
        if renderer_id in {"langchain", "pydantic_ai", "openai_agents", "mastra", "google_adk", "vercel_ai", "langchain_js", "copilot_sdk", "google_adk_java", "langchain4j", "ms_agent_framework_dotnet"}
        else "lossy"
    )
    assert annotations["title"].category == expected_title_category

    scorecard = build_scorecard(
        Report(
            mcp_server=handle.spec,
            mcp_spec_version=source_spec_version,
            generated_with="mcp-mirror/test",
            tools=sorted(source_by_name),
            runs=[
                FrameworkRun(
                    framework=renderer_id,
                    framework_version=RENDERERS[renderer_id].versions()["framework"],
                    differences=diffs,
                )
            ],
        )
    )
    jobs = scorecard["frameworks"][renderer_id]["jobs"]
    if renderer_id == "crewai":
        assert jobs["J2"]["verdict"] == "lossy"
        assert jobs["J2"]["count"] == len(schema_diffs)
        assert jobs["J3"] == {
            "verdict": "transformative",
            "count": 1,
            "details": [structure[0].detail],
        }
    if renderer_id == "mastra":
        assert jobs["J1"]["verdict"] == "transformative"
        assert jobs["J1"]["count"] == len(names) == 5


@pytest.mark.skipif(
    "pydantic_ai" not in RENDERERS,
    reason="pydantic_ai renderer not installed",
)
def test_pydantic_ai_renderer_uses_current_mcp_toolset():
    from pydantic_ai.mcp import MCPToolset

    renderer = RENDERERS["pydantic_ai"]
    toolset = renderer._build_server(_handle())

    assert isinstance(toolset, MCPToolset)


@pytest.mark.skipif(
    "google_adk" not in RENDERERS,
    reason="google_adk renderer not installed",
)
def test_google_adk_reads_the_negotiated_version_on_both_connect_paths():
    """``observe_negotiation`` sees the version whichever way the session connected.

    On mcp 2.x both the legacy handshake and the SDK's ``negotiate_auto`` (ADK's
    ``ADK_ENABLE_MCP_MODERN_PROTOCOL=1`` path, which probes ``server/discover``) end in
    ``ClientSession.adopt``; on mcp 1.x only ``initialize`` exists. #27 left the opt-in
    scan unfinished because the renderer wrapped ``initialize`` alone.
    """
    import asyncio

    from mcp_mirror.renderers.google_adk_renderer import observe_negotiation

    class Result:
        def __init__(self, version):
            self.protocol_version = version

    class ModernSession:  # the mcp 2.x shape
        def __init__(self):
            self.initialize_result = None
            self.protocol_version = None

        def adopt(self, result):
            self.protocol_version = result.protocol_version
            self.initialize_result = result if result.protocol_version == "2025-11-25" else None

        async def initialize(self):
            self.adopt(Result("2025-11-25"))

        async def negotiate_auto(self):
            self.adopt(Result("2026-07-28"))

    original_adopt = ModernSession.adopt
    with observe_negotiation(ModernSession) as seen:
        asyncio.run(ModernSession().negotiate_auto())
        asyncio.run(ModernSession().initialize())
    assert seen == [("2026-07-28", "server/discover"), ("2025-11-25", "initialize")]
    assert ModernSession.adopt is original_adopt

    class LegacySession:  # the mcp 1.x shape: no adopt
        async def initialize(self):
            return Result("2025-11-25")

    original_initialize = LegacySession.initialize
    with observe_negotiation(LegacySession) as seen:
        asyncio.run(LegacySession().initialize())
    assert seen == [("2025-11-25", "initialize")]
    assert LegacySession.initialize is original_initialize


@pytest.mark.skipif(
    "google_adk" not in RENDERERS,
    reason="google_adk renderer not installed",
)
def test_google_adk_fences_property_descriptions_on_the_model_path():
    """The differ flags a lost property description, not a rewritten one.

    ADK's McpTool.process_llm_request fences every schema ``description`` as well as
    the tool's own, so the property-descriptions row derives ``y`` while the text the
    model reads is wrapped. This pins the wrapping.
    """

    _version, source_reps = load_source(_handle())
    source = {rep.name: rep for rep in source_reps}["create_report"]
    rendered = {rep.name: rep for rep in RENDERERS["google_adk"].render(_handle())}["create_report"]

    source_title = source.params["properties"]["title"]["description"]
    rendered_title = rendered.params["properties"]["title"]["description"]
    assert rendered_title == f"{ADK_FENCE_BEGIN}\n{source_title}\n{ADK_FENCE_END}"
    nested = rendered.params["properties"]["config"]["properties"]["format"]
    assert nested["description"].startswith(ADK_FENCE_BEGIN)
    assert nested["enum"] == source.params["properties"]["config"]["properties"]["format"]["enum"]


@pytest.mark.skipif(
    "google_adk" not in RENDERERS,
    reason="google_adk renderer not installed",
)
def test_google_adk_model_reads_the_tool_result_as_json():
    """The tool returns the dumped CallToolResult; the model gets it inside a
    FunctionResponse, as JSON rather than content blocks, with isError still set."""

    renderer = RENDERERS["google_adk"]
    arguments = {"account_id": "a", "confirm": False}

    direct = renderer.render_result(_handle(), "delete_account", arguments, "direct")
    agent = renderer.render_result(_handle(), "delete_account", arguments, "agent")

    assert direct.block_kinds == ["text"]
    assert direct.is_error is True
    assert agent.block_kinds == []
    assert agent.is_error is True
    assert json.loads(agent.text)["content"][0]["text"] == direct.text


@pytest.mark.skipif(
    "ms_agent_framework" not in RENDERERS,
    reason="ms_agent_framework renderer not installed",
)
def test_ms_agent_framework_hides_prompts_from_definitions_and_errors_from_the_model():
    """Pins two Agent Framework findings about prompts and errors.

    ``load_prompts`` turns each MCP prompt into a FunctionTool beside the tools, and the
    renderer reports only the tool-built ones as definitions. On the agent path an
    ``isError`` result reaches the model as a generic error, without the server's text.
    """

    renderer = RENDERERS["ms_agent_framework"]
    handle = _handle()
    names = {rep.name for rep in renderer.render(handle)}
    assert "summarize_records" not in names
    prompts = renderer.probe_surface(handle)["prompts.list"]
    assert [entry["name"] for entry in prompts["entries"]] == ["summarize_records"]

    results = parse_server(shlex.join([sys.executable, str(FIXTURE.with_name("results_server.py"))]))
    direct = renderer.render_result(results, "tool_error", {}, "direct")
    agent = renderer.render_result(results, "tool_error", {}, "agent")
    assert direct.is_error and "SENTINEL_TOOL_ERROR" in (direct.text or "")
    assert agent.is_error and "SENTINEL_TOOL_ERROR" not in (agent.text or "")


@pytest.mark.skipif(
    "strands" not in RENDERERS,
    reason="strands renderer not installed",
)
def test_strands_agent_fills_in_missing_top_level_property_descriptions():
    """ToolRegistry.validate_tool_spec writes a ``Property <name>`` description into any
    top-level property that lacks one before the spec reaches Model.stream. The fixture
    describes every top-level property, so the captured spec matches it; this pins that
    the capture is the registry's output rather than MCPAgentTool.tool_spec."""

    from strands.tools.registry import ToolRegistry

    spec = {"name": "t", "description": "d", "inputSchema": {"json": {"type": "object", "properties": {"x": {}}}}}
    registry = ToolRegistry()
    registry.validate_tool_spec(spec)
    assert spec["inputSchema"]["json"]["properties"]["x"] == {"type": "string", "description": "Property x"}

    _version, source_reps = load_source(_handle())
    rendered = {rep.name: rep for rep in RENDERERS["strands"].render(_handle())}
    for rep in source_reps:
        assert rendered[rep.name].params == rep.params


@pytest.mark.skipif(
    "strands" not in RENDERERS,
    reason="strands renderer not installed",
)
def test_strands_tool_error_reaches_the_model_with_its_flag():
    """MCPClient maps isError onto the ToolResult's status and keeps isError beside it;
    the agent appends that ToolResult unchanged to the conversation the model reads."""

    renderer = RENDERERS["strands"]
    arguments = {"account_id": "a", "confirm": False}

    direct = renderer.render_result(_handle(), "delete_account", arguments, "direct")
    agent = renderer.render_result(_handle(), "delete_account", arguments, "agent")

    assert direct.is_error is True and agent.is_error is True
    assert direct.raw["status"] == agent.raw["status"] == "error"
    assert direct.text == agent.text


@pytest.mark.skipif(
    "openai_agents_js" not in RENDERERS,
    reason="openai_agents_js renderer not installed",
)
def test_openai_agents_js_model_reads_errors_and_structured_content_only_as_text():
    """What the adapter does with results, pinned.

    The function_call_result the next model turn receives has no isError and no
    structuredContent field: the failure arrives as ordinary text, and the typed value
    only replaces the content when the non-default useStructuredContent option is on.
    """

    renderer = RENDERERS["openai_agents_js"]
    handle = _handle()
    arguments = {"account_id": "a", "confirm": False}

    agent = renderer.render_result(handle, "delete_account", arguments, "agent")
    direct = renderer.render_result(handle, "delete_account", arguments, "direct")

    assert agent.is_error is None
    assert agent.raw["type"] == "function_call_result"
    assert "isError" not in agent.raw
    assert direct.is_error is True


@pytest.mark.skipif(
    "openai_agents_js" not in RENDERERS,
    reason="openai_agents_js renderer not installed",
)
def test_openai_agents_js_fills_in_the_top_level_object_keywords():
    """mcpToFunctionTool spreads the inputSchema and fills in type, properties and
    required where the server left them out, so every tool reaches the model as an
    object schema with a required list."""

    rendered = {rep.name: rep for rep in RENDERERS["openai_agents_js"].render(_handle())}

    for rep in rendered.values():
        assert rep.params.get("type") == "object"
        assert "required" in rep.params


@pytest.mark.skipif(
    "semantic_kernel" not in RENDERERS,
    reason="semantic_kernel renderer not installed",
)
def test_semantic_kernel_offers_every_prompt_to_the_model_as_a_tool():
    """load_prompts (on by default) turns each MCP prompt into a KernelFunction, so the
    prompt reaches the request's tools array beside the tools, with every argument
    marked required. render() returns only the tool-born functions; this pins the rest."""

    import asyncio

    run = asyncio.run(RENDERERS["semantic_kernel"]._run_agent(_handle(), call=None))
    offered = {t["function"]["name"]: t["function"] for t in run["bodies"][0]["tools"]}
    prompt = offered[f"{SK_PREFIX}summarize_records"]

    assert run["origins"][f"{SK_PREFIX}summarize_records"] == "prompt"
    assert prompt["description"] == "SENTINEL_PROMPT_DESCRIPTION"
    # The fixture declares only `status` as required; `tone` is optional.
    assert sorted(prompt["parameters"]["required"]) == ["status", "tone"]


@pytest.mark.skipif(
    "semantic_kernel" not in RENDERERS,
    reason="semantic_kernel renderer not installed",
)
def test_semantic_kernel_model_reads_the_python_repr_of_the_result():
    """FunctionResultContent serializes the returned list with str(), so the tool
    message carries the repr of SK's content objects, and isError is gone at both
    boundaries."""

    renderer = RENDERERS["semantic_kernel"]
    arguments = {"account_id": "a", "confirm": False}

    direct = renderer.render_result(_handle(), "delete_account", arguments, "direct")
    agent = renderer.render_result(_handle(), "delete_account", arguments, "agent")

    assert direct.block_kinds == ["text"]
    assert direct.is_error is None
    assert agent.block_kinds == []
    assert agent.is_error is None
    assert agent.text.startswith("[TextContent(inner_content=TextContent(type='text', text=")
    assert direct.text in agent.text


@pytest.mark.skipif(
    "llamaindex" not in RENDERERS,
    reason="llamaindex renderer not installed",
)
def test_llamaindex_model_reads_the_repr_of_the_call_tool_result():
    """FunctionTool._parse_tool_output has no case for CallToolResult, so the ToolOutput
    and the tool message the FunctionAgent sends back hold one text block: the result's
    Python repr. isError survives only inside that text, never as a flag."""

    renderer = RENDERERS["llamaindex"]
    arguments = {"account_id": "a", "confirm": False}

    direct = renderer.render_result(_handle(), "delete_account", arguments, "direct")
    agent = renderer.render_result(_handle(), "delete_account", arguments, "agent")

    for rep in (direct, agent):
        assert rep.block_kinds == ["text"]
        assert "TextContent(type='text'" in rep.text
    assert direct.is_error is False
    assert agent.is_error is None
    assert agent.text == direct.text


@pytest.mark.skipif("haystack" not in RENDERERS, reason="haystack renderer not installed")
def test_haystack_results_are_call_tool_result_json_and_errors_are_exceptions():
    """What the adapter does with results, pinned.

    MCPClient.call_tool returns CallToolResult.model_dump_json, so the direct Tool
    return keeps every block and structuredContent once parsed, and the Agent hands the
    model that same JSON as one text. isError raises MCPInvocationError, which
    Tool.invoke wraps in ToolInvocationError; the Agent sends its text with error=True.
    """

    renderer = RENDERERS["haystack"]
    handle = _handle()

    direct = renderer.render_result(handle, "search_records", {"query": "x"}, "direct")
    agent = renderer.render_result(handle, "search_records", {"query": "x"}, "agent")
    assert direct.block_kinds == ["text", "image", "audio", "resource", "resource_link"]
    assert agent.block_kinds == ["text"]
    assert '"structuredContent"' in agent.text
    assert "SENTINEL_TEXT_BLOCK" in agent.text

    arguments = {"account_id": "a", "confirm": False}
    failed = renderer.render_result(handle, "delete_account", arguments, "direct")
    assert failed.is_error is True
    assert failed.error_channel == "exception"
    assert failed.text.startswith("ToolInvocationError: ")
    assert "SENTINEL_TOOL_ERROR" in failed.text
    told = renderer.render_result(handle, "delete_account", arguments, "agent")
    assert told.is_error is True
    assert told.block_kinds == ["text"]


@pytest.mark.skipif("pi" not in RENDERERS, reason="pi renderer not installed")
def test_pi_model_content_drops_the_error_flag_the_client_returns():
    """What the adapter does with results, pinned.

    McpClient.callTool returns isError; toLlmContent hands the model the failure text
    alone, with no error or structured field.
    """

    renderer = RENDERERS["pi"]
    handle = _handle()
    arguments = {"account_id": "a", "confirm": False}

    llm = renderer.render_result(handle, "delete_account", arguments, "llm")
    direct = renderer.render_result(handle, "delete_account", arguments, "direct")

    assert llm.is_error is None
    assert isinstance(llm.raw, list) and all(set(block) <= {"type", "text", "data", "mimeType"} for block in llm.raw)
    assert direct.is_error is True


@pytest.mark.skipif("pi" not in RENDERERS, reason="pi renderer not installed")
def test_pi_hands_back_the_tools_list_entry_unchanged():
    """McpClient.listTools returns each Tool as parsed JSON, so annotations survive."""

    rendered = {rep.name: rep for rep in RENDERERS["pi"].render(_handle())}

    assert rendered["delete_account"].annotations.get("destructiveHint") is True


@pytest.mark.skipif("smolagents" not in RENDERERS, reason="smolagents renderer not installed")
def test_smolagents_observation_drops_the_error_flag_and_rejects_omitted_optionals():
    """What the adapter does with results, pinned.

    MCPAdaptTool.forward returns content[0] as a bare string and never reads isError,
    so the agent's observation is an ordinary one. Every input is required, so a call
    that omits an optional parameter fails in smolagents before reaching the server.
    """

    renderer = RENDERERS["smolagents"]
    handle = _handle()

    direct = renderer.render_result(handle, "delete_account", {"account_id": "a", "confirm": False}, "direct")
    agent = renderer.render_result(handle, "delete_account", {"account_id": "a", "confirm": False}, "agent")
    omitted = renderer.render_result(handle, "explain_topic", {"topic": "x"}, "agent")

    assert direct.is_error is None and direct.block_kinds == ["text"]
    assert agent.is_error is None and agent.text == "Observation:\n" + direct.text
    assert omitted.is_error is True and "Argument max_words is required" in omitted.text


@pytest.mark.skipif(
    "crewai_native" not in RENDERERS,
    reason="crewai_native renderer not installed",
)
def test_crewai_native_hashes_away_the_tool_name_behind_a_long_launch_command(monkeypatch, tmp_path):
    """What the adapter does with names, pinned.

    The stdio server name is the command and its args, so an absolute path makes
    f"{server}_{tool}" longer than 64 characters, and sanitize_tool_name truncates it
    and appends a hash. The model then sees none of the MCP tool name. Run from
    outside the checkout, so the renderer keeps the fixture's absolute path. The
    fixture is copied under a long directory so the result does not depend on where
    the repository is checked out.
    """

    long_dir = tmp_path / ("a-path-long-enough-that-crewai-must-hash-the-tool-name-" * 2)
    long_dir.mkdir()
    shutil.copy(FIXTURE, long_dir / FIXTURE.name)
    monkeypatch.chdir(tmp_path)
    reps = RENDERERS["crewai_native"].render(parse_server(shlex.join([sys.executable, str(long_dir / FIXTURE.name)])))

    assert len(reps) == 5
    # Truncated to the 64-character limit with a hash suffix; the exact length depends
    # on where the cut lands in the path, so only the limit and uniqueness are fixed.
    assert all(len(rep.name) <= 64 for rep in reps)
    assert len({rep.name for rep in reps}) == 5
    for tool in ("search_records", "create_report", "deliver_payload", "delete_account", "explain_topic"):
        assert not any(tool in rep.name for rep in reps)


@pytest.mark.skipif(
    "crewai_native" not in RENDERERS,
    reason="crewai_native renderer not installed",
)
def test_crewai_native_model_reads_an_error_result_as_plain_text():
    """What the adapter does with results, pinned.

    MCPNativeTool turns isError into a ToolFailure, so BaseTool.run carries the flag,
    but the native executor renders it as its message alone: the model's tool message
    is indistinguishable from a successful one.
    """

    renderer = RENDERERS["crewai_native"]
    handle = parse_server(shlex.join([sys.executable, str(RESULTS_FIXTURE)]))

    direct = renderer.render_result(handle, "tool_error", {}, "direct")
    agent = renderer.render_result(handle, "tool_error", {}, "agent")
    mixed = renderer.render_result(handle, "mixed_content", {}, "agent")

    assert direct.is_error is True and direct.error_channel == "error_field"
    assert agent.is_error is None and agent.text == direct.text
    assert mixed.block_kinds == ["text"]


@pytest.mark.skipif(
    "google_adk_js" not in RENDERERS,
    reason="google_adk_js renderer not installed",
)
def test_google_adk_js_model_reads_the_whole_call_tool_result_as_one_object():
    """What the adapter does with results, pinned.

    MCPTool.runAsync returns the CallToolResult unchanged, and ADK puts that object in
    the functionResponse the next LlmRequest carries: isError survives as a field, but
    no content block reaches the model as its own part.
    """

    renderer = RENDERERS["google_adk_js"]
    arguments = {"account_id": "a", "confirm": False}

    direct = renderer.render_result(_handle(), "delete_account", arguments, "direct")
    agent = renderer.render_result(_handle(), "delete_account", arguments, "agent")

    assert direct.block_kinds == ["text"]
    assert direct.is_error is True
    assert agent.raw["name"] == "delete_account"
    assert agent.raw["response"] == direct.raw
    assert agent.block_kinds == []
    assert agent.is_error is True


@pytest.mark.skipif(
    "google_adk_js" not in RENDERERS,
    reason="google_adk_js renderer not installed",
)
def test_google_adk_js_raw_keeps_the_gemini_schema_dialect():
    """The comparison reads the genai Schema back as JSON Schema; raw keeps what ADK built."""

    rendered = {rep.name: rep for rep in RENDERERS["google_adk_js"].render(_handle())}

    search = rendered["search_records"]
    assert search.raw["parameters"]["type"] == "OBJECT"
    assert search.raw["parameters"]["properties"]["limit"]["type"] == "INTEGER"
    assert search.params["properties"]["limit"]["type"] == "integer"
    assert rendered["deliver_payload"].raw["parameters"]["properties"]["mode"]["type"] == "TYPE_UNSPECIFIED"
    assert "type" not in rendered["deliver_payload"].params["properties"]["mode"]
    assert "response" in rendered["explain_topic"].raw
    assert "response" not in search.raw


# Agno is managed-only (its manifest says why), so CI's local environment never has it;
# these drive the real adapter through the manifest-pinned runner instead.
_AGNO_MANAGED = pytest.mark.skipif(
    shutil.which("uv") is None, reason="the agno renderer runs only through the managed runner, which needs uv"
)
RESULTS_FIXTURE = FIXTURE.parent / "results_server.py"


def _agno_managed(request: dict) -> dict:
    from mcp_mirror.cli import _managed_environment
    from mcp_mirror.runner_manifests import build_managed_worker_command

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "agno.json"
        proc = subprocess.run(
            build_managed_worker_command("agno", out),
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=600,
            env=_managed_environment(),
        )
        assert out.exists() and out.read_text().strip(), proc.stderr[-2000:]
        payload = json.loads(out.read_text())
    assert payload.get("ok"), payload.get("error")
    return payload


@_AGNO_MANAGED
def test_agno_managed_render_passes_schemas_through_and_drops_tool_metadata():
    handle = _handle()
    source_spec_version, source_reps = load_source(handle)
    payload = _agno_managed({"server": handle.spec, "headers": {}})

    assert payload["versions"] == {"framework": "3.1.2", "adapter": "3.1.2"}
    evidence = payload["evidence"]
    expected = EXPECTED_BOUNDARIES["agno"]
    assert evidence["capture_api"] == expected["capture_api"]
    assert evidence["capture_object"] == expected["capture_object"]
    assert evidence["capture_stage"] == expected["capture_stage"]
    assert evidence["provider_request_captured"] is False
    # protocol_mode="auto" falls back to the fixture's handshake-era revision.
    assert evidence["negotiated_mcp_spec_version"] == source_spec_version

    rendered = {rep["name"]: rep for rep in payload["reps"]}
    source = {rep.name: rep for rep in source_reps}
    assert set(rendered) == set(source)
    for name, rep in rendered.items():
        assert rep["description"] == source[name].description
        assert rep["params"] == source[name].params, name
        assert not rep.get("annotations")
        assert not (rep.get("framework_metadata") or {}).get("annotations")


@_AGNO_MANAGED
def test_agno_managed_results_lose_the_error_flag_and_forward_media_to_the_model():
    server = shlex.join([sys.executable, str(RESULTS_FIXTURE)])
    calls = [{"tool": tool, "arguments": {}} for tool in ("tool_error", "image_content", "text_and_structured")]
    payload = _agno_managed({"server": server, "headers": {}, "calls": calls, "result_boundaries": ["*"]})
    results = {(entry["tool"], entry["boundary"]): entry["rep"] for entry in payload["results"]}

    for boundary in ("direct", "agent"):
        error = results[("tool_error", boundary)]
        assert error["is_error"] is not True
        assert error["text"].startswith("Error from MCP tool 'tool_error'")
        # Model._handle_function_call_media moves the image into a follow-up user message.
        assert "image" in results[("image_content", boundary)]["block_kinds"]
    assert results[("text_and_structured", "direct")]["structured_content"]
    assert results[("text_and_structured", "agent")]["structured_content"] is None


# DSPy is managed-only too (its manifest says why).
def _dspy_managed(request: dict) -> dict:
    from mcp_mirror.cli import _managed_environment
    from mcp_mirror.runner_manifests import build_managed_worker_command

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "dspy.json"
        proc = subprocess.run(
            build_managed_worker_command("dspy", out),
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=600,
            env=_managed_environment(),
        )
        assert out.exists() and out.read_text().strip(), proc.stderr[-2000:]
        payload = json.loads(out.read_text())
    assert payload.get("ok"), payload.get("error")
    return payload


@_AGNO_MANAGED
def test_dspy_managed_render_marks_every_parameter_without_a_default_required():
    handle = _handle()
    source_spec_version, source_reps = load_source(handle)
    payload = _dspy_managed({"server": handle.spec, "headers": {}})

    assert payload["versions"] == {"framework": "3.4.0", "adapter": "3.4.0"}
    evidence = payload["evidence"]
    expected = EXPECTED_BOUNDARIES["dspy"]
    assert evidence["capture_api"] == expected["capture_api"]
    assert evidence["capture_object"] == expected["capture_object"]
    assert evidence["capture_stage"] == expected["capture_stage"]
    assert evidence["provider_request_captured"] is False
    assert evidence["negotiated_mcp_spec_version"] == source_spec_version

    rendered = {rep["name"]: rep for rep in payload["reps"]}
    source = {rep.name: rep for rep in source_reps}
    assert set(rendered) == set(source)
    for name, rep in rendered.items():
        assert rep["description"] == source[name].description
        # format_as_litellm_function_call keeps the properties and recomputes required
        # as every property without a default.
        assert rep["params"]["properties"] == source[name].params["properties"], name
        properties = rep["params"]["properties"]
        assert set(rep["params"]["required"]) == {k for k, v in properties.items() if "default" not in v}, name
        assert not rep.get("annotations")
    assert "mode" in rendered["deliver_payload"]["params"]["required"]
    assert "mode" not in (source["deliver_payload"].params.get("required") or [])


@_AGNO_MANAGED
def test_dspy_managed_results_raise_on_error_and_drop_media_next_to_text():
    server = shlex.join([sys.executable, str(RESULTS_FIXTURE)])
    calls = [{"tool": tool, "arguments": {}} for tool in ("tool_error", "mixed_content", "text_and_structured")]
    payload = _dspy_managed({"server": server, "headers": {}, "calls": calls, "result_boundaries": ["*"]})
    results = {(entry["tool"], entry["boundary"]): entry["rep"] for entry in payload["results"]}

    # _convert_mcp_tool_result raises for isError; ReAct turns the raise into text.
    assert results[("tool_error", "direct")]["is_error"] is True
    assert results[("tool_error", "react")]["is_error"] is not True
    assert results[("tool_error", "react")]["text"].startswith("Execution error in tool_error")
    for boundary in ("direct", "react"):
        assert results[("mixed_content", boundary)]["block_kinds"] == ["text"]
        assert results[("text_and_structured", boundary)]["structured_content"] is None
    assert results[("text_and_structured", "structured")]["structured_content"]


@pytest.mark.skipif(
    "spring_ai" not in RENDERERS,
    reason="spring_ai renderer not installed",
)
def test_spring_ai_errors_are_flagless_text_for_the_model_and_raise_when_direct():
    """What the adapter does with results, pinned. The model reads an isError
    result as ordinary text holding a Java record's toString; a direct
    ToolCallback.call throws ToolExecutionException instead."""

    renderer = RENDERERS["spring_ai"]
    arguments = {"account_id": "a", "confirm": False}

    direct = renderer.render_result(_handle(), "delete_account", arguments, "direct")
    agent = renderer.render_result(_handle(), "delete_account", arguments, "agent")

    assert agent.is_error is None
    assert agent.error_channel is None
    assert agent.text.startswith("Error calling tool: [TextContent[annotations=null, text=")
    assert direct.is_error is True
    assert direct.error_channel == "exception"
    assert direct.text.startswith("ToolExecutionException: Error calling tool:")


# BeeAI is managed-only too (its manifest says why).
def _beeai_managed(request: dict) -> dict:
    from mcp_mirror.cli import _managed_environment
    from mcp_mirror.runner_manifests import build_managed_worker_command

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "beeai.json"
        proc = subprocess.run(
            build_managed_worker_command("beeai", out),
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=600,
            env=_managed_environment(),
        )
        assert out.exists() and out.read_text().strip(), proc.stderr[-2000:]
        payload = json.loads(out.read_text())
    assert payload.get("ok"), payload.get("error")
    return payload


@_AGNO_MANAGED
def test_beeai_managed_render_sends_strict_schemas_and_drops_annotations():
    handle = _handle()
    source_spec_version, source_reps = load_source(handle)
    payload = _beeai_managed({"server": handle.spec, "headers": {}})

    assert payload["versions"] == {"framework": "0.1.85", "adapter": "0.1.85"}
    evidence = payload["evidence"]
    expected = EXPECTED_BOUNDARIES["beeai"]
    assert evidence["capture_api"] == expected["capture_api"]
    assert evidence["capture_object"] == expected["capture_object"]
    assert evidence["capture_stage"] == expected["capture_stage"]
    assert evidence["provider_request_captured"] is False
    assert evidence["negotiated_mcp_spec_version"] == "2025-11-25"

    rendered = {rep["name"]: rep for rep in payload["reps"]}
    source = {rep.name: rep for rep in source_reps}
    # RequirementAgent's own final_answer tool is not an MCP tool.
    assert set(rendered) == set(source)
    for name, rep in rendered.items():
        assert rep["description"] == source[name].description
        # The strict schema marks every property required and closes the object.
        assert set(rep["params"]["required"]) == set(rep["params"]["properties"]), name
        assert rep["params"]["additionalProperties"] is False, name
        assert not rep.get("annotations")


@_AGNO_MANAGED
def test_beeai_managed_results_keep_structured_content_alone_and_raise_on_error():
    server = shlex.join([sys.executable, str(RESULTS_FIXTURE)])
    calls = [{"tool": tool, "arguments": {}} for tool in ("tool_error", "mixed_content", "image_content")]
    payload = _beeai_managed({"server": server, "headers": {}, "calls": calls, "result_boundaries": ["*"]})
    results = {(entry["tool"], entry["boundary"]): entry["rep"] for entry in payload["results"]}

    # MCPTool._run raises ToolError for isError; RequirementAgent writes the
    # tool_error template into the tool message.
    assert results[("tool_error", "direct")]["error_channel"] == "exception"
    assert results[("tool_error", "agent")]["text"].startswith("The tool has failed;")
    # structuredContent replaces the content blocks entirely.
    assert results[("mixed_content", "direct")]["structured_content"]
    assert results[("mixed_content", "direct")]["block_kinds"] == []
    # The model sees every result as one JSON string.
    assert results[("image_content", "direct")]["block_kinds"] == ["image"]
    assert results[("image_content", "agent")]["block_kinds"] == ["text"]


@pytest.mark.skipif(
    "google_adk_java" not in RENDERERS,
    reason="google_adk_java renderer not installed",
)
def test_google_adk_java_wraps_text_and_flags_errors_with_an_error_key():
    """What the adapter does with results, pinned. AbstractMcpTool
    wraps text as {"text_output": [...]} and turns an isError result into an
    {"error": ...} map, the same at both boundaries."""

    renderer = RENDERERS["google_adk_java"]
    server = parse_server(shlex.join([sys.executable, str(RESULTS_FIXTURE)]))

    for boundary in ("direct", "agent"):
        text = renderer.render_result(server, "text_content", {}, boundary)
        assert json.loads(text.text) == {"text_output": [{"text": "SENTINEL_TEXT_BLOCK"}]}
        assert text.is_error is None

        error = renderer.render_result(server, "tool_error", {}, boundary)
        assert error.is_error is True
        assert error.error_channel == "error_field"
        assert json.loads(error.text)["error"].startswith("Tool execution failed. Details: SENTINEL_TOOL_ERROR")


@pytest.mark.skipif(
    "langchain4j" not in RENDERERS,
    reason="langchain4j renderer not installed",
)
def test_langchain4j_errors_are_flagged_for_the_model_and_raise_when_direct():
    """What the adapter does with errors, pinned. The model reads an isError
    result flagged as an error with the server's text intact; a direct
    McpToolExecutor call throws McpApplicationErrorException instead (since
    1.22.0-beta32; earlier releases threw ToolExecutionException)."""

    renderer = RENDERERS["langchain4j"]
    arguments = {"account_id": "a", "confirm": False}

    direct = renderer.render_result(_handle(), "delete_account", arguments, "direct")
    agent = renderer.render_result(_handle(), "delete_account", arguments, "agent")

    assert agent.is_error is True
    assert agent.error_channel == "flag"
    assert direct.is_error is True
    assert direct.error_channel == "exception"
    assert direct.text.startswith("McpApplicationErrorException: ")
    assert direct.text.removeprefix("McpApplicationErrorException: ") == agent.text


@pytest.mark.skipif(
    "ms_agent_framework_dotnet" not in RENDERERS,
    reason="ms_agent_framework_dotnet renderer not installed",
)
def test_ms_agent_framework_dotnet_falls_back_to_call_tool_result_json():
    """What the adapter does with results, pinned.
    McpClientTool maps a lone image to DataContent, but returns a resource_link result as
    the CallToolResult's JSON, the same at both boundaries."""

    renderer = RENDERERS["ms_agent_framework_dotnet"]
    server = parse_server(shlex.join([sys.executable, str(RESULTS_FIXTURE)]))

    for boundary in ("direct", "agent"):
        image = renderer.render_result(server, "image_content", {}, boundary)
        assert image.block_kinds == ["image"]

        link = renderer.render_result(server, "resource_link_content", {}, boundary)
        assert link.block_kinds == []
        assert json.loads(link.text)["content"][0]["type"] == "resource_link"


@pytest.mark.skipif(
    "google_adk_go" not in RENDERERS,
    reason="google_adk_go renderer not installed",
)
def test_google_adk_go_flattens_content_to_one_output_string_and_errors_to_go_errors():
    """What the adapter does with results, pinned. mcpTool.Run returns
    {"output": text} with every block flattened to text, {"output": structuredContent}
    when there is structured content, and a Go error for an isError result, which ADK
    hands the model as {"error": ...}."""

    renderer = RENDERERS["google_adk_go"]
    server = parse_server(shlex.join([sys.executable, str(RESULTS_FIXTURE)]))

    direct = renderer.render_result(server, "text_content", {}, "direct")
    assert direct.text == "SENTINEL_TEXT_BLOCK"
    agent = renderer.render_result(server, "text_content", {}, "agent")
    assert json.loads(agent.text) == {"output": "SENTINEL_TEXT_BLOCK"}

    image = renderer.render_result(server, "image_content", {}, "agent")
    assert image.block_kinds == []
    assert json.loads(image.text)["output"].startswith("[MCP image: ")

    structured = renderer.render_result(server, "text_and_structured", {}, "direct")
    assert structured.structured_content
    assert "SENTINEL_TEXT_BLOCK" not in structured.text

    error = renderer.render_result(server, "tool_error", {}, "direct")
    assert error.is_error is True
    assert error.error_channel == "exception"
    assert error.text.startswith("Tool execution failed. Details: SENTINEL_TOOL_ERROR")
    error = renderer.render_result(server, "tool_error", {}, "agent")
    assert error.error_channel == "error_field"
    assert json.loads(error.text)["error"].startswith("Tool execution failed. Details: SENTINEL_TOOL_ERROR")

