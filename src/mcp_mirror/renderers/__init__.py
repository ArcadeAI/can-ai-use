"""Renderer protocol + registry of installed renderers (DESIGN.md section 8).

A renderer turns a live MCP server into the tool definitions exposed at one precise
framework boundary. The cardinal rule (decision D1) is to always drive the
framework's *real* adapter and never reimplement it. Each renderer declares whether
that boundary is a framework object, a provider-shaped format, or a captured request.

Renderers live behind optional extras. Importing a renderer module must not import
its heavy framework at module load time; availability is checked with
``importlib.util.find_spec`` so ``mcp-mirror frameworks`` works with nothing installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Protocol, runtime_checkable

from ..models import RendererEvidence, ToolRep
from ..runner_manifests import load_runner_manifests
from ..source import ServerHandle


@runtime_checkable
class Renderer(Protocol):
    id: str

    def versions(self) -> dict: ...

    def evidence(self) -> RendererEvidence: ...

    def render(self, server: ServerHandle) -> list[ToolRep]: ...


class RenderError(RuntimeError):
    """Raised when a framework adapter fails to render the server's tools."""


# The manifest registry is the single source of truth for runner IDs and modules.
_RENDERER_MODULES: dict[str, str] = {
    renderer_id: manifest.module
    for renderer_id, manifest in load_runner_manifests().items()
}

# Friendly aliases users might pass on the CLI.
_ALIASES = {
    "pydantic-ai": "pydantic_ai",
    "pydanticai": "pydantic_ai",
    "lc": "langchain",
    "openai": "openai_agents",
    "openai-agents": "openai_agents",
    "openai_agents_sdk": "openai_agents",
    "agents": "openai_agents",
    "google-adk": "google_adk",
    "adk": "google_adk",
    "vercel-ai": "vercel_ai",
    "ai-sdk": "vercel_ai",
    "ms-agent-framework": "ms_agent_framework",
    "agent-framework": "ms_agent_framework",
    "agent_framework": "ms_agent_framework",
    "maf": "ms_agent_framework",
    "autogen-ext": "autogen",
    "autogen_ext": "autogen",
    "strands-agents": "strands",
    "strands_agents": "strands",
    "openai-agents-js": "openai_agents_js",
    "openai_agents_ts": "openai_agents_js",
    "pi-mcp": "pi",
    "pi_mcp": "pi",
    "semantic-kernel": "semantic_kernel",
    "sk": "semantic_kernel",
    "llama-index": "llamaindex",
    "llama_index": "llamaindex",
    "google-adk-js": "google_adk_js",
    "google_adk_ts": "google_adk_js",
    "adk-js": "google_adk_js",
    "agno-agi": "agno",
    "mcp-haystack": "haystack",
    "haystack-ai": "haystack",
    "smolagent": "smolagents",
    "smol": "smolagents",
    "dspy-ai": "dspy",
    "spring-ai": "spring_ai",
    "springai": "spring_ai",
    "langchain-js": "langchain_js",
    "langchainjs": "langchain_js",
    "genkit": "genkit_js",
    "genkit-js": "genkit_js",
    "genkit-ai-mcp": "genkit_js",
    "beeai-framework": "beeai",
    "beeai_framework": "beeai",
    "bee": "beeai",
    "copilot-sdk": "copilot_sdk",
    "github-copilot-sdk": "copilot_sdk",
    "crewai-native": "crewai_native",
    "crewai_mcp": "crewai_native",
    "google-adk-java": "google_adk_java",
    "adk-java": "google_adk_java",
    "langchain-4j": "langchain4j",
    "lc4j": "langchain4j",
    "ms-agent-framework-dotnet": "ms_agent_framework_dotnet",
    "agent-framework-dotnet": "ms_agent_framework_dotnet",
    "maf-dotnet": "ms_agent_framework_dotnet",
    "google-adk-go": "google_adk_go",
    "adk-go": "google_adk_go",
    "koog-mcp": "koog",
    "jetbrains-koog": "koog",
}

_EXTRAS = {
    "langchain": "langchain",
    "pydantic_ai": "pydantic-ai",
    "crewai": "crewai",
    "crewai_native": "crewai-native",
    "openai_agents": "openai-agents",
    "google_adk": "google-adk",
    "ms_agent_framework": "ms-agent-framework",
    "autogen": "autogen",
    "strands": "strands",
    "semantic_kernel": "semantic-kernel",
    "llamaindex": "llamaindex",
    "haystack": "haystack",
    "smolagents": "smolagents",
}


def canonical_id(name: str) -> str:
    name = name.strip().lower()
    return _ALIASES.get(name, name)


def all_renderer_ids() -> list[str]:
    return list(_RENDERER_MODULES)


def local_renderer_ids() -> list[str]:
    """Registered renderers expected to install into the project's own environment."""

    manifests = load_runner_manifests()
    return [rid for rid in _RENDERER_MODULES if not manifests[rid].managed_only]


def install_hint(renderer_id: str) -> str:
    """Return the shortest supported setup path for one renderer."""

    renderer_id = canonical_id(renderer_id)
    manifest = load_runner_manifests().get(renderer_id)
    if manifest is not None and manifest.managed_only:
        return f"run it with --runner managed ({manifest.managed_only_reason})"
    extra = _EXTRAS.get(renderer_id)
    if extra:
        return f"python -m pip install 'mcp-mirror[{extra}]'"
    if renderer_id == "mastra":
        return (
            "set up the bundled Mastra Node runner from a source checkout "
            "(src/mcp_mirror/renderers/mastra_node)"
        )
    if renderer_id == "vercel_ai":
        return "npm ci --prefix src/mcp_mirror/renderers/vercel_ai_node (from a source checkout)"
    if renderer_id == "openai_agents_js":
        return (
            "set up the bundled OpenAI Agents SDK (JS) Node runner from a source checkout "
            "(src/mcp_mirror/renderers/openai_agents_js_node)"
        )
    if renderer_id == "pi":
        return "npm ci --prefix src/mcp_mirror/renderers/pi_node (from a source checkout)"
    if renderer_id == "genkit_js":
        return "npm ci --prefix src/mcp_mirror/renderers/genkit_js_node (from a source checkout)"
    if renderer_id == "google_adk_js":
        return (
            "set up the bundled Google ADK (TypeScript) Node runner from a source checkout "
            "(src/mcp_mirror/renderers/google_adk_js_node)"
        )
    if renderer_id == "langchain_js":
        return "npm ci --prefix src/mcp_mirror/renderers/langchain_js_node (from a source checkout)"
    if renderer_id == "spring_ai":
        return (
            "build the bundled Spring AI Java worker from a source checkout with a JDK 21 "
            "(mvn -f src/mcp_mirror/renderers/spring_ai_java/pom.xml package)"
        )
    if renderer_id == "google_adk_java":
        return (
            "build the bundled Google ADK (Java) worker from a source checkout with a JDK 21 "
            "(mvn -f src/mcp_mirror/renderers/google_adk_java/pom.xml package)"
        )
    if renderer_id == "langchain4j":
        return (
            "build the bundled LangChain4j Java worker from a source checkout with a JDK 21 "
            "(mvn -f src/mcp_mirror/renderers/langchain4j_java/pom.xml package)"
        )
    if renderer_id == "ms_agent_framework_dotnet":
        return (
            "publish the bundled Microsoft Agent Framework (.NET) worker from a source checkout "
            "with a .NET 10 SDK (dotnet publish src/mcp_mirror/renderers/ms_agent_framework_dotnet -c Release)"
        )
    if renderer_id == "google_adk_go":
        return (
            "build the bundled Google ADK (Go) worker from a source checkout with Go 1.27 "
            "(go -C src/mcp_mirror/renderers/google_adk_go build -o worker .)"
        )
    if renderer_id == "koog":
        return (
            "build the bundled Koog worker from a source checkout with a JDK 21 "
            "(mvn -f src/mcp_mirror/renderers/koog_kotlin/pom.xml package)"
        )
    if renderer_id == "copilot_sdk":
        return "npm ci --prefix src/mcp_mirror/renderers/copilot_sdk_node (from a source checkout)"
    return f"unknown renderer {renderer_id!r}"


def available_renderers() -> dict[str, Renderer]:
    """Return ``{id: Renderer}`` for every framework that is actually installed."""

    found: dict[str, Renderer] = {}
    for rid, module_suffix in _RENDERER_MODULES.items():
        try:
            module = import_module(module_suffix, __name__)
        except Exception:
            continue
        try:
            if module.available():
                found[rid] = module.get_renderer()
        except Exception:
            continue
    return found


def select_renderers(requested: list[str] | None) -> tuple[dict[str, Renderer], list[str]]:
    """Resolve a requested id list against installed renderers.

    Returns ``(selected, missing)`` where ``missing`` are requested ids that are not
    installed. When ``requested`` is ``None`` every installed renderer is selected.
    """

    available = available_renderers()
    if requested is None:
        return available, []

    selected: dict[str, Renderer] = {}
    missing: list[str] = []
    for raw in requested:
        rid = canonical_id(raw)
        if rid in available:
            selected[rid] = available[rid]
        else:
            missing.append(rid)
    return selected, missing
