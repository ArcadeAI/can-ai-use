"""The read-only mcp-mirror MCP server over stdio. The queries live in catalog.py."""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from .catalog import (
    SUPPORT_LABELS,
    _resolve_snapshot,
    capability_data,
    default_revision,
    inspect_tool_definition_data,
    inspect_tool_result_data,
    list_snapshots_data,
    load_snapshots,
    search_capabilities_data,
)


def create_server() -> FastMCP:
    server = FastMCP(
        "mcp-mirror",
        instructions=(
            "Use mcp-mirror while designing MCP tools and results. It reports "
            "versioned observations at explicitly named framework capture boundaries; "
            "it does not rank frameworks or predict model behavior."
        ),
        log_level="ERROR",
    )

    @server.tool()
    def list_snapshots() -> dict[str, Any]:
        """List MCP revisions, framework versions, and measurement coverage."""

        return list_snapshots_data()

    @server.tool()
    def search_capabilities(query: str, revision: str = "default") -> dict[str, Any]:
        """Search the published MCP and JSON Schema capability catalog."""

        return search_capabilities_data(query, revision)

    @server.tool()
    def lookup_capability(
        capability_id: str,
        revision: str = "default",
        frameworks: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return versioned framework observations and evidence for one capability."""

        return capability_data(capability_id, revision, frameworks)

    @server.tool()
    def inspect_tool_definition(
        tool: dict[str, Any],
        revision: str = "default",
        frameworks: list[str] | None = None,
    ) -> dict[str, Any]:
        """Identify features used by an MCP tool definition and return known adapter observations."""

        return inspect_tool_definition_data(tool, revision, frameworks)

    @server.tool()
    def inspect_tool_result(
        result: dict[str, Any],
        revision: str = "default",
        frameworks: list[str] | None = None,
    ) -> dict[str, Any]:
        """Identify features used by an MCP CallToolResult and return known adapter observations."""

        return inspect_tool_result_data(result, revision, frameworks)

    @server.resource("mcp-mirror://snapshots")
    def snapshots_resource() -> str:
        """Machine-readable snapshot index."""

        return json.dumps(list_snapshots_data(), indent=2)

    @server.resource("mcp-mirror://capabilities/{revision}")
    def capabilities_resource(revision: str) -> str:
        """Machine-readable capability index for one MCP revision."""

        selected_revision, snapshot = _resolve_snapshot(revision)
        rows = [
            {
                "id": capability["id"],
                "title": capability.get("title"),
                "question": capability.get("question"),
                "categories": capability.get("categories", []),
            }
            for capability in snapshot.get("data", {}).values()
        ]
        return json.dumps({"revision": selected_revision, "capabilities": rows}, indent=2)

    return server


def run_server() -> None:
    """Run the read-only mcp-mirror server over stdio."""

    create_server().run(transport="stdio")


__all__ = [
    "capability_data",
    "create_server",
    "inspect_tool_definition_data",
    "inspect_tool_result_data",
    "list_snapshots_data",
    "load_snapshots",
    "run_server",
    "search_capabilities_data",
]
