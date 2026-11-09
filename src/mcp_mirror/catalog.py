"""The published support data as queries: what the MCP servers (local and Arcade-hosted) answer.

Standard library only, so the Arcade-hosted server (deploy/arcade) can vendor this file
unchanged. Data comes from MCP_MIRROR_DATA_URL (the published site, fetched and cached),
MCP_MIRROR_DATA (a directory), or the data built in this repository.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


SUPPORT_LABELS = {
    "y": "preserved at the declared capture boundary",
    "a": "changed or retained at another boundary",
    "n": "not exposed at the declared capture boundary",
    "x": "adapter could not negotiate this protocol revision",
    "u": "not measured",
}

def _data_directories() -> list[Path]:
    directories: list[Path] = []
    if configured := os.environ.get("MCP_MIRROR_DATA"):
        directories.append(Path(configured).expanduser())
    directories.extend(
        [
            Path(__file__).resolve().parents[2] / "data" / "fulldata",
            Path(__file__).resolve().parent / "data",
        ]
    )
    return directories


CACHE_SECONDS = 600
_cache: dict[str, Any] = {"at": 0.0, "snapshots": None, "default": None}


def _fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "canaiuse-mcp"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read())


def _published_only(payload: dict[str, Any]) -> dict[str, Any]:
    # Preview rows are built so the site can show them on request; the server has no
    # such request, so it answers for published rows only.
    payload["agents"] = {
        agent_id: agent
        for agent_id, agent in (payload.get("agents") or {}).items()
        if agent.get("visibility", "published") != "preview"
    }
    return payload


def _load_from_url(base: str) -> tuple[dict[str, dict[str, Any]], str | None]:
    base = base.rstrip("/")
    index = _fetch_json(f"{base}/specs.json")
    snapshots = {}
    for entry in index.get("specs", []):
        payload = _published_only(_fetch_json(f"{base}/{entry['file']}"))
        revision = payload.get("mcp_spec")
        if isinstance(revision, str) and revision:
            snapshots[revision] = payload
    return snapshots, index.get("default_spec")


def _load_from_directories() -> tuple[dict[str, dict[str, Any]], str | None]:
    snapshots: dict[str, dict[str, Any]] = {}
    default = None
    for directory in _data_directories():
        if not directory.is_dir():
            continue
        if default is None and (directory / "specs.json").is_file():
            try:
                default = json.loads((directory / "specs.json").read_text()).get("default_spec")
            except (OSError, json.JSONDecodeError):
                pass
        for path in sorted(directory.glob("data-*.json")):
            if path.name == "data-1.0.json":
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            revision = payload.get("mcp_spec")
            if isinstance(revision, str) and revision:
                snapshots.setdefault(revision, _published_only(payload))
    return snapshots, default


def _load() -> tuple[dict[str, dict[str, Any]], str | None]:
    now = time.monotonic()
    if _cache["snapshots"] is not None and now - _cache["at"] < CACHE_SECONDS:
        return _cache["snapshots"], _cache["default"]
    url = os.environ.get("MCP_MIRROR_DATA_URL")
    snapshots, default = _load_from_url(url) if url else _load_from_directories()
    if not snapshots:
        where = url or ", ".join(str(path) for path in _data_directories())
        raise RuntimeError(f"no mcp-mirror snapshots found; searched: {where}")
    _cache.update(at=now, snapshots=snapshots, default=default)
    return snapshots, default


def load_snapshots() -> dict[str, dict[str, Any]]:
    """Every published compatibility snapshot, keyed by MCP revision (cached for ten minutes)."""

    return _load()[0]


def default_revision() -> str:
    """The revision the site opens on (specs.json default_spec), else the newest."""

    snapshots, default = _load()
    return default if default in snapshots else sorted(snapshots)[-1]


def _resolve_snapshot(revision: str | None) -> tuple[str, dict[str, Any]]:
    snapshots = load_snapshots()
    if revision in (None, "", "default"):
        selected = default_revision()
        return selected, snapshots[selected]
    if revision == "latest":
        selected = sorted(snapshots)[-1]
        return selected, snapshots[selected]
    if revision not in snapshots:
        available = ", ".join(sorted(snapshots))
        raise ValueError(f"unknown MCP revision {revision!r}; available: {available}")
    return revision, snapshots[revision]


def list_snapshots_data() -> dict[str, Any]:
    rows = []
    for revision, snapshot in sorted(load_snapshots().items(), reverse=True):
        coverage = snapshot.get("coverage", {})
        rows.append(
            {
                "revision": revision,
                "data_updated": snapshot.get("updated"),
                "measured_on": snapshot.get("measured_on"),
                "measured_features": coverage.get("measured", coverage.get("measured_features")),
                "total_features": coverage.get("total", coverage.get("total_features")),
                "frameworks": [
                    {
                        "id": framework_id,
                        "name": framework.get("name", framework_id),
                        "version": framework.get("current_version"),
                        "version_released": _release_date(framework),
                        "measured_on": framework.get("measured_on"),
                    }
                    for framework_id, framework in snapshot.get("agents", {}).items()
                ],
            }
        )
    return {"snapshots": rows}


def _release_date(framework: dict[str, Any]) -> str | None:
    """The release date of the measured version (era 0 in the version list)."""
    current = framework.get("current_version")
    for entry in framework.get("version_list") or []:
        if entry.get("version") == current:
            return (entry.get("release_date") or "")[:10] or None
    return None


def _direct_result_boundary(snapshot: dict[str, Any], framework_id: str) -> dict[str, str]:
    """Where the published tool-result cell was read: the row's direct (non-agent) boundary
    from the data, following an alias to the row it shares an adapter with."""
    framework = snapshot["agents"][framework_id]
    owner = snapshot["agents"].get(framework.get("aliases_of") or framework_id, framework)
    boundaries = owner.get("result_boundaries") or []
    direct = next((b for b in boundaries if not b.get("agent_path")), boundaries[0] if boundaries else {})
    return {"capture_api": direct.get("capture_api", "unknown"),
            "capture_object": direct.get("capture_object", "unknown")}


def _cell_detail(
    capability: dict[str, Any],
    snapshot: dict[str, Any],
    framework_id: str,
) -> dict[str, Any]:
    framework = snapshot["agents"][framework_id]
    version = framework.get("current_version")
    raw = capability.get("stats", {}).get(framework_id, {}).get(version, "u")
    code = str(raw).split()[0]
    note = None
    match = re.search(r"#(\d+)", str(raw))
    if match:
        note = capability.get("notes_by_num", {}).get(match.group(1))
    if "Tool results" in capability.get("categories", []):
        result_boundary = _direct_result_boundary(snapshot, framework_id)
        capture_boundary = {
            **result_boundary,
            "capture_stage": "framework_tool_result",
            "provider_request_captured": False,
        }
    else:
        capture_boundary = framework.get("capture_boundary")
    return {
        "framework_id": framework_id,
        "framework": framework.get("name", framework_id),
        "version": version,
        "code": code,
        "meaning": SUPPORT_LABELS.get(code, "unknown support code"),
        "mechanism": note,
        "capture_boundary": capture_boundary,
        # The day this framework's cell was captured; falls back to the day the
        # framework was last measured when the row itself carries no date for it.
        "measured_on": (
            (capability.get("measured") or {}).get("measured_on", {}).get(framework_id)
            or framework.get("measured_on")
        ),
    }


def capability_data(
    capability_id: str,
    revision: str | None = None,
    frameworks: list[str] | None = None,
) -> dict[str, Any]:
    selected_revision, snapshot = _resolve_snapshot(revision)
    capabilities = snapshot.get("data", {})
    if capability_id not in capabilities:
        matches = [
            item["id"]
            for item in _search_rows(snapshot, capability_id)
        ][:8]
        suffix = f"; close matches: {', '.join(matches)}" if matches else ""
        raise ValueError(f"unknown capability {capability_id!r}{suffix}")
    capability = capabilities[capability_id]
    selected_frameworks = frameworks or list(snapshot.get("agents", {}))
    unknown = [item for item in selected_frameworks if item not in snapshot.get("agents", {})]
    if unknown:
        raise ValueError(f"unknown framework id(s): {', '.join(unknown)}")
    return {
        "revision": selected_revision,
        "id": capability["id"],
        "title": capability.get("title"),
        "question": capability.get("question"),
        "description": capability.get("description"),
        "observations": [
            _cell_detail(capability, snapshot, framework_id)
            for framework_id in selected_frameworks
        ],
        "summary": capability.get("notes"),
        "limitations": capability.get("known_issues", []),
        "source": capability.get("docs_url") or capability.get("spec"),
        "reproduce": capability.get("reproduce"),
        "measured": capability.get("measured"),
    }


def _search_rows(snapshot: dict[str, Any], query: str) -> list[dict[str, Any]]:
    terms = [term.casefold() for term in query.split() if term]
    scored: list[tuple[int, dict[str, Any]]] = []
    for capability in snapshot.get("data", {}).values():
        haystack = " ".join(
            str(capability.get(key, ""))
            for key in ("id", "title", "description", "question", "keywords")
        ).casefold()
        if not terms or not all(term in haystack for term in terms):
            continue
        score = sum(
            3 if term in str(capability.get("id", "")).casefold() else 1
            for term in terms
        )
        scored.append(
            (
                score,
                {
                    "id": capability["id"],
                    "title": capability.get("title"),
                    "question": capability.get("question"),
                    "categories": capability.get("categories", []),
                    "measured": capability.get("measured"),
                },
            )
        )
    return [row for _, row in sorted(scored, key=lambda item: (-item[0], item[1]["id"]))]


def search_capabilities_data(query: str, revision: str | None = None) -> dict[str, Any]:
    selected_revision, snapshot = _resolve_snapshot(revision)
    return {
        "revision": selected_revision,
        "query": query,
        "matches": _search_rows(snapshot, query)[:20],
    }


def _schema_capabilities(schema: Any) -> set[str]:
    found: set[str] = set()
    if not isinstance(schema, dict):
        return found
    if "required" in schema:
        found.add("required")
    if "enum" in schema:
        found.add("enum")
    if "format" in schema:
        found.add("format")
    if "default" in schema:
        found.add("default-value")
    if "additionalProperties" in schema:
        found.add("additional-properties")
    if "oneOf" in schema or "anyOf" in schema:
        found.add("one-of-any-of")
    # A constraint inside a branch is still a constraint the tool declares.
    for key in ("oneOf", "anyOf"):
        if isinstance(schema.get(key), list):
            for branch in schema[key]:
                found.update(_schema_capabilities(branch))
    if any(key in schema for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")):
        found.add("numeric-range")
    schema_type = schema.get("type")
    if schema_type == "object" and isinstance(schema.get("properties"), dict):
        if any(isinstance(value, dict) and value.get("description") for value in schema["properties"].values()):
            found.add("property-descriptions")
        if any(isinstance(value, dict) and value.get("type") == "object" for value in schema["properties"].values()):
            found.add("nested-objects")
        for child in schema["properties"].values():
            found.update(_schema_capabilities(child))
    if schema_type == "array":
        if isinstance(schema.get("items"), dict) and schema["items"].get("type") == "object":
            found.add("arrays-of-objects")
        found.update(_schema_capabilities(schema.get("items")))
    if isinstance(schema_type, list) and "null" in schema_type:
        found.add("null-acceptance")
    return found


def inspect_tool_definition_data(
    tool: dict[str, Any],
    revision: str | None = None,
    frameworks: list[str] | None = None,
) -> dict[str, Any]:
    identified = {"tool-name"}
    description = tool.get("description")
    if isinstance(description, str) and description:
        identified.add("description-fidelity")
        if len(description) >= 512:
            identified.add("long-descriptions")
    annotations = tool.get("annotations")
    if isinstance(annotations, dict):
        mapping = {
            "title": "title-annotation",
            "readOnlyHint": "read-only-hint",
            "destructiveHint": "destructive-hint",
            "idempotentHint": "idempotent-hint",
            "openWorldHint": "open-world-hint",
        }
        identified.update(mapping[key] for key in mapping.keys() & annotations.keys())
    if tool.get("icons"):
        identified.add("tool-icons")
    identified.update(_schema_capabilities(tool.get("inputSchema")))
    selected_revision, snapshot = _resolve_snapshot(revision)
    published = set(snapshot.get("data", {}))
    capabilities = [
        capability_data(capability_id, selected_revision, frameworks)
        for capability_id in sorted(identified & published)
    ]
    return {
        "revision": selected_revision,
        "tool": tool.get("name"),
        "identified_capabilities": [item["id"] for item in capabilities],
        "compatibility": capabilities,
        "note": "Results describe published capture boundaries, not overall framework quality.",
    }


def inspect_tool_result_data(
    result: dict[str, Any],
    revision: str | None = None,
    frameworks: list[str] | None = None,
) -> dict[str, Any]:
    identified = {"tools-call"}
    if result.get("isError") is True:
        identified.add("tool-execution-errors")
    if "structuredContent" in result:
        identified.add("structured-content")
    for block in result.get("content", []) if isinstance(result.get("content"), list) else []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        mapping = {
            "text": "tool-text-content",
            "image": "tool-image-content",
            "audio": "tool-audio-content",
            "resource": "embedded-resource-content",
            "resource_link": "resource-link-content",
        }
        if block_type in mapping:
            identified.add(mapping[block_type])
        if block.get("annotations"):
            identified.add("content-annotations")
    selected_revision, snapshot = _resolve_snapshot(revision)
    published = set(snapshot.get("data", {}))
    capabilities = [
        capability_data(capability_id, selected_revision, frameworks)
        for capability_id in sorted(identified & published)
    ]
    return {
        "revision": selected_revision,
        "identified_capabilities": [item["id"] for item in capabilities],
        "compatibility": capabilities,
        "note": "Results describe published capture boundaries, not model behavior.",
    }
