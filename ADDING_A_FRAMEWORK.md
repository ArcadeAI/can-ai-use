# Adding an agent framework

`mcp-mirror` is designed so a framework contribution plugs into the existing
source loader, differ, Jobs-To-Be-Done scorecard, reporters, and regression
harness. The contributor still has to prove one framework-specific fact: which
tool definition the framework's real adapter presents at the model boundary.

Once that renderer is registered, the shared engine provides the rest:

- stdio and streamable-HTTP server configuration;
- process isolation from the other framework packages, plus dependency
  isolation in managed mode;
- normalization into `ToolRep`;
- field-level differences and J1-J5 aggregation;
- terminal, JSON, and markdown reports;
- baseline comparison; and
- the common fixture integration test.

This is a multi-file contribution, not an external plug-in API. The checklist
below names every required surface so adding a framework does not depend on
repository folklore.

## 1. Establish and declare the capture boundary

Find the published framework or adapter API that performs the MCP-to-tool
conversion. Drive that API directly. Do not copy its implementation or write an
equivalent converter inside `mcp-mirror`. Name the exact API and object inspected;
do not silently describe an intermediate framework object as a provider request.

The renderer must extract the tool definition produced at that path:

```text
name
description
parameters  (JSON Schema)
```

Use the strongest deterministic boundary the framework exposes without making a
model call. Classify it as:

- `framework_tool_definition` for an adapted framework object;
- `provider_format` for a provider-shaped tool dictionary that has not been
  serialized into a request; or
- `provider_request` only when the test actually intercepts the serialized
  request body.

`provider_request_captured` may be true only for the last case. If the adapter
does not expose its independently negotiated MCP protocol version, record
`None` and explain that limitation. Never copy the direct source connection's
version into a framework run as an assumption.

Also inspect what happens to MCP annotations:

- put annotations in `ToolRep.annotations` only when the declared capture
  object contains them;
- put annotations in
  `framework_metadata={"annotations": ...}` when the framework retains them
  outside the captured object; and
- leave both empty when the adapted object no longer carries them.

That distinction drives J5. Treating retained metadata as dropped, or treating
model-blind metadata as surfaced, produces a false authorization result.

## 2. Choose the implementation route

### Python framework

Add `src/mcp_mirror/renderers/<id>_renderer.py`. Follow an existing renderer and
provide this module interface:

```python
ID = "framework_id"

def available() -> bool:
    ...

def get_renderer():
    return FrameworkRenderer()

class FrameworkRenderer:
    id = ID

    def versions(self) -> dict:
        return {"framework": "...", "adapter": "..."}

    def evidence(self) -> RendererEvidence:
        return RendererEvidence(
            capture_api="package.module.Adapter.method",
            capture_object="package.TypeName",
            capture_stage="framework_tool_definition",
            provider_request_captured=False,
            negotiated_mcp_spec_version=None,
            protocol_version_evidence="the adapter does not expose it",
            limitation="no serialized provider request was captured",
        )

    def render(self, server: ServerHandle) -> list[ToolRep]:
        ...
```

Import the heavy framework package inside `render()`, not at module import
time. `mcp-mirror frameworks` must continue to work when the optional
dependency is absent. Wrap adapter failures in `RenderError`.

Use `normalize_function_tool()` after the real adapter has produced its tool
shape. Do not import MCP in the differ, scorecard, or reporter.

### TypeScript or JavaScript framework

Follow the Mastra bridge:

```text
src/mcp_mirror/renderers/<id>_renderer.py
src/mcp_mirror/renderers/<id>_node/package.json
src/mcp_mirror/renderers/<id>_node/package-lock.json
src/mcp_mirror/renderers/<id>_node/_worker.mjs
```

The Node worker drives the published adapter and emits JSON. The Python wrapper
handles availability, subprocess isolation, timeout and error translation,
version reporting, evidence declaration, and conversion to `ToolRep`.

Commit the worker lockfile. Do not commit `node_modules/`.

For example, a Vercel AI SDK contribution would use this route. The PR must
identify and call the current AI SDK MCP/tool conversion API; the framework
name alone is not evidence that a hand-built OpenAI-style schema matches its
tool object or any provider-bound payload.

## 3. Register installation and discovery

1. Add the renderer id, module, runtime, exact top-level packages, supported
   protocol evidence, and capture expectation to
   `src/mcp_mirror/runner-manifests.json`.
2. Add only useful spelling variants to `_ALIASES`.
3. For Python, add a same-named optional dependency in `pyproject.toml` for
   local mode. Do not force mutually incompatible adapters into one
   environment. If the adapter's pins cannot resolve alongside the `all` extra
   (an MCP SDK major it excludes, say), add no extra and set `"managed_only": true`
   with a one-line `"managed_only_reason"` in its manifest instead; CI's
   availability check then skips it, and its tests run it through the managed runner.
4. For a Node worker, commit both `package.json` and `package-lock.json` and
   document its `npm ci --prefix ...` command.
5. Make `versions()` report the installed framework and adapter distributions,
   not versions copied from documentation.

After registration, `available_renderers()` automatically includes the
framework when its dependency is present. Local mode runs it in an isolated
process. Managed mode also creates a separate `uv` environment from its
manifest so another adapter's MCP SDK constraint cannot affect it.

## 4. Prove the renderer against the fixture

Install the complete locked development environment:

```bash
uv sync --locked --extra dev --extra all
npm ci
npm ci --prefix src/mcp_mirror/renderers/mastra_node
```

Add the new framework's equivalent npm installation when applicable, then run:

```bash
uv run pytest tests -q
uv run mcp-mirror frameworks
uv run mcp-mirror scan "python fixtures/tricky_server.py" \
  --frameworks framework_id \
  --runner managed \
  --fail-on-incomplete
```

The shared test verifies that the renderer:

- connects through the real adapter;
- returns tools from the fixture; and
- reports `destructiveHint` as present at the boundary, retained elsewhere, or
  dropped.

Add framework-specific assertions for transformations that the generic test
cannot establish. At minimum cover the framework's expected handling of:

- names, tool descriptions, and per-property descriptions;
- required parameters, enums, formats, defaults, numeric constraints, and
  whether optional wrappers newly accept `null`;
- nested objects, arrays of objects, `additionalProperties`, and every
  `oneOf` / `anyOf` branch;
- injected wrapper text or fields; and
- `readOnlyHint`, `destructiveHint`, `idempotentHint`, and `openWorldHint`.

An integration test may not replace the adapter with a mock converter. Missing
optional dependencies may skip local tests, but CI installs all declared
Python adapters and the checked-in Mastra worker before running the suite. CI
also fails if any registered renderer is unavailable or reports an unknown
framework or adapter version, preventing a broken registration from silently
skipping its own tests.

## 5. Add the framework to the compatibility data

The scanner and site are separate public surfaces. A renderer makes scans
possible; it does not silently publish support claims.

1. Add the framework to `data/frameworks.json` with its package, registry,
   language, renderer path, exact tested version, release history, and current
   era, and pin it in `src/mcp_mirror/runner-manifests.json`.
2. Run the instruments against the fixture for each revision the adapter can
   reach. They write artifacts under `measurements/`; the cells are derived from
   those, not typed:

   ```bash
   uv run mcp-mirror scan "python fixtures/tricky_server.py" --runner managed \
     --frameworks <id> --spec-version <revision> --out <snapshot>/measurements/scan/<id>.json
   uv run python scripts/measure_result_boundaries.py --spec-version <revision> --renderers <id>
   uv run python scripts/probe_surface.py --spec-version <revision> --renderers <id>
   uv run python scripts/probe_utilities.py --spec-version <revision> --renderers <id>
   uv run python scripts/probe_client_features.py --spec-version <revision> --renderers <id>
   uv run python scripts/probe_notifications.py --spec-version <revision> --renderers <id>
   uv run python scripts/measure_protocol.py --spec-version <revision> --renderers <id>
   uv run python scripts/measure_transports.py --spec-version <revision> --renderers <id>
   uv run python scripts/run_conformance.py   # if the adapter has an HTTP transport
   ```

   Every instrument that defaults to `all-frameworks.json` must be given
   `--out <snapshot>/measurements/<kind>/<id>.json` for a single framework, so one
   framework's run never rewrites another's capture; the cell rules read every file
   in the directory. `measure_protocol.py` and `run_conformance.py` already write one
   directory per renderer, so they keep their default output. Wrap each command in `uv run python scripts/measure_locked.py --`
   so measurements on the machine run one at a time, then run
   `uv run python scripts/scrub_home_paths.py`.

   The renderer answers each instrument through a method: `render`, `render_result`,
   `probe_surface`, `probe_client_features`, `probe_utilities`,
   `probe_notifications`, plus stdio, Streamable HTTP and HTTP+SSE connections for
   the protocol and transports instruments. A renderer without one of them derives
   `u` for those rows, which is honest but incomplete.

   `probe_utilities.py` needs a `probe_utilities(servers)` method on the renderer. It
   gets one recorded fixture per question in `src/mcp_mirror/probe.py`
   (`UTILITY_FEATURES`) and should answer each one through the adapter's own public
   API. Mark an answer `route: "session"` when it only works on a raw SDK session the
   adapter hands out, and `unsupported` when there is no public route at all.

3. If the adapter cannot negotiate a revision, record it in that snapshot's
   `attempts.json` with the handshake evidence. Every cell in the revision is then
   `x`, which is an answer for every row rather than a reason to leave them `u`.
4. Write a note in each capability overlay for every derived cell that is not `y`.
   The build fails until you do; that is the review step.
5. Keep framework ordering alphabetical within language group. There is no
   aggregate framework score.
6. Land the entry with `"visibility": "preview"`. A preview row is built into
   every aggregate but the site shows it only with `?preview=1` in the URL, the
   bundled MCP server omits it, and `specs.json` does not count it. That lets
   you show the maintainer the row before it is public. Say in the pull request
   what they should look at, and delete the field (or set `"published"`) once
   the row is ready.

### Frameworks that reuse another framework's adapter

Some frameworks reach MCP only through another framework's adapter. LangGraph,
for example, loads MCP tools with `langchain-mcp-adapters`. Publish these as an
alias rather than measuring the same adapter twice:

```json
"langgraph": {
  "name": "LangGraph",
  "abbr": "LG",
  "aliases_of": "langchain",
  "visibility": "preview",
  "alias_note": "Why this framework's MCP path is the source row's adapter."
}
```

An alias may declare only `name`, `abbr`, `aliases_of`, `alias_note` and
`visibility`. The build copies every measurement field (versions, package,
capture and result boundaries, measurement status) from the source, so its
cells are the source's derived cells and cannot drift from them. A note keyed
by the alias id in a capability overlay replaces the source's note on that
cell; without one, the alias carries the source's note. Aliases and preview rows
are left out of verdict headlines, which count distinct adapters.

Validate and regenerate the published aggregate:

```bash
python3 data/validator/validate.py data
python3 scripts/build_data.py
npm run build
```

## 6. Pull request evidence

Include the following in the PR description:

- framework and adapter package names;
- exact versions tested;
- the real adapter/conversion API called by the renderer;
- the exact extracted object and capture-stage classification;
- whether a serialized provider request was actually captured;
- whether the adapter-negotiated MCP version is observed or unknown;
- the fate of MCP annotations and the object fields that prove it;
- fixture command and test output;
- data files changed; and
- any transport not exercised.

Do not describe a framework as better or worse. Describe what the adapter
preserved, retained elsewhere, changed, added, or dropped.

## Current limits

The registry is internal and explicit; third-party packages cannot register a
renderer through Python entry points. Non-Python adapters need a small process
bridge. Cells are generated only from artifacts committed to this repository,
never from an arbitrary scan, and the notes that explain them are reviewed.

Those limits are intentional for the current census: every published framework
and result remains auditable in this repository. A future external plug-in API
would need an owner decision about trust, packaging, and whether third-party
renderers can produce publishable support data.
