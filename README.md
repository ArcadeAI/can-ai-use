# mcp-mirror

> An open compatibility tracker and deterministic scanner for the Model Context Protocol across agent frameworks.

- **Live site**: <https://canaiuse.com>
- **Maintained by**: [Arcade.dev](https://arcade.dev) engineers, as a community project. See [Support](#support).

MCP defines three server primitives, a set of client features, and a base
protocol of transports, notifications and authorization. An agent framework
adapts some part of that into its own objects. `mcp-mirror` records the source
and one declared framework boundary, then shows what stayed present, what moved
to retained metadata, and what disappeared.

Coverage is published alongside the results. Each protocol snapshot carries a
`surface.json` enumerating what that revision of the specification defines, and
every feature in it appears as a row whether or not a scan has reached it yet.
The site reports measured features against defined features, so the size of the
gap is visible rather than implied by absence.

The project has two public surfaces:

- **The compatibility site** answers, "Can I rely on this MCP capability?" Each
  result names the tested framework version, the exact mechanism, and a command
  that reproduces the measurement.
- **The scanner** runs the same deterministic comparison against your own MCP
  server, with no model call and no model API key.

The support data is versioned separately from the presentation so other sites,
framework docs, and CI checks can consume it directly.

## Read a result

Every framework-version cell has one of five factual states:

- **Present at the capture boundary**: the value survives in the exact adapted
  tool definition named by that framework's measurement.
- **Changed or retained elsewhere**: a numbered note says whether the value was
  transformed at the boundary, partly preserved, or retained only in framework
  metadata.
- **Dropped during adaptation**: the value is not recoverable from the adapted
  tool object.
- **Cannot reach the revision**: the adapter cannot negotiate this MCP revision
  at all, so it exposes none of its features. This is a measured result carrying
  the dependency or negotiation evidence, not a gap in coverage.
- **Not yet measured**: the release exists in PyPI or npm, but no scan has
  covered it.

Every retained or dropped result cites a numbered note that names the mechanism.
Untested releases stay visible instead of disappearing from the history.

This is a compatibility census, not a leaderboard. Frameworks are included when
they ship a first-party or officially documented MCP adapter. They are ordered
alphabetically within language group, and no overall score is computed.

## Why it matters

A valid JSON Schema can become weaker after adaptation. A long description can
be truncated. A nested object can be flattened. A risk annotation can be kept
for application policy while remaining absent from the captured definition, or it can be
dropped entirely.

`destructiveHint` shows why those distinctions matter. In the releases measured
against MCP 2025-11-25, Strands and pi expose the hint at their declared capture
boundary and most other frameworks do not. LangChain,
Pydantic AI, and Mastra 2.x retain it in framework metadata. CrewAI, OpenAI
Agents SDK, and Mastra 1.x do not retain that behavioral hint on the captured
tool object. Those are different integration facts even when their normalized
definitions look alike.

## Evidence boundaries

Every scanner run records the API and object it inspected:

- LangChain: the OpenAI-compatible dictionary returned by
  `convert_to_openai_tool`.
- Pydantic AI: `ToolDefinition` objects returned by `MCPToolset.get_tools`.
- CrewAI: adapted `BaseTool` name, description, and `args_schema` from
  `MCPServerAdapter`.
- OpenAI Agents SDK: the SDK `FunctionTool` returned by
  `MCPUtil.to_function_tool`.
- Mastra: tool actions returned by `MCPClient.listTools`, with JSON Schema
  extracted from Mastra's Standard Schema wrapper.

None of the v0.1 renderers captures a serialized provider request. Reports say
that explicitly. A result therefore proves behavior at the named framework
boundary, not the final bytes sent to OpenAI, Anthropic, or another provider.

## Install

`mcp-mirror` is not published to PyPI, so it runs from a source checkout.
`uv run` builds the environment on first use, which is why every reproduce
command the site publishes starts with it. Later examples name the bare
`mcp-mirror` command; prefix them the same way unless the project environment
is already active.

From the root of a clone of this repository:

```bash
uv run mcp-mirror version               # base install, no renderers
uv run --extra all mcp-mirror version   # every renderer
uv run --extra langchain mcp-mirror version
uv run --extra pydantic-ai mcp-mirror version
uv run --extra crewai mcp-mirror version
uv run --extra openai-agents mcp-mirror version
```

Renderers are optional so a scan installs only the frameworks it needs.
Python 3.11 or newer is required. Run `uv run mcp-mirror frameworks` for the
exact setup command for every missing renderer. Install framework extras
separately when their MCP SDK constraints conflict.

`--runner managed` needs none of these extras: it reads
`runner-manifests.json` and builds one pinned environment per adapter, which is
why the published reproduce commands use it rather than a local install.

Mastra is a TypeScript framework. Its renderer drives the real `@mastra/mcp`
client through the included Node worker. `--runner managed` needs npm on PATH and
nothing else: it installs the manifest's npm packages and its pinned Node into
`~/.cache/mcp-mirror/node-runners/<digest>` (override with `MCP_MIRROR_CACHE_DIR`),
once per manifest digest and bundled lockfile, and runs the worker from there. For `--runner local`,
install the bundled project instead:

```bash
npm ci --prefix src/mcp_mirror/renderers/mastra_node
npm ci --prefix src/mcp_mirror/renderers/vercel_ai_node
```

## Quickstart

The fixture server exercises description fidelity, enum and format constraints,
nested structures, unions, authorization annotations, and Unicode.

```bash
# Scan the fixture through every installed renderer.
mcp-mirror scan "python fixtures/tricky_server.py"

# Scan an HTTP MCP server through selected frameworks.
mcp-mirror scan https://example.com/mcp \
  --frameworks langchain,pydantic_ai \
  --output md \
  --out mcp-mirror-report.json

# Run an adapter without installing it into mcp-mirror's environment.
# Managed mode requires uv and caches a separate environment per manifest.
mcp-mirror scan "python fixtures/tricky_server.py" \
  --frameworks pydantic_ai,openai_agents \
  --runner managed \
  --spec-version 2026-07-28 \
  --fail-on-incomplete \
  --out mcp-mirror-report.json

# Focus on parameter and authorization jobs.
mcp-mirror scan "python fixtures/tricky_server.py" --job J2,J5
```

`<server>` may be an HTTP(S) URL or a command string for a stdio server.

## Use it from your agent

The data is also an MCP server at `https://canaiuse.com/mcp`. It is open and read-only:
no account, no token, no session, one URL any MCP client can use. Add it to Claude Code,
Cursor or any client that speaks Streamable HTTP:

```json
{
  "mcpServers": {
    "canaiuse": {"url": "https://canaiuse.com/mcp"}
  }
}
```

The endpoint reads the data the site publishes, so an answer is always the deployed data; `worker/` holds the
Worker and `tests/test_worker_parity.py` keeps its catalog identical to `catalog.py`.

### Run it locally over stdio

`mcp-mirror serve` exposes the same evidence as a read-only MCP server over stdio.
From a source checkout, add it to Cursor or another MCP client:

```json
{
  "mcpServers": {
    "mcp-mirror": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/mcp-mirror",
        "run",
        "mcp-mirror",
        "serve"
      ]
    }
  }
}
```

After a release containing this command is published, the portable command is
`uvx mcp-mirror serve`.

The server provides five tools:

- `inspect_tool_definition` identifies the capabilities used by a proposed MCP
  tool and returns versioned adapter observations.
- `inspect_tool_result` does the same for a proposed `CallToolResult`.
- `lookup_capability` returns the evidence for one named feature.
- `search_capabilities` searches the catalog.
- `list_snapshots` lists measured revisions, versions, and coverage.

For example, ask the connected agent:

> Inspect this `delete_account` tool definition with mcp-mirror. Which declared
> capture boundaries retain `destructiveHint`, and where does each framework
> place it?

The server does not execute the submitted tool, call a model, or rank
frameworks. It reads the same versioned data published by the website.

## Jobs to be done

Field-level differences are grouped by the production job a tool author needs
to complete:

- **J1, identity**: the tool name and full purpose remain at the capture boundary.
- **J2, parameters**: types, required fields, enums, formats, and constraints
  remain available after adaptation.
- **J3, structure**: nested and structured input survives adaptation.
- **J4, injection**: the adapted definition contains nothing the tool author did
  not write.
- **J5, authorization**: dangerous or out-of-scope behavior remains legible,
  reported as present at the boundary, retained elsewhere, or destroyed.

Jobs return `pass`, `degraded`, or `fail`. Numerical difference counts are
diagnostic detail, not intent scores. J5 is always present in JSON, terminal,
and markdown scorecards even when `--job` narrows the other columns. No overall
framework verdict is computed.

## Regression checks

Save a baseline and fail CI when an adapter introduces new drift:

```bash
mcp-mirror scan "python fixtures/tricky_server.py" \
  --frameworks langchain \
  --out baseline.json

mcp-mirror scan "python fixtures/tricky_server.py" \
  --frameworks langchain \
  --baseline baseline.json \
  --fail-on-drift \
  --fail-on-incomplete
```

Reports include the MCP protocol version negotiated by mcp-mirror's direct
source connection. Baselines are compared only within that same protocol
version. Each framework run also records the version negotiated by that
adapter's own connection. Handshake-era source evidence comes from `initialize`;
MCP `2026-07-28` source evidence comes from `server/discover`. If a renderer's
evidence seam disappears, the renderer fails instead of copying the source
connection's version by assumption. Baseline comparison also rejects
per-renderer protocol mismatches and unknown mixed-version evidence. A live scan
exits with code `3` rather than diffing a renderer response established under a
different MCP revision from the source snapshot.

JSON reports use `mcp-mirror/report@2`. They keep every requested framework in
the result, including unsupported adapters, adapter errors, and protocol
mismatches. Source and framework observations carry stable artifact hashes, and
the report records non-secret scan inputs plus runtime provenance. Human reports
print a copyable reproduction command. Header names are retained, but header
values are never written to the report. URL user info, common credential query
parameters, and common stdio credential flags are redacted before the server
identifier is stored.

`--runner local` uses installed extras in separate subprocesses.
`--runner managed` also separates dependencies: the direct source connection
and each Python adapter run in disposable `uv` environments selected from
`runner-manifests.json`. Mastra runs the same way, with its npm packages and
Node version installed from the manifest into a cache keyed by the manifest
digest and the bundled lockfile. Manifest digests and the actual framework and adapter versions are
written to the report.

The compatibility site publishes separate MCP `2025-11-25` and `2026-07-28`
snapshots and never merges their cells. `2025-11-25` is the default the site
opens on (`data/framework_defaults.json`). `2026-07-28` is published beside it. Pydantic AI
2.54.0, OpenAI Agents SDK 0.23.1, and Mastra 2.1.0 produced same-protocol
captures. LangChain MCP Adapters 0.3.2 and CrewAI 1.15.26 have incompatible
Python SDK constraints; those rows are explicitly unmeasured in the 2026
snapshot. Mastra reached the revision when 2.0 moved it onto the
`@modelcontextprotocol/client` 2.0 SDK; the 2025 snapshot keeps its 1.17.3 row.

Each 2026-07-28 row is its own scan file, `data/specs/2026-07-28/measurements/scan/<framework>.json`,
and failed same-protocol attempts are recorded in `data/specs/2026-07-28/attempts.json`.
The command that reproduces a row, pinned to the version it measured, is on that
framework's page on the site (`--runner managed`, no local install). Copying it from
there keeps this README from going stale every time a row is re-measured.

## CLI

```text
mcp-mirror scan <server>
    --frameworks langchain,pydantic_ai,crewai
    --runner local|managed
    --spec-version 2026-07-28
    --output table|json|md
    --job J1,J5
    --baseline baseline.json --fail-on-drift
    --fail-on-incomplete
    --out report.json

mcp-mirror frameworks
mcp-mirror show report.json --output table|json|md
mcp-mirror version
```

Exit codes are `0` for success, `1` for detected drift, `2` for a connection or
handshake failure, `3` for a protocol-version mismatch, and `4` when
`--fail-on-incomplete` finds an unsupported or failed framework run. When both
gates are on, `4` wins over `1`, because drift measured from a partial scan is
not a signal worth acting on. A `3` still writes the report, so the failed
assertion ships with the evidence of what the server actually negotiated.

## Data and site architecture

The repository uses the same useful separation as caniuse:

```text
data/frameworks.json + data/capabilities/*.json
data/specs/2026-07-28/frameworks.json
data/specs/2026-07-28/capabilities/*.json
        |
        +--> */data-2025-11-25.json
        +--> */data-2026-07-28.json
        +--> */specs.json
        +--> */data-1.0.json       newest snapshot alias
        +--> docs/standalone.html  both snapshots inlined
```

Per-capability files remain the source of truth within each protocol snapshot.
`scripts/build_data.py` validates and aggregates each snapshot independently.
The React site loads one complete dataset at a time through the protocol
snapshot selector.

The site uses React 19, HeroUI v3, Tailwind CSS 4, and Vite. The production build
is written to `docs/` for static hosting. The build also inlines the compiled
CSS, JavaScript, and support data into `docs/standalone.html`, which opens
directly from `file://` without a server.

```bash
npm install
npm run dev       # http://127.0.0.1:8900
npm run build     # validate data, type-check, and build docs/
npm run preview
```

## Development

```bash
uv sync --locked --extra dev --extra all
npm ci --prefix src/mcp_mirror/renderers/mastra_node
npm ci --prefix src/mcp_mirror/renderers/vercel_ai_node
uv run pytest tests -q
```

Run the data validator directly when editing support records:

```bash
python3 data/validator/validate.py data
python3 scripts/build_data.py
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for correction and tone rules. The
complete [`ADDING_A_FRAMEWORK.md`](ADDING_A_FRAMEWORK.md) runbook covers
framework inclusion, Python and Node renderer patterns, J5 handling, tests,
compatibility data, and PR evidence. Renderers must drive the framework's real
adapter rather than a reimplementation.

## Support

This is a community project built by Arcade.dev engineers. It is not an official
Arcade.dev product and is not covered by Arcade.dev support or SLAs. Issues and pull
requests are monitored on a best-effort basis, with no guarantee of support.
Report security issues as [`SECURITY.md`](SECURITY.md) describes, not in a public issue.

## License

Code is licensed under MIT. Files in `data/` are licensed under CC BY 4.0.

## Deploying the site

canaiuse.com is a Cloudflare Worker serving `docs/` as static assets, with the MCP endpoint
at `/mcp` in `worker/index.ts`, configured in `wrangler.toml` on the Arcade Cloudflare account.
From a checkout with wrangler logged in:

```bash
npm run build
npx wrangler deploy
```

The custom-domain routes for `canaiuse.com` and `www.canaiuse.com` create the DNS records
and certificates. GitHub Pages is off.
