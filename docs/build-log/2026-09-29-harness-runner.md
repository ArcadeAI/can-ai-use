# 2026-09-29: harnesses are measured on the wire

`scripts/measure_harness.py <harness>` runs a coding harness headless against the five
fixtures, each mounted through `mcp-mirror record` as its own server (`def`, `res`, `ntf`,
`cf`, `util`). Every JSON-RPC message the harness sends or receives lands in a JSONL
capture. How each harness is launched lives in `plan/harness_sessions.yaml`. What each
cell observes lives in `plan/harness_surface.yaml`.

- **Level 1 (startup)** needs no model. Claude Code's model call goes to a local stub
  that replies "ready" and accepts a placeholder key, so level 1 needs no credential
  and reruns from a clean checkout. It runs three times: over stdio, over streamable
  HTTP, and with `util` starting 8 seconds late.
- **Level 2 (session)** needs a model. It runs fixed prompts three times each under
  `op run --env-file .env.harness`. That file does not exist yet, so level 2 is
  recorded as skipped and its cells publish `u`.

The harness table has its own snapshot, `data/harnesses/`, with its own `surface.json`
(generated from the plan) and `harnesses.json`. The build writes it to
`data/fulldata/harnesses.json`, and it never merges with the framework table. Cells
are derived by the wire rules in `cells.py`, and each cell's evidence is a capture path
and message sequence number.

First capture: Claude Code 2.1.285 asks for 2026-07-28 with `server/discover` and falls
back to `initialize` at 2025-11-25, because the project's fixtures run the 1.x SDK.
It declares `roots` and `elicitation` (form and url), but not `sampling`. On every server
it calls `tools/list`, `prompts/list` and `resources/list` before the first model turn,
follows every cursor, and never calls `resources/templates/list`. It answers a
server-initiated ping within 10 ms and echoes `Mcp-Session-Id` over HTTP. When a server
answers 8 seconds late, it waits for it rather than dropping it.
