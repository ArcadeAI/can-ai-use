# 2026-09-30: server utilities are measured

A new instrument, `scripts/probe_utilities.py`, measures the server utilities: pagination,
progress, logging, log level, completion, cancellation, tasks and caching hints. It writes
`measurements/utilities/all-frameworks.json` in each snapshot. `derive_utility_cells` in
`src/mcp_mirror/cells.py` reads that file.

## How the instrument works

Each question gets its own fixture process behind `mcp-mirror record`, and every process the
adapter launches writes its own capture. Each adapter answers through its own public API
(`probe_utilities` in each renderer). The artifact keeps two halves for each question: the wire
the server saw, and what the adapter handed its caller.

A cell is `y` only when the adapter's own API does the job. It is `a` in two cases: the job works
only on a raw SDK session the adapter hands out, or progress needs a token the caller builds by
hand.

## Fixture changes

`fixtures/utilities_server.py` now serves 60 tools, resources and prompts in pages of 20. It also
adds `task_capable`, a tool marked `taskSupport: optional`, and advertises task-augmented
`tools/call`. At 2026-07-28 that is the Tasks extension. Also at 2026-07-28, every `tools/list`
page carries `ttlMs: 1500` and `cacheScope: private`.

## Cells, 2025-11-25

| Row | CrewAI | LangChain | OpenAI Agents | Pydantic AI | Mastra |
| --- | --- | --- | --- | --- | --- |
| pagination-cursor | n | y | y | y | y |
| progress-notifications | n | y | a | y | y |
| logging-notifications | n | y | y | y | y |
| logging-set-level | n | a | a | y | n |
| completion-complete | n | a | a | a | n |
| cancellation | n | n | y | y | y |
| tasks-experimental | n | n | n | n | n |

## Cells, 2026-07-28

CrewAI and LangChain are `x` in every row.

| Row | OpenAI Agents | Pydantic AI | Mastra |
| --- | --- | --- | --- |
| pagination-cursor | y | y | y |
| progress-notifications | a | y | y |
| logging-notifications | n | y | y |
| completion-complete | a | a | n |
| cancellation | y | y | y |
| extension-tasks | n | n | n |
| caching-hints | a | a | y |

## Limits

- Cancellation is measured over stdio only.
- `cacheScope` partitioning is not exercised.
- The task rows are a probe only. No adapter asked for a task, so the fixture never had to
  return one.
