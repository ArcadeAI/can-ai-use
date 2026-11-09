# 2026-09-29: the fixture is split by instrument

`fixtures/tricky_server.py` served everything from one file, and one mixed result was
the only way tool results were measured. The fixture is now five servers, one per
instrument. All of them build on both Python SDK lines through `fixtures/_dual.py`.

| Fixture | What it serves |
| --- | --- |
| `fixtures/definitions_server.py` | What `tricky_server.py` served, unchanged: tool definitions plus the calls, resources and prompts the scan and probe were measured against. |
| `fixtures/results_server.py` | One no-argument tool per content kind (`text_content`, `image_content`, `audio_content`, `embedded_resource_content`, `resource_link_content`, `annotated_content`), plus `mixed_content`, `text_and_structured` and `tool_error`. |
| `fixtures/notify_server.py` | `add_tool`, `remove_tool`, `add_resource`, `update_resource`, `add_prompt`. Each one changes the catalogue and fires the matching `list_changed` or `resources/updated`: on the session for handshake revisions, on `subscriptions/listen` at 2026-07-28. |
| `fixtures/client_features_server.py` | `ask_sampling`, `ask_elicitation_form`, `ask_elicitation_url`, `list_client_roots`. Handshake revisions send the request on the session; 2026-07-28 returns an `InputRequiredResult`. |
| `fixtures/utilities_server.py` | 25 tools, resources and prompts in pages of 10 with cursors, `slow_with_progress`, `cancellable_wait`, `emit_logs` with `logging/setLevel`, and completion for a prompt argument and a template variable. |

`fixtures/tricky_server.py` is now an alias for the definitions server, so every
published reproduce command that names it still runs.

`scripts/measure_result_boundaries.py` now measures against the results server, one
call per kind. The old mixed result hid two things:

- LangChain's audio conversion raises and aborts the whole call. Measured one kind at
  a time, its image, embedded-resource, resource-link and content-annotation cells are
  now measured instead of `u`: image `y`, embedded resource `n`, resource link `a`,
  annotations `n`.
- Mastra returns `structuredContent` in place of the whole result whenever a result
  carries it, and otherwise passes the result through untouched. Both old calls carried
  `structuredContent`, so Mastra's content cells read `n`. They are now `a` in both
  snapshots.

Tests: `tests/test_fixtures.py` drives every new server with a real client on both SDK
lines, and covers the 2026-07-28 wire (listen stream, input requests) with the 2.x
client.
