# Contributing

This repository is two things, and they are contributed to differently:

- **The scanner** (`src/mcp_mirror/`), code that drives each framework's *real*
  MCP adapter and diffs its declared capture object against what the server sent. MIT.
- **The support data** (`data/`), the tables the site and other tools read. CC BY 4.0.

The data is the part most people want to change, so start there.

## Commits and pull requests

Small, atomic changes are how this repository stays reviewable. The data is derived,
so a reader has to be able to see which code change moved which cell. A commit that
mixes three changes makes that impossible.

### One commit, one change

- **A commit does one thing.** Fix a renderer, add a capability, bump a pin. If the
  message needs "and", it is probably two commits.
- **Every commit passes on its own.** Code, the measurement it produces and the
  regenerated data land in the same commit, so
  `uv run python scripts/generate_cells.py --check` and the test suite pass at each
  commit, not just at the tip. Never commit a code change and leave "rebuild data" for
  later.
- **Measurement changes travel with their cause.** A renderer fix commits with the
  re-measured artifact and the cells that moved. A re-measure with no code change, such
  as a version bump, is its own commit.
- **No WIP, fixup or "address review" commits on `main`.** Squash or rebase them into
  the commit they belong to before merging.
- **Write the subject as what the commit does,** in the imperative and under about 72
  characters, for example `Measure Google ADK over the three transports`. Use the body
  for why and for anything the diff does not show, like the evidence behind a changed
  cell.

### One pull request, one concern

- **Keep a PR to one coherent concern.** One framework row, one capability, one
  instrument, one fix. Split unrelated changes, even small ones. A drive-by cleanup
  goes in its own PR.
- **Title it with what changes.** Fill in
  [the PR template](.github/PULL_REQUEST_TEMPLATE.md) briefly. A sentence per section is
  often enough. Delete sections that do not apply.
- **Say what you verified, not what you plan to verify.** List the commands you ran and
  what happened. If a check could not run, say which and why. Passing tests alone do
  not prove a cell is right; check the capture.
- **Open it as a draft until it is ready to merge.** Required checks pass on the
  current revision, review findings are addressed, and you know of no blocker.
- **Rebase on `main` before merging,** and rebuild generated files instead of
  hand-merging their conflicts: `python3 scripts/build_data.py && npm run build`.
- **You own the result,** including any part an agent wrote. Read the diff and the
  description yourself before asking for review.

## Correcting a cell

Cells are not typed by hand. Each one is derived by `src/mcp_mirror/cells.py` from a
measurement artifact under `data/measurements/` (or `data/specs/<revision>/measurements/`
for a newer revision): a definition scan, a tool-result capture, or a resources-and-prompts
probe. A correction is therefore a disagreement about a *measurement*. Reproduce it with
the command printed on the capability's page, for example:

```bash
uv run python scripts/probe_surface.py --spec-version 2026-07-28 --renderers mastra
```

If the capture disagrees with your framework's actual behaviour, that is a bug in the
renderer or in a derivation rule. Open an issue with the output and we will fix it. If
the capture is right, re-run the instrument and commit the new artifact; the cells follow.

What a person writes lives in `data/capabilities/<id>.json`: the description, why it
matters, and one note per framework explaining any cell that is not `y`. `git log` on
that file is the capability's editorial history; `git log data/measurements/` is its
measurement history.

## Shipped a fix in your framework?

Say so, that's the best kind of PR. Bump the pin in
`src/mcp_mirror/runner-manifests.json`, re-run the instruments, and commit the new
artifacts. If your cell turns `y`, delete its note. We'd rather celebrate a fix than sit
on a stale red cell.

## Adding a framework

A framework is listed if it ships a first-party or officially documented MCP
adapter. Nothing is excluded on quality grounds.

Follow [`ADDING_A_FRAMEWORK.md`](ADDING_A_FRAMEWORK.md) for the complete
renderer contract, Python and Node worker patterns, dependency and registry
wiring, J5 annotation handling, fixture tests, support-data changes, and PR
evidence.

The short version is:

1. Drive the framework's **real** adapter and declare the exact capture API,
   object, and evidence stage.
2. Register the renderer and its pinned dependency path.
3. Prove it against the tricky fixture, including annotation retention.
4. Add the framework to `data/frameworks.json` and its pin to the runner manifests, run
   the instruments, and write a note for every cell that is not `y`.

Once registered, the existing process isolation, `ToolRep` differ, J1-J5
scorecard, reporters, baseline comparison, and common integration test apply
automatically. Extracting the correct framework object and publishing measured
data remain deliberate, reviewable work.

## Adding a capability

Add the feature to the revision's `surface.json` first; it is published as unmeasured
until something measures it, so the gap is visible rather than implied by absence. To
measure it, add a fixture case, a derivation rule in `src/mcp_mirror/cells.py`, and an
overlay in `data/capabilities/` holding the prose. `scripts/generate_cells.py --check`
confirms the committed build matches a fresh derivation.

## The rules the data must satisfy

These are enforced in CI by `data/validator/validate.py`:

- Statuses are `y` (present unchanged at the declared boundary), `a`
  (transformed, partly preserved, or retained elsewhere), `n` (dropped), `x`
  (the adapter cannot negotiate this revision, so it reaches nothing in it), or
  `u` (not yet measured).
- `x` is a result, not a gap. Use it when the adapter's SDK pin or negotiated
  version rules the revision out, and cite the evidence recorded in
  `attempts.json`. Never leave such a cell `u`, which claims we did not look.
- **Every `a`, `n` and `x` cell must cite a note that names the exact mechanism.** The
  build fails on a derived `a` or `n` with no note in the overlay. A
  status has to be checkable, not trusted, "dropped" alone is an opinion,
  "the adapted tool object exposes no annotations field" is a fact someone can verify.
- Framework ids must exist in `data/frameworks.json`.

Run the checks and regenerate the published aggregate:

```bash
uv sync --locked --extra dev --extra all
uv run pytest tests -q
python3 data/validator/validate.py data
python3 scripts/build_data.py
npm run build
```

## A note on tone

This project measures what each framework **changed**, not who is "best". It is
not a ranking and shouldn't read like one. Notes should describe mechanisms;
they shouldn't editorialise about a framework's choices. If a PR's wording makes
a maintainer's Monday worse without adding information, it'll get edited.
