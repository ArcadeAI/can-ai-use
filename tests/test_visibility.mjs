import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";

import {
  frameworkHref,
  initialSpecFor,
  isAgentsRoute,
  isFrameworksRoute,
  parseSupport,
  previewRequested,
  supportCounts,
  routeFromHash,
  visibleDatabase,
} from "../site/src/lib/data.ts";
import {
  defaultsRule,
  filteredDatabase,
  frameworkReproduceCommands,
  groupFrameworks,
  languageGroup,
  shortFrameworkName,
  unreachableNotes,
} from "../site/src/lib/frameworks.ts";
import {
  agentHref,
  agentsHref,
  harnessCodeCounts,
  harnessLiveNote,
  harnessMeasuredOn,
  harnessRowFor,
  harnessRule,
  harnessUnmeasured,
  matchesHarnessRow,
  rowCodeCounts,
  rowLabelTally,
  shownHarnesses,
  visibleHarnesses,
} from "../site/src/lib/harnesses.ts";

const load = (spec) =>
  JSON.parse(readFileSync(new URL(`../site/public/data/data-${spec}.json`, import.meta.url)));

test("previewRequested reads only ?preview=1", () => {
  assert.equal(previewRequested(""), false);
  assert.equal(previewRequested("?spec=2025-11-25"), false);
  assert.equal(previewRequested("?preview=0"), false);
  assert.equal(previewRequested("?preview=1"), true);
  assert.equal(previewRequested("?spec=2025-11-25&preview=1"), true);
});

for (const spec of ["2025-11-25", "2026-07-28"]) {
  test(`${spec}: a preview row is absent by default and present with the flag`, () => {
    // Every row may be published, so mark one alias and one framework preview here
    // rather than relying on the data to still hold a preview row.
    const database = load(spec);
    for (const id of ["langgraph", "agno"]) database.agents[id].visibility = "preview";
    const preview = Object.keys(database.agents).filter(
      (id) => database.agents[id].visibility === "preview",
    );
    assert.ok(preview.includes("langgraph") && preview.includes("agno"));

    const shown = visibleDatabase(database, previewRequested(""));
    for (const id of preview) {
      assert.ok(!(id in shown.agents));
      for (const capability of Object.values(shown.data)) {
        assert.ok(!(id in capability.stats), `${capability.id} still has ${id}`);
        assert.ok(!(capability.boundary_dependent ?? []).includes(id));
      }
    }
    for (const capability of Object.values(shown.data)) {
      const refs = Object.values(capability.stats).flatMap((versions) =>
        Object.values(versions).flatMap((raw) => parseSupport(raw).noteRefs),
      );
      assert.deepEqual(
        Object.keys(capability.notes_by_num ?? {}).map(Number).sort((a, b) => a - b),
        [...refs].sort((a, b) => a - b),
        `${capability.id}: notes should be exactly the cited ones`,
      );
    }

    const flagged = visibleDatabase(database, previewRequested("?preview=1"));
    assert.equal(flagged.agents, database.agents);
    for (const id of preview) {
      assert.ok(id in flagged.agents);
    }
  });

  test(`${spec}: notes open with a tally of the rows the page shows`, () => {
    const database = load(spec);
    const views = [
      ["public", visibleDatabase(database, false)],
      ["preview", visibleDatabase(database, true)],
      ["filtered", filteredDatabase(visibleDatabase(database, true), defaults)],
    ];
    for (const [name, view] of views) {
      const rows = Object.keys(view.agents).length;
      for (const capability of Object.values(view.data)) {
        const tallied = capability.notes.match(
          /^(\d+) of (\d+) frameworks? present at the capture boundary, (\d+) changed or retained elsewhere, (\d+) dropped(?:, (\d+) unmeasured)?(?:, (\d+) cannot reach MCP [\d-]+)?\./,
        );
        assert.ok(tallied, `${name} ${capability.id}: notes do not open with the tally`);
        // The Observed header on the same page counts these, so the two have to agree.
        const counts = supportCounts(view, capability);
        assert.deepEqual(
          tallied.slice(1).map((value) => Number(value ?? 0)),
          [counts.y, rows, counts.a, counts.n, counts.u, counts.x],
          `${name} ${capability.id}: ${capability.notes}`,
        );
      }
    }
    // The site recounts with its own copy of the sentence; it has to match the build's.
    const shown = visibleDatabase(database, false);
    for (const [capabilityId, capability] of Object.entries(database.data)) {
      assert.equal(shown.data[capabilityId].notes, capability.notes, capabilityId);
    }
  });
}

const manifests = JSON.parse(
  readFileSync(new URL("../src/mcp_mirror/runner-manifests.json", import.meta.url)),
);
const defaults = JSON.parse(
  readFileSync(new URL("../data/framework_defaults.json", import.meta.url)),
);

test("every framework falls in the language group its runner runtime names", () => {
  const database = load("2026-07-28");
  const groups = groupFrameworks(database, manifests);
  assert.deepEqual(
    groups.map((group) => group.label),
    ["Python", "TypeScript", "Java/Kotlin", ".NET & Go"],
  );
  assert.deepEqual(
    groups.flatMap((group) => group.agentIds).sort(),
    Object.keys(database.agents).sort(),
  );
  const expected = {python: "python", typescript: "typescript", java: "jvm", csharp: "dotnet-go", go: "dotnet-go"};
  for (const group of groups) {
    for (const agentId of group.agentIds) {
      assert.equal(expected[database.agents[agentId].type], group.id, agentId);
    }
  }
  // An alias row runs on its source's runner, so it groups with the source.
  assert.equal(languageGroup("langgraph", database.agents, manifests), "python");
});

test("an empty language group is dropped rather than drawn", () => {
  // Hide every JVM row so its group is empty, whatever the data publishes.
  const raw = load("2026-07-28");
  for (const id of Object.keys(raw.agents)) {
    if (languageGroup(id, raw.agents, manifests) === "jvm") raw.agents[id].visibility = "preview";
  }
  const database = visibleDatabase(raw, false);
  const groups = groupFrameworks(database, manifests);
  assert.ok(groups.every((group) => group.agentIds.length > 0));
  assert.ok(!groups.some((group) => group.id === "jvm"));
});

test("short names drop only the language suffix the group label already states", () => {
  assert.equal(shortFrameworkName("OpenAI Agents SDK (JS)"), "OpenAI Agents SDK");
  assert.equal(shortFrameworkName("Google ADK (TypeScript)"), "Google ADK");
  assert.equal(shortFrameworkName("Google ADK (Java)"), "Google ADK");
  assert.equal(shortFrameworkName("Microsoft Agent Framework (.NET)"), "Microsoft Agent Framework");
  assert.equal(shortFrameworkName("Google ADK (Go)"), "Google ADK");
  assert.equal(shortFrameworkName("CrewAI (native MCP)"), "CrewAI (native MCP)");
});

test("Filtered keeps exactly the chosen default frameworks", () => {
  for (const spec of ["2025-11-25", "2026-07-28"]) {
    const database = load(spec);
    for (const id of defaults.default_frameworks) {
      assert.ok(id in database.agents, `${spec}: default framework ${id} is not a published row`);
    }
    const filtered = filteredDatabase(database, defaults);
    assert.deepEqual(
      Object.keys(filtered.agents).sort(),
      [...defaults.default_frameworks].sort(),
    );
    // Note numbers survive filtering, and only notes a kept cell cites remain.
    for (const capability of Object.values(filtered.data)) {
      const refs = Object.values(capability.stats).flatMap((versions) =>
        Object.values(versions).flatMap((raw) => parseSupport(raw).noteRefs),
      );
      assert.deepEqual(
        Object.keys(capability.notes_by_num ?? {}).map(Number).sort((a, b) => a - b),
        [...new Set(refs)].sort((a, b) => a - b),
        capability.id,
      );
    }
    const rule = defaultsRule(database, defaults);
    for (const id of defaults.default_frameworks) {
      assert.ok(rule.includes(database.agents[id].name), `${spec}: rule omits ${id}`);
    }
  }
});

test("harness links point at real rows and capabilities", () => {
  const links = JSON.parse(readFileSync(new URL("../data/harness_links.json", import.meta.url)));
  const harnesses = JSON.parse(
    readFileSync(new URL("../site/public/data/harnesses.json", import.meta.url)),
  );
  const capabilityIds = new Set(
    ["1.0", "2026-07-28", "2025-11-25"].flatMap((spec) => Object.keys(load(spec).data)),
  );
  for (const id of links.default_harnesses) {
    assert.ok(harnesses.agents[id], `default harness ${id} is not in harnesses.json`);
  }
  for (const [capabilityId, rowId] of Object.entries(links.capabilities)) {
    assert.ok(capabilityIds.has(capabilityId), `${capabilityId} is not a framework capability`);
    assert.ok(harnesses.data[rowId], `${rowId} is not a harness row`);
    assert.equal(harnessRowFor(capabilityId, harnesses, links).id, rowId);
  }
  assert.equal(harnessRowFor("tool-icons", harnesses, links), undefined);
});

test("the harness filter keeps the default set in data order, All keeps every row", () => {
  const harnesses = JSON.parse(
    readFileSync(new URL("../site/public/data/harnesses.json", import.meta.url)),
  );
  const links = {schema: "", default_harnesses: ["codex", "claude_code"], capabilities: {}};
  const entries = Object.entries(harnesses.agents);
  const order = entries.map(([id]) => id);
  const filtered = shownHarnesses(entries, links, false).map(([id]) => id);
  assert.deepEqual(filtered, ["claude_code", "codex"].sort((a, b) => order.indexOf(a) - order.indexOf(b)));
  assert.equal(shownHarnesses(entries, links, true).length, entries.length);
  assert.match(harnessRule(harnesses, links), /^Filtered: .+, .+$/);
});

test("#/fw/<id> routes to that framework's page", () => {
  assert.deepEqual(routeFromHash("#/fw/vercel_ai"), {name: "framework", frameworkId: "vercel_ai"});
  assert.deepEqual(routeFromHash("#/fw/crewai-native?note=2"), {
    name: "framework",
    frameworkId: "crewai-native",
  });
  assert.deepEqual(routeFromHash("#/fw/"), {name: "framework", frameworkId: ""});
  assert.equal(frameworkHref("langgraph"), "#/fw/langgraph");
  assert.deepEqual(routeFromHash(frameworkHref("pydantic-ai")), {
    name: "framework",
    frameworkId: "pydantic-ai",
  });
  assert.deepEqual(routeFromHash("#/cap/enum"), {name: "capability", capabilityId: "enum"});
  // A malformed escape is an unknown id, not a crash.
  assert.deepEqual(routeFromHash("#/cap/%"), {name: "capability", capabilityId: "%"});
  assert.deepEqual(routeFromHash("#/agent/%E0%A4%A"), {name: "agent", agentId: "%E0%A4%A"});
});

test("the agents section has an index, a page per behaviour and a page per agent", () => {
  assert.deepEqual(routeFromHash("#/agents"), {name: "agents"});
  assert.deepEqual(routeFromHash("#/agents/"), {name: "agents"});
  assert.deepEqual(routeFromHash("#/agents/h-destructive-blocked-by-default"), {
    name: "agentRow",
    rowId: "h-destructive-blocked-by-default",
  });
  assert.deepEqual(routeFromHash("#/agent/claude_code"), {name: "agent", agentId: "claude_code"});
  assert.deepEqual(routeFromHash("#/agent/claude_code?preview=1"), {
    name: "agent",
    agentId: "claude_code",
  });
  assert.equal(agentsHref(), "#/agents");
  assert.equal(agentsHref("h-startup-timeout"), "#/agents/h-startup-timeout");
  assert.equal(agentHref("codex"), "#/agent/codex");
  assert.deepEqual(routeFromHash(agentsHref("h-startup-timeout")), {
    name: "agentRow",
    rowId: "h-startup-timeout",
  });
  // The old address keeps working: /harnesses is the agents index.
  assert.deepEqual(routeFromHash("#/harnesses"), {name: "agents"});
  assert.deepEqual(routeFromHash("#/harnesses/"), {name: "agents"});
  assert.deepEqual(routeFromHash("#/frameworks"), {name: "frameworks"});
  assert.deepEqual(routeFromHash("#/frameworks/"), {name: "frameworks"});
  assert.deepEqual(routeFromHash("#/"), {name: "index"});
  for (const hash of ["#/frameworks", "#/cap/enum", "#/fw/langchain"]) {
    assert.ok(isFrameworksRoute(routeFromHash(hash)), hash);
  }
  for (const hash of ["#/", "#/agents", "#/agent/claude_code", "#/stats"]) {
    assert.ok(!isFrameworksRoute(routeFromHash(hash)), hash);
  }
  for (const route of [routeFromHash("#/agents"), routeFromHash("#/agents/x"), routeFromHash("#/agent/x")]) {
    assert.ok(isAgentsRoute(route), JSON.stringify(route));
  }
  for (const hash of ["#/", "#/cap/enum", "#/fw/langgraph", "#/stats", "#/changes", "#/method"]) {
    assert.ok(!isAgentsRoute(routeFromHash(hash)), hash);
  }
});

test("agent tallies count every shown cell once, by code and by label", () => {
  const harnesses = JSON.parse(
    readFileSync(new URL("../site/public/data/harnesses.json", import.meta.url)),
  );
  const ids = Object.keys(harnesses.agents);
  const rows = Object.values(harnesses.data);
  const total = (counts) => counts.y + counts.a + counts.n + counts.u;
  for (const row of rows) {
    const counts = rowCodeCounts(row, ids);
    assert.equal(total(counts), ids.length, `${row.id}: a cell is counted twice or not at all`);
    const tally = rowLabelTally(row, ids);
    assert.equal(
      tally.reduce((sum, entry) => sum + entry.count, 0),
      ids.length,
      `${row.id}: label tally does not cover every agent`,
    );
    assert.ok(
      tally.every((entry) => entry.label && harnesses.support_codes[entry.code]),
      `${row.id}: a tally entry has no label or an unknown code`,
    );
    assert.ok(matchesHarnessRow(row, row.title.slice(0, 5)), `${row.id}: title does not match itself`);
    assert.ok(matchesHarnessRow(row, ""), `${row.id}: an empty query hides the row`);
  }
  for (const id of ids) {
    assert.equal(total(harnessCodeCounts(rows, id)), rows.length, `${id}: a row is counted twice or not at all`);
  }
  assert.ok(!rows.some((row) => matchesHarnessRow(row, "zzz-no-such-behaviour")));

  // An agent whose run opened no session has no date and says why; it still counts
  // in every tally as "not measured" and keeps its place in the lists.
  const blocked = Object.entries(harnesses.agents).filter(([, agent]) => agent.measurement_blocked);
  assert.ok(blocked.length > 0, "the published data has no unmeasured agent to check");
  for (const [id, agent] of blocked) {
    assert.equal(agent.measured_on, null, `${id}: an unmeasured agent carries a date`);
    assert.ok(harnessUnmeasured(agent), `${id}: no reason for being unmeasured`);
    for (const row of rows) {
      assert.equal(harnessMeasuredOn(row, id, agent), undefined, `${row.id}: a date for unmeasured ${id}`);
      assert.equal(row.cells[id].code, "u", `${row.id}: unmeasured ${id} has a measured cell`);
    }
  }
  const measured = ids.find((id) => !harnesses.agents[id].measurement_blocked);
  assert.equal(harnessUnmeasured(harnesses.agents[measured]), undefined);
  assert.equal(harnessMeasuredOn(rows[0], measured, harnesses.agents[measured]), harnesses.agents[measured].measured_on);

  // An agent measured against its live service says so on every surface: the note
  // names the service, and only such an agent's capture lives under wire-live.
  assert.equal(harnessLiveNote({...harnesses.agents[measured], measured_live: false}), undefined);
  assert.equal(
    harnessLiveNote({...harnesses.agents[measured], measured_live: true, live_service: "Cursor"}),
    "measured live against Cursor's service; results can vary between runs",
  );
  for (const [id, agent] of Object.entries(harnesses.agents)) {
    const live = agent.capture.startsWith("measurements/wire-live/");
    assert.equal(Boolean(agent.measured_live), live, `${id}: measured_live disagrees with its capture path`);
    assert.equal(Boolean(harnessLiveNote(agent)), live, `${id}: the live note disagrees with its capture path`);
    if (live) {
      assert.ok(agent.live_service, `${id}: a live row names its service`);
      assert.ok(!agent.measurement_blocked, `${id}: a live row that measured nothing stays unmeasured, not live`);
    }
  }

  // The index hides preview agents unless the flag is on, and never reorders them.
  const database = structuredClone(harnesses);
  database.agents[ids[0]].visibility = "preview";
  assert.deepEqual(
    visibleHarnesses(database, false).map(([id]) => id),
    ids.slice(1).filter((id) => database.agents[id].visibility !== "preview"),
  );
  assert.deepEqual(visibleHarnesses(database, true).map(([id]) => id), ids);
});

test("a framework's reproduce commands name only that framework's runner", () => {
  const database = load("2026-07-28");
  const capabilities = Object.values(database.data);
  for (const [agentId, runner] of [["vercel_ai", "vercel_ai"], ["openai-agents", "openai_agents"]]) {
    const commands = frameworkReproduceCommands(agentId, database, capabilities);
    assert.ok(commands.length > 1, `${agentId}: ${commands.length} commands`);
    for (const command of commands) {
      assert.ok(!command.includes("&&"), command);
      const named = command.match(/--(?:frameworks|renderers)\s+(\S+)/)[1].split(",");
      assert.ok(named.includes(runner), command);
    }
  }
  // An alias reruns through its source's runner.
  assert.deepEqual(
    frameworkReproduceCommands("langgraph", database, capabilities),
    frameworkReproduceCommands("langchain", database, capabilities),
  );
});

test("a framework page opens on the default it reaches, else the newest it reaches", () => {
  const index = JSON.parse(
    readFileSync(new URL("../site/public/data/specs.json", import.meta.url)),
  );
  const langchain = {name: "framework", frameworkId: "langchain"};
  const vercel = {name: "framework", frameworkId: "vercel_ai"};
  assert.equal(initialSpecFor(index, null, langchain), "2025-11-25");
  assert.equal(initialSpecFor(index, null, vercel), index.default_spec);
  // With the default moved to a snapshot LangChain cannot reach, it falls back to the
  // newest one it does, and Vercel AI (which reaches 2026-07-28) opens there.
  const newer = {...index, default_spec: "2026-07-28"};
  assert.equal(initialSpecFor(newer, null, langchain), "2025-11-25");
  assert.equal(initialSpecFor(newer, null, vercel), "2026-07-28");
  // A snapshot named in the URL wins; an unknown one falls back.
  assert.equal(initialSpecFor(index, "2026-07-28", langchain), "2026-07-28");
  assert.equal(initialSpecFor(index, "1999-01-01", langchain), "2025-11-25");
  assert.equal(initialSpecFor(index, null, {name: "home"}), index.default_spec);
});

test("unreachable notes are grouped by text with the capabilities that share them", () => {
  const database = load("2026-07-28");
  const capabilities = Object.values(database.data).filter((capability) => capability.shown !== false);
  const notes = unreachableNotes("langchain", database, capabilities);
  assert.ok(notes.length >= 1);
  assert.equal(
    notes.reduce((sum, note) => sum + note.capabilityIds.length, 0),
    capabilities.length,
  );
  assert.equal(new Set(notes.map((note) => note.text)).size, notes.length);
  assert.deepEqual(unreachableNotes("vercel_ai", database, capabilities), []);
});

test("feedback posts JSON to /feedback and reports the Worker's error", async () => {
  const {sendFeedback} = await import("../site/src/lib/feedback.ts");
  const payload = {kind: "capability", subject: "tool-output-schema", page: "https://canaiuse.com/#/cap/x", message: "hi"};
  let seen;
  const ok = await sendFeedback(payload, async (url, init) => {
    seen = {url, init};
    return new Response(JSON.stringify({ok: true}), {status: 200});
  });
  assert.deepEqual(ok, {ok: true});
  assert.equal(seen.url, "/feedback");
  assert.equal(seen.init.method, "POST");
  assert.deepEqual(JSON.parse(seen.init.body), payload);
  const limited = await sendFeedback(payload, async () => new Response(JSON.stringify({error: "Too many"}), {status: 429}));
  assert.deepEqual(limited, {ok: false, error: "Too many"});
  const offline = await sendFeedback(payload, async () => { throw new Error("offline"); });
  assert.equal(offline.ok, false);
});
