// Smoke test for the agents section: the #/agents index, one page per behaviour
// (#/agents/<row>) and one per agent (#/agent/<id>) at 390px and 1440px, every cell
// linked to the wire capture it was read from, no console errors, the old #/harnesses
// address still answering, and the framework table left exactly as it was.
import {existsSync} from "node:fs";
import {readFile} from "node:fs/promises";
import path from "node:path";
import {pathToFileURL} from "node:url";

import {chromium} from "playwright-core";
import {preview} from "vite";

const chromePath =
  process.env.CHROME_PATH ??
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const widths = [390, 1440];

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

// The release the pages should mark as newer than the measured one, as NewerRelease.tsx decides it.
function newerRelease(row) {
  if (!row?.stable_version || row.stable_version === row.current_version) return null;
  const date = (version) => row.version_list.find((entry) => entry.version === version)?.release_date;
  const measured = date(row.current_version);
  const stable = date(row.stable_version);
  return measured && stable && stable > measured ? row.stable_version : null;
}

const harnessData = JSON.parse(await readFile("docs/data/harnesses.json", "utf8"));
const specIndex = JSON.parse(await readFile("docs/data/specs.json", "utf8"));
const database = JSON.parse(await readFile("docs/data/data-1.0.json", "utf8"));
const harnessIds = Object.keys(harnessData.agents);
const links = JSON.parse(await readFile("data/harness_links.json", "utf8"));
const defaults = new Set(links.default_harnesses);
// The page opens Filtered: published harnesses in the default set. All shows the rest.
const publicHarnesses = Object.entries(harnessData.agents).filter(
  ([id, harness]) => harness.visibility !== "preview" && defaults.has(id),
);
const rows = Object.values(harnessData.data);
const repo = harnessData.repo;

// The harness table is its own file, named by the index but never one of its specs.
assert(specIndex.harnesses?.file === "harnesses.json", "specs.json does not point at harnesses.json");
assert(
  specIndex.specs.every((entry) => entry.file !== specIndex.harnesses.file),
  "harnesses.json is listed as a framework spec",
);
for (const aggregate of [database]) {
  assert(
    harnessIds.every((id) => !(id in aggregate.agents)),
    "a harness leaked into the framework aggregate",
  );
  assert(
    rows.every((row) => !(row.id in aggregate.data)),
    "a harness row leaked into the framework aggregate",
  );
}

const server = await preview({
  configFile: "vite.config.ts",
  preview: {host: "127.0.0.1", port: 0, strictPort: false},
});
const address = server.httpServer.address();
assert(address && typeof address !== "string", "Preview server did not bind");
const baseUrl = `http://127.0.0.1:${address.port}/`;
const browser = await chromium.launch({executablePath: chromePath, headless: true});
const consoleErrors = [];

const defaultIds = harnessIds.filter((id) => defaults.has(id));
const publicIds = publicHarnesses.map(([id]) => id);
const published = (harnessId) => harnessData.agents[harnessId].visibility !== "preview";
// A published harness links to the copy of its capture shipped with the site (the
// repository is private); a preview one links to the capture in the repository.
const captureLink = (harnessId, cell) =>
  published(harnessId)
    ? `./${cell.capture}`
    : `${repo}/blob/main/${cell.capture}${cell.line ? `#L${cell.line}` : ""}`;
const longValue = (cell) => (cell.value ?? "").length > 16;

function checkCapture(label, rowId, harnessId, cell) {
  const expected = harnessData.data[rowId].cells[harnessId];
  assert(cell, `${label}: ${rowId} / ${harnessId} has no link`);
  assert(
    cell.href === captureLink(harnessId, expected),
    `${label}: ${rowId} / ${harnessId} links to ${cell.href}, not its capture`,
  );
  if (published(harnessId)) {
    assert(existsSync(`docs/${expected.capture}`), `${label}: ${expected.capture} is not shipped with the site`);
  }
  assert(
    expected.capture.startsWith(
      `data/harnesses/measurements/${harnessData.agents[harnessId].measured_live ? "wire-live" : "wire"}/`,
    ),
    `${label}: ${rowId} / ${harnessId} capture is outside the wire snapshot`,
  );
  assert(
    cell.label === expected.label && cell.code === expected.code,
    `${label}: ${rowId} / ${harnessId} shows ${cell.label}, data says ${expected.label}`,
  );
}

async function assertFits(page, label) {
  const metrics = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    innerWidth: window.innerWidth,
    frameworkGrids: document.querySelectorAll(".support-grid, .stats-matrix").length,
  }));
  assert(
    metrics.scrollWidth === metrics.innerWidth,
    `${label}: document is ${metrics.scrollWidth}px wide in a ${metrics.innerWidth}px viewport`,
  );
  assert(metrics.frameworkGrids === 0, `${label}: an agents page rendered a framework grid`);
}

// The index: one bar per agent linking to its page, one bar per behaviour linking to
// its page, grouped by area, with the legend's words.
async function checkAgentsIndex(page, label, expectedAgentIds) {
  await page.locator(".agents-page .agents-scores").waitFor();
  await assertFits(page, label);
  const index = await page.evaluate(() => ({
    title: document.querySelector(".agents-page h1")?.textContent,
    agents: [...document.querySelectorAll(".agents-scores tr[data-agent-id]")].map((li) => ({
      id: li.getAttribute("data-agent-id"),
      href: li.querySelector("a")?.getAttribute("href"),
      bar: Boolean(li.querySelector(".home-framework-score-bars")),
      version: li.getAttribute("data-summary-version"),
      newer: li.querySelector(".newer-release")?.getAttribute("data-newer-version") ?? null,
      dated: Boolean(li.querySelector("td time")),
      unmeasured: li.querySelector(".harness-unmeasured-reason")?.textContent ?? null,
    })),
    areas: [...document.querySelectorAll(".agents-behaviours")].map((section) => ({
      area: section.getAttribute("aria-label"),
      rows: [...section.querySelectorAll(".agents-behaviour[data-row-id]")].map((li) => ({
        id: li.getAttribute("data-row-id"),
        href: li.querySelector("a")?.getAttribute("href"),
        bar: Boolean(li.querySelector(".home-framework-score-bars")),
      })),
    })),
    terms: [...document.querySelectorAll(".harness-legend dt")].map((dt) => dt.textContent),
    codes: document.querySelectorAll(".harness-legend-codes li[data-code]").length,
  }));
  assert(index.title === "Agents", `${label}: the agents index is titled "${index.title}"`);
  // An agent with no measurement keeps its place but shows "Not measured yet" and the
  // reason instead of a date and a bar of results.
  for (const agent of index.agents) {
    const reason = harnessData.agents[agent.id].measurement_blocked ?? null;
    assert(
      agent.unmeasured === reason && agent.dated === !reason,
      `${label}: ${agent.id} shows ${JSON.stringify(agent)}; expected reason ${JSON.stringify(reason)}`,
    );
  }
  const blocked = expectedAgentIds.filter((id) => harnessData.agents[id].measurement_blocked);
  assert(
    index.agents.some((agent) => agent.unmeasured) === blocked.length > 0,
    `${label}: the index shows ${index.agents.filter((agent) => agent.unmeasured).length} unmeasured agents, but the listed agents with no session are ${blocked.join(", ") || "none"}`,
  );
  assert(
    JSON.stringify(index.agents.map(({id}) => id)) === JSON.stringify(expectedAgentIds),
    `${label}: index lists ${index.agents.map(({id}) => id)}, expected ${expectedAgentIds}`,
  );
  for (const agent of index.agents) {
    assert(
      agent.href === `#/agent/${agent.id}` && agent.bar &&
        agent.version === harnessData.agents[agent.id].current_version,
      `${label}: ${agent.id} index entry is incomplete: ${JSON.stringify(agent)}`,
    );
    assert(
      agent.newer === newerRelease(harnessData.agents[agent.id]),
      `${label}: ${agent.id} marks ${agent.newer} as the newer release; expected ${newerRelease(harnessData.agents[agent.id])}`,
    );
  }
  assert(
    JSON.stringify(index.areas.map(({area}) => area)) === JSON.stringify(harnessData.areas),
    `${label}: behaviour groups ${index.areas.map(({area}) => area)} are not the data's areas`,
  );
  const listed = index.areas.flatMap(({rows}) => rows);
  assert(listed.length === rows.length, `${label}: ${listed.length} of ${rows.length} behaviours listed`);
  for (const entry of listed) {
    assert(
      entry.href === `#/agents/${entry.id}` && entry.bar && harnessData.data[entry.id],
      `${label}: behaviour entry ${JSON.stringify(entry)} is incomplete`,
    );
  }
  for (const word of harnessData.vocabulary) {
    assert(index.terms.includes(word), `${label}: vocabulary is missing "${word}"`);
  }
  assert(
    index.codes === Object.keys(harnessData.support_codes).length,
    `${label}: the legend shows ${index.codes} codes`,
  );
}

// A behaviour page: the row as one table of agents, measured-on in the column titles,
// compact cells, numbered notes with summaries over closed evidence, the legend.
async function checkBehaviourPage(page, label, rowId, expectedAgentIds) {
  const row = harnessData.data[rowId];
  const module = page.locator(`.agent-row-module[data-row-id="${rowId}"]`);
  await module.locator(".harness-table").waitFor();
  await assertFits(page, label);
  const shown = await module.evaluate((root) => ({
    title: root.querySelector("h2")?.textContent,
    tables: root.querySelectorAll(".harness-table").length,
    headers: [...root.querySelectorAll(".harness-table thead th[data-harness-id]")].map((th) => ({
      id: th.getAttribute("data-harness-id"),
      href: th.querySelector("a.agent-name")?.getAttribute("href"),
      measured: th.querySelector(".harness-measured-on time")?.getAttribute("datetime") ?? null,
      notMeasured: th.querySelector(".harness-not-measured")?.textContent ?? null,
      revision: th.querySelector(".harness-revision")?.textContent ?? null,
      title: th.getAttribute("title"),
    })),
    revisionNote: root.querySelector(".harness-revision-note")?.textContent ?? "",
    rowIds: [...root.querySelectorAll(".harness-table tbody tr:not(.harness-newer-row)")].map((tr) => tr.getAttribute("data-row-id")),
    newer: [...root.querySelectorAll(".harness-table tbody tr.harness-newer-row td")].map(
      (td) => td.querySelector(".harness-newer-cell")?.getAttribute("data-version") ?? null,
    ),
    newerNote: root.querySelector(".harness-newer-note")?.textContent ?? "",
    deprecated: root.querySelector(".behaviour-deprecated")?.textContent ?? null,
    specHref: root.querySelector(".feature-title-line .spec-link")?.getAttribute("href") ?? null,
    cells: [...root.querySelectorAll(".harness-table tbody tr[data-row-id] td")].map((td) => {
      const cell = td.querySelector(".harness-cell");
      const capture = td.querySelector("a.harness-cell-capture");
      return cell
        ? {
            href: capture?.getAttribute("href") ?? null,
            label: cell.dataset.label,
            code: cell.dataset.code,
            text: cell.textContent.trim(),
            marker: cell.querySelector(".support-note-ref")?.getAttribute("href") ?? null,
            longNote: Boolean(cell.querySelector(".harness-cell-note")),
          }
        : null;
    }),
    notes: [...root.querySelectorAll(".agent-handling-notes li")].map((item) => ({
      id: `#${item.id}`,
      summary: item.querySelector(".harness-note-summary")?.textContent ?? null,
      evidence: item.querySelector("details.harness-evidence > summary")?.textContent ?? null,
      open: Boolean(item.querySelector("details.harness-evidence[open]")),
    })),
    observed: root.querySelector(".feature-observed-summary strong")?.textContent ?? "",
    legendTerms: root.querySelectorAll(".harness-legend dt").length,
    baseline: root.querySelector(".measurement-baseline")?.textContent ?? "",
  }));
  assert(shown.title === row.title && shown.tables === 1, `${label}: ${rowId} page lacks its one table`);
  assert(
    JSON.stringify(shown.headers.map(({id}) => id)) === JSON.stringify(expectedAgentIds),
    `${label}: ${rowId} columns ${shown.headers.map(({id}) => id)} do not match ${expectedAgentIds}`,
  );
  for (const header of shown.headers) {
    const agent = harnessData.agents[header.id];
    const measured = agent.measurement_blocked
      ? null
      : row.measured?.measured_on?.[header.id] ?? agent.measured_on ?? null;
    assert(
      header.href === `#/agent/${header.id}` && header.measured === measured &&
        (!measured || header.title.includes("measured")) &&
        (!agent.measurement_blocked ||
          (header.notMeasured === "Not measured yet" && header.title.includes(agent.measurement_blocked))),
      `${label}: ${rowId} / ${header.id} column title is ${JSON.stringify(header)}`,
    );
    // The revision under the name is what the agent settled on against the 2026-07-28 server.
    const negotiated = harnessData.data["h-negotiated-revision"].by_revision?.[header.id]?.["2026-07-28"]?.value;
    const settled = [...new Set([...(negotiated ?? "").matchAll(/(\d{4}-\d{2}-\d{2}) \(requested/g)].map((m) => m[1]))];
    const revision = agent.measurement_blocked || !settled.length ? null : `MCP ${settled.join(" / ")}`;
    assert(header.revision === revision, `${label}: ${rowId} / ${header.id} shows revision ${header.revision}, expected ${revision}`);
  }
  assert(
    !shown.headers.some(({revision}) => revision) || shown.revisionNote.includes("settled on when offered 2026-07-28"),
    `${label}: ${rowId} shows revisions without saying what they are`,
  );
  assert(JSON.stringify(shown.rowIds) === JSON.stringify([rowId]), `${label}: ${rowId} table shows ${shown.rowIds}`);
  // Under each measured cell, the agent's newer stable release when there is one, unmeasured.
  const newer = expectedAgentIds.map((id) => {
    const agent = harnessData.agents[id];
    return !agent.measurement_blocked && agent.stable_version && agent.stable_version !== agent.current_version
      ? agent.stable_version
      : null;
  });
  assert(
    newer.some(Boolean)
      ? JSON.stringify(shown.newer) === JSON.stringify(newer) && shown.newerNote.includes("newest release")
      : !shown.newer.length,
    `${label}: ${rowId} newer releases ${JSON.stringify(shown.newer)}, expected ${JSON.stringify(newer)}`,
  );
  // The spec link and any deprecation come from the paired capability, not typed on the page.
  const paired = row.capability;
  assert(
    shown.specHref === (paired?.docs_url ?? null) &&
      shown.deprecated === (paired?.status === "deprecated" ? `Deprecated in MCP ${paired.revision}` : null),
    `${label}: ${rowId} spec link ${shown.specHref} / badge ${shown.deprecated} do not match ${JSON.stringify(paired)}`,
  );
  shown.cells.forEach((cell, index) => {
    const harnessId = expectedAgentIds[index];
    checkCapture(label, rowId, harnessId, cell);
    assert(
      cell.text.length < 60 && !cell.longNote,
      `${label}: ${rowId} / ${harnessId} cell is not compact: ${cell.text}`,
    );
  });
  const expectedNotes = expectedAgentIds.filter((id) => row.cells[id]?.note || longValue(row.cells[id]));
  assert(
    JSON.stringify(shown.cells.map(({marker}) => marker).filter(Boolean)) ===
      JSON.stringify(shown.notes.map(({id}) => id)) && shown.notes.length === expectedNotes.length,
    `${label}: ${rowId} note markers ${shown.cells.map(({marker}) => marker)} do not match notes ${shown.notes.map(({id}) => id)}`,
  );
  const summaries = expectedAgentIds.filter((id) => row.cells[id]?.note).map((id) => row.cells[id].summary);
  assert(
    JSON.stringify(shown.notes.filter(({summary}) => summary).map(({summary}) => summary)) ===
      JSON.stringify(summaries) &&
      shown.notes.every(({summary, evidence, open}) => !summary || (evidence === "Run evidence" && !open)),
    `${label}: ${rowId} notes do not show summaries over closed evidence: ${JSON.stringify(shown.notes)}`,
  );
  // The tally counts every shown agent once, in the row's own words.
  const counted = [...shown.observed.matchAll(/(\d+) /g)].reduce((sum, match) => sum + Number(match[1]), 0);
  assert(counted === expectedAgentIds.length, `${label}: ${rowId} tally "${shown.observed}" counts ${counted} agents`);
  assert(shown.legendTerms > 0 && shown.baseline.includes(row.area), `${label}: ${rowId} lacks its legend or baseline`);
}

// An agent page: facts, a tally over every behaviour, each behaviour's cell with its
// note, and a reproduce command naming only that agent.
async function checkAgentPage(page, label, harnessId) {
  const harness = harnessData.agents[harnessId];
  const root = page.locator(`.agent-page[data-agent-id="${harnessId}"]`);
  await root.locator(".framework-summary").waitFor();
  await assertFits(page, label);
  const gaps = rows.filter((row) => row.cells[harnessId]?.code !== "y").map((row) => row.id);
  const shownGaps = await root.locator(".framework-row[data-row-id]").evaluateAll((items) =>
    items.map((item) => item.getAttribute("data-row-id")),
  );
  assert(
    JSON.stringify([...shownGaps].sort()) === JSON.stringify([...gaps].sort()),
    `${label}: ${harnessId} opens on ${shownGaps}, its gaps are ${gaps}`,
  );
  await root.getByRole("radio", {name: /All behaviours/}).click();
  const shown = await root.evaluate((node, id) => ({
    title: node.querySelector("h1")?.textContent,
    facts: [...node.querySelectorAll(".framework-facts dt")].map((dt) => dt.textContent),
    version: node.querySelector(".framework-facts dd strong")?.textContent,
    measuredOn: node.querySelector(".framework-facts time")?.getAttribute("datetime") ?? null,
    notMeasured: node.querySelector(".framework-facts .harness-not-measured")?.textContent ?? null,
    rows: [...node.querySelectorAll(".framework-row[data-row-id]")].map((item) => {
      const cell = item.querySelector("a.harness-cell");
      return {
        id: item.getAttribute("data-row-id"),
        titleHref: item.querySelector(".framework-row-title")?.getAttribute("href"),
        href: cell?.getAttribute("href") ?? null,
        label: cell?.dataset.label,
        code: cell?.dataset.code,
        support: item.getAttribute("data-support"),
        noteText: item.querySelector(".framework-row-notes li")?.textContent ?? null,
        evidence: item.querySelector("details.harness-evidence > summary")?.textContent ?? null,
        open: Boolean(item.querySelector("details.harness-evidence[open]")),
      };
    }),
    command: node.querySelector(".framework-reproduce code, .framework-reproduce pre")?.textContent ?? "",
    legendTerms: node.querySelectorAll(".harness-legend dt").length,
    summary: node.querySelector(".framework-summary-line")?.textContent ?? "",
  }), harnessId);
  assert(
    shown.title.startsWith(harness.name) && shown.version === harness.current_version &&
      ["Measured version", "Measured on", "Vendor", "Approval mode"].every((fact) => shown.facts.includes(fact)),
    `${label}: ${harnessId} facts are ${JSON.stringify([shown.title, shown.version, shown.facts])}`,
  );
  assert(
    harness.measurement_blocked
      ? shown.measuredOn === null && shown.notMeasured === `Not measured yet. ${harness.measurement_blocked}`
      : shown.measuredOn === harness.measured_on && shown.notMeasured === null,
    `${label}: ${harnessId} measured-on reads ${JSON.stringify([shown.measuredOn, shown.notMeasured])}`,
  );
  assert(shown.rows.length === rows.length, `${label}: ${harnessId} shows ${shown.rows.length} of ${rows.length} behaviours`);
  for (const entry of shown.rows) {
    const cell = harnessData.data[entry.id].cells[harnessId];
    checkCapture(label, entry.id, harnessId, entry);
    assert(
      entry.titleHref === `#/agents/${entry.id}` && entry.support === cell.code,
      `${label}: ${harnessId} / ${entry.id} row is ${JSON.stringify(entry)}`,
    );
    if (cell.note) {
      assert(
        entry.noteText?.includes(cell.summary ?? "") && entry.evidence === "Run evidence" && !entry.open,
        `${label}: ${harnessId} / ${entry.id} note is ${JSON.stringify(entry)}`,
      );
    } else if (longValue(cell)) {
      assert(entry.noteText?.includes(cell.value), `${label}: ${harnessId} / ${entry.id} long value is not under the cell`);
    } else {
      assert(entry.noteText === null, `${label}: ${harnessId} / ${entry.id} has a note the data lacks`);
    }
  }
  assert(shown.summary.startsWith(`${rows.length} behaviours`), `${label}: ${harnessId} summary "${shown.summary}"`);
  const command = harnessData.reproduce_by_agent?.[harnessId];
  assert(!command || shown.command.includes(command), `${label}: ${harnessId} reproduce command is "${shown.command}"`);
  assert(shown.legendTerms > 0, `${label}: ${harnessId} page has no legend`);
}

const behaviourRows = Object.keys(harnessData.data);
const sampleRow = "h-destructive-blocked-by-default" in harnessData.data
  ? "h-destructive-blocked-by-default"
  : behaviourRows[0];

try {
  for (const width of widths) {
    const page = await browser.newPage({viewport: {width, height: 900}});
    page.on("console", (message) => {
      if (message.type() === "error") {
        consoleErrors.push(`${width}px: ${message.text()}`);
      }
    });
    page.on("pageerror", (error) => consoleErrors.push(`${width}px: ${error.message}`));

    // The home page's Measured results panel opens on frameworks; its Agents tab carries the one
    // way in to the agents index.
    await page.goto(`${baseUrl}#/`, {waitUntil: "networkidle"});
    const frameworkRows = await page.locator(".home-framework-score[data-agent-id]").count();
    // The home list shows the default frameworks (data/framework_defaults.json).
    const defaultFrameworks = JSON.parse(
      await readFile("data/framework_defaults.json", "utf8"),
    ).default_frameworks;
    assert(
      frameworkRows === defaultFrameworks.filter(
        (id) => database.agents[id] && database.agents[id].visibility !== "preview",
      ).length,
      `${width}px: home framework scores changed (${frameworkRows} rows)`,
    );
    await page.getByRole("radio", {name: "Agents"}).click();
    const homeLinks = page.locator('main a[href="#/agents"]');
    assert((await homeLinks.count()) === 1, `${width}px: home page should link to #/agents once`);
    await homeLinks.click();
    await page.waitForURL(/#\/agents$/);
    await checkAgentsIndex(page, `${width}px`, publicIds);
    await page.screenshot({path: `/tmp/mcp-mirror-agents-${width}.png`, fullPage: true});

    // Filtered opens on the default set; All shows every published agent.
    await page.getByRole("radio", {name: "All"}).click();
    await checkAgentsIndex(
      page,
      `${width}px All`,
      harnessIds.filter((id) => published(id)),
    );

    // Every behaviour has its own page at 1440px; the sample row at every width.
    const rowsToCheck = width === 1440 ? behaviourRows : [sampleRow];
    for (const rowId of rowsToCheck) {
      await page.goto(`${baseUrl}#/agents/${rowId}`, {waitUntil: "networkidle"});
      await checkBehaviourPage(page, `${width}px`, rowId, publicIds);
    }
    await page.screenshot({path: `/tmp/mcp-mirror-agents-${sampleRow}-${width}.png`, fullPage: true});

    // Every published agent has its own page.
    for (const harnessId of width === 1440 ? publicIds : publicIds.slice(0, 1)) {
      await page.goto(`${baseUrl}#/agent/${harnessId}`, {waitUntil: "networkidle"});
      await checkAgentPage(page, `${width}px`, harnessId);
    }

    // The old address still answers with the index.
    await page.goto(`${baseUrl}#/harnesses`, {waitUntil: "networkidle"});
    await checkAgentsIndex(page, `${width}px #/harnesses`, publicIds);
    assert(
      (await page.locator('.site-nav a[aria-current="page"]').textContent()) === "Agents",
      `${width}px: the Agents tab is not active on #/harnesses`,
    );

    // The framework table is unaffected: no harness column appears on it.
    await page.goto(`${baseUrl}#/stats?view=frameworks`, {waitUntil: "networkidle"});
    const statsText = await page.locator("main").innerText();
    for (const [, harness] of publicHarnesses) {
      assert(
        !statsText.includes(`${harness.name} ${harness.current_version}`),
        `${width}px: ${harness.name} appears on the framework table`,
      );
    }
    await page.close();
  }

  // A cell the harness answers differently per transport shows one band per transport
  // (preview agents included, with the flag).
  const splits = rows.flatMap((row) =>
    Object.entries(row.cells)
      .filter(([, cell]) => cell.transports?.length)
      .map(([harnessId, cell]) => ({row: row.id, harnessId, cell})),
  );
  {
    const previewPage = await browser.newPage({viewport: {width: 1440, height: 900}});
    await previewPage.goto(`${baseUrl}?preview=1#/agents`, {waitUntil: "networkidle"});
    await checkAgentsIndex(previewPage, "preview", defaultIds);
    await previewPage.getByRole("radio", {name: "All"}).click();
    await checkAgentsIndex(previewPage, "preview All", harnessIds);
    for (const rowId of new Set(splits.map(({row}) => row))) {
      await previewPage.goto(`${baseUrl}?preview=1#/agents/${rowId}`, {waitUntil: "networkidle"});
      await checkBehaviourPage(previewPage, "preview", rowId, defaultIds);
      await previewPage.getByRole("radio", {name: "All"}).click();
      await checkBehaviourPage(previewPage, "preview All", rowId, harnessIds);
      for (const {harnessId, cell} of splits.filter((split) => split.row === rowId)) {
        const column = harnessIds.indexOf(harnessId);
        const bands = await previewPage
          .locator(`tr[data-row-id="${rowId}"] td`)
          .nth(column)
          .locator(".harness-cell-transport")
          .evaluateAll((nodes) => nodes.map((node) => [node.dataset.transport, node.dataset.code]));
        assert(
          JSON.stringify(bands) === JSON.stringify(cell.transports.map((t) => [t.transport, t.code])),
          `${rowId} / ${harnessId}: transport bands ${JSON.stringify(bands)} do not match the data`,
        );
      }
    }
    await previewPage.close();
  }

  // Every cell with a note has a one-line summary (the page shows it over the evidence).
  const summarized = rows.flatMap((row) =>
    Object.entries(row.cells)
      .filter(([, cell]) => cell.note)
      .map(([harnessId, cell]) => ({row, harnessId, cell})),
  );
  assert(
    summarized.every(({cell}) => cell.summary),
    "a harness cell with a note has no summary",
  );

  // A paired capability page gets the harness row as its own table below the
  // framework grid, titled with a link to the behaviour page; an unpaired one gets none.
  const capabilityPage = await browser.newPage({viewport: {width: 1440, height: 900}});
  for (const [capabilityId, rowId] of Object.entries(links.capabilities)) {
    if (!(capabilityId in database.data)) continue;
    await capabilityPage.goto(`${baseUrl}?preview=1#/cap/${capabilityId}`, {
      waitUntil: "networkidle",
    });
    const section = capabilityPage.locator(`.agent-handling[data-harness-row="${rowId}"]`);
    await section.waitFor();
    const row = harnessData.data[rowId];
    const placement = await capabilityPage.evaluate(() => {
      const grid = document.querySelector(".support-grid");
      const agents = document.querySelector(".agent-handling");
      return {
        below: grid.getBoundingClientRect().bottom <= agents.getBoundingClientRect().top,
        mixed: Boolean(grid.querySelector("[data-harness-id]")),
      };
    });
    assert(placement.below && !placement.mixed, `${capabilityId}: harness table is not apart from and below the grid`);
    assert(
      (await section.locator("h2").first().textContent()) === "How agents handle this" &&
        (await section.locator(".agent-handling-question").textContent()).includes(
          row.question || row.observe,
        ) &&
        (await section.locator(`a.agent-handling-row-link[href="#/agents/${rowId}"]`).count()) === 1 &&
        (await section.locator(".harness-legend dt").count()) > 0,
      `${capabilityId}: harness table lacks its title, link to #/agents/${rowId}, question or legend`,
    );
    assert(
      JSON.stringify(
        await section
          .locator("thead th[data-harness-id]")
          .evaluateAll((ths) => ths.map((th) => th.getAttribute("data-harness-id"))),
      ) === JSON.stringify(defaultIds),
      `${capabilityId}: harness table does not open on the default harnesses`,
    );
    // Cells stay as compact as framework cells: label, #n and a capture link. The
    // evidence is a numbered note under the table, one per cell that has one.
    const compact = await section.locator("tbody .harness-cell").evaluateAll((cells) =>
      cells.map((cell) => ({
        text: cell.textContent.trim(),
        marker: cell.querySelector(".support-note-ref")?.getAttribute("href") ?? null,
        capture: Boolean(cell.querySelector("a.harness-cell-capture[href]")),
        longNote: Boolean(cell.querySelector(".harness-cell-note")),
      })),
    );
    const noteIds = await section
      .locator(".agent-handling-notes li")
      .evaluateAll((items) => items.map((item) => `#${item.id}`));
    const expectedNotes = defaultIds.filter((id) => row.cells[id]?.note || longValue(row.cells[id]));
    assert(
      compact.every(({text, capture, longNote}) => text.length < 60 && capture && !longNote),
      `${capabilityId}: an agent cell is not compact: ${JSON.stringify(compact)}`,
    );
    assert(
      JSON.stringify(compact.map(({marker}) => marker).filter(Boolean)) ===
        JSON.stringify(noteIds) && noteIds.length === expectedNotes.length,
      `${capabilityId}: note markers ${compact.map(({marker}) => marker)} do not match notes ${noteIds}`,
    );
    const noteParts = await section.locator(".agent-handling-notes li").evaluateAll((items) =>
      items.map((item) => ({
        summary: item.querySelector(".harness-note-summary")?.textContent ?? null,
        evidence: item.querySelector("details.harness-evidence > summary")?.textContent ?? null,
        open: Boolean(item.querySelector("details.harness-evidence[open]")),
      })),
    );
    const summaries = defaultIds.filter((id) => row.cells[id]?.note).map((id) => row.cells[id].summary);
    assert(
      JSON.stringify(noteParts.filter(({summary}) => summary).map(({summary}) => summary)) ===
        JSON.stringify(summaries) &&
        noteParts.every(({summary, evidence, open}) => !summary || (evidence === "Run evidence" && !open)),
      `${capabilityId}: agent notes do not show summaries over closed evidence: ${JSON.stringify(noteParts)}`,
    );
  }
  await capabilityPage.goto(`${baseUrl}?preview=1#/cap/tool-icons`, {waitUntil: "networkidle"});
  assert(
    (await capabilityPage.locator(".agent-handling").count()) === 0,
    "tool-icons has no paired harness row but shows one",
  );
  await capabilityPage.close();

  // The standalone file carries the section too, agent pages included.
  const offline = await browser.newPage({viewport: {width: 1440, height: 900}});
  offline.on("console", (message) => {
    if (message.type() === "error") {
      consoleErrors.push(`standalone: ${message.text()}`);
    }
  });
  const standalone = pathToFileURL(path.resolve("docs", "standalone.html")).href;
  await offline.goto(`${standalone}#/agents`, {waitUntil: "load"});
  await checkAgentsIndex(offline, "standalone", publicIds);
  await offline.goto(`${standalone}#/agents/${sampleRow}`, {waitUntil: "load"});
  await checkBehaviourPage(offline, "standalone", sampleRow, publicIds);
  await offline.goto(`${standalone}#/agent/${publicIds[0]}`, {waitUntil: "load"});
  await checkAgentPage(offline, "standalone", publicIds[0]);
  await offline.close();

  assert(consoleErrors.length === 0, `Browser console errors: ${consoleErrors.join(" | ")}`);
  console.log(
    `agents smoke passed: ${widths.join(", ")}px and standalone, ${rows.length} behaviour pages x ${publicHarnesses.length} agent(s), ${publicIds.length} agent pages, every cell linked to its capture, ${splits.length} per-transport split(s) rendered`,
  );
} finally {
  await browser.close();
  await new Promise((resolve, reject) => {
    server.httpServer.close((error) => (error ? reject(error) : resolve()));
  });
}
