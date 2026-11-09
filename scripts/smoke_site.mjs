import {chromium} from "playwright-core";
import {preview} from "vite";
import {copyFileSync, mkdtempSync, readFileSync} from "node:fs";
import {tmpdir} from "node:os";
import path from "node:path";
import {pathToFileURL} from "node:url";

// Framework counts come from the built data, so adding a framework never needs this file.
const SPEC_INDEX = JSON.parse(readFileSync(path.join("docs", "data", "specs.json"), "utf8"));
const reachTextFor = (id) => {
  const spec = SPEC_INDEX.specs.find((entry) => entry.id === id);
  return `${spec.measured_frameworks} of ${spec.total_frameworks} frameworks reach this revision`;
};

// Derived from the built dataset rather than hardcoded, so adding a protocol feature to
// the surface does not require editing an expected count here.
const database = JSON.parse(
  readFileSync(`site/public/data/data-${SPEC_INDEX.default_spec}.json`, "utf8"),
);
const capabilityCount = Object.values(database.data).filter(
  (capability) => capability.shown !== false,
).length;
const coverage = database.coverage;
const news = JSON.parse(
  readFileSync("site/src/data/news.json", "utf8"),
);

// The version each snapshot measures at moves whenever the runner manifests are
// repinned, so read it rather than restate it.
// LangChain is in the home page's default list (data/framework_defaults.json).
const legacyLangchainVersion = JSON.parse(
  readFileSync("site/public/data/data-2025-11-25.json", "utf8"),
).agents.langchain.current_version;

const chromePath =
  process.env.CHROME_PATH ??
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

const server = await preview({
  configFile: "vite.config.ts",
  preview: {
    host: "127.0.0.1",
    port: 0,
    strictPort: false,
  },
});

const address = server.httpServer.address();
assert(address && typeof address !== "string", "Preview server did not bind a TCP port");
const baseUrl = `http://127.0.0.1:${address.port}/`;
const browser = await chromium.launch({executablePath: chromePath, headless: true});
const consoleErrors = [];

try {
  const mobile = await browser.newPage({viewport: {width: 390, height: 844}});
  mobile.on("console", (message) => {
    if (message.type() === "error") {
      consoleErrors.push(message.text());
    }
  });
  await mobile.goto(baseUrl, {waitUntil: "networkidle"});
  const widths = await mobile.evaluate(() => ({
    viewport: window.innerWidth,
    document: document.documentElement.scrollWidth,
  }));
  assert(
    widths.document <= widths.viewport,
    `Mobile layout overflows: ${widths.document}px document in ${widths.viewport}px viewport`,
  );

  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  page.on("console", (message) => {
    if (message.type() === "error") {
      consoleErrors.push(message.text());
    }
  });
  await page.goto(baseUrl, {waitUntil: "networkidle"});

  const reachText = reachTextFor(SPEC_INDEX.default_spec);
  const specSelector = page.getByLabel("MCP specification revision");
  assert(
    (await specSelector.inputValue()) === SPEC_INDEX.default_spec &&
      (await page.getByText(reachText).count()) ===
        1,
    "The current MCP snapshot is not explicit",
  );
  await page
    .getByRole("navigation", {name: "Primary navigation"})
    .getByRole("link", {name: "Stats", exact: true})
    .waitFor();
  const firstHeadline = page
    .locator('.news-crawl-group:not([data-clone="true"]) .news-ticker-link')
    .first();
  const featureHref = await firstHeadline.getAttribute("href");
  assert(
    featureHref === news[0].url,
    `News crawl has an invalid newest destination: ${featureHref}`,
  );
  await page.goto(`${baseUrl}#/cap/destructive-hint`, {waitUntil: "networkidle"});
  await page.locator(".feature-module.is-detail").waitFor();
  // A row no scan covers has every cell unmeasured, so none of its releases count as tested.
  const unmeasuredId = Object.keys(database.data).find(
    (id) => database.data[id].shown !== false && !database.data[id].measured.run_date,
  );
  if (unmeasuredId) {
    await page.goto(`${baseUrl}#/cap/${unmeasuredId}`, {waitUntil: "networkidle"});
    const observed = await page.locator(".feature-observed-summary small").innerText();
    assert(
      observed.startsWith("0 of "),
      `Unmeasured capability ${unmeasuredId} claims tested releases: ${observed}`,
    );
  }
  // A ?note= link opens on that note, here one only a framework outside the defaults cites.
  const defaults = new Set(
    JSON.parse(readFileSync("data/framework_defaults.json", "utf8")).default_frameworks,
  );
  const linked = Object.entries(database.data)
    .flatMap(([id, capability]) =>
      Object.keys(capability.notes_by_num ?? {}).map((note) => {
        const citers = Object.entries(capability.stats)
          .filter(([, versions]) =>
            Object.values(versions).some((raw) => raw.split(" ").slice(1).includes(`#${note}`)),
          )
          .map(([agentId]) => agentId);
        return {id, note, citers};
      }),
    )
    .find(({citers}) => citers.length && citers.every((agentId) => !defaults.has(agentId)));
  if (linked) {
    await page.goto(`${baseUrl}#/cap/${linked.id}?note=${linked.note}`, {waitUntil: "networkidle"});
    await page.waitForFunction(
      (note) => document.activeElement?.id.endsWith(`-note-${note}`),
      linked.note,
      {timeout: 5000},
    );
  }
  await page.goto(baseUrl, {waitUntil: "networkidle"});

  const homeSearch = page.getByRole("searchbox", {name: "Search capabilities and agent behaviours"});
  await homeSearch.fill("hint");
  // A search covers both sections and labels which is which.
  await page.locator('.search-results-group[data-results="capabilities"]').waitFor();
  assert(
    (await page.locator('.search-results-group[data-results="behaviours"] .home-behaviour-score').count()) > 0 &&
      (await page
        .locator('.search-results-group[data-results="behaviours"] .home-behaviour-score a')
        .first()
        .getAttribute("href"))?.startsWith("#/agents/"),
    "Searching did not list matching agent behaviours beside the capabilities",
  );
  await page
    .getByRole("navigation", {name: "Primary navigation"})
    .getByRole("link", {name: "Frameworks"})
    .click();
  await page.locator(".frameworks-page").waitFor();
  assert(
    page.url().endsWith("#/frameworks") &&
      (await homeSearch.inputValue()) === "" &&
      (await page.locator(".home-section").count()) === 0 &&
      (await page.locator(".framework-directory tbody tr[data-agent-id]").count()) ===
        Object.values(database.agents).filter((agent) => agent.visibility !== "preview").length,
    "The Frameworks tab did not open the framework directory",
  );

  assert(
    coverage &&
      (await page
        .getByText(`${coverage.measured} of ${coverage.total} capabilities measured.`, {exact: false})
        .count()) === 1,
    "The framework directory does not say how many capabilities are measured",
  );
  await homeSearch.focus();
  await page.locator(".recent-searches").waitFor();
  await page.getByRole("button", {name: "Settings"}).click();
  const settingsDialog = page.getByRole("dialog", {name: "Settings"});
  await settingsDialog.waitFor();
  assert(
    (await settingsDialog.getByLabel("MCP protocol snapshot").count()) === 1 &&
      (await settingsDialog.getByRole("radio", {name: "Support status"}).count()) === 1 &&
      (await page.locator(".recent-searches").count()) === 0,
    "Settings did not open accessibly or close search history",
  );
  await settingsDialog.getByRole("button", {name: "Done"}).click();

  await page
    .getByRole("navigation", {name: "Primary navigation"})
    .getByRole("link", {name: "Stats"})
    .click();
  await page.getByRole("heading", {name: "Stats", exact: true}).waitFor();
  assert(
    page.url().endsWith("#/stats") &&
      (await page.locator(".stats-matrix tbody tr").count()) ===
        capabilityCount &&
      (await page.getByRole("tab", {name: /Capabilities/}).count()) === 1 &&
      (await page.getByRole("tab", {name: /Frameworks/}).count()) === 1,
    "Stats route did not expose the exhaustive capability view",
  );
  await page
    .locator(".stats-filter-grid label")
    .filter({hasText: "Framework"})
    .locator("select")
    .selectOption("openai-agents");
  await page.getByLabel("Support state").selectOption("n");
  assert(
    page.url().includes("framework=openai-agents") &&
      page.url().includes("support=n") &&
      (await page.locator(".stats-matrix tbody tr").count()) < capabilityCount,
    "Stats filters were not reflected in the shareable URL or table",
  );
  const [csvDownload] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", {name: "Export CSV"}).click(),
  ]);
  assert(
    csvDownload.suggestedFilename().includes(`mcp-${SPEC_INDEX.default_spec}`) &&
      csvDownload.suggestedFilename().endsWith(".csv") &&
      (await page.getByRole("button", {name: "Export JSON"}).count()) === 1,
    "Stats exports are not available",
  );
  await page.getByRole("tab", {name: /Frameworks/}).click();
  assert(
    page.url().includes("view=frameworks") &&
      (await page.locator(".framework-card").count()) === SPEC_INDEX.specs.find((entry) => entry.id === SPEC_INDEX.default_spec).total_frameworks &&
      (await page.locator(".framework-measurement-note").count()) ===
        SPEC_INDEX.specs.find((entry) => entry.id === SPEC_INDEX.default_spec).total_frameworks - SPEC_INDEX.specs.find((entry) => entry.id === SPEC_INDEX.default_spec).measured_frameworks,
    "Compare frameworks did not move inside Stats",
  );
  await page.getByRole("button", {name: "Reset filters"}).click();
  await page.getByRole("tab", {name: /Capabilities/}).click();
  await page.getByRole("link", {name: "Reset Can AI use search"}).click();
  await page.locator(".home-overview").waitFor();
  // Picking the other snapshot pins it in the URL; picking the default clears it again.
  const otherSpec = SPEC_INDEX.specs.find((entry) => entry.id !== SPEC_INDEX.default_spec).id;
  await specSelector.selectOption(otherSpec);
  await page.getByText(reachTextFor(otherSpec)).waitFor();
  assert(page.url().includes(`?spec=${otherSpec}`), "The protocol selector did not pin the other snapshot");
  await specSelector.selectOption("2025-11-25");
  await page.getByText(reachTextFor("2025-11-25")).waitFor();
  assert(
    (await page
      .locator('.home-framework-score[data-agent-id="langchain"]')
      .getAttribute("data-summary-version")) === legacyLangchainVersion,
    "The protocol selector did not load the complete 2025-11-25 snapshot",
  );
  // Back restores the earlier entry's URL, and its snapshot with it.
  await page.evaluate(() => {
    window.location.hash = "#/cap/enum";
  });
  await page.locator(".feature-module.is-detail").waitFor();
  await specSelector.selectOption(otherSpec);
  await page.getByText(reachTextFor(otherSpec)).waitFor();
  await page.goBack();
  await page.locator(".home-overview").waitFor();
  await page.getByText(reachTextFor(SPEC_INDEX.default_spec)).waitFor({timeout: 5000});
  assert(!page.url().includes("spec="), `Back left a pinned snapshot in the URL: ${page.url()}`);

  await homeSearch.fill("format");
  await page.getByRole("link", {name: "Reset Can AI use search"}).click();
  assert(
    page.url().endsWith("#/") && (await homeSearch.inputValue()) === "",
    "The wordmark did not reset while already on the index route",
  );
  await page
    .getByRole("navigation", {name: "Primary navigation"})
    .getByRole("link", {name: "Agents"})
    .click();
  await page.getByRole("heading", {name: "Agents", exact: true, level: 1}).waitFor();
  assert(
    (await page.locator(".spec-selector").count()) === 0 &&
      (await page.locator('.site-nav .nav-link[aria-current="page"]').textContent()) === "Agents",
    "The agents pages still carry the framework protocol snapshot bar",
  );
  await page.getByRole("link", {name: "Reset Can AI use search"}).click();
  await page.locator(".home-overview").waitFor();
  await homeSearch.fill("hint");
  await page.getByRole("link", {name: "Reset Can AI use search"}).click();
  assert(
    (await homeSearch.inputValue()) === "",
    "Search wordmark did not share the home reset behavior",
  );

  for (const recentQuery of [
    "hint",
    "schema",
    "tool",
    "object",
    "description",
    "format",
    "enum",
  ]) {
    await homeSearch.fill(recentQuery);
    await homeSearch.press("Enter");
  }
  await homeSearch.fill("HINT");
  await homeSearch.press("Enter");
  await homeSearch.fill("x");
  await homeSearch.press("Enter");
  await homeSearch.fill("zzzznotfound");
  await homeSearch.press("Enter");
  const storedRecents = await page.evaluate(() =>
    JSON.parse(localStorage.getItem("mcp-mirror:recent-searches:v1") ?? "[]"),
  );
  assert(
    storedRecents.length === 6 &&
      storedRecents[0] === "HINT" &&
      !storedRecents.includes("hint") &&
      !storedRecents.includes("x") &&
      !storedRecents.includes("zzzznotfound"),
    `Recent search normalization or eligibility failed: ${JSON.stringify(storedRecents)}`,
  );
  await page.reload({waitUntil: "networkidle"});
  const reloadedSearch = page.getByRole("searchbox", {name: "Search capabilities and agent behaviours"});
  const recentPanel = page.getByRole("region", {name: "Recent searches"});
  // The field autofocuses on load; that must not open the panel. A click does.
  assert(
    (await recentPanel.count()) === 0 || !(await recentPanel.isVisible()),
    "Recent searches opened on page load without the visitor asking",
  );
  await reloadedSearch.fill("");
  await reloadedSearch.blur();
  await reloadedSearch.click();
  await recentPanel.waitFor();
  const recentGeometry = await recentPanel.evaluate((panel) => {
    const panelRect = panel.getBoundingClientRect();
    const fieldRect = panel.closest(".capability-search-wrap").getBoundingClientRect();
    const hero = panel.closest(".search-hero-inner");
    const heroRect = hero.getBoundingClientRect();
    const visibleElement = document.elementFromPoint(
      panelRect.left + panelRect.width / 2,
      panelRect.bottom - 4,
    );
    return {
      anchored: panelRect.top >= fieldRect.bottom,
      contained:
        panelRect.left >= 0 && panelRect.right <= document.documentElement.clientWidth,
      extendsPastHero: panelRect.bottom > heroRect.bottom,
      heroOverflow: getComputedStyle(hero).overflow,
      visiblePastHero: panel.contains(visibleElement),
    };
  });
  assert(
    recentGeometry.anchored &&
      recentGeometry.contained &&
      recentGeometry.extendsPastHero &&
      recentGeometry.heroOverflow === "visible" &&
      recentGeometry.visiblePastHero,
    `Recent searches panel is not safely anchored: ${JSON.stringify(recentGeometry)}`,
  );
  await recentPanel.getByRole("button", {name: "HINT", exact: true}).focus();
  await recentPanel.getByRole("button", {name: "HINT", exact: true}).press("Enter");
  assert(
    (await reloadedSearch.inputValue()) === "HINT",
    "Keyboard activation did not run a recent search",
  );
  await page.getByRole("link", {name: "Reset Can AI use search"}).click();
  await reloadedSearch.click();
  await recentPanel.getByRole("button", {name: /Remove recent search HINT/}).click();
  assert(
    !(await page.evaluate(() =>
      JSON.parse(localStorage.getItem("mcp-mirror:recent-searches:v1") ?? "[]"),
    )).includes("HINT"),
    "Per-item recent search removal failed",
  );
  await recentPanel.getByRole("button", {name: "Clear recent searches"}).click();
  assert(
    (await page.evaluate(() =>
      JSON.parse(localStorage.getItem("mcp-mirror:recent-searches:v1") ?? "[]"),
    )).length === 0,
    "Clear recent searches failed",
  );

  const searchBox = page.getByRole("searchbox", {name: "Search capabilities and agent behaviours"});
  const readSearchStyle = () =>
    searchBox.evaluate((node) => {
      const input = window.getComputedStyle(node);
      const group = window.getComputedStyle(node.closest(".search-field__group"));
      return {
        inputBackground: input.backgroundColor,
        inputBoxShadow: input.boxShadow,
        inputOutline: input.outlineStyle,
        groupBackground: group.backgroundColor,
        groupBorderBottomColor: group.borderBottomColor,
        groupBoxShadow: group.boxShadow,
        groupOutline: group.outlineStyle,
      };
    });
  await searchBox.evaluate((node) => node.blur());
  const unfocusedSearchStyle = await readSearchStyle();
  await searchBox.focus();
  const focusedSearchStyle = await readSearchStyle();
  const isTransparent = (value) =>
    value === "rgba(0, 0, 0, 0)" || value === "transparent";
  assert(
    [
      unfocusedSearchStyle,
      focusedSearchStyle,
    ].every(
      (style) =>
        isTransparent(style.inputBackground) &&
        !isTransparent(style.groupBackground) &&
        style.inputOutline === "none" &&
        style.groupOutline === "none" &&
        style.inputBoxShadow === "none",
    ),
    `Search field lost its restrained highlighted surface: ${JSON.stringify({
      unfocusedSearchStyle,
      focusedSearchStyle,
    })}`,
  );
  assert(
    focusedSearchStyle.groupBorderBottomColor !==
      unfocusedSearchStyle.groupBorderBottomColor &&
      focusedSearchStyle.groupBackground !==
        unfocusedSearchStyle.groupBackground &&
      focusedSearchStyle.groupBoxShadow === "none",
    "Search focus state is not visibly stronger",
  );
  await searchBox.fill("hint");
  const typedSearchStyle = await readSearchStyle();
  assert(
    isTransparent(typedSearchStyle.inputBackground) &&
      !isTransparent(typedSearchStyle.groupBackground) &&
      typedSearchStyle.inputOutline === "none" &&
      typedSearchStyle.groupOutline === "none" &&
      typedSearchStyle.inputBoxShadow === "none" &&
      typedSearchStyle.groupBoxShadow === "none",
    `Typed search field lost its focus highlight: ${JSON.stringify(
      typedSearchStyle,
    )}`,
  );

  const tableCount = await page.locator(".feature-module.is-search-result").count();
  assert(tableCount > 1, "Live search did not default to full feature cards");
  const tableWidths = await page
    .locator(".feature-module.is-search-result")
    .evaluateAll((nodes) =>
      nodes.map((node) => Math.round(node.getBoundingClientRect().width)),
    );
  assert(
    tableWidths.every((width) => width > 1000),
    `A full table result did not span the page: ${tableWidths.join(", ")}`,
  );
  assert(
    (await page.locator(".feature-module.is-search-result .capability-grid-section").count()) ===
      tableCount,
    "A full feature card is missing its support grid",
  );

  await page.getByRole("radio", {name: "List"}).click();
  const cardCount = await page.locator(".capability-card-compact").count();
  assert(
    tableCount === cardCount,
    `List mode changed result count from ${tableCount} to ${cardCount}`,
  );
  const cardWidths = await page
    .locator(".capability-card-compact")
    .evaluateAll((nodes) =>
      nodes.map((node) => Math.round(node.getBoundingClientRect().width)),
    );
  assert(
    cardWidths.every((width) => width < 500),
    `A compact list result expanded into a page-wide panel: ${cardWidths.join(", ")}`,
  );

  await Promise.all([
    page.waitForURL(/#\/cap\//),
    page.getByRole("link", {name: /Open measurement/}).first().click(),
  ]);
  await page.locator(".feature-module.is-detail").waitFor();
  await page.locator('button[aria-label="Star this capability"]').click();
  await page.getByRole("tab", {name: "Test a feature"}).click();
  await page.getByText(/no model call or model API key is required/i).waitFor();
  // The tested release copies the scan that produced its cell, not a placeholder server.
  const openedId = decodeURIComponent(new URL(page.url()).hash.split("/")[2]);
  if (database.data[openedId]?.reproduce?.includes("mcp-mirror scan")) {
    const command = await page.locator(".feature-test-panel .command-box").innerText();
    assert(
      !command.includes("your_server.py") && command.includes("mcp-mirror scan"),
      `Test a feature on ${openedId} offers a placeholder instead of the recorded scan: ${command}`,
    );
  }

  await page.goto(`${baseUrl}#/`, {waitUntil: "networkidle"});
  await page.getByRole("searchbox", {name: "Search capabilities and agent behaviours"}).fill("hint");
  assert(
    (await page.evaluate(() =>
      JSON.parse(localStorage.getItem("mcpMirrorStars") ?? "[]"),
    )).length > 0,
    "Star selection did not persist",
  );

  const offline = await browser.newPage({viewport: {width: 1280, height: 800}});
  offline.on("console", (message) => {
    if (message.type() === "error") {
      consoleErrors.push(message.text());
    }
  });
  // Opened from an empty directory, the way it travels, so a link that needs a
  // neighbouring file cannot pass by finding it in docs/.
  const alone = path.join(mkdtempSync(path.join(tmpdir(), "standalone-")), "standalone.html");
  copyFileSync(path.resolve("docs", "standalone.html"), alone);
  await offline.goto(pathToFileURL(alone).href, {waitUntil: "load"});
  await offline
    .getByLabel("Capability and behaviour search")
    .getByRole("link", {name: "Reset Can AI use search"})
    .waitFor();
  const localLinks = await offline
    .locator("a[href]")
    .evaluateAll((links) =>
      links.map((link) => link.href).filter((href) => href.startsWith("file:") && !href.includes("#")),
    );
  assert(localLinks.length === 0, `Standalone links point at neighbouring files: ${localLinks.slice(0, 3).join(", ")}`);
  assert(
    await offline.evaluate(
      () =>
        Boolean(window.__MCP_MIRROR_DATA__) &&
        Boolean(window.__MCP_MIRROR_SPEC_INDEX__) &&
        Object.keys(window.__MCP_MIRROR_DATASETS__ ?? {}).length === 2,
    ),
    "Standalone build did not expose both inlined protocol snapshots",
  );
  await offline
    .getByLabel("MCP specification revision")
    .selectOption("2025-11-25");
  await offline.getByText(reachTextFor("2025-11-25")).waitFor();

  // The framework page a maintainer is sent: every capability, and the full note behind
  // every cell that is not a plain yes, with nothing wider than the viewport.
  const current = JSON.parse(
    readFileSync(`site/public/data/data-${SPEC_INDEX.default_spec}.json`, "utf8"),
  );
  const vercel = current.agents.vercel_ai;
  const vercelCapabilities = Object.values(current.data).filter(
    (capability) => capability.shown !== false,
  );
  const cellOf = (capability) => {
    const match = (capability.stats.vercel_ai?.[vercel.current_version] ?? "u").match(
      /^([yanxu])((?:\s+#\d+)*)$/,
    );
    return {
      code: match?.[1] ?? "u",
      notes: (match?.[2].match(/#(\d+)/g) ?? []).map((ref) => ref.slice(1)),
    };
  };
  const gaps = vercelCapabilities.filter((capability) => cellOf(capability).code !== "y");
  const plain = (text) => text.replaceAll("`", "").replace(/\s+/g, " ").trim();
  for (const width of [390, 1440]) {
    const fw = await browser.newPage({viewport: {width, height: 900}});
    fw.on("console", (message) => {
      if (message.type() === "error") {
        consoleErrors.push(message.text());
      }
    });
    await fw.goto(`${baseUrl}?preview=1#/fw/vercel_ai`, {waitUntil: "networkidle"});
    await fw.locator(".framework-page").waitFor();
    const shownGaps = await fw.locator(".framework-row").evaluateAll((rows) =>
      rows.map((row) => ({
        id: row.getAttribute("data-capability-id"),
        support: row.getAttribute("data-support"),
        notes: Object.fromEntries(
          [...row.querySelectorAll("[data-note]")].map((note) => [
            note.getAttribute("data-note"),
            note.textContent,
          ]),
        ),
      })),
    );
    assert(
      shownGaps.length === gaps.length,
      `${width}px: /fw/vercel_ai shows ${shownGaps.length} gaps, data has ${gaps.length}`,
    );
    for (const capability of gaps) {
      const {code, notes} = cellOf(capability);
      const row = shownGaps.find((entry) => entry.id === capability.id);
      assert(row?.support === code, `${width}px: ${capability.id} is not shown as ${code}`);
      for (const note of notes) {
        assert(
          plain(row.notes[note] ?? "").includes(plain(capability.notes_by_num[note])),
          `${width}px: ${capability.id} note #${note} is missing or cut`,
        );
      }
    }
    await fw.getByText(/^All capabilities/).click();
    const allIds = await fw
      .locator(".framework-row")
      .evaluateAll((rows) => rows.map((row) => row.getAttribute("data-capability-id")));
    assert(
      allIds.length === vercelCapabilities.length &&
        vercelCapabilities.every((capability) => allIds.includes(capability.id)),
      `${width}px: /fw/vercel_ai lists ${allIds.length} of ${vercelCapabilities.length} capabilities`,
    );
    const fwWidths = await fw.evaluate(() => [
      document.documentElement.scrollWidth,
      window.innerWidth,
    ]);
    assert(
      fwWidths[0] <= fwWidths[1],
      `${width}px: /fw/vercel_ai overflows to ${fwWidths[0]}px`,
    );
    await fw.close();
  }

  // LangChain reaches 2025-11-25 and not 2026-07-28: its page opens on 2025-11-25, and the
  // unreachable snapshot shows a single banner.
  const langchain = await browser.newPage({viewport: {width: 1440, height: 900}});
  await langchain.goto(`${baseUrl}#/fw/langchain`, {waitUntil: "networkidle"});
  await langchain.locator(".framework-row").first().waitFor();
  assert(
    (await langchain.getByLabel("MCP specification revision").inputValue()) === "2025-11-25",
    `/fw/langchain opened on ${langchain.url()}, not 2025-11-25`,
  );
  assert(
    (await langchain.locator('.framework-row:not([data-support="x"])').count()) > 0,
    "/fw/langchain on 2025-11-25 shows no measured cells",
  );
  await langchain.getByLabel("MCP specification revision").selectOption("2026-07-28");
  await langchain.locator(".framework-unreachable").waitFor();
  assert(
    (await langchain.locator(".framework-row").count()) === 0,
    "/fw/langchain on 2026-07-28 still lists unreachable rows",
  );
  assert(
    (await langchain.locator(".framework-unreachable-reason").count()) === 1,
    "/fw/langchain on 2026-07-28 does not state its reason once",
  );
  assert(
    (await langchain.locator(".framework-reached-link").getAttribute("href")) ===
      "?spec=2025-11-25#/fw/langchain",
    "/fw/langchain banner does not link to the snapshot it reaches",
  );
  await langchain.close();

  const home = await browser.newPage({viewport: {width: 1440, height: 900}});
  await home.goto(baseUrl, {waitUntil: "networkidle"});
  assert(
    (await home
      .locator('.home-framework-score[data-agent-id="langchain"] a')
      .getAttribute("href")) === "#/fw/langchain",
    "The home framework list does not link to the framework page",
  );
  await home.goto(`${baseUrl}#/cap/enum`, {waitUntil: "networkidle"});
  assert(
    (await home.locator('a.agent-name[href="#/fw/langchain"]').count()) > 0,
    "Capability grid column headers do not link to the framework page",
  );
  await home.close();

  assert(
    consoleErrors.length === 0,
    `Browser console errors: ${consoleErrors.join(" | ")}`,
  );

  console.log(
    `site smoke passed: ${tableCount} full search tables, compact list toggle, detail tabs, ${widths.viewport}px mobile viewport`,
  );
} finally {
  await browser.close();
  await new Promise((resolve, reject) => {
    server.httpServer.close((error) => (error ? reject(error) : resolve()));
  });
}
