import {Card, Link, ToggleButton, ToggleButtonGroup} from "@heroui/react";
import {useState} from "react";

import {CapabilityCard} from "../components/CapabilityCard";
import {CapabilityTitle} from "../components/CapabilityText";
import {CopyCodeBlock} from "../components/CopyCodeBlock";
import type {GridMode} from "../components/CapabilityGrid";
import {FilterIcon, GridIcon, ListIcon} from "../components/Icons";
import {NewCapabilities} from "../components/NewCapabilities";
import {capabilityHref, currentSupport, frameworkHref, previewRequested, siteFileHref} from "../lib/data";
import {filteredDatabase} from "../lib/frameworks";
import {FRAMEWORK_DEFAULTS} from "../lib/frameworkSources";
import {
  agentHref,
  agentsHref,
  harnessCodeCounts,
  harnessUnmeasured,
  rowCodeCounts,
  shownHarnesses,
  visibleHarnesses,
} from "../lib/harnesses";
import {HARNESS_LINKS} from "../lib/harnessSources";
import {MCP_FACTS, popularCapabilities} from "../lib/home";
import type {
  Capability,
  HarnessDatabase,
  HarnessRow,
  MirrorDatabase,
  SpecIndex,
  SupportCode,
} from "../types";
import {AgentScoreBar, HarnessBar} from "./AgentsPage";

export type ResultsMode = "tables" | "list";

const FRAMEWORK_CODES = ["y", "a", "n", "x", "u"] as const;

function frameworkCounts(
  database: MirrorDatabase,
  capabilities: Capability[],
  agentId: string,
): Record<SupportCode, number> {
  const counts: Record<SupportCode, number> = {y: 0, a: 0, n: 0, x: 0, u: 0};
  for (const capability of capabilities) {
    counts[currentSupport(database, capability, agentId).code] += 1;
  }
  return counts;
}

/** One framework as a bar, the way the framework index lists it. */
function FrameworkScore({
  database,
  capabilities,
  agentId,
}: {
  database: MirrorDatabase;
  capabilities: Capability[];
  agentId: string;
}) {
  const agent = database.agents[agentId];
  const counts = frameworkCounts(database, capabilities, agentId);
  const total = capabilities.length || 1;
  return (
    <li
      className="home-framework-score"
      data-agent-id={agentId}
      data-summary-version={agent.current_version}
      aria-label={`${agent.name} ${agent.current_version}`}
    >
      <span className="home-framework-score-label">
        <a className="home-framework-link" href={frameworkHref(agentId)}>
          {agent.name}
        </a>{" "}
        <strong className="framework-summary-version">{agent.current_version}</strong>
        {agent.aliases_of
          ? ` via ${database.agents[agent.aliases_of]?.name ?? agent.aliases_of}`
          : ""}
        {agent.visibility === "preview" ? " (preview)" : ""}
      </span>
      <span className="home-framework-score-bars" aria-hidden="true">
        {FRAMEWORK_CODES.map((code) =>
          counts[code] ? (
            <i
              className={`home-score-segment support-${code}`}
              key={code}
              style={{width: `${(counts[code] / total) * 100}%`}}
            />
          ) : null,
        )}
      </span>
    </li>
  );
}

/** One agent behaviour as a bar over the shown agents, linking to its page. */
function BehaviourScore({
  row,
  shownIds,
}: {
  row: HarnessRow;
  shownIds: string[];
}) {
  return (
    <li className="home-framework-score home-behaviour-score" data-row-id={row.id}>
      <span className="home-framework-score-label">
        <a className="home-framework-link" href={agentsHref(row.id)}>
          {row.title}
        </a>
        <small className="agents-behaviour-question">{row.question || row.observe}</small>
      </span>
      <HarnessBar counts={rowCodeCounts(row, shownIds)} total={shownIds.length} />
    </li>
  );
}

type HomeFact = {text: string; link?: string; href?: string};

/** caniuse's "Did you know?": one line at a time, MCP first, then how to read the site. */
const homeFacts = (spec: string): HomeFact[] => [
  ...MCP_FACTS.map((fact) => ({
    text: fact.label,
    link: "Spec",
    href: `https://modelcontextprotocol.io/specification/${spec}/${fact.path}`,
  })),
  {
    text: "Every cell links to the recording it was read from, so you can check our reading yourself.",
    link: "How it works",
    href: "#/method",
  },
  {text: "Search covers agent behaviours as well as framework capabilities. Try “pagination”."},
  {
    text: "Spot a wrong cell? The Feedback tab on any capability page comes straight to us.",
  },
];

/**
 * caniuse's "Index of features" bar: opens every capability by category, and the
 * agent behaviours by area; the filter narrows the capabilities to chosen categories.
 */
function HomeIndex({
  database,
  capabilities,
  harnesses,
}: {
  database: MirrorDatabase;
  capabilities: Capability[];
  harnesses: HarnessDatabase | null;
}) {
  const [open, setOpen] = useState(false);
  const [filterOpen, setFilterOpen] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const shown = new Map(capabilities.map((capability) => [capability.id, capability]));
  const categories = Object.entries(database.cats).filter(
    ([category, ids]) =>
      (!selected.length || selected.includes(category)) && ids.some((id) => shown.has(id)),
  );
  const rows = harnesses ? Object.values(harnesses.data) : [];
  const listed = open || selected.length > 0;
  return (
    <section className="feature-filter-shell home-index" aria-label="Index of capabilities">
      <div className="feature-strip">
        <button
          className="home-index-toggle"
          type="button"
          aria-expanded={listed}
          aria-controls="home-index-list"
          onClick={() => (selected.length ? setSelected([]) : setOpen((value) => !value))}
        >
          {selected.length
            ? `${categories.reduce((sum, [, ids]) => sum + ids.filter((id) => shown.has(id)).length, 0)} capabilities in ${selected.length} ${selected.length === 1 ? "category" : "categories"}`
            : "Index of capabilities"}
        </button>
        <button
          className={`filter-button${filterOpen ? " is-selected" : ""}`}
          type="button"
          aria-controls="capability-filter-panel"
          aria-expanded={filterOpen}
          onClick={() => setFilterOpen((value) => !value)}
        >
          <FilterIcon />
          Filter capabilities
        </button>
      </div>
      {filterOpen ? (
        <fieldset className="capability-filter-panel" id="capability-filter-panel">
          <legend>Capability categories</legend>
          <div className="filter-options">
            {Object.keys(database.cats).map((category) => (
              <label key={category}>
                <input
                  type="checkbox"
                  checked={selected.includes(category)}
                  onChange={(event) => {
                    const checked = event.currentTarget.checked;
                    setSelected((current) =>
                      checked ? [...current, category] : current.filter((item) => item !== category),
                    );
                  }}
                />
                <span>{category}</span>
              </label>
            ))}
          </div>
        </fieldset>
      ) : null}
      {listed ? (
        <div className="home-index-list" id="home-index-list">
          {categories.map(([category, ids]) => (
            <section className="home-index-group" key={category}>
              <h3>{category}</h3>
              <ul>
                {ids
                  .map((id) => shown.get(id))
                  .filter((capability): capability is Capability => Boolean(capability))
                  .map((capability) => (
                    <li key={capability.id}>
                      <Link href={capabilityHref(capability.id)}>
                        <CapabilityTitle capability={capability} />
                      </Link>
                    </li>
                  ))}
              </ul>
            </section>
          ))}
          {!selected.length && harnesses
            ? harnesses.areas.map((area) => (
                <section className="home-index-group home-index-agents" key={area}>
                  <h3>Agents: {area}</h3>
                  <ul>
                    {rows
                      .filter((row) => row.area === area)
                      .map((row) => (
                        <li key={row.id}>
                          <Link href={agentsHref(row.id)}>{row.title}</Link>
                        </li>
                      ))}
                  </ul>
                </section>
              ))
            : null}
        </div>
      ) : null}
    </section>
  );
}

/**
 * The home page, laid out like caniuse.com's: the index bar, then latest, popular,
 * test, did-you-know, other ways in, and the scores, with frameworks and agents as the
 * two score tabs. A search covers both: the framework capabilities and the agent
 * behaviours, in two labelled groups.
 */
export function HomePage({
  database,
  capabilities,
  harnesses,
  behaviours,
  query,
  resultsMode,
  onResultsModeChange,
  stars,
  onToggleStar,
  gridMode,
  onGridModeChange,
  specIndex,
}: {
  database: MirrorDatabase;
  capabilities: Capability[];
  harnesses: HarnessDatabase | null;
  behaviours: HarnessRow[];
  query: string;
  resultsMode: ResultsMode;
  onResultsModeChange: (mode: ResultsMode) => void;
  stars: string[];
  onToggleStar: (capabilityId: string) => void;
  gridMode: GridMode;
  onGridModeChange: (mode: GridMode) => void;
  specIndex: SpecIndex | null;
}) {
  const [scoreView, setScoreView] = useState<"frameworks" | "agents">("frameworks");
  const [factIndex, setFactIndex] = useState(0);
  const preview = previewRequested(window.location.search);
  const visibleAgents = harnesses ? visibleHarnesses(harnesses, preview) : [];
  const defaultAgents = shownHarnesses(visibleAgents, HARNESS_LINKS, false);
  const defaultAgentIds = defaultAgents.map(([id]) => id);
  const rows = harnesses ? Object.values(harnesses.data) : [];

  if (query.trim()) {
    const total = capabilities.length + behaviours.length;
    return (
      <main className="site-shell page-main" id="main-content">
        <div className="results-heading search-results-heading">
          <div>
            <p className="eyebrow">Search results</p>
            <h2>{total ? `Matches for “${query.trim()}”` : "No matching capability or behaviour"}</h2>
          </div>

          <ToggleButtonGroup
            aria-label="Search result layout"
            disallowEmptySelection
            selectionMode="single"
            selectedKeys={[resultsMode]}
            size="sm"
            onSelectionChange={(keys) => {
              const selected = Array.from(keys)[0];
              if (selected === "tables" || selected === "list") {
                onResultsModeChange(selected);
              }
            }}
          >
            <ToggleButton id="tables">
              <GridIcon width={16} height={16} />
              Tables
            </ToggleButton>
            <ToggleButton id="list">
              <ToggleButtonGroup.Separator />
              <ListIcon width={16} height={16} />
              List
            </ToggleButton>
          </ToggleButtonGroup>
        </div>

        {total ? (
          <div className="search-results-groups">
            {capabilities.length ? (
              <section
                className="search-results-group"
                data-results="capabilities"
                aria-labelledby="search-capabilities-title"
              >
                <h3 id="search-capabilities-title" className="search-results-group-title">
                  Framework capabilities{" "}
                  <small>
                    {capabilities.length} of {Object.keys(database.data).length} ·{" "}
                    <Link href="#/frameworks">every framework</Link>
                  </small>
                </h3>
                <div
                  className={
                    resultsMode === "tables"
                      ? "search-results-tables"
                      : "capability-card-grid search-results-cards"
                  }
                >
                  {capabilities.map((capability) => (
                    <CapabilityCard
                      capability={capability}
                      database={database}
                      key={capability.id}
                      presentation={resultsMode === "tables" ? "table" : "compact"}
                      starred={stars.includes(capability.id)}
                      onToggleStar={() => onToggleStar(capability.id)}
                      gridMode={gridMode}
                      onGridModeChange={onGridModeChange}
                    />
                  ))}
                </div>
              </section>
            ) : null}
            {behaviours.length ? (
              <section
                className="search-results-group"
                data-results="behaviours"
                aria-labelledby="search-behaviours-title"
              >
                <h3 id="search-behaviours-title" className="search-results-group-title">
                  Agent behaviours{" "}
                  <small>
                    {behaviours.length} of {rows.length} · measured on the wire ·{" "}
                    <Link href={agentsHref()}>every agent</Link>
                  </small>
                </h3>
                <ol className="home-list home-framework-scores search-behaviour-results">
                  {behaviours.map((row) => (
                    <BehaviourScore row={row} shownIds={defaultAgentIds} key={row.id} />
                  ))}
                </ol>
              </section>
            ) : null}
          </div>
        ) : (
          <Card className="empty-card" variant="secondary">
            <Card.Header>
              <Card.Title>Nothing measured under that name</Card.Title>
              <Card.Description>
                Try a schema term such as enum, required, format, nested, an MCP
                annotation name, or an agent behaviour such as prompt, pagination
                or sampling.
              </Card.Description>
            </Card.Header>
          </Card>
        )}
      </main>
    );
  }

  const listedFrameworks = filteredDatabase(database, FRAMEWORK_DEFAULTS);
  const frameworkTotal = Object.keys(database.agents).length;
  const popular = popularCapabilities(database, capabilities);
  const facts = homeFacts(database.mcp_spec);
  const fact = facts[factIndex % facts.length];

  return (
    <main className="site-shell page-main index-home home-overview" id="main-content">
      <h1 className="sr-only">Can AI use MCP? Frameworks and agents, measured.</h1>
      <HomeIndex database={database} capabilities={capabilities} harnesses={harnesses} />

      <div className="home-sections">
        <NewCapabilities specIndex={specIndex} titleId="home-new-title" />

        <section className="home-section home-section-popular" aria-labelledby="home-popular-title">
          <h2 id="home-popular-title">
            {popular.observed ? "Most searched capabilities" : "Popular capabilities"}
          </h2>
          <ol className="home-list home-list-numbered">
            {popular.capabilities.map((capability) => (
              <li data-capability-id={capability.id} key={capability.id}>
                <Link href={capabilityHref(capability.id)}>
                  <CapabilityTitle capability={capability} />
                </Link>
              </li>
            ))}
          </ol>
        </section>

        <section className="home-section home-section-test" aria-labelledby="home-test-title">
          <h2 id="home-test-title">Test a capability</h2>
          <p>
            Scan your MCP server against the current framework adapters. The command runs
            locally and does not call a model.
          </p>
          <CopyCodeBlock
            className="home-command"
            value={`uv run mcp-mirror scan "python your_server.py" --runner managed --spec-version ${database.mcp_spec}`}
          />
        </section>

        <section className="home-section home-section-dyk" aria-labelledby="home-dyk-title">
          <h2 id="home-dyk-title">Did you know?</h2>
          <p className="home-fact">
            {fact.text}{" "}
            {fact.href ? (
              <Link
                href={fact.href}
                {...(fact.href.startsWith("http") ? {target: "_blank", rel: "noopener noreferrer"} : {})}
              >
                {fact.link}
              </Link>
            ) : null}
          </p>
          <button
            className="home-fact-next"
            type="button"
            onClick={() => setFactIndex((index) => index + 1)}
          >
            Next
          </button>
        </section>

        <section className="home-section home-section-tools" aria-labelledby="home-tools-title">
          <h2 id="home-tools-title">Other ways in</h2>
          <p>Ask the catalog from your agent: it is an MCP server too.</p>
          <CopyCodeBlock
            className="home-command"
            value="claude mcp add --transport http canaiuse https://canaiuse.com/mcp"
          />
          <ul className="home-list">
            <li>
              <Link href={siteFileHref(`data/data-${database.mcp_spec}.json`)} target="_blank">
                Published JSON data for MCP {database.mcp_spec}
              </Link>
            </li>
            <li>
              <Link href="#/stats">Filter and export measured support</Link>
            </li>
            <li>
              <Link href={siteFileHref("standalone.html")} target="_blank">
                The whole site as one offline page
              </Link>
            </li>
          </ul>
        </section>

        <section className="home-section home-section-scores" aria-labelledby="home-scores-title">
          <h2 id="home-scores-title">Measured results</h2>
          <div className="home-score-tabs" role="radiogroup" aria-label="Results for">
            {(["frameworks", "agents"] as const).map((view) => (
              <button
                className="home-score-tab"
                type="button"
                role="radio"
                aria-checked={scoreView === view}
                key={view}
                onClick={() => setScoreView(view)}
              >
                {view === "frameworks" ? "Frameworks" : "Agents"}
              </button>
            ))}
          </div>
          {scoreView === "frameworks" ? (
            <div className="home-panel-frameworks">
              <ol className="home-list home-framework-scores" data-version-kind="current_version">
                {Object.keys(listedFrameworks.agents).map((agentId) => (
                  <FrameworkScore
                    database={database}
                    capabilities={capabilities}
                    agentId={agentId}
                    key={agentId}
                  />
                ))}
              </ol>
              <Link className="home-section-more" href="#/frameworks">
                All {frameworkTotal} frameworks
              </Link>
            </div>
          ) : (
            <div className="home-panel-agents">
              <ol className="home-list home-framework-scores home-agent-scores">
                {defaultAgents.map(([harnessId, harness]) => (
                  <li
                    className="home-framework-score home-agent-score"
                    data-harness-id={harnessId}
                    data-summary-version={harness.current_version}
                    key={harnessId}
                    aria-label={`${harness.name} ${harness.current_version}`}
                  >
                    <span className="home-framework-score-label">
                      <a className="home-framework-link" href={agentHref(harnessId)}>
                        {harness.name}
                      </a>{" "}
                      <strong className="framework-summary-version">{harness.current_version}</strong>
                      {harnessUnmeasured(harness) ? (
                        <small className="agents-score-date agents-score-unmeasured"> · not measured yet</small>
                      ) : null}
                      {harness.visibility === "preview" ? " (preview)" : ""}
                    </span>
                    <AgentScoreBar harness={harness} counts={harnessCodeCounts(rows, harnessId)} total={rows.length} />
                  </li>
                ))}
              </ol>
              <Link className="home-section-more" href={agentsHref()}>
                All {visibleAgents.length} agents
              </Link>
            </div>
          )}
          <Link className="home-score-note" href="#/method">
            About these scores
          </Link>
        </section>
      </div>
    </main>
  );
}
