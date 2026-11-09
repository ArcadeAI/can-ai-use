import {Card, Link, Spinner} from "@heroui/react";
import {useState} from "react";

import {CopyCodeBlock} from "../components/CopyCodeBlock";
import {HarnessLegend, HarnessToggle} from "../components/HarnessParts";
import {ArrowUpRightIcon} from "../components/Icons";
import {previewRequested, shortDate} from "../lib/data";
import {
  agentHref,
  agentsHref,
  HARNESS_CODE_LABELS,
  HARNESS_CODES,
  harnessCodeCounts,
  harnessRule,
  harnessCoverageLine,
  harnessUnmeasured,
  rowCodeCounts,
  shownHarnesses,
  visibleHarnesses,
} from "../lib/harnesses";
import {HARNESS_LINKS} from "../lib/harnessSources";
import type {Harness, HarnessDatabase} from "../types";
import {LIFECYCLE_LABELS} from "../components/Lifecycle";
import {NewerReleaseMark} from "../components/NewerRelease";

// The tallies that get a column; "not measured" stays in the bar only.
const COUNTED = ["y", "a", "n"] as const;
const COUNTED_HEADINGS: Record<(typeof COUNTED)[number], string> = {
  y: "Yes",
  a: "Partly",
  n: "No",
};

/** The loading and error states every agents page shares. */
export function AgentsLoadState({error}: {error: string | null}) {
  if (error) {
    return (
      <main className="site-shell page-main" id="main-content">
        <Card>
          <Card.Header>
            <Card.Title>Agent data could not be loaded</Card.Title>
            <Card.Description>{error}</Card.Description>
          </Card.Header>
        </Card>
      </main>
    );
  }
  return (
    <main className="site-shell page-main load-state" id="main-content" role="status">
      <Spinner size="lg" />
      <span>Loading agent captures</span>
    </main>
  );
}

/** A y/a/n/u bar for one agent or one behaviour, like the home framework bars. */
/** The bar for an agent's row, or the "Not measured yet" mark when no session ran. */
export function AgentScoreBar({
  harness,
  counts,
  total,
}: {
  harness: Harness;
  counts: Record<(typeof HARNESS_CODES)[number], number>;
  total: number;
}) {
  const reason = harnessUnmeasured(harness);
  if (!reason) return <HarnessBar counts={counts} total={total} />;
  return (
    <>
      <span className="home-framework-score-bars harness-unmeasured" role="img" aria-label="Not measured yet" />
      <small className="harness-unmeasured-reason">{reason}</small>
    </>
  );
}

export function HarnessBar({
  counts,
  total,
}: {
  counts: Record<(typeof HARNESS_CODES)[number], number>;
  total: number;
}) {
  const label = HARNESS_CODES.filter((code) => counts[code])
    .map((code) => `${counts[code]} ${HARNESS_CODE_LABELS[code]}`)
    .join(", ");
  return (
    <span className="home-framework-score-bars" role="img" aria-label={label} title={label}>
      {HARNESS_CODES.map((code) =>
        counts[code] ? (
          <i
            className={`home-score-segment support-${code}`}
            key={code}
            style={{width: `${(counts[code] / (total || 1)) * 100}%`}}
          />
        ) : null,
      )}
    </span>
  );
}

/**
 * The agents directory: one table row per agent with what was measured, when, against
 * what, and how its behaviours came out; Filtered | All; then every behaviour by area
 * with its own bar. Each name opens the
 * agent's page; each behaviour opens its page, where the cells are.
 */
export function AgentsPage({
  data,
  error,
}: {
  data: HarnessDatabase | null;
  error: string | null;
}) {
  const [showAll, setShowAll] = useState(false);
  if (!data) return <AgentsLoadState error={error} />;

  const visible = visibleHarnesses(data, previewRequested(window.location.search));
  const harnesses = shownHarnesses(visible, HARNESS_LINKS, showAll);
  const shownIds = harnesses.map(([id]) => id);
  const rows = Object.values(data.data);
  const repo = data.repo;
  // Only the harnesses on show: a preview row's command stays private with its cells.
  const commands = [
    ...new Set(
      rows.flatMap((row) =>
        Object.entries(row.reproduce_by_agent ?? {})
          .filter(([id, command]) => shownIds.includes(id) && Boolean(command))
          .map(([, command]) => command),
      ),
    ),
  ];

  return (
    <main className="site-shell page-main harness-page agents-page" id="main-content">
      <header className="page-intro">
        <p className="eyebrow">Measured on the wire</p>
        <h1>Agents</h1>
        <p>
          What an agent actually sends to an MCP server. Each agent runs headless
          against five fixture servers behind a recorder, and every cell is read off
          the recorded JSON-RPC. This is a separate table from the framework support
          tables and uses its own words. Every cell links to the capture it came from.
        </p>
        <p className="home-section-intro">
          {data.measured_on ? (
            <>
              Data as of <time dateTime={data.measured_on}>{shortDate(data.measured_on)}</time>.{" "}
            </>
          ) : null}
          {harnessCoverageLine(visible, rows.length)}
        </p>
      </header>

      <HarnessLegend data={data} titleId="harness-legend-title" />

      <section className="agents-index" aria-labelledby="agents-index-title">
        <h2 id="agents-index-title">Agents</h2>
        <HarnessToggle
          showAll={showAll}
          onChange={setShowAll}
          rule={showAll ? `All ${visible.length} agents` : harnessRule(data, HARNESS_LINKS)}
        />
        <div className="directory-table-wrap">
          <table className="directory-table agents-directory agents-scores">
            <thead>
              <tr>
                <th scope="col">Agent</th>
                <th scope="col" className="directory-optional">Vendor</th>
                <th scope="col">Measured</th>
                <th scope="col" className="directory-optional">On</th>
                <th scope="col" className="directory-optional directory-wide">Run against</th>
                {COUNTED.map((code) => (
                  <th scope="col" className="directory-count" key={code}>
                    {COUNTED_HEADINGS[code]}
                  </th>
                ))}
                <th scope="col" className="directory-bar-head">
                  <span className="sr-only">All {rows.length} behaviours</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {harnesses.map(([harnessId, harness]) => {
                const counts = harnessCodeCounts(rows, harnessId);
                const unmeasured = harnessUnmeasured(harness);
                return (
                  <tr
                    data-agent-id={harnessId}
                    data-summary-version={harness.current_version}
                    key={harnessId}
                  >
                    <th scope="row">
                      <a className="home-framework-link" href={agentHref(harnessId)}>
                        {harness.name}
                      </a>
                      {harness.lifecycle ? (
                        <small className="directory-note directory-lifecycle">
                          {" "}{LIFECYCLE_LABELS[harness.lifecycle.status] ?? harness.lifecycle.status}
                        </small>
                      ) : null}
                      {harness.visibility === "preview" ? " (preview)" : ""}
                    </th>
                    <td className="directory-optional">{harness.vendor}</td>
                    <td>
                      <strong className="framework-summary-version">{harness.current_version}</strong>
                      <NewerReleaseMark row={harness} />
                    </td>
                    <td className="directory-optional">
                      {unmeasured ? (
                        "—"
                      ) : harness.measured_on ? (
                        <time dateTime={harness.measured_on}>{shortDate(harness.measured_on)}</time>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td className="directory-optional directory-wide">
                      {harness.measured_live
                        ? `the live ${harness.live_service ?? "service"}`
                        : "a scripted model"}
                    </td>
                    {unmeasured ? (
                      <td className="directory-bar" colSpan={COUNTED.length + 1}>
                        <AgentScoreBar harness={harness} counts={counts} total={rows.length} />
                      </td>
                    ) : (
                      <>
                        {COUNTED.map((code) => (
                          <td className={`directory-count directory-count-${code}`} key={code}>
                            {counts[code]}
                          </td>
                        ))}
                        <td className="directory-bar">
                          <AgentScoreBar harness={harness} counts={counts} total={rows.length} />
                        </td>
                      </>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {data.areas.map((area) => {
        const areaRows = rows.filter((row) => row.area === area);
        if (!areaRows.length) return null;
        return (
          <section className="harness-area agents-behaviours" key={area} aria-label={area}>
            <h2>{area}</h2>
            <ol className="home-list home-framework-scores agents-behaviour-list">
              {areaRows.map((row) => (
                <li
                  className="home-framework-score agents-behaviour"
                  data-row-id={row.id}
                  key={row.id}
                >
                  <span className="home-framework-score-label">
                    <a className="home-framework-link" href={agentsHref(row.id)}>
                      {row.title}
                    </a>
                    <small className="agents-behaviour-question">{row.question || row.observe}</small>
                  </span>
                  <HarnessBar counts={rowCodeCounts(row, shownIds)} total={shownIds.length} />
                </li>
              ))}
            </ol>
          </section>
        );
      })}

      <section className="harness-reproduce">
        <h2>Reproduce</h2>
        <p>
          Level 1 needs no model credentials. Captures updated{" "}
          {shortDate(data.updated)}
          {data.measured_on ? `, newest session measured ${shortDate(data.measured_on)}` : ""}.{" "}
          <Link
            href={`${repo}/tree/main/data/harnesses/measurements`}
            target="_blank"
            rel="noopener noreferrer"
          >
            Browse every capture
            <ArrowUpRightIcon width={13} height={13} />
          </Link>
        </p>
        {commands.map((command) => (
          <CopyCodeBlock key={command} value={command} />
        ))}
      </section>
    </main>
  );
}
