import {Link} from "@heroui/react";

import {LIFECYCLE_LABELS} from "../components/Lifecycle";
import {NewerReleaseMark} from "../components/NewerRelease";
import {currentSupport, frameworkHref, shortDate, siteFileHref, SUPPORT_META} from "../lib/data";
import {groupFrameworks, shortFrameworkName} from "../lib/frameworks";
import {RUNNER_MANIFESTS} from "../lib/frameworkSources";
import type {Agent, Capability, MirrorDatabase, SupportCode} from "../types";

const CODES = ["y", "a", "n", "x", "u"] as const;
// The columns that carry a count; Unreachable and Unmeasured stay in the bar only.
const COUNTED = ["y", "a", "n"] as const;
const COUNTED_HEADINGS: Record<(typeof COUNTED)[number], string> = {
  y: "Present",
  a: "Changed",
  n: "Dropped",
};

/** What the recorded run negotiated, or why this revision's cells could not be set. */
function revisionReached(agent: Agent, spec: string): string {
  const negotiated = agent.capture_boundary?.negotiated_mcp_spec_version;
  if (agent.measurement_status === "sdk_incompatible") return "cannot negotiate";
  if (agent.measurement_status === "protocol_mismatch") return negotiated ?? "another revision";
  return negotiated ?? spec;
}

/**
 * The framework directory at #/frameworks: every framework in one table per language,
 * with what was measured, when, whether a newer release is out, the revision the run
 * reached, and how the capabilities came through. The home page holds the overview.
 */
export function FrameworksPage({
  database,
  capabilities,
}: {
  database: MirrorDatabase;
  capabilities: Capability[];
}) {
  const total = capabilities.length || 1;
  const agentCount = Object.keys(database.agents).length;

  return (
    <main className="site-shell page-main frameworks-page" id="main-content">
      <header className="page-intro">
        <p className="eyebrow">MCP {database.mcp_spec}</p>
        <h1>Frameworks</h1>
        <p>
          Every agent framework we measure, and the adapter that turns an MCP server's tools
          into the framework's own tool definitions, captured where that adapter hands them
          on. Each row is one release, scanned against the same
          fixture servers, and each count is the number of {database.mcp_spec} capabilities
          that came through present, changed or dropped. Open a framework for its cells and
          the recordings behind them.
        </p>
        <p className="home-section-intro">
          {database.measured_on ? (
            <>
              Data as of{" "}
              <time dateTime={database.measured_on}>{shortDate(database.measured_on)}</time>.{" "}
            </>
          ) : null}
          {agentCount} frameworks
          {database.coverage
            ? `, ${database.coverage.measured} of ${database.coverage.total} capabilities measured.`
            : "."}{" "}
          <Link href="#/stats?view=frameworks">Compare them capability by capability</Link>
        </p>
      </header>

      {groupFrameworks(database, RUNNER_MANIFESTS).map((group) => (
        <section
          className="framework-directory-group"
          data-group={group.id}
          key={group.id}
          aria-labelledby={`framework-directory-${group.id}`}
        >
          <h2 id={`framework-directory-${group.id}`}>{group.label}</h2>
          <div className="directory-table-wrap">
            <table className="directory-table framework-directory">
              <thead>
                <tr>
                  <th scope="col">Framework</th>
                  <th scope="col" className="directory-optional directory-wide">Adapter</th>
                  <th scope="col">Measured</th>
                  <th scope="col" className="directory-optional">On</th>
                  <th scope="col" className="directory-optional">Revision</th>
                  {COUNTED.map((code) => (
                    <th scope="col" className="directory-count" key={code} title={SUPPORT_META[code].label}>
                      {COUNTED_HEADINGS[code]}
                    </th>
                  ))}
                  <th scope="col" className="directory-bar-head">
                    <span className="sr-only">All {capabilities.length} capabilities</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {group.agentIds.map((agentId) => {
                  const agent = database.agents[agentId];
                  const counts: Record<SupportCode, number> = {y: 0, a: 0, n: 0, x: 0, u: 0};
                  capabilities.forEach((capability) => {
                    counts[currentSupport(database, capability, agentId).code] += 1;
                  });
                  const label = CODES.filter((code) => counts[code])
                    .map((code) => `${counts[code]} ${SUPPORT_META[code].shortLabel.toLowerCase()}`)
                    .join(", ");
                  return (
                    <tr
                      data-agent-id={agentId}
                      data-summary-version={agent.current_version}
                      key={agentId}
                    >
                      <th scope="row">
                        <a className="home-framework-link" href={frameworkHref(agentId)}>
                          {shortFrameworkName(agent.name)}
                        </a>
                        {agent.aliases_of ? (
                          <small className="directory-note">
                            {" "}via {database.agents[agent.aliases_of]?.name ?? agent.aliases_of}
                          </small>
                        ) : null}
                        {agent.lifecycle ? (
                          <small className="directory-note directory-lifecycle">
                            {" "}{LIFECYCLE_LABELS[agent.lifecycle.status] ?? agent.lifecycle.status}
                          </small>
                        ) : null}
                        {agent.visibility === "preview" ? " (preview)" : ""}
                      </th>
                      <td className="directory-optional directory-wide">
                        <code>{agent.package}</code>
                      </td>
                      <td>
                        <strong className="framework-summary-version">{agent.current_version}</strong>
                        <NewerReleaseMark row={agent} />
                      </td>
                      <td className="directory-optional">
                        {agent.measured_on ? (
                          <time dateTime={agent.measured_on}>{shortDate(agent.measured_on)}</time>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td className="directory-optional" data-status={agent.measurement_status ?? "measured"}>
                        {revisionReached(agent, database.mcp_spec)}
                      </td>
                      {COUNTED.map((code) => (
                        <td className={`directory-count directory-count-${code}`} key={code}>
                          {counts[code]}
                        </td>
                      ))}
                      <td className="directory-bar">
                        <span className="home-framework-score-bars" role="img" aria-label={label} title={label}>
                          {CODES.map((code) =>
                            counts[code] ? (
                              <i
                                className={`home-score-segment support-${code}`}
                                key={code}
                                style={{width: `${(counts[code] / total) * 100}%`}}
                              />
                            ) : null,
                          )}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      ))}

      <p className="directory-footer">
        <Link href={siteFileHref(`data/data-${database.mcp_spec}.json`)} target="_blank">
          Published JSON data for MCP {database.mcp_spec}
        </Link>
        {" · "}
        <Link href="#/method">How measurements work</Link>
      </p>
    </main>
  );
}
