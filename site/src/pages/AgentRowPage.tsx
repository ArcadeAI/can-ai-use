import {Button, Link, Tabs, Tooltip} from "@heroui/react";
import {useId, useState} from "react";

import {CopyCodeBlock, writeClipboardText} from "../components/CopyCodeBlock";
import {FeedbackForm} from "../components/FeedbackForm";
import {HarnessLegend, HarnessToggle} from "../components/HarnessParts";
import {HarnessRowNotes, HarnessRowTable} from "../components/HarnessRowTable";
import {ArrowUpRightIcon, DocumentIcon} from "../components/Icons";
import {capabilityHref, previewRequested, shortDate} from "../lib/data";
import {
  agentsHref,
  harnessMeasuredOn,
  harnessRule,
  harnessUnmeasured,
  rowLabelTally,
  shownHarnesses,
  visibleHarnesses,
} from "../lib/harnesses";
import {HARNESS_LINKS} from "../lib/harnessSources";
import type {Harness, HarnessDatabase, HarnessRow} from "../types";
import {AgentsLoadState} from "./AgentsPage";

type DetailKey = "notes" | "test" | "rule" | "feedback";

const capitalized = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);

/** "A, B and C", or "A, B, C and 2 more" past `limit`. */
function namedAgents(names: string[], limit = 3): string {
  const listed = names.length > limit ? [...names.slice(0, limit), `${names.length - limit} more`] : names;
  return listed.length > 1 ? `${listed.slice(0, -1).join(", ")} and ${listed.at(-1)}` : listed[0];
}

/**
 * The behaviour's finding in one line, the way a capability's reads: which agents miss
 * it, by what they did instead, the maintainer's default agents named first. Counted
 * over every measured agent, whatever the table below is filtered to.
 */
function headline(row: HarnessRow, harnesses: [string, Harness][]): string {
  const rank = (id: string) => {
    const index = HARNESS_LINKS.default_harnesses.indexOf(id);
    return index === -1 ? Number.MAX_SAFE_INTEGER : index;
  };
  const answered = harnesses
    .filter(([id, harness]) => !harnessUnmeasured(harness) && row.cells[id] && row.cells[id].code !== "u")
    .sort(([left], [right]) => rank(left) - rank(right));
  if (!answered.length) return "Not yet measured";
  const tally = rowLabelTally(row, answered.map(([id]) => id));
  if (tally.length === 1) return `${capitalized(tally[0].label)} from all ${answered.length} measured agents`;
  const missCode = answered.some(([id]) => row.cells[id].code === "n") ? "n" : "a";
  const missing = answered.filter(([id]) => row.cells[id].code === missCode);
  if (!missing.length) return `${tally.length} different answers from ${answered.length} measured agents`;
  const byLabel = new Map<string, string[]>();
  for (const [id, harness] of missing) {
    const label = row.cells[id].label;
    byLabel.set(label, [...(byLabel.get(label) ?? []), harness.name]);
  }
  const groups = [...byLabel.entries()].map(([label, names]) => `${capitalized(label)}: ${namedAgents(names)}`);
  return `${groups.join("; ")} (${missing.length} of ${answered.length})`;
}

/** The behaviour's spec section, read off the framework capability it is paired with. */
function SpecLink({row}: {row: HarnessRow}) {
  if (!row.capability) return null;
  return (
    <Link
      className="spec-link"
      href={row.capability.docs_url}
      target="_blank"
      rel="noopener noreferrer"
      aria-label={`${row.capability.docs_label} for ${row.title}`}
    >
      <DocumentIcon className="spec-link-document-icon" width={14} height={14} />
      <span>{row.capability.docs_label}</span>
      <ArrowUpRightIcon width={14} height={14} />
    </Link>
  );
}

/** The command that re-measures one agent; every behaviour comes from the same session. */
function TestPanel({row, harnesses}: {row: HarnessRow; harnesses: [string, Harness][]}) {
  const measured = harnesses.filter(([id]) => row.reproduce_by_agent?.[id]);
  const [agentId, setAgentId] = useState(measured[0]?.[0] ?? "");
  if (!measured.length) return <p>No reproduce command is recorded for this behaviour.</p>;
  const command = row.reproduce_by_agent?.[agentId] ?? row.reproduce ?? "";
  return (
    <div className="feature-test-panel">
      <p>
        Re-run one agent against the five fixture servers and read this behaviour off the new
        capture. One session answers every behaviour, so an agent&apos;s command is the same on
        every behaviour page.
      </p>
      <div className="feature-test-controls">
        <label>
          Agent
          <select value={agentId} onChange={(event) => setAgentId(event.target.value)}>
            {measured.map(([id, harness]) => (
              <option value={id} key={id}>{harness.name} {harness.current_version}</option>
            ))}
          </select>
        </label>
      </div>
      <CopyCodeBlock className="command-box" value={command} />
    </div>
  );
}

/**
 * One agent behaviour, laid out like a capability page: the finding, the spec section,
 * the plain description, a grid of agents with the newer release under each, and the
 * notes, a reproduce command, the exact rule and feedback in tabs.
 */
export function AgentRowPage({
  data,
  error,
  rowId,
}: {
  data: HarnessDatabase | null;
  error: string | null;
  rowId: string;
}) {
  const [showAll, setShowAll] = useState(false);
  const [selectedDetail, setSelectedDetail] = useState<DetailKey>("notes");
  const [linkCopied, setLinkCopied] = useState(false);
  const scopeId = useId();
  const scope = `behaviour-${scopeId.replace(/[^a-zA-Z0-9_-]/g, "")}`;
  if (!data) return <AgentsLoadState error={error} />;
  const row: HarnessRow | undefined = data.data[rowId];
  if (!row) {
    return (
      <main className="site-shell page-main" id="main-content">
        <div className="page-intro">
          <h1>Behaviour not found</h1>
          <p>No agent behaviour with this id is published.</p>
          <Link href={agentsHref()}>Return to the agents index</Link>
        </div>
      </main>
    );
  }
  const visible = visibleHarnesses(data, previewRequested(window.location.search));
  const harnesses = shownHarnesses(visible, HARNESS_LINKS, showAll);
  const shownIds = harnesses.map(([id]) => id);
  const tally = rowLabelTally(row, shownIds);
  const dates = harnesses
    .map(([id, harness]) => harnessMeasuredOn(row, id, harness))
    .filter((date): date is string => Boolean(date))
    .sort();
  const earliest = dates[0];
  const latest = dates[dates.length - 1];
  const captures = new Set(shownIds.map((id) => row.cells[id]?.capture.split("/").pop()).filter(Boolean));
  const deprecated = row.capability?.status === "deprecated" ? row.capability : undefined;

  const permalink = () => {
    const url = `${window.location.href.split("#")[0]}${agentsHref(row.id)}`;
    void writeClipboardText(url).then(() => {
      setLinkCopied(true);
      window.setTimeout(() => setLinkCopied(false), 1600);
    });
  };

  return (
    <main className="site-shell page-main agent-row-page" id="main-content">
      <p className="page-breadcrumb">
        <Link href={agentsHref()}>Agents</Link> · {row.area}
      </p>
      <article
        className="feature-module is-detail agent-row-module"
        data-row-id={row.id}
        aria-labelledby={`${scope}-title`}
      >
        <aside className="feature-action-rail" aria-label="Behaviour actions">
          <Tooltip delay={250}>
            <Tooltip.Trigger>
              <Button
                aria-label="Copy a link to this behaviour"
                isIconOnly
                size="sm"
                variant="ghost"
                onPress={permalink}
              >
                <span className="permalink-symbol" aria-hidden="true">#</span>
              </Button>
            </Tooltip.Trigger>
            <Tooltip.Content>{linkCopied ? "Link copied" : "Copy link"}</Tooltip.Content>
          </Tooltip>
        </aside>

        <div className="feature-module-main">
          <header className="feature-module-header">
            <div className="feature-heading-copy">
              <div className="feature-title-line">
                <h2 id={`${scope}-title`}>{row.title}</h2>
                <SpecLink row={row} />
                {deprecated ? (
                  <span
                    className="behaviour-deprecated"
                    data-status="deprecated"
                    title={`${deprecated.title.replaceAll("`", "")} is deprecated in MCP ${deprecated.revision}`}
                  >
                    Deprecated in MCP {deprecated.revision}
                  </span>
                ) : null}
              </div>
              <section
                className="measurement-baseline"
                aria-label="Measurement baseline"
                data-region="measurement-baseline"
              >
                <strong>{headline(row, visible)}</strong>
                <span>
                  {row.area} · measured{" "}
                  {earliest && latest && earliest !== latest ? (
                    <>
                      <time dateTime={earliest}>{earliest}</time> to{" "}
                      <time dateTime={latest}>{latest}</time>
                    </>
                  ) : (
                    <time dateTime={latest ?? row.measured.run_date ?? ""}>
                      {shortDate(latest ?? row.measured.run_date ?? undefined)}
                    </time>
                  )}{" "}
                  · {shownIds.length} agents · <code>{[...captures].join(", ") || "wire capture"}</code>
                </span>
              </section>
            </div>
            <p className="feature-observed-summary" aria-label="Observed summary">
              <span>Observed</span>
              <strong>
                {tally.map((entry) => `${entry.count} ${entry.label}`).join(" · ")}
              </strong>
              <small>{shownIds.length} of {visible.length} agents shown</small>
            </p>
          </header>

          <div className="feature-description" data-region="description">
            {row.question ? <p className="agent-row-question">{row.question}</p> : null}
            <p className="agent-row-description">{row.description || row.observe}</p>
            {row.capability ? (
              <p className="agent-row-capability">
                Frameworks:{" "}
                <Link href={capabilityHref(row.capability.id)}>
                  how adapters handle {row.capability.title.replaceAll("`", "")}
                </Link>
              </p>
            ) : null}
          </div>

          <div className="feature-display-controls" data-region="display-controls">
            <HarnessToggle
              showAll={showAll}
              onChange={setShowAll}
              rule={showAll ? `All ${visible.length} agents` : harnessRule(data, HARNESS_LINKS)}
            />
          </div>

          <div className="feature-matrix" data-region="agent-matrix">
            <HarnessRowTable
              data={data}
              row={row}
              harnesses={harnesses}
              noteScope={scope}
              showNotes={false}
              onNoteActivate={() => setSelectedDetail("notes")}
            />
          </div>

          <Tabs
            className="feature-tabs"
            selectedKey={selectedDetail}
            variant="secondary"
            onSelectionChange={(key) => setSelectedDetail(String(key) as DetailKey)}
          >
            <Tabs.ListContainer>
              <Tabs.List aria-label={`${row.title} details`}>
                <Tabs.Tab id="notes" aria-label="Notes">
                  <Tabs.Indicator /><span>Notes</span>
                </Tabs.Tab>
                <Tabs.Tab id="test" aria-label="Reproduce">
                  <Tabs.Indicator /><span>Reproduce</span>
                </Tabs.Tab>
                <Tabs.Tab id="rule" aria-label="Exact rule">
                  <Tabs.Indicator /><span>Exact rule</span>
                </Tabs.Tab>
                <Tabs.Tab id="feedback" aria-label="Feedback">
                  <Tabs.Indicator /><span>Feedback</span>
                </Tabs.Tab>
              </Tabs.List>
            </Tabs.ListContainer>

            <Tabs.Panel id="notes">
              <div className="feature-panel-content">
                {deprecated ? (
                  <p className="behaviour-deprecated-note">
                    MCP {deprecated.revision} deprecates{" "}
                    <code>{deprecated.title.replaceAll("`", "")}</code>. It still works, but new
                    implementations should not rely on it.{" "}
                    <Link href={deprecated.docs_url} target="_blank" rel="noopener noreferrer">
                      Read the spec<Link.Icon />
                    </Link>
                  </p>
                ) : null}
                <HarnessRowNotes data={data} row={row} harnesses={harnesses} noteScope={scope} />
              </div>
            </Tabs.Panel>
            <Tabs.Panel id="test">
              <div className="feature-panel-content">
                <TestPanel row={row} harnesses={visible} />
              </div>
            </Tabs.Panel>
            <Tabs.Panel id="rule">
              <div className="feature-panel-content">
                <p>How a cell is read off the capture:</p>
                <p className="agent-row-observe">{row.observe}</p>
              </div>
            </Tabs.Panel>
            <Tabs.Panel id="feedback">
              <div className="feature-panel-content feature-feedback">
                <FeedbackForm kind="behaviour" subject={row.id} />
              </div>
            </Tabs.Panel>
          </Tabs>

          <HarnessLegend data={data} titleId={`${scope}-legend`} />
        </div>
      </article>
    </main>
  );
}
