import {Link, ToggleButton, ToggleButtonGroup} from "@heroui/react";
import {useState} from "react";

import {CopyCodeBlock} from "../components/CopyCodeBlock";
import {FeedbackForm} from "../components/FeedbackForm";
import {captureHref, HarnessEvidence, HarnessLegend} from "../components/HarnessParts";
import {shortValue} from "../components/HarnessRowTable";
import {ArrowUpRightIcon} from "../components/Icons";
import {shortDate} from "../lib/data";
import {
  agentsHref,
  HARNESS_CODE_LABELS,
  HARNESS_CODES,
  harnessCodeCounts,
  harnessLiveNote,
  harnessUnmeasured,
} from "../lib/harnesses";
import type {HarnessDatabase} from "../types";
import {AgentsLoadState} from "./AgentsPage";
import {LifecycleNote} from "../components/Lifecycle";
import {NewerReleaseFact} from "../components/NewerRelease";

type RowView = "gaps" | "all";

/**
 * One agent's column of the table, turned on its side: every behaviour with its
 * cell and the note behind every cell that is not a plain yes. This is the page an
 * agent's maintainer is sent to check.
 */
export function AgentPage({
  data,
  error,
  agentId,
}: {
  data: HarnessDatabase | null;
  error: string | null;
  agentId: string;
}) {
  const [view, setView] = useState<RowView>("gaps");
  if (!data) return <AgentsLoadState error={error} />;
  const harness = data.agents[agentId];
  if (!harness) {
    return (
      <main className="site-shell page-main" id="main-content">
        <div className="page-intro">
          <h1>Agent not found</h1>
          <p>No agent with this id is published.</p>
          <Link href={agentsHref()}>Return to the agents index</Link>
        </div>
      </main>
    );
  }
  const rows = Object.values(data.data);
  const counts = harnessCodeCounts(rows, agentId);
  const total = rows.length || 1;
  const summary = HARNESS_CODES.filter((code) => counts[code])
    .map((code) => `${counts[code]} ${HARNESS_CODE_LABELS[code]}`)
    .join(", ");
  const command = rows
    .map((row) => row.reproduce_by_agent?.[agentId])
    .find((value): value is string => Boolean(value));
  const published = harness.visibility !== "preview";
  const unmeasured = harnessUnmeasured(harness);
  const liveNote = harnessLiveNote(harness);
  const approvalModes = data.surface?.approval_modes ?? {};

  return (
    <main
      className="site-shell page-main framework-page agent-page"
      id="main-content"
      data-agent-id={agentId}
    >
      <header className="page-intro framework-intro">
        <p className="eyebrow">
          <Link href={agentsHref()}>Agent</Link> · measured on the wire
        </p>
        <h1>
          {harness.name}
          {harness.visibility === "preview" ? " (preview)" : ""}
        </h1>
        <LifecycleNote lifecycle={harness.lifecycle} />
        <dl className="framework-facts">
          <dt>Measured version</dt>
          <dd>
            <strong>{harness.current_version}</strong>
          </dd>
          <NewerReleaseFact row={harness} />
          <dt>Measured on</dt>
          <dd>
            {unmeasured ? (
              <span className="harness-not-measured">
                <strong>Not measured yet.</strong> {unmeasured}
              </span>
            ) : harness.measured_on ? (
              <time dateTime={harness.measured_on}>{shortDate(harness.measured_on)}</time>
            ) : (
              "no dated capture"
            )}
          </dd>
          {liveNote ? (
            <>
              <dt>Measured how</dt>
              <dd className="harness-live-note">
                <strong>Live.</strong> This row was {liveNote}. Every other row is measured against a
                scripted stand-in for the model and repeats exactly.
              </dd>
              {harness.model ? (
                <>
                  <dt>Model</dt>
                  <dd>{harness.model}</dd>
                </>
              ) : null}
            </>
          ) : null}
          <dt>Vendor</dt>
          <dd>{harness.vendor}</dd>
          {harness.install ? (
            <>
              <dt>Install</dt>
              <dd>
                <code>{harness.install}</code>
              </dd>
            </>
          ) : null}
          {harness.headless ? (
            <>
              <dt>Headless run</dt>
              <dd>
                <code>{harness.headless}</code>
              </dd>
            </>
          ) : null}
          {harness.mcp_config ? (
            <>
              <dt>MCP config</dt>
              <dd>
                <code>{harness.mcp_config}</code>
              </dd>
            </>
          ) : null}
          <dt>Approval mode</dt>
          <dd>
            Measured in {approvalModes.default ?? "default approval mode"}.
            <small>
              {harness.unattended
                ? <>Unattended: <code>{harness.unattended}</code></>
                : "No documented switch removes prompts."}
            </small>
          </dd>
        </dl>
        <details className="agent-row-feedback page-feedback">
          <summary>Something wrong on this page?</summary>
          <FeedbackForm kind="agent" subject={agentId} about={`${harness.name} ${harness.current_version}`} />
        </details>
      </header>

      <HarnessLegend data={data} titleId="agent-legend-title" />

      <section className="framework-summary" aria-label="Behaviour summary">
        <p className="framework-summary-line">
          {rows.length} behaviours: {summary}.
        </p>
        <div className="support-bar" role="img" aria-label={summary}>
          {HARNESS_CODES.map((code) =>
            counts[code] ? (
              <span
                className={`support-segment support-${code}`}
                key={code}
                style={{width: `${(counts[code] / total) * 100}%`}}
              />
            ) : null,
          )}
        </div>
        <ToggleButtonGroup
          aria-label="Behaviours shown"
          disallowEmptySelection
          selectionMode="single"
          selectedKeys={[view]}
          size="sm"
          onSelectionChange={(keys) => {
            const selected = Array.from(keys)[0];
            if (selected === "gaps" || selected === "all") setView(selected);
          }}
        >
          <ToggleButton id="gaps">Only gaps ({rows.length - counts.y})</ToggleButton>
          <ToggleButtonGroup.Separator />
          <ToggleButton id="all">All behaviours ({rows.length})</ToggleButton>
        </ToggleButtonGroup>
      </section>

      {data.areas.map((area) => {
        const areaRows = rows.filter((row) => row.area === area);
        if (!areaRows.length) return null;
        const shown = areaRows
          .filter((row) => view === "all" || row.cells[agentId]?.code !== "y")
          // Gaps first, so the rows a maintainer has to check lead each area.
          .sort(
            (left, right) =>
              Number(left.cells[agentId]?.code === "y") -
              Number(right.cells[agentId]?.code === "y"),
          );
        const present = areaRows.length - shown.length;
        return (
          <section className="framework-area" key={area} aria-label={area}>
            <h2>{area}</h2>
            {shown.length ? (
              <ul className="framework-rows">
                {shown.map((row) => {
                  const cell = row.cells[agentId];
                  if (!cell) return null;
                  const note = cell.note ? row.notes_by_num?.[String(cell.note)] : undefined;
                  // A value longer than a revision string goes under the cell, not in it.
                  const longValue = cell.value && !shortValue(cell.value) ? cell.value : undefined;
                  const captureName = cell.capture.split("/").slice(-2).join("/");
                  const href = captureHref(data.repo, cell, published);
                  return (
                    <li
                      className="framework-row"
                      data-row-id={row.id}
                      data-support={cell.code}
                      key={row.id}
                    >
                      <Link className="framework-row-title" href={agentsHref(row.id)}>
                        {row.title}
                      </Link>
                      <a
                        className={`support-cell framework-cell harness-cell support-${cell.code}`}
                        data-code={cell.code}
                        data-label={cell.label}
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        title={`${cell.label}. Open the wire capture ${captureName}`}
                      >
                        {cell.transports?.length ? (
                          <span className="harness-cell-transports">
                            {cell.transports.map((side) => (
                              <span
                                key={side.transport}
                                className={`harness-cell-transport support-${side.code}`}
                                data-transport={side.transport}
                                data-code={side.code}
                              >
                                <em>{side.transport}</em>{" "}
                                {shortValue(side.value) ? <strong>{side.value}</strong> : side.label}
                              </span>
                            ))}
                          </span>
                        ) : (
                          <span className="harness-cell-label">
                            {shortValue(cell.value) ? <strong>{cell.value}</strong> : null}{" "}
                            {cell.label}
                          </span>
                        )}
                        {cell.requested && !cell.transports?.length ? (
                          <small>requested {cell.requested}</small>
                        ) : null}
                        <small className="harness-cell-capture">
                          {captureName}
                          {cell.line ? `:${cell.line}` : ""}
                          <ArrowUpRightIcon width={11} height={11} />
                        </small>
                      </a>
                      {note || longValue ? (
                        <ul className="framework-row-notes">
                          <li data-note={cell.note}>
                            {cell.note ? <a href={agentsHref(row.id)}>#{cell.note}</a> : null}{" "}
                            {longValue ? <span className="harness-long-value">{longValue}</span> : null}{" "}
                            {cell.summary ?? note}
                            {cell.summary && note ? (
                              <HarnessEvidence
                                evidence={note}
                                href={href}
                                captureName={captureName}
                                line={cell.line}
                              />
                            ) : null}
                          </li>
                        </ul>
                      ) : null}
                    </li>
                  );
                })}
              </ul>
            ) : null}
            {view === "gaps" && present ? (
              <p className="framework-area-present">
                {shown.length ? "Plus " : ""}
                {present} on the wire in every judged run.
              </p>
            ) : null}
          </section>
        );
      })}

      <section className="framework-reproduce" aria-labelledby="agent-reproduce-title">
        <h2 id="agent-reproduce-title">Reproduce</h2>
        <p>
          This reruns every measurement on this page. Level 1 needs no model
          credentials.{" "}
          <Link
            href={`${data.repo}/tree/main/data/harnesses/${harness.capture ? harness.capture.split("/").slice(0, -1).join("/") : `measurements/wire/${encodeURIComponent(agentId)}`}`}
            target="_blank"
            rel="noopener noreferrer"
          >
            Browse the captures
            <ArrowUpRightIcon width={13} height={13} />
          </Link>
        </p>
        {command ? <CopyCodeBlock className="home-command" value={command} /> : null}
        {data.feedback_url ? (
          <p>
            Something here wrong?{" "}
            <Link href={data.feedback_url} target="_blank" rel="noopener noreferrer">
              Send a correction
            </Link>
            .
          </p>
        ) : null}
      </section>
    </main>
  );
}
