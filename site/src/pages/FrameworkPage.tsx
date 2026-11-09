import {Link, ToggleButton, ToggleButtonGroup} from "@heroui/react";
import {useState} from "react";

import {CapabilityTitle, InlineCode} from "../components/CapabilityText";
import {CopyCodeBlock} from "../components/CopyCodeBlock";
import {FeedbackForm} from "../components/FeedbackForm";
import {
  capabilityHref,
  currentSupport,
  frameworkHref,
  measuredRelease,
  shortDate,
  SUPPORT_META,
} from "../lib/data";
import {frameworkReproduceCommands, unreachableNotes} from "../lib/frameworks";
import type {Capability, MirrorDatabase, SpecIndex, SupportCode} from "../types";
import {BoundaryList} from "./ReferencePages";
import {LifecycleNote} from "../components/Lifecycle";
import {NewerReleaseFact} from "../components/NewerRelease";

const CODES: SupportCode[] = ["y", "a", "n", "x", "u"];

type RowView = "gaps" | "all";

/** Capabilities grouped by protocol area, in the order the coverage summary lists them. */
function byArea(database: MirrorDatabase, capabilities: Capability[]) {
  const order = (database.coverage?.by_area ?? []).map((entry) => entry.area);
  for (const capability of capabilities) {
    const area = capability.area ?? "Other";
    if (!order.includes(area)) {
      order.push(area);
    }
  }
  return order
    .map((area) => ({
      area,
      capabilities: capabilities.filter(
        (capability) => (capability.area ?? "Other") === area,
      ),
    }))
    .filter((group) => group.capabilities.length > 0);
}

/**
 * One framework's row of the table, turned on its side: every capability with its
 * cell, and the full note behind every cell that is not a plain yes. This is the page a
 * maintainer is sent to check.
 */
export function FrameworkPage({
  database,
  frameworkId,
  capabilities,
  specIndex,
}: {
  database: MirrorDatabase;
  frameworkId: string;
  capabilities: Capability[];
  specIndex: SpecIndex | null;
}) {
  const [view, setView] = useState<RowView>("gaps");
  const agent = database.agents[frameworkId];
  const release = measuredRelease(agent);
  const boundary = agent.capture_boundary;
  const source = agent.aliases_of ? database.agents[agent.aliases_of] : undefined;
  const rows = capabilities.map((capability) => ({
    capability,
    support: currentSupport(database, capability, frameworkId),
  }));
  const counts: Record<SupportCode, number> = {y: 0, a: 0, n: 0, x: 0, u: 0};
  rows.forEach(({support}) => {
    counts[support.code] += 1;
  });
  const total = rows.length || 1;
  const summary = CODES.map(
    (code) => `${counts[code]} ${SUPPORT_META[code].shortLabel.toLowerCase()}`,
  ).join(", ");
  const supportByCapability = new Map(
    rows.map(({capability, support}) => [capability.id, support]),
  );
  const commands = frameworkReproduceCommands(frameworkId, database, capabilities);
  const xNotes = unreachableNotes(frameworkId, database, capabilities);
  // A reason given on more than one capability is stated once, above the areas.
  const sharedNotes = xNotes.filter((note) => note.capabilityIds.length > 1);
  const sharedText = new Set(sharedNotes.map((note) => note.text));
  const allUnreachable = rows.length > 0 && counts.x === rows.length;
  const reached = [...(specIndex?.specs ?? [])]
    .sort((left, right) => right.id.localeCompare(left.id))
    .find(
      (entry) =>
        entry.id !== database.mcp_spec && entry.reached_by?.includes(frameworkId),
    );
  const reachedHref = (() => {
    if (!reached) {
      return "";
    }
    const params = new URLSearchParams(window.location.search);
    params.set("spec", reached.id);
    return `?${params}${frameworkHref(frameworkId)}`;
  })();

  return (
    <main
      className="site-shell page-main framework-page"
      id="main-content"
      data-agent-id={frameworkId}
    >
      <header className="page-intro framework-intro">
        <p className="eyebrow">Framework · MCP {database.mcp_spec}</p>
        <h1>
          {agent.name}
          {agent.visibility === "preview" ? " (preview)" : ""}
        </h1>
        <LifecycleNote lifecycle={agent.lifecycle} />
        <dl className="framework-facts">
          <dt>Measured version</dt>
          <dd>
            <strong>{agent.current_version}</strong>
            {release?.release_date ? `, released ${shortDate(release.release_date)}` : ""}
          </dd>
          <NewerReleaseFact row={agent} />
          <dt>Measured on</dt>
          <dd>
            {agent.measured_on ? (
              <time dateTime={agent.measured_on}>{shortDate(agent.measured_on)}</time>
            ) : (
              "no dated capture"
            )}
          </dd>
          <dt>Adapter</dt>
          <dd>
            <code>
              {agent.package}@{agent.current_version}
            </code>{" "}
            via <code>{agent.adapter}</code>
          </dd>
          {source ? (
            <>
              <dt>Measured through</dt>
              <dd>
                <Link href={frameworkHref(agent.aliases_of!)}>{source.name}</Link>
                {agent.alias_note ? <>. <InlineCode text={agent.alias_note} /></> : null}
              </dd>
            </>
          ) : null}
          <dt>Capture boundary</dt>
          <dd>
            <code>{boundary.capture_api}</code>
            <small>
              {boundary.capture_object} ({boundary.capture_stage.replaceAll("_", " ")})
            </small>
          </dd>
          <dt>MCP revision reached</dt>
          <dd>
            {boundary.negotiated_mcp_spec_version ?? "None negotiated"}
            {agent.measurement_note ? (
              <small>
                <InlineCode text={agent.measurement_note} />
              </small>
            ) : null}
          </dd>
        </dl>
        <BoundaryList boundaries={agent.result_boundaries} />
        <details className="agent-row-feedback page-feedback">
          <summary>Something wrong on this page?</summary>
          <FeedbackForm
            kind="framework"
            subject={frameworkId}
            about={`${agent.name} ${agent.current_version}`}
            revision={database.mcp_spec}
          />
        </details>
      </header>

      <section className="framework-summary" aria-label="Support summary">
        <p className="framework-summary-line">
          {rows.length} capabilities: {summary}.
        </p>
        <div className="support-bar" role="img" aria-label={summary}>
          {CODES.map((code) =>
            counts[code] ? (
              <span
                className={`support-segment support-${code}`}
                key={code}
                style={{width: `${(counts[code] / total) * 100}%`}}
              />
            ) : null,
          )}
        </div>
        {allUnreachable ? null : <ToggleButtonGroup
          aria-label="Capabilities shown"
          disallowEmptySelection
          selectionMode="single"
          selectedKeys={[view]}
          size="sm"
          onSelectionChange={(keys) => {
            const selected = Array.from(keys)[0];
            if (selected === "gaps" || selected === "all") {
              setView(selected);
            }
          }}
        >
          <ToggleButton id="gaps">Only gaps ({rows.length - counts.y})</ToggleButton>
          <ToggleButtonGroup.Separator />
          <ToggleButton id="all">All capabilities ({rows.length})</ToggleButton>
        </ToggleButtonGroup>}
      </section>

      {allUnreachable ? (
        <section className="framework-unreachable" aria-label="Unreachable at this revision">
          <h2>
            Every capability is unreachable on MCP {database.mcp_spec}
          </h2>
          {xNotes.length ? (
            xNotes.map((note) => (
              <p className="framework-unreachable-reason" key={note.text}>
                <InlineCode text={note.text} />
              </p>
            ))
          ) : (
            <p className="framework-unreachable-reason">
              {SUPPORT_META.x.label}.
            </p>
          )}
          <p>
            {reached ? (
              <Link className="framework-reached-link" href={reachedHref}>
                See {agent.name} on {reached.label}, a revision it reaches
              </Link>
            ) : (
              "No published snapshot has a revision this adapter reaches yet."
            )}
          </p>
        </section>
      ) : null}

      {!allUnreachable && sharedNotes.length ? (
        <section className="framework-shared-notes" aria-label="Unreachable capabilities">
          <h2>Unreachable</h2>
          <ul>
            {sharedNotes.map((note) => (
              <li data-capabilities={note.capabilityIds.length} key={note.text}>
                <strong>{note.capabilityIds.length} capabilities:</strong>{" "}
                <InlineCode text={note.text} />
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {allUnreachable ? null : byArea(database, capabilities).map(({area, capabilities: areaCapabilities}) => {
        const shown = areaCapabilities
          .filter(
            (capability) =>
              view === "all" || supportByCapability.get(capability.id)!.code !== "y",
          )
          // Gaps first, so the rows a maintainer has to check lead each area.
          .sort(
            (left, right) =>
              Number(supportByCapability.get(left.id)!.code === "y") -
              Number(supportByCapability.get(right.id)!.code === "y"),
          );
        const present = areaCapabilities.length - shown.length;
        return (
          <section className="framework-area" key={area} aria-label={area}>
            <h2>{area}</h2>
            {shown.length ? (
              <ul className="framework-rows">
                {shown.map((capability) => {
                  const support = supportByCapability.get(capability.id)!;
                  const boundaryDependent = Boolean(
                    capability.boundary_dependent?.includes(frameworkId),
                  );
                  return (
                    <li
                      className="framework-row"
                      data-capability-id={capability.id}
                      data-support={support.code}
                      key={capability.id}
                    >
                      <Link
                        className="framework-row-title"
                        href={capabilityHref(capability.id)}
                      >
                        <CapabilityTitle capability={capability} />
                      </Link>
                      <span
                        className={`support-cell framework-cell support-${support.code}${
                          boundaryDependent ? " is-boundary-dependent" : ""
                        }`}
                        title={SUPPORT_META[support.code].label}
                      >
                        {SUPPORT_META[support.code].symbol}{" "}
                        {SUPPORT_META[support.code].shortLabel}
                      </span>
                      {support.code !== "y" &&
                      (support.noteRefs.length || boundaryDependent) ? (
                        <ul className="framework-row-notes">
                          {boundaryDependent ? (
                            <li className="framework-row-boundary">
                              Depends on which result boundary is measured.
                            </li>
                          ) : null}
                          {support.noteRefs.map((noteRef) => {
                            const text = capability.notes_by_num?.[String(noteRef)] ?? "";
                            const shared = support.code === "x" && sharedText.has(text);
                            return (
                              <li
                                data-note={noteRef}
                                data-shared={shared || undefined}
                                key={noteRef}
                              >
                                <a href={`${capabilityHref(capability.id)}?note=${noteRef}`}>
                                  #{noteRef}
                                </a>{" "}
                                {shared ? (
                                  "Unreachable for the shared reason stated above."
                                ) : (
                                  <InlineCode text={text} />
                                )}
                              </li>
                            );
                          })}
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
                {present} present at the capture boundary.
              </p>
            ) : null}
          </section>
        );
      })}

      <section className="framework-reproduce" aria-labelledby="framework-reproduce-title">
        <h2 id="framework-reproduce-title">Reproduce</h2>
        <p>
          These rerun every measurement on this page. No model call or model API key is
          required.
        </p>
        {commands.map((command) => (
          <CopyCodeBlock className="home-command" key={command} value={command} />
        ))}
        {database.feedback_url ? (
          <p>
            Something here wrong?{" "}
            <Link href={database.feedback_url} target="_blank" rel="noopener noreferrer">
              Send a correction
            </Link>
            .
          </p>
        ) : null}
      </section>
    </main>
  );
}
