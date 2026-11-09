import {
  ToggleButton,
  ToggleButtonGroup,
} from "@heroui/react";
import {useId, useState} from "react";

import {
  capabilityHref,
  frameworkHref,
  measuredOn,
  parseSupport,
  shortDate,
  SUPPORT_META,
} from "../lib/data";
import {groupFrameworks, shortFrameworkName} from "../lib/frameworks";
import {RUNNER_MANIFESTS} from "../lib/frameworkSources";
import type {Capability, MirrorDatabase} from "../types";

export type GridMode = "support" | "date";

export function CapabilityGrid({
  database,
  capability,
  compact = false,
  hideToolbar = false,
  showAllVersions = true,
  showCaption = true,
  mode: controlledMode,
  onModeChange,
  noteTargetPrefix,
  onNoteActivate,
}: {
  database: MirrorDatabase;
  capability: Capability;
  compact?: boolean;
  hideToolbar?: boolean;
  showAllVersions?: boolean;
  showCaption?: boolean;
  mode?: GridMode;
  onModeChange?: (mode: GridMode) => void;
  noteTargetPrefix?: string;
  onNoteActivate?: (note: number) => void;
}) {
  const [localMode, setLocalMode] = useState<GridMode>("support");
  const mode = controlledMode ?? localMode;
  const setMode = onModeChange ?? setLocalMode;
  const headingId = useId();
  const agentEntries = Object.entries(database.agents).map(
    ([agentId, agent]) =>
      [
        agentId,
        agent,
        showAllVersions
          ? agent.version_list
          : (() => {
              const measuredIndex = agent.version_list.findIndex(
                (version) => version.era === 0,
              );
              if (measuredIndex < 0) return agent.version_list.slice(-3);
              return agent.version_list.slice(
                Math.max(0, measuredIndex - 1),
                measuredIndex + 2,
              );
            })(),
      ] as const,
  );
  const groups = groupFrameworks(database, RUNNER_MANIFESTS);
  const entriesById = new Map(agentEntries.map((entry) => [entry[0], entry]));
  const rowCount = Math.max(
    ...agentEntries.map(([, , versions]) => versions.length),
    0,
  );

  return (
    <section
      className={`capability-grid-section${compact ? " is-compact" : ""}`}
      aria-labelledby={headingId}
    >
      {hideToolbar ? (
        <h3 className="sr-only" id={headingId}>Framework support</h3>
      ) : (
        <div className="grid-toolbar">
          <div>
            <h3 id={headingId}>Framework support</h3>
            <p>Columns are adapters. The outlined cell is the tested release.</p>
          </div>
          {!compact ? (
          <ToggleButtonGroup
            aria-label="Version cell detail"
            disallowEmptySelection
            selectionMode="single"
            selectedKeys={[mode]}
            size="sm"
            onSelectionChange={(keys) => {
              const selected = Array.from(keys)[0];
              if (selected === "support" || selected === "date") {
                setMode(selected);
              }
            }}
          >
            <ToggleButton id="support">Result</ToggleButton>
            <ToggleButtonGroup.Separator />
            <ToggleButton id="date">Release date</ToggleButton>
          </ToggleButtonGroup>
          ) : null}
        </div>
      )}

      <div className="support-grid-scroll">
        <div className="support-grid" data-group-count={groups.length}>
          {groups.map((group) => (
            <section
              className={`framework-group framework-group-${group.id}`}
              data-group={group.id}
              key={group.id}
              aria-label={`${group.label} frameworks`}
              style={{flexGrow: group.agentIds.length}}
            >
              <h4 className="framework-group-label">{group.label}</h4>
              <div
                className="framework-group-columns"
                style={{
                  gridTemplateColumns: `repeat(${group.agentIds.length}, minmax(max-content, 1fr))`,
                }}
              >
                {group.agentIds.map((agentId) => entriesById.get(agentId)!).map(([agentId, agent, versions]) => (
                  <div className="agent-column" key={agentId}>
                    <div
                      className={`agent-heading language-${agent.type}`}
                      title={
                        measuredOn(capability, agentId, agent)
                          ? `${agent.name} · measured ${measuredOn(capability, agentId, agent)}`
                          : agent.name
                      }
                    >
                      <a className="agent-name" href={frameworkHref(agentId)}>
                        <span aria-hidden="true">{shortFrameworkName(agent.name)}</span>
                        <span className="sr-only">{agent.name}</span>
                      </a>
                    </div>

                    {Array.from({length: rowCount}, (_, rowIndex) => {
                      const leadingGapCount = rowCount - versions.length;
                      const version =
                        rowIndex < leadingGapCount
                          ? undefined
                          : versions[rowIndex - leadingGapCount];
                      if (!version) {
                        return (
                          <div
                            className="support-cell is-empty"
                            aria-hidden="true"
                            data-agent-id={agentId}
                            data-gap="leading"
                            data-row-index={rowIndex}
                            key={`gap-${rowIndex}`}
                          />
                        );
                      }

                      const support = parseSupport(
                        capability.stats[agentId]?.[version.version],
                      );
                      const detail =
                        mode === "date"
                          ? shortDate(version.release_date)
                          : SUPPORT_META[support.code].shortLabel;
                      const notes = support.noteRefs.length
                        ? `, notes ${support.noteRefs.join(", ")}`
                        : "";
                      // A cell whose answer changes with the result boundary says so on
                      // the cell, because the code alone cannot carry "it depends where
                      // you look" and a reader should not have to find that in prose.
                      const boundaryDependent =
                        version.era === 0 &&
                        Boolean(capability.boundary_dependent?.includes(agentId));
                      const boundaryNote = boundaryDependent
                        ? ", depends on which result boundary is measured"
                        : "";

                      return (
                        <div
                          className={`support-cell support-${support.code}${
                            version.era === 0 ? " is-current" : ""
                          }${boundaryDependent ? " is-boundary-dependent" : ""}`}
                          key={version.version}
                          data-agent-id={agentId}
                          data-current={version.era === 0 || undefined}
                          data-boundary-dependent={boundaryDependent || undefined}
                          data-row-index={rowIndex}
                          data-version={version.version}
                          title={`${agent.name} ${version.version}: ${SUPPORT_META[support.code].label}${notes}${boundaryNote}`}
                          aria-label={`${agent.name} ${version.version}, ${SUPPORT_META[support.code].label}${notes}${boundaryNote}`}
                        >
                          <strong>{version.version}</strong>
                          <small>
                            {mode === "support"
                              ? `${SUPPORT_META[support.code].symbol} ${detail}`
                              : detail}
                          </small>
                          {support.noteRefs.length || boundaryDependent ? (
                            <span className="support-cell-marks">
                              {boundaryDependent ? (
                                <span className="boundary-flag" aria-hidden="true">
                                  boundary
                                </span>
                              ) : null}
                              {support.noteRefs.length ? (
                                <sup className="support-note-refs">
                                  {support.noteRefs.map((noteRef) => (
                                    <a
                                      className="support-note-ref"
                                      href={
                                        noteTargetPrefix
                                          ? `#${noteTargetPrefix}-note-${noteRef}`
                                          : `${capabilityHref(capability.id)}?note=${noteRef}`
                                      }
                                      aria-label={`Read note ${noteRef} for ${agent.name} ${version.version}`}
                                      key={noteRef}
                                      onClick={(event) => {
                                        if (onNoteActivate) {
                                          event.preventDefault();
                                          onNoteActivate(noteRef);
                                        }
                                      }}
                                    >
                                      #{noteRef}
                                    </a>
                                  ))}
                                </sup>
                              ) : null}
                            </span>
                          ) : null}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            </section>
          ))}
        </div>
      </div>

      {!compact && showCaption ? (
        <p className="grid-caption">
          Newest listed releases align on the bottom row. Shorter histories use
          leading neutral cells. Release numbers and dates come from PyPI and
          npm. Superscript <strong>#n</strong> links to that cell&apos;s
          explanation in Notes.
        </p>
      ) : null}
    </section>
  );
}
