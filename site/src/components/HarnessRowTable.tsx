import {shortDate} from "../lib/data";
import {
  agentHref,
  harnessLiveNote,
  harnessMeasuredOn,
  harnessSettledRevision,
  harnessUnmeasured,
  NEWEST_FIXTURE_REVISION,
} from "../lib/harnesses";
import type {Harness, HarnessCell, HarnessDatabase, HarnessRow} from "../types";
import {captureHref, HarnessEvidence} from "./HarnessParts";
import {ArrowUpRightIcon} from "./Icons";

/** A value short enough to sit in the cell, like a revision; longer ones go to the notes. */
const MAX_CELL_VALUE = 16;
export const shortValue = (value: string | null) =>
  value && value.length <= MAX_CELL_VALUE ? value : null;

/** The note anchor for a harness: its numbered note, or its long value when it has none. */
const noteKey = (harnessId: string, cell: HarnessCell) =>
  cell.note ? String(cell.note) : harnessId;

/**
 * One harness cell sized like a framework cell: the label, a #n marker for its
 * evidence note below the table, and a small link to the capture it was read from.
 */
function CompactHarnessCell({
  repo,
  harness,
  row,
  cell,
  noteId,
  hasNote,
  onNoteActivate,
}: {
  repo: string;
  harness: Harness;
  row: HarnessRow;
  cell: HarnessCell;
  noteId: string;
  hasNote: boolean;
  onNoteActivate?: () => void;
}) {
  const captureName = cell.capture.split("/").slice(-2).join("/");
  return (
    <div
      className={`harness-cell is-compact support-${cell.code}`}
      data-code={cell.code}
      data-label={cell.label}
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
              <em>{side.transport}</em> {shortValue(side.value) ?? side.label}
            </span>
          ))}
        </span>
      ) : (
        <span className="harness-cell-label">
          {shortValue(cell.value) ? <strong>{cell.value}</strong> : null} {cell.label}
        </span>
      )}
      <span className="harness-cell-marks">
        {hasNote ? (
          <a
            className="support-note-ref"
            href={`#${noteId}`}
            aria-label={`Read the ${cell.note ? `note ${cell.note}` : "evidence"} for ${harness.name} ${harness.current_version}`}
            onClick={(event) => {
              // Routes live in the hash, so focus the note instead of navigating.
              event.preventDefault();
              // A page that keeps the notes in a tab opens it first, then the note renders.
              onNoteActivate?.();
              window.requestAnimationFrame(() => {
                window.requestAnimationFrame(() => {
                  const target = document.getElementById(noteId);
                  target?.scrollIntoView({block: "nearest"});
                  target?.focus();
                });
              });
            }}
          >
            {cell.note ? `#${cell.note}` : "details"}
          </a>
        ) : null}
        <a
          className="harness-cell-capture"
          href={captureHref(repo, cell, harness.visibility !== "preview")}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={`${harness.name}, ${row.title}: ${cell.label}. Open the wire capture ${captureName}`}
          title={captureName}
        >
          capture
          <ArrowUpRightIcon width={10} height={10} />
        </a>
      </span>
    </div>
  );
}

/** The release after the measured one, when the registry has a newer stable version. */
export function newerRelease(harness: Harness): string | undefined {
  return harness.stable_version && harness.stable_version !== harness.current_version
    ? harness.stable_version
    : undefined;
}

/**
 * One note per cell with evidence to show: its numbered note, a value too long for the
 * cell, or both. A note with a summary shows that line, the rest one click away.
 */
function harnessNotes(row: HarnessRow, harnesses: [string, Harness][]) {
  return harnesses.flatMap(([harnessId, harness]) => {
    const cell = row.cells[harnessId];
    if (!cell) return [];
    const note = cell.note ? row.notes_by_num?.[String(cell.note)] : undefined;
    const longValue = cell.value && !shortValue(cell.value) ? cell.value : undefined;
    const text = [longValue, note].filter(Boolean).join(" ");
    const summary = note ? cell.summary ?? undefined : undefined;
    return text ? [{key: noteKey(harnessId, cell), harnessId, harness, cell, text, summary}] : [];
  });
}

/** The numbered notes under a harness row's grid, each linking its capture. */
export function HarnessRowNotes({
  data,
  row,
  harnesses,
  noteScope,
}: {
  data: HarnessDatabase;
  row: HarnessRow;
  harnesses: [string, Harness][];
  noteScope: string;
}) {
  const repo = data.repo;
  const noteId = (key: string) => `${noteScope}-note-${key}`;
  const notes = harnessNotes(row, harnesses);
  if (!notes.length) return null;
  return (
    <ul className="numbered-notes agent-handling-notes">
      {notes.map(({key, harness, cell, text, summary}) => {
        const label = `${cell.note ? `#${cell.note} · ` : ""}${harness.name} ${harness.current_version}`;
        const captureName = cell.capture.split("/").slice(-2).join("/");
        const href = captureHref(repo, cell, harness.visibility !== "preview");
        return summary ? (
          <li id={noteId(key)} key={key} tabIndex={-1} aria-label={`${label}: ${summary}`}>
            <strong className="note-reference-label">{label}</strong>
            <span className="harness-note-summary">{summary}</span>
            <HarnessEvidence
              evidence={text}
              href={href}
              captureName={captureName}
              line={cell.line}
            />
          </li>
        ) : (
          <li id={noteId(key)} key={key} tabIndex={-1} aria-label={`${label}: ${text}`}>
            <strong className="note-reference-label">{label}</strong>
            <span>
              {text}{" "}
              <a
                className="harness-note-capture"
                href={href}
                target="_blank"
                rel="noopener noreferrer"
              >
                {captureName}
                {cell.line ? `:${cell.line}` : ""}
                <ArrowUpRightIcon width={11} height={11} />
              </a>
            </span>
          </li>
        );
      })}
    </ul>
  );
}

/**
 * One harness row as a grid of agents, the way a capability page shows a grid of
 * frameworks: 112px a column, each column titled with the agent and the day it was
 * measured, the evidence in numbered notes under the table. Shared by the behaviour
 * page and the "How agents handle this" block on a capability page.
 */
export function HarnessRowTable({
  data,
  row,
  harnesses,
  noteScope,
  showNotes = true,
  onNoteActivate,
}: {
  data: HarnessDatabase;
  row: HarnessRow;
  harnesses: [string, Harness][];
  noteScope: string;
  /** False when the page shows the notes elsewhere, in its Notes tab. */
  showNotes?: boolean;
  onNoteActivate?: () => void;
}) {
  const repo = data.repo;
  const noteId = (key: string) => `${noteScope}-note-${key}`;
  const noted = new Set(harnessNotes(row, harnesses).map(({harnessId}) => harnessId));
  // Like a framework column's later releases: the newer version, unmeasured, under the
  // measured cell, so a reader sees the result may not hold for what they would install.
  const hasNewer = harnesses.some(([, harness]) => !harnessUnmeasured(harness) && newerRelease(harness));
  const showsRevision = harnesses.some(
    ([harnessId, harness]) => !harnessUnmeasured(harness) && harnessSettledRevision(data, harnessId),
  );

  return (
    <>
      <div className="harness-table-wrap">
        <table
          className="harness-table agent-handling-table"
          style={{minWidth: `${harnesses.length * 112}px`}}
        >
          <thead>
            <tr>
              {harnesses.map(([harnessId, harness]) => {
                const measured = harnessMeasuredOn(row, harnessId, harness);
                const unmeasured = harnessUnmeasured(harness);
                const liveNote = harnessLiveNote(harness);
                const revision = unmeasured ? undefined : harnessSettledRevision(data, harnessId);
                return (
                  <th
                    scope="col"
                    key={harnessId}
                    data-harness-id={harnessId}
                    data-measured-live={liveNote ? "true" : undefined}
                    title={
                      unmeasured
                        ? `${harness.name} · not measured yet: ${unmeasured}`
                        : liveNote
                          ? `${harness.name} · ${liveNote}${measured ? `; measured ${shortDate(measured)}` : ""}`
                          : measured
                            ? `${harness.name} · measured ${shortDate(measured)}`
                            : harness.name
                    }
                  >
                    <a className="agent-name" href={agentHref(harnessId)}>
                      {harness.name}
                    </a>
                    <small>{harness.current_version}</small>
                    {unmeasured ? (
                      <small className="harness-measured-on harness-not-measured">Not measured yet</small>
                    ) : measured ? (
                      <small className="harness-measured-on">
                        <time dateTime={measured}>{shortDate(measured)}</time>
                      </small>
                    ) : null}
                    {revision ? (
                      <small
                        className="harness-revision"
                        title={`Settled on MCP ${revision} when the server offered ${NEWEST_FIXTURE_REVISION}`}
                      >
                        MCP {revision}
                      </small>
                    ) : null}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            <tr data-row-id={row.id}>
              {harnesses.map(([harnessId, harness]) => {
                const cell = row.cells[harnessId];
                return (
                  <td key={harnessId}>
                    {cell ? (
                      <CompactHarnessCell
                        repo={repo}
                        harness={harness}
                        row={row}
                        cell={cell}
                        noteId={noteId(noteKey(harnessId, cell))}
                        hasNote={noted.has(harnessId)}
                        onNoteActivate={onNoteActivate}
                      />
                    ) : null}
                  </td>
                );
              })}
            </tr>
            {hasNewer ? (
              <tr className="harness-newer-row">
                {harnesses.map(([harnessId, harness]) => {
                  const newer = harnessUnmeasured(harness) ? undefined : newerRelease(harness);
                  const released = newer
                    ? harness.version_list.find((entry) => entry.version === newer)?.release_date
                    : undefined;
                  return (
                    <td key={harnessId}>
                      {newer ? (
                        <div
                          className="support-cell support-u harness-newer-cell"
                          data-version={newer}
                          title={`${harness.name} ${newer}${released ? `, released ${shortDate(released)}` : ""}: not measured yet`}
                          aria-label={`${harness.name} ${newer}, a newer release, not measured yet`}
                        >
                          <strong>{newer}</strong>
                        </div>
                      ) : null}
                    </td>
                  );
                })}
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
      {showsRevision ? (
        <p className="harness-revision-note">
          Each agent ran against an MCP 2025-11-25 server and a {NEWEST_FIXTURE_REVISION} server. The
          revision under its name is the one it settled on when offered {NEWEST_FIXTURE_REVISION}.
        </p>
      ) : null}
      {hasNewer ? (
        <p className="harness-revision-note harness-newer-note">
          The grey cell under a result is the agent&apos;s newest release, out since it was
          measured and not measured yet.
        </p>
      ) : null}
      {showNotes ? (
        <HarnessRowNotes data={data} row={row} harnesses={harnesses} noteScope={noteScope} />
      ) : null}
    </>
  );
}
