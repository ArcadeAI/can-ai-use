import type {Harness, HarnessDatabase, HarnessRow} from "../types";

/**
 * Which harnesses the page shows by default, and which harness row answers the
 * agent-side half of a framework capability. Kept in data/harness_links.json so a
 * pairing is a data change, never a component change.
 */
export interface HarnessLinks {
  schema: string;
  default_harnesses: string[];
  // How the default set was picked, in words; printed under the toggle.
  rule?: string;
  capabilities: Record<string, string>;
}

/** The harnesses on show, in data order: the default set unless `showAll`. */
export function shownHarnesses(
  harnesses: [string, Harness][],
  links: HarnessLinks,
  showAll: boolean,
): [string, Harness][] {
  if (showAll) return harnesses;
  const defaults = new Set(links.default_harnesses);
  return harnesses.filter(([id]) => defaults.has(id));
}

/** The harness row paired with a framework capability, if there is one. */
export function harnessRowFor(
  capabilityId: string,
  database: HarnessDatabase,
  links: HarnessLinks,
): HarnessRow | undefined {
  const rowId = links.capabilities[capabilityId];
  return rowId ? database.data[rowId] : undefined;
}

/** The line under the Filtered | All toggle, naming the default harnesses. */
export function harnessRule(database: HarnessDatabase, links: HarnessLinks): string {
  const names = links.default_harnesses
    .map((id) => database.agents[id]?.name)
    .filter(Boolean);
  return `Filtered: ${names.join(", ")}.${links.rule ? ` ${links.rule}` : ""}`;
}

export function agentsHref(rowId?: string): string {
  return rowId ? `#/agents/${encodeURIComponent(rowId)}` : "#/agents";
}

export function agentHref(agentId: string): string {
  return `#/agent/${encodeURIComponent(agentId)}`;
}

/** The harnesses a visitor sees, in data order: preview rows only with ?preview=1. */
export function visibleHarnesses(
  database: HarnessDatabase,
  showPreview: boolean,
): [string, Harness][] {
  return Object.entries(database.agents).filter(
    ([, harness]) => showPreview || harness.visibility !== "preview",
  );
}

/** Harness codes in the order the bars and tallies list them. */
export const HARNESS_CODES = ["y", "a", "n", "u"] as const;
export type HarnessCode = (typeof HARNESS_CODES)[number];

/**
 * A short word for each code, for a bar's label and an agent page's summary line. A code
 * answers the row's question, so on "is the destructive call held?" a block is a yes.
 */
export const HARNESS_CODE_LABELS: Record<HarnessCode, string> = {
  y: "yes",
  a: "in part",
  n: "no",
  u: "not measured",
};

/** How many of the given harnesses' cells in a row carry each code. */
export function rowCodeCounts(
  row: HarnessRow,
  harnessIds: string[],
): Record<HarnessCode, number> {
  const counts: Record<HarnessCode, number> = {y: 0, a: 0, n: 0, u: 0};
  for (const id of harnessIds) {
    const code = row.cells[id]?.code;
    if (code && code in counts) counts[code as HarnessCode] += 1;
  }
  return counts;
}

/** How many of a harness's cells across the rows carry each code. */
export function harnessCodeCounts(
  rows: HarnessRow[],
  harnessId: string,
): Record<HarnessCode, number> {
  const counts: Record<HarnessCode, number> = {y: 0, a: 0, n: 0, u: 0};
  for (const row of rows) {
    const code = row.cells[harnessId]?.code;
    if (code && code in counts) counts[code as HarnessCode] += 1;
  }
  return counts;
}

/**
 * A row's tally in its own words: each cell label with how many harnesses carry it,
 * in code order so the answer the row asks about comes first.
 */
export function rowLabelTally(
  row: HarnessRow,
  harnessIds: string[],
): {label: string; code: HarnessCode; count: number}[] {
  const tally = new Map<string, {label: string; code: HarnessCode; count: number}>();
  for (const id of harnessIds) {
    const cell = row.cells[id];
    if (!cell) continue;
    const entry = tally.get(cell.label) ?? {label: cell.label, code: cell.code as HarnessCode, count: 0};
    entry.count += 1;
    tally.set(cell.label, entry);
  }
  return [...tally.values()].sort(
    (left, right) =>
      HARNESS_CODES.indexOf(left.code) - HARNESS_CODES.indexOf(right.code) ||
      right.count - left.count,
  );
}

/** The day a harness's cell in a row was captured, else the day it was last measured. */
export function harnessMeasuredOn(row: HarnessRow, harnessId: string, harness: Harness): string | undefined {
  if (harnessUnmeasured(harness)) return undefined;
  return row.measured.measured_on?.[harnessId] ?? harness.measured_on ?? undefined;
}

/** The newest revision a fixture offers; what an agent settles on against it is its ceiling. */
export const NEWEST_FIXTURE_REVISION = "2026-07-28";

/**
 * The MCP revision a harness settled on when the server offered the newest revision,
 * read from the negotiated-revision row ("2025-11-25", or "2024-11-05 / 2025-11-25" when
 * stdio and HTTP differ). Undefined when that session was not measured.
 */
export function harnessSettledRevision(database: HarnessDatabase, harnessId: string): string | undefined {
  const value = database.data["h-negotiated-revision"]?.by_revision?.[harnessId]?.[NEWEST_FIXTURE_REVISION]?.value;
  if (!value) return undefined;
  const settled = [...value.matchAll(/(\d{4}-\d{2}-\d{2}) \(requested/g)].map((match) => match[1]);
  return [...new Set(settled)].join(" / ") || undefined;
}

/**
 * Why a harness has no measurement yet, when its run opened no session at any level.
 * Such an agent keeps its place in the lists but shows "Not measured yet" and this
 * line, never a date and never a bar that reads as results.
 */
export function harnessUnmeasured(harness: Harness): string | undefined {
  return harness.measurement_blocked || undefined;
}

/**
 * The line the agent's page and its column tooltips show for a live-measured agent, so a
 * reader never takes its cells for the stubbed, repeatable kind. Undefined for a stub-measured row.
 */
export function harnessLiveNote(harness: Harness): string | undefined {
  if (!harness.measured_live) return undefined;
  const service = harness.live_service || harness.name;
  return `measured live against ${service}'s service; results can vary between runs`;
}

/** "14 agents measured against 24 behaviours; 2 not measurable yet." An agent that ran no session is not counted as measured. */
export function harnessCoverageLine(agents: [string, Harness][], behaviours: number): string {
  const blocked = agents.filter(([, harness]) => harnessUnmeasured(harness)).length;
  const line = `${agents.length - blocked} agents measured against ${behaviours} behaviours`;
  return blocked ? `${line}; ${blocked} not measurable yet.` : `${line}.`;
}

/** Whether a harness row answers a search, by its title, question, id or what it observes. */
export function matchesHarnessRow(row: HarnessRow, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  return [row.title, row.question ?? "", row.observe, row.id, row.area]
    .join(" ")
    .toLowerCase()
    .includes(needle);
}
