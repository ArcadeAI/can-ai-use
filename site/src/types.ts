export type SupportCode = "y" | "a" | "n" | "x" | "u";

export interface VersionEntry {
  version: string;
  release_date?: string;
  release_date_unix?: number | null;
  era: number;
}

/**
 * One place a framework can be asked what a tool call returned.
 *
 * A tool definition has one obvious capture point. A tool result often has more than
 * one, and they can disagree, so each is named and exactly one is the path a real
 * agent takes.
 */
export interface ResultBoundary {
  id: string;
  label: string;
  capture_api: string;
  capture_object: string;
  agent_path: boolean;
  limitation?: string;
}

/** A project's status when it has been archived, shut down or replaced since it was measured. */
export interface Lifecycle {
  status: "archived" | "discontinued" | "transitioning" | string;
  since?: string;
  note: string;
  source?: string;
}

export interface Agent {
  lifecycle?: Lifecycle;
  name: string;
  abbr: string;
  type: "python" | "typescript" | "java" | "go" | "csharp" | string;
  package: string;
  registry: "pypi" | "npm" | "maven" | "go" | "nuget" | string;
  adapter: string;
  renderer: string;
  /** Where the tool *definition* was read. */
  capture_boundary: {
    capture_api: string;
    capture_object: string;
    capture_stage:
      | "framework_tool_definition"
      | "provider_format"
      | "provider_request";
    provider_request_captured: boolean;
    negotiated_mcp_spec_version: string | null;
    protocol_version_evidence: string;
  };
  /** Where a tool *result* was read. Every boundary the renderer can be asked at. */
  result_boundaries: ResultBoundary[];
  measurement_status?:
    | "measured"
    | "sdk_incompatible"
    | "protocol_mismatch";
  measurement_note?: string;
  /** Set on a row that publishes another row's cells because it uses that adapter. */
  aliases_of?: string;
  alias_note?: string;
  /** Preview rows are in the data but shown only with ?preview=1. */
  visibility?: "published" | "preview";
  current_version: string;
  stable_version: string;
  dev_version: string | null;
  version_list: VersionEntry[];
  /** The day this row was measured: the newest capture date across its cells (ISO date). */
  measured_on?: string | null;
}

export interface CapabilityExample {
  before_label?: string;
  before: string[];
  after_label?: string;
  after: string[];
}

export interface Capability {
  schema: string;
  id: string;
  title: string;
  description: string;
  question?: string;
  spec: string;
  spec_label?: string;
  docs_url?: string;
  docs_label?: string;
  status: string;
  categories?: string[];
  keywords?: string;
  stats: Record<string, Record<string, string>>;
  /** A tally of the rows shown, then the summary. The build counts the public rows. */
  notes?: string;
  /** Hand-written mechanism prose that names no count, kept so a view can recount. */
  summary?: string;
  notes_by_num?: Record<string, string>;
  /**
   * Agent ids whose answer here changes with which result boundary you look at.
   * Those cells are 'a', and their note has to say what each boundary yields.
   */
  boundary_dependent?: string[];
  verdict: {
    headline: string;
    detail?: string;
  };
  usage_perc_y?: number | null;
  usage_perc_a?: number | null;
  usage_note?: string;
  links?: Array<{url: string; title: string}>;
  known_issues?: string[];
  example?: CapabilityExample;
  why: string;
  reproduce?: string;
  measured: {
    // Null on a row the protocol defines but no scan has exercised yet.
    run_date: string | null;
    /** The day each framework's cell in this row was captured, in agent order. */
    measured_on?: Record<string, string>;
    mcp_spec: string;
    fixture?: string | null;
  };
  shown?: boolean;
  /** Which part of the protocol this row belongs to, from the spec surface. */
  area?: string;
  measurement_state?: "measured" | "not_measured";
  /** What a scan would have to do to turn this row into a measurement. */
  probe?: string;
  /** The wire methods, notifications or fields the feature is carried by. */
  wire?: string[];
}

export interface CoverageArea {
  area: string;
  measured: number;
  total: number;
}

/** How much of the protocol revision we have measured, computed at build time. */
export interface Coverage {
  revision: string;
  catalogued: boolean;
  source: string;
  measured: number;
  total: number;
  by_area: CoverageArea[];
  note: string;
  omissions: string[];
}

export interface MirrorDatabase {
  schema: string;
  updated: string;
  /** The newest day any published row was measured. */
  measured_on?: string;
  mcp_spec: string;
  repo: string;
  feedback_url?: string | null;
  eras: Record<string, string>;
  agents: Record<string, Agent>;
  statuses: Record<string, {short: string; label: string}>;
  support_codes: Record<SupportCode, string>;
  defaults: Record<string, string>;
  popularity?: {
    schema: string;
    mode: "mock" | "observed";
    generated_at: string;
    windows: string[];
    entries: Array<{
      capability_id: string;
      counts: Record<string, [number, number]>;
    }>;
  };
  coverage?: Coverage | null;
  cats: Record<string, string[]>;
  data: Record<string, Capability>;
}

export interface SpecIndexEntry {
  id: string;
  label: string;
  file: string;
  updated: string;
  measured_on?: string;
  measured_frameworks: number;
  total_frameworks: number;
  coverage?: Coverage | null;
  /** Rows with at least one cell that is not 'x' in this snapshot. */
  reached_by?: string[];
  /** Capabilities this revision adds over the next-older one; none for the oldest. */
  new_capabilities?: {id: string; title: string}[];
}

export interface SpecIndex {
  schema: string;
  default_spec: string;
  specs: SpecIndexEntry[];
  /** The harness table's own file. It is never one of the framework specs. */
  harnesses?: {file: string; updated: string; measured_on?: string; harnesses: number; rows: number};
}

/** A harness measured on the wire. Its rows never merge with the framework table. */
export interface Harness {
  name: string;
  abbr?: string;
  vendor: string;
  type: string;
  capture: string;
  measurement_status: string;
  visibility?: "published" | "preview";
  current_version: string;
  /** The newest stable release on the agent's registry; absent when it has none to read. */
  stable_version?: string;
  version_list: VersionEntry[];
  /** The day the harness's wire session was captured (ISO date); null when no session ran. */
  measured_on?: string | null;
  /** Why no session ran at any level, for a harness with no measurement yet. */
  measurement_blocked?: string;
  lifecycle?: Lifecycle;
  /**
   * The row was measured against the harness's real service (a --live run under op run)
   * because no stub can answer it; its results can vary between runs. Every other row is
   * stub-measured and carries none of these.
   */
  measured_live?: boolean;
  /** The service the live run ran against, as the note names it ("Cursor"). */
  live_service?: string;
  /** The model that service reported in the harness's own output. */
  model?: string | null;
  /** Inventory facts shown on the agent's page, from plan/harness_inventory. */
  tier?: number;
  install?: string | null;
  headless?: string | null;
  mcp_config?: string | null;
  /** The documented switch that removes approval prompts, when there is one. */
  unattended?: string | null;
}

/** One harness cell as build_data.py derives it from the wire capture. */
export interface HarnessCell {
  code: SupportCode;
  note: number | null;
  label: string;
  value: string | null;
  requested: string | null;
  /** The cell on each level-1 transport, present only when the transports differ. */
  transports?: HarnessTransportCell[] | null;
  /** Repository path of the capture the cell was read from. */
  capture: string;
  line: number | null;
  /** One line read off the cell's evidence note, present when the cell has a note. */
  summary?: string | null;
}

/** One transport's side of a cell the harness answers differently per transport. */
export interface HarnessTransportCell {
  transport: string;
  code: SupportCode;
  label: string;
  value: string | null;
  requested: string | null;
}

export interface HarnessRow {
  id: string;
  area: string;
  level: number;
  title: string;
  question?: string;
  /** What the behaviour means and how it is checked, for a reader new to MCP. */
  description?: string;
  observe: string;
  value_cell: boolean;
  stats: Record<string, Record<string, string>>;
  notes_by_num?: Record<string, string>;
  cells: Record<string, HarnessCell>;
  /** The same cell against each fixture revision, per harness. */
  by_revision?: Record<string, Record<string, {code: SupportCode; value?: string | null}>>;
  reproduce?: string;
  reproduce_by_agent?: Record<string, string>;
  measured: {run_date: string | null; measured_on?: Record<string, string>};
  /** The framework capability this behaviour is paired with, as the newest catalogue has it. */
  capability?: {
    id: string;
    title: string;
    docs_url: string;
    docs_label: string;
    status: string;
    revision: string;
  };
}

export interface HarnessDatabase {
  schema: string;
  updated: string;
  measured_on?: string;
  repo: string;
  feedback_url?: string | null;
  support_codes: Record<string, string>;
  vocabulary: string[];
  agents: Record<string, Harness>;
  areas: string[];
  data: Record<string, HarnessRow>;
  /** What the measurement runs: fixtures, prompts and the approval modes it measures in. */
  surface?: {
    prompts?: Record<string, string>;
    approval_modes?: Record<string, string>;
  };
}

declare global {
  interface Window {
    __MCP_MIRROR_DATA__?: MirrorDatabase;
    __MCP_MIRROR_DATASETS__?: Record<string, MirrorDatabase>;
    __MCP_MIRROR_SPEC_INDEX__?: SpecIndex;
    __MCP_MIRROR_HARNESSES__?: HarnessDatabase;
    /** Set only in standalone.html: the hosted site its file links resolve against. */
    __MCP_MIRROR_SITE_URL__?: string;
  }
}

export interface ParsedSupport {
  code: SupportCode;
  noteRefs: number[];
}

export type Route =
  | {name: "index"}
  | {name: "frameworks"}
  | {name: "capability"; capabilityId: string}
  | {name: "framework"; frameworkId: string}
  | {name: "stats"}
  | {name: "changes"}
  | {name: "method"}
  | {name: "agents"}
  | {name: "agentRow"; rowId: string}
  | {name: "agent"; agentId: string};
