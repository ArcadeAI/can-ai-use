// The published support data as queries: a TypeScript port of src/mcp_mirror/catalog.py.
//
// tests/test_worker_parity.py runs both on the built data and requires identical JSON,
// so every function here mirrors its Python twin line for line, including Python's
// truthiness and `None` (always emitted as null, never dropped from the output).

type Json = Record<string, unknown>;
export type Snapshots = Record<string, Json>;
export type FetchJson = (path: string) => Promise<unknown>;

export const SUPPORT_LABELS: Record<string, string> = {
  y: "preserved at the declared capture boundary",
  a: "changed or retained at another boundary",
  n: "not exposed at the declared capture boundary",
  x: "adapter could not negotiate this protocol revision",
  u: "not measured",
};

const CACHE_MS = 600_000;

const isObject = (value: unknown): value is Json =>
  typeof value === "object" && value !== null && !Array.isArray(value);

// Python truthiness: "", 0, None, False, [] and {} are all false.
const truthy = (value: unknown): boolean => {
  if (Array.isArray(value)) return value.length > 0;
  if (isObject(value)) return Object.keys(value).length > 0;
  return Boolean(value);
};

const orNull = (value: unknown): unknown => (value === undefined ? null : value);

// Python's dict.get(key, fallback): the fallback applies only when the key is absent.
const getOr = (obj: Json, key: string, fallback: unknown): unknown => (key in obj ? obj[key] : fallback);

// Python's str(value): used only to build search haystacks.
const pyStr = (value: unknown): string => {
  if (value === undefined) return "";
  if (value === null) return "None";
  if (value === true) return "True";
  if (value === false) return "False";
  if (Array.isArray(value)) return `[${value.map(pyRepr).join(", ")}]`;
  if (isObject(value))
    return `{${Object.entries(value).map(([k, v]) => `${pyRepr(k)}: ${pyRepr(v)}`).join(", ")}}`;
  return String(value);
};

export const pyRepr = (value: unknown): string => {
  if (typeof value !== "string") return pyStr(value);
  const quote = value.includes("'") && !value.includes('"') ? '"' : "'";
  const escaped = value.replace(/\\/g, "\\\\").replace(new RegExp(quote, "g"), `\\${quote}`);
  return `${quote}${escaped}${quote}`;
};

const sortedStrings = (values: Iterable<string>): string[] =>
  [...values].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));

const publishedOnly = (payload: Json): Json => {
  // Preview rows are built so the site can show them on request; the server has no
  // such request, so it answers for published rows only.
  const agents = isObject(payload.agents) ? payload.agents : {};
  payload.agents = Object.fromEntries(
    Object.entries(agents).filter(
      ([, agent]) => (isObject(agent) ? (agent.visibility ?? "published") : "published") !== "preview",
    ),
  );
  return payload;
};

export class Catalog {
  fetchJson: FetchJson;
  cache: {at: number; snapshots: Snapshots; defaultRevision: string | null} | null = null;

  constructor(fetchJson: FetchJson) {
    this.fetchJson = fetchJson;
  }

  // The published site's index and one file per revision, cached for ten minutes.
  async load(): Promise<{snapshots: Snapshots; defaultRevision: string | null}> {
    const now = Date.now();
    if (this.cache && now - this.cache.at < CACHE_MS) return this.cache;
    const index = await this.fetchJson("specs.json");
    const specs = isObject(index) && Array.isArray(index.specs) ? index.specs : [];
    const snapshots: Snapshots = {};
    for (const entry of specs) {
      if (!isObject(entry) || typeof entry.file !== "string") continue;
      const payload = await this.fetchJson(entry.file);
      if (!isObject(payload)) continue;
      const revision = payload.mcp_spec;
      if (typeof revision === "string" && revision) snapshots[revision] = publishedOnly(payload);
    }
    if (Object.keys(snapshots).length === 0) throw new Error("no canaiuse snapshots found in the deployed data");
    const defaultRevision = isObject(index) && typeof index.default_spec === "string" ? index.default_spec : null;
    this.cache = {at: now, snapshots, defaultRevision};
    return this.cache;
  }

  async listSnapshots(): Promise<Json> {
    return listSnapshotsData((await this.load()).snapshots);
  }

  async searchCapabilities(query: string, revision: string | null = null): Promise<Json> {
    const data = await this.load();
    return searchCapabilitiesData(data, query, revision);
  }

  async lookupCapability(capabilityId: string, revision: string | null = null, frameworks: string[] | null = null): Promise<Json> {
    const data = await this.load();
    return capabilityData(data, capabilityId, revision, frameworks);
  }

  async inspectToolDefinition(tool: Json, revision: string | null = null, frameworks: string[] | null = null): Promise<Json> {
    const data = await this.load();
    return inspectToolDefinitionData(data, tool, revision, frameworks);
  }

  async inspectToolResult(result: Json, revision: string | null = null, frameworks: string[] | null = null): Promise<Json> {
    const data = await this.load();
    return inspectToolResultData(data, result, revision, frameworks);
  }
}

type Loaded = {snapshots: Snapshots; defaultRevision: string | null};

const defaultRevision = ({snapshots, defaultRevision}: Loaded): string =>
  defaultRevision !== null && defaultRevision in snapshots ? defaultRevision : sortedStrings(Object.keys(snapshots)).at(-1)!;

const resolveSnapshot = (data: Loaded, revision: string | null): [string, Json] => {
  const {snapshots} = data;
  if (revision === null || revision === "" || revision === "default") {
    const selected = defaultRevision(data);
    return [selected, snapshots[selected]];
  }
  if (revision === "latest") {
    const selected = sortedStrings(Object.keys(snapshots)).at(-1)!;
    return [selected, snapshots[selected]];
  }
  if (!(revision in snapshots)) {
    const available = sortedStrings(Object.keys(snapshots)).join(", ");
    throw new Error(`unknown MCP revision ${pyRepr(revision)}; available: ${available}`);
  }
  return [revision, snapshots[revision]];
};

const agentsOf = (snapshot: Json): Record<string, Json> =>
  isObject(snapshot.agents) ? (snapshot.agents as Record<string, Json>) : {};
const capabilitiesOf = (snapshot: Json): Record<string, Json> =>
  isObject(snapshot.data) ? (snapshot.data as Record<string, Json>) : {};

export function listSnapshotsData(snapshots: Snapshots): Json {
  const rows = [];
  for (const revision of sortedStrings(Object.keys(snapshots)).reverse()) {
    const snapshot = snapshots[revision];
    const coverage = isObject(snapshot.coverage) ? snapshot.coverage : {};
    rows.push({
      revision,
      data_updated: orNull(snapshot.updated),
      measured_on: orNull(snapshot.measured_on),
      measured_features: orNull(getOr(coverage, "measured", getOr(coverage, "measured_features", null))),
      total_features: orNull(getOr(coverage, "total", getOr(coverage, "total_features", null))),
      frameworks: Object.entries(agentsOf(snapshot)).map(([frameworkId, framework]) => ({
        id: frameworkId,
        name: getOr(framework, "name", frameworkId),
        version: orNull(framework.current_version),
        version_released: releaseDate(framework),
        measured_on: orNull(framework.measured_on),
      })),
    });
  }
  return {snapshots: rows};
}

// The release date of the measured version (era 0 in the version list).
const releaseDate = (framework: Json): string | null => {
  const current = framework.current_version;
  const versions = Array.isArray(framework.version_list) ? framework.version_list : [];
  for (const entry of versions) {
    if (isObject(entry) && entry.version === current) {
      const date = (truthy(entry.release_date) ? String(entry.release_date) : "").slice(0, 10);
      return date || null;
    }
  }
  return null;
};

// Where the published tool-result cell was read: the row's direct (non-agent) boundary
// from the data, following an alias to the row it shares an adapter with.
const directResultBoundary = (snapshot: Json, frameworkId: string): Json => {
  const agents = agentsOf(snapshot);
  const framework = agents[frameworkId];
  const ownerId = truthy(framework.aliases_of) ? String(framework.aliases_of) : frameworkId;
  const owner = agents[ownerId] ?? framework;
  const boundaries = Array.isArray(owner.result_boundaries) ? owner.result_boundaries : [];
  const direct =
    boundaries.find((b) => isObject(b) && !truthy(b.agent_path)) ?? (boundaries.length ? boundaries[0] : {});
  const boundary = isObject(direct) ? direct : {};
  return {
    capture_api: getOr(boundary, "capture_api", "unknown"),
    capture_object: getOr(boundary, "capture_object", "unknown"),
  };
};

const cellMeasuredOn = (capability: Json, framework: Json, frameworkId: string): unknown => {
  const measured = isObject(capability.measured) ? capability.measured : {};
  const byFramework = isObject(measured.measured_on) ? (measured.measured_on as Json) : {};
  return byFramework[frameworkId] || framework.measured_on || null;
};

const cellDetail = (capability: Json, snapshot: Json, frameworkId: string): Json => {
  const framework = agentsOf(snapshot)[frameworkId];
  const version = framework.current_version;
  const stats = isObject(capability.stats) ? capability.stats : {};
  const byVersion = isObject(stats[frameworkId]) ? (stats[frameworkId] as Json) : {};
  const raw = byVersion[String(version)] ?? "u";
  const code = pyStr(raw).trim().split(/\s+/)[0];
  let note: unknown = null;
  const match = /#(\d+)/.exec(pyStr(raw));
  if (match) {
    const notes = isObject(capability.notes_by_num) ? capability.notes_by_num : {};
    note = orNull(notes[match[1]]);
  }
  const categories = Array.isArray(capability.categories) ? capability.categories : [];
  let captureBoundary: unknown;
  if (categories.includes("Tool results")) {
    captureBoundary = {
      ...directResultBoundary(snapshot, frameworkId),
      capture_stage: "framework_tool_result",
      provider_request_captured: false,
    };
  } else {
    captureBoundary = orNull(framework.capture_boundary);
  }
  return {
    framework_id: frameworkId,
    framework: getOr(framework, "name", frameworkId),
    version: orNull(version),
    code,
    meaning: SUPPORT_LABELS[code] ?? "unknown support code",
    mechanism: note,
    capture_boundary: captureBoundary,
    // The day this framework's cell was captured; falls back to the day the framework
    // was last measured when the row itself carries no date for it (as catalog.py).
    measured_on: cellMeasuredOn(capability, framework, frameworkId),
  };
};

export function capabilityData(data: Loaded, capabilityId: string, revision: string | null, frameworks: string[] | null): Json {
  const [selectedRevision, snapshot] = resolveSnapshot(data, revision);
  const capabilities = capabilitiesOf(snapshot);
  if (!(capabilityId in capabilities)) {
    const matches = searchRows(snapshot, capabilityId).slice(0, 8).map((item) => item.id);
    const suffix = matches.length ? `; close matches: ${matches.join(", ")}` : "";
    throw new Error(`unknown capability ${pyRepr(capabilityId)}${suffix}`);
  }
  const capability = capabilities[capabilityId];
  const agents = agentsOf(snapshot);
  const selectedFrameworks = truthy(frameworks) ? frameworks! : Object.keys(agents);
  const unknown = selectedFrameworks.filter((item) => !(item in agents));
  if (unknown.length) throw new Error(`unknown framework id(s): ${unknown.join(", ")}`);
  return {
    revision: selectedRevision,
    id: capability.id,
    title: orNull(capability.title),
    question: orNull(capability.question),
    description: orNull(capability.description),
    observations: selectedFrameworks.map((frameworkId) => cellDetail(capability, snapshot, frameworkId)),
    summary: orNull(capability.notes),
    limitations: getOr(capability, "known_issues", []),
    source: truthy(capability.docs_url) ? capability.docs_url : orNull(capability.spec),
    reproduce: orNull(capability.reproduce),
    measured: orNull(capability.measured),
  };
}

type SearchRow = {id: string; title: unknown; question: unknown; categories: unknown; measured: unknown};

const searchRows = (snapshot: Json, query: string): SearchRow[] => {
  const terms = query.split(/\s+/).filter((term) => term).map((term) => term.toLowerCase());
  const scored: [number, SearchRow][] = [];
  for (const capability of Object.values(capabilitiesOf(snapshot))) {
    const haystack = ["id", "title", "description", "question", "keywords"]
      .map((key) => pyStr(capability[key]))
      .join(" ")
      .toLowerCase();
    if (!terms.length || !terms.every((term) => haystack.includes(term))) continue;
    const id = pyStr(capability.id).toLowerCase();
    const score = terms.reduce((total, term) => total + (id.includes(term) ? 3 : 1), 0);
    scored.push([
      score,
      {
        id: capability.id as string,
        title: orNull(capability.title),
        question: orNull(capability.question),
        categories: getOr(capability, "categories", []),
        measured: orNull(capability.measured),
      },
    ]);
  }
  return scored
    .sort(([scoreA, a], [scoreB, b]) => scoreB - scoreA || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))
    .map(([, row]) => row);
};

export function searchCapabilitiesData(data: Loaded, query: string, revision: string | null): Json {
  const [selectedRevision, snapshot] = resolveSnapshot(data, revision);
  return {revision: selectedRevision, query, matches: searchRows(snapshot, query).slice(0, 20)};
}

const schemaCapabilities = (schema: unknown): Set<string> => {
  const found = new Set<string>();
  if (!isObject(schema)) return found;
  if ("required" in schema) found.add("required");
  if ("enum" in schema) found.add("enum");
  if ("format" in schema) found.add("format");
  if ("default" in schema) found.add("default-value");
  if ("additionalProperties" in schema) found.add("additional-properties");
  if ("oneOf" in schema || "anyOf" in schema) found.add("one-of-any-of");
  // A constraint inside a branch is still a constraint the tool declares.
  for (const key of ["oneOf", "anyOf"]) {
    const branches = schema[key];
    if (Array.isArray(branches)) for (const branch of branches) for (const item of schemaCapabilities(branch)) found.add(item);
  }
  if (["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"].some((key) => key in schema)) found.add("numeric-range");
  const schemaType = schema.type;
  if (schemaType === "object" && isObject(schema.properties)) {
    const properties = Object.values(schema.properties);
    if (properties.some((value) => isObject(value) && truthy(value.description))) found.add("property-descriptions");
    if (properties.some((value) => isObject(value) && value.type === "object")) found.add("nested-objects");
    for (const child of properties) for (const item of schemaCapabilities(child)) found.add(item);
  }
  if (schemaType === "array") {
    if (isObject(schema.items) && schema.items.type === "object") found.add("arrays-of-objects");
    for (const item of schemaCapabilities(schema.items)) found.add(item);
  }
  if (Array.isArray(schemaType) && schemaType.includes("null")) found.add("null-acceptance");
  return found;
};

const identifiedCapabilities = (data: Loaded, identified: Set<string>, revision: string | null, frameworks: string[] | null) => {
  const [selectedRevision, snapshot] = resolveSnapshot(data, revision);
  const published = new Set(Object.keys(capabilitiesOf(snapshot)));
  const capabilities = sortedStrings([...identified].filter((id) => published.has(id))).map((id) =>
    capabilityData(data, id, selectedRevision, frameworks),
  );
  return {selectedRevision, capabilities};
};

export function inspectToolDefinitionData(data: Loaded, tool: Json, revision: string | null, frameworks: string[] | null): Json {
  const identified = new Set(["tool-name"]);
  const description = tool.description;
  if (typeof description === "string" && description) {
    identified.add("description-fidelity");
    if ([...description].length >= 512) identified.add("long-descriptions");
  }
  const annotations = tool.annotations;
  if (isObject(annotations)) {
    const mapping: Record<string, string> = {
      title: "title-annotation",
      readOnlyHint: "read-only-hint",
      destructiveHint: "destructive-hint",
      idempotentHint: "idempotent-hint",
      openWorldHint: "open-world-hint",
    };
    for (const key of Object.keys(mapping)) if (key in annotations) identified.add(mapping[key]);
  }
  if (truthy(tool.icons)) identified.add("tool-icons");
  for (const item of schemaCapabilities(tool.inputSchema)) identified.add(item);
  const {selectedRevision, capabilities} = identifiedCapabilities(data, identified, revision, frameworks);
  return {
    revision: selectedRevision,
    tool: orNull(tool.name),
    identified_capabilities: capabilities.map((item) => item.id),
    compatibility: capabilities,
    note: "Results describe published capture boundaries, not overall framework quality.",
  };
}

export function inspectToolResultData(data: Loaded, result: Json, revision: string | null, frameworks: string[] | null): Json {
  const identified = new Set(["tools-call"]);
  if (result.isError === true) identified.add("tool-execution-errors");
  if ("structuredContent" in result) identified.add("structured-content");
  const mapping: Record<string, string> = {
    text: "tool-text-content",
    image: "tool-image-content",
    audio: "tool-audio-content",
    resource: "embedded-resource-content",
    resource_link: "resource-link-content",
  };
  const content = Array.isArray(result.content) ? result.content : [];
  for (const block of content) {
    if (!isObject(block)) continue;
    const blockType = block.type;
    if (typeof blockType === "string" && blockType in mapping) identified.add(mapping[blockType]);
    if (truthy(block.annotations)) identified.add("content-annotations");
  }
  const {selectedRevision, capabilities} = identifiedCapabilities(data, identified, revision, frameworks);
  return {
    revision: selectedRevision,
    identified_capabilities: capabilities.map((item) => item.id),
    compatibility: capabilities,
    note: "Results describe published capture boundaries, not model behavior.",
  };
}
