import type {
  Agent,
  Capability,
  MirrorDatabase,
  ParsedSupport,
  Route,
  SpecIndex,
  SupportCode,
} from "../types";

export const SUPPORT_META: Record<
  SupportCode,
  {label: string; shortLabel: string; symbol: string}
> = {
  y: {
    label: "Present at capture boundary",
    shortLabel: "Present",
    symbol: "✓",
  },
  a: {
    label: "Changed or retained elsewhere",
    shortLabel: "Changed/retained",
    symbol: "◐",
  },
  n: {
    label: "Dropped during adaptation",
    shortLabel: "Dropped",
    symbol: "×",
  },
  x: {
    label: "Adapter cannot negotiate this revision",
    shortLabel: "Unreachable",
    symbol: "⊘",
  },
  u: {
    label: "Not yet measured",
    shortLabel: "Unmeasured",
    symbol: "?",
  },
};

export function parseSupport(raw?: string): ParsedSupport {
  if (!raw) {
    return {code: "u", noteRefs: []};
  }

  const match = raw.trim().match(/^([yanxu])((?:\s+#\d+)*)$/);
  if (!match) {
    return {code: "u", noteRefs: []};
  }

  return {
    code: match[1] as SupportCode,
    noteRefs: (match[2].match(/#(\d+)/g) ?? []).map((value) =>
      Number(value.slice(1)),
    ),
  };
}

export function currentSupport(
  database: MirrorDatabase,
  capability: Capability,
  agentId: string,
): ParsedSupport {
  const agent = database.agents[agentId];
  return parseSupport(capability.stats[agentId]?.[agent.current_version]);
}

export function supportCounts(
  database: MirrorDatabase,
  capability: Capability,
): Record<SupportCode, number> {
  const counts: Record<SupportCode, number> = {y: 0, a: 0, n: 0, x: 0, u: 0};
  for (const agentId of Object.keys(database.agents)) {
    counts[currentSupport(database, capability, agentId).code] += 1;
  }
  return counts;
}

/**
 * A row backed by an actual adapter run. Rows the protocol defines but no scan has
 * exercised are published with every cell unmeasured, and they have no run date, so they
 * are excluded anywhere the page is reporting measurement history.
 */
export function isMeasured(capability: Capability): boolean {
  return (
    capability.measurement_state !== "not_measured" &&
    Boolean(capability.measured.run_date)
  );
}

export function previewRequested(search: string): boolean {
  return new URLSearchParams(search).get("preview") === "1";
}

/**
 * The dataset a visitor sees. Preview rows are built into the data so a maintainer can
 * be shown one before it is public, and they are removed here unless ?preview=1 is set.
 * The build numbers preview notes after every published one, so dropping them leaves the
 * public numbering contiguous.
 */
export function visibleDatabase(
  database: MirrorDatabase,
  showPreview: boolean,
): MirrorDatabase {
  const hidden = new Set(
    Object.entries(database.agents)
      .filter(([, agent]) => agent.visibility === "preview")
      .map(([agentId]) => agentId),
  );
  return showPreview ? withTally(database) : withoutAgents(database, hidden);
}

/**
 * The counting sentence that opens a capability's notes, over the rows given. Mirrors
 * `tally` in src/mcp_mirror/cells.py, which writes the public view into the data. It
 * counts every row given, as the Observed header does, so the two agree.
 */
export function tally(
  agents: MirrorDatabase["agents"],
  capability: Capability,
): string {
  const codes = Object.keys(agents).map((agentId) => parseSupport(Object.values(capability.stats[agentId] ?? {})[0]).code);
  if (!codes.length) {
    return "";
  }
  const count = (code: SupportCode) => codes.filter((value) => value === code).length;
  const parts = [
    `${count("y")} of ${codes.length} ${codes.length === 1 ? "framework" : "frameworks"} present at the capture boundary`,
    `${count("a")} changed or retained elsewhere`,
    `${count("n")} dropped`,
  ];
  if (count("u")) {
    parts.push(`${count("u")} unmeasured`);
  }
  if (count("x")) {
    parts.push(`${count("x")} cannot reach MCP ${capability.measured.mcp_spec}`);
  }
  return `${parts.join(", ")}.`;
}

/** Recount every capability's notes over the rows this dataset still has. */
function withTally(database: MirrorDatabase): MirrorDatabase {
  const data = Object.fromEntries(
    Object.entries(database.data).map(([capabilityId, capability]) => [
      capabilityId,
      {
        ...capability,
        notes: [tally(database.agents, capability), capability.summary]
          .filter(Boolean)
          .join(" "),
      },
    ]),
  );
  return {...database, data};
}

/**
 * The dataset with some rows removed: their cells, the notes only they cited and their
 * boundary flags all go with them. Note numbers are kept, so a note link means the
 * same note whichever rows are on screen. The tally in the notes is recounted.
 */
export function withoutAgents(
  database: MirrorDatabase,
  hidden: Set<string>,
): MirrorDatabase {
  if (hidden.size === 0) {
    return database;
  }
  const keep = <T,>(record: Record<string, T>) =>
    Object.fromEntries(
      Object.entries(record).filter(([agentId]) => !hidden.has(agentId)),
    );
  const data = Object.fromEntries(
    Object.entries(database.data).map(([capabilityId, capability]) => {
      const stats = keep(capability.stats);
      const cited = new Set(
        Object.values(stats).flatMap((versions) =>
          Object.values(versions).flatMap((raw) => parseSupport(raw).noteRefs),
        ),
      );
      return [
        capabilityId,
        {
          ...capability,
          stats,
          notes_by_num: Object.fromEntries(
            Object.entries(capability.notes_by_num ?? {}).filter(([number]) =>
              cited.has(Number(number)),
            ),
          ),
          boundary_dependent: capability.boundary_dependent?.filter(
            (agentId) => !hidden.has(agentId),
          ),
        },
      ];
    }),
  );
  return withTally({...database, agents: keep(database.agents), data});
}

export function shownCapabilities(database: MirrorDatabase): Capability[] {
  return Object.values(database.data)
    .filter((capability) => capability.shown !== false)
    .sort((left, right) => left.title.localeCompare(right.title));
}

export function matchesCapability(
  capability: Capability,
  rawQuery: string,
): boolean {
  const query = rawQuery.trim().toLowerCase();
  if (!query) {
    return true;
  }

  const haystack = [
    capability.title,
    capability.description,
    capability.question ?? "",
    capability.keywords ?? "",
    ...(capability.categories ?? []),
  ]
    .join(" ")
    .toLowerCase();

  return query
    .split(/\s+/)
    .filter(Boolean)
    .every((term) => haystack.includes(term));
}

export function shortDate(iso?: string): string {
  return iso ? iso.slice(0, 10) : "Unknown";
}

/** The day a framework's cell in this row was captured, falling back to the day the
 * framework was last measured. */
export function measuredOn(
  capability: Capability,
  agentId: string,
  agent: Agent,
): string | undefined {
  return capability.measured.measured_on?.[agentId] ?? agent.measured_on ?? undefined;
}

/** The earliest and latest day any of the row's frameworks were captured. */
export function measuredRange(
  database: MirrorDatabase,
  capability: Capability,
): {earliest: string; latest: string} | null {
  const dates = Object.entries(database.agents)
    .map(([agentId, agent]) => measuredOn(capability, agentId, agent))
    .filter((date): date is string => Boolean(date))
    .sort();
  if (!dates.length) return null;
  return {earliest: dates[0], latest: dates[dates.length - 1]};
}

export function measuredRelease(agent: Agent) {
  return (
    agent.version_list.find((version) => version.era === 0) ??
    agent.version_list.find((version) => version.version === agent.current_version)
  );
}

export function titleCaseRule(name: string): string {
  return name
    .replace(/_/g, " ")
    .replace(/^\w/, (letter) => letter.toUpperCase());
}

/** A route segment as an id; a malformed escape such as a bare "%" stays as typed, so it reads as not found. */
function decodeSegment(segment: string): string {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}

export function routeFromHash(hash: string): Route {
  const path = hash.split("?")[0];
  if (path.startsWith("#/cap/")) {
    return {
      name: "capability",
      capabilityId: decodeSegment(path.slice("#/cap/".length)),
    };
  }
  if (path.startsWith("#/fw/")) {
    return {
      name: "framework",
      frameworkId: decodeSegment(path.slice("#/fw/".length)),
    };
  }
  if (path === "#/frameworks" || path === "#/frameworks/") {
    return {name: "frameworks"};
  }
  if (path === "#/stats" || path === "#/compare") {
    return {name: "stats"};
  }
  if (path === "#/changes") {
    return {name: "changes"};
  }
  if (path === "#/method") {
    return {name: "method"};
  }
  // #/harnesses was the agents page's first address; links to it keep working.
  if (["#/agents", "#/agents/", "#/harnesses", "#/harnesses/"].includes(path)) {
    return {name: "agents"};
  }
  if (path.startsWith("#/agents/")) {
    return {name: "agentRow", rowId: decodeSegment(path.slice("#/agents/".length))};
  }
  if (path.startsWith("#/agent/")) {
    return {name: "agent", agentId: decodeSegment(path.slice("#/agent/".length))};
  }
  return {name: "index"};
}

/** The framework pages: the framework index, one capability, or one framework. */
export function isFrameworksRoute(route: Route): boolean {
  return route.name === "frameworks" || route.name === "capability" || route.name === "framework";
}

/** The agents pages: the agents index, one behaviour, or one agent. */
export function isAgentsRoute(route: Route): boolean {
  return route.name === "agents" || route.name === "agentRow" || route.name === "agent";
}

export function capabilityHref(capabilityId: string): string {
  return `#/cap/${encodeURIComponent(capabilityId)}`;
}

/**
 * The snapshot a page opens on. A ?spec= in the URL wins. Otherwise a framework page
 * opens on the default when it reaches it, else on the newest snapshot it reaches, because a page of nothing but
 * unreachable cells tells its maintainer one sentence 66 times. Every other page, and a
 * framework that reaches none, opens on the default.
 */
export function initialSpecFor(
  index: SpecIndex,
  requested: string | null,
  route: Route,
): string {
  if (requested && index.specs.some((entry) => entry.id === requested)) {
    return requested;
  }
  if (route.name === "framework") {
    // The site default when this framework reaches it, so every page agrees by default.
    const reachesDefault = index.specs.find(
      (entry) => entry.id === index.default_spec && entry.reached_by?.includes(route.frameworkId),
    );
    if (reachesDefault) {
      return reachesDefault.id;
    }
    const reached = [...index.specs]
      .sort((left, right) => right.id.localeCompare(left.id))
      .find((entry) => entry.reached_by?.includes(route.frameworkId));
    if (reached) {
      return reached.id;
    }
  }
  return index.default_spec;
}

/**
 * A file the hosted site serves beside the page, such as a dataset or a capture. The
 * standalone file has no neighbours, so there it links the hosted copy instead.
 */
export function siteFileHref(file: string): string {
  const site = typeof window === "undefined" ? undefined : window.__MCP_MIRROR_SITE_URL__;
  return site ? new URL(file, site).href : `./${file}`;
}

export function frameworkHref(frameworkId: string): string {
  return `#/fw/${encodeURIComponent(frameworkId)}`;
}

export function reproduceCommand(
  agentId: string,
  agent: Agent,
  version: string,
  capability?: Capability,
): string {
  if (capability?.reproduce && !capability.reproduce.includes("mcp-mirror scan")) {
    return capability.reproduce;
  }
  // An alias row was measured through its source's adapter, so that is what reruns it.
  const framework = agent.aliases_of ?? agentId;
  const job = capability?.reproduce?.match(/--job\s+(\S+)/)?.[1];
  const jobArgument = job ? ` --job ${job}` : "";
  const specArgument = capability?.measured.mcp_spec
    ? ` --spec-version ${capability.measured.mcp_spec}`
    : "";

  if (agent.registry === "maven") {
    // A Java adapter runs as a worker jar the managed runner builds against the pins.
    return `uv run mcp-mirror scan "python3 your_server.py" --frameworks ${framework} --runner managed${jobArgument}${specArgument}`;
  }
  if (agent.registry === "nuget") {
    // A .NET adapter runs as a worker the managed runner publishes against the pins.
    return `uv run mcp-mirror scan "python3 your_server.py" --frameworks ${framework} --runner managed${jobArgument}${specArgument}`;
  }
  if (agent.registry === "go") {
    // A Go adapter runs as a worker binary the managed runner builds against the pins.
    return `uv run mcp-mirror scan "python3 your_server.py" --frameworks ${framework} --runner managed${jobArgument}${specArgument}`;
  }
  if (agent.registry === "npm") {
    return `npm i ${agent.package}@${version} && uv run mcp-mirror scan "python3 your_server.py" --frameworks ${framework}${jobArgument}${specArgument}`;
  }
  return `uv run --with ${agent.package}==${version} mcp-mirror scan "python3 your_server.py" --frameworks ${framework}${jobArgument}${specArgument}`;
}
