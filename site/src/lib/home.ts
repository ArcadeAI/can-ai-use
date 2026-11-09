import type {Capability, MirrorDatabase, SpecIndex} from "../types";

/** The capabilities readers ask about most, until the site has observed counts. */
export const EDITORIAL_POPULAR_IDS = [
  "destructive-hint",
  "read-only-hint",
  "required",
  "nested-objects",
  "enum",
] as const;

export const MCP_FACTS = [
  {
    label: "MCP messages use JSON-RPC 2.0.",
    path: "basic",
  },
  {
    label: "Clients and servers negotiate a protocol revision during initialization.",
    path: "basic/lifecycle",
  },
  {
    label: "Servers can expose tools, resources, and reusable prompts.",
    path: "server",
  },
] as const;

/**
 * What the newest revision with additions brought in: its id, its label and the
 * capabilities it adds. Every capability in a snapshot shares one run date, so a list
 * sorted by date would only be alphabetical.
 */
export function newestAdditions(
  specIndex: SpecIndex | null,
): {spec: string; label: string; capabilities: {id: string; title: string}[]} | null {
  const entry = specIndex?.specs
    .slice()
    .sort((left, right) => right.id.localeCompare(left.id))
    .find((candidate) => candidate.new_capabilities?.length);
  if (!entry?.new_capabilities) {
    return null;
  }
  return {
    spec: entry.id,
    label: entry.label,
    capabilities: entry.new_capabilities,
  };
}

/** The five most-viewed capabilities when the site has counts, else the editorial five. */
export function popularCapabilities(
  database: MirrorDatabase,
  capabilities: Capability[],
): {capabilities: Capability[]; observed: boolean} {
  const observed = database.popularity?.mode === "observed" ? database.popularity : undefined;
  if (observed) {
    const window = observed.windows.includes("week") ? "week" : observed.windows[0];
    const counts = new Map(
      observed.entries.map((entry) => [
        entry.capability_id,
        window ? (entry.counts[window]?.[0] ?? 0) : 0,
      ]),
    );
    return {
      observed: true,
      capabilities: [...capabilities]
        .sort(
          (left, right) =>
            (counts.get(right.id) ?? 0) - (counts.get(left.id) ?? 0) ||
            left.title.localeCompare(right.title),
        )
        .slice(0, 5),
    };
  }
  const byId = new Map(capabilities.map((capability) => [capability.id, capability]));
  return {
    observed: false,
    capabilities: EDITORIAL_POPULAR_IDS.map((id) => byId.get(id)).filter(
      (capability): capability is Capability => Boolean(capability),
    ),
  };
}
