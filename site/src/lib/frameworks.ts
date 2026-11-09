import type {Agent, Capability, MirrorDatabase} from "../types";
// The .ts extension lets node run this module directly under the site tests.
import {parseSupport, reproduceCommand, withoutAgents} from "./data.ts";

/**
 * How the framework grid is laid out and which rows the default view keeps.
 *
 * Kept free of JSON imports so the rules can be tested under node directly; the site
 * passes in the runner manifests and data/usage.json (see frameworkSources.ts).
 */

export type LanguageGroupId = "python" | "typescript" | "jvm" | "dotnet-go";

export const LANGUAGE_GROUPS: {id: LanguageGroupId; label: string}[] = [
  {id: "python", label: "Python"},
  {id: "typescript", label: "TypeScript"},
  {id: "jvm", label: "Java/Kotlin"},
  {id: "dotnet-go", label: ".NET & Go"},
];

const GROUP_BY_RUNTIME: Record<string, LanguageGroupId> = {
  python: "python",
  node: "typescript",
  java: "jvm",
  dotnet: "dotnet-go",
  go: "dotnet-go",
};

export type RunnerManifests = {runners: Record<string, {runtime: string}>};

export type UsageEntry = {
  package: string;
  source: string | null;
  monthly_downloads: number | null;
  reason: string | null;
  fetched: string;
};

export type Usage = {fetched: string; frameworks: Record<string, UsageEntry>};

/** data/framework_defaults.json: the rows the default view keeps. */
// `rule` says in words how the default set was picked; the site prints it under the toggle.
export type FrameworkDefaults = {default_frameworks: string[]; rule?: string};

/** The runner-manifest key behind a site row; an alias row runs on its source's runner. */
export function runnerId(
  agentId: string,
  agents: Record<string, Agent>,
): string | undefined {
  const agent = agents[agentId];
  if (agent?.aliases_of && agents[agent.aliases_of]) {
    return runnerId(agent.aliases_of, agents);
  }
  const renderer = agent?.renderer;
  return renderer
    ? renderer.split("/").pop()?.replace(/\.py$/, "").replace(/_renderer$/, "")
    : undefined;
}

export function languageGroup(
  agentId: string,
  agents: Record<string, Agent>,
  manifests: RunnerManifests,
): LanguageGroupId {
  const runtime = manifests.runners[runnerId(agentId, agents) ?? ""]?.runtime;
  const group = runtime ? GROUP_BY_RUNTIME[runtime] : undefined;
  if (!group) {
    throw new Error(`${agentId}: no language group for runtime ${runtime ?? "(none)"}`);
  }
  return group;
}

/** Column groups in display order, each keeping the data's own row order. Empty groups are dropped. */
export function groupFrameworks(
  database: MirrorDatabase,
  manifests: RunnerManifests,
): {id: LanguageGroupId; label: string; agentIds: string[]}[] {
  const agentIds = Object.keys(database.agents);
  return LANGUAGE_GROUPS.map((group) => ({
    ...group,
    agentIds: agentIds.filter(
      (agentId) => languageGroup(agentId, database.agents, manifests) === group.id,
    ),
  })).filter((group) => group.agentIds.length > 0);
}

/** Inside a language group the group label already names the language, so the suffix goes. */
export function shortFrameworkName(name: string): string {
  return name.replace(/\s+\((?:JS|TypeScript|Java|\.NET|Go)\)$/, "");
}

/**
 * The default view: the maintainer's chosen frameworks (data/framework_defaults.json).
 * All shows every published row.
 */
export function filteredDatabase(
  database: MirrorDatabase,
  defaults: FrameworkDefaults,
): MirrorDatabase {
  const keep = new Set(defaults.default_frameworks);
  return withoutAgents(
    database,
    new Set(Object.keys(database.agents).filter((agentId) => !keep.has(agentId))),
  );
}

export function defaultsRule(database: MirrorDatabase, defaults: FrameworkDefaults): string {
  const names = defaults.default_frameworks
    .filter((agentId) => database.agents[agentId])
    .map((agentId) => database.agents[agentId].name);
  return `Filtered: ${names.join(", ")}.${defaults.rule ? ` ${defaults.rule}` : ""}`;
}

/**
 * The commands that rerun one framework's measurements. A capability's published
 * reproduce line chains every framework's run with &&, so this keeps the runs that name
 * this framework's runner and drops repeats. A framework no run names (one that cannot
 * negotiate the revision) gets the generic scan command for its measured release.
 */
export function frameworkReproduceCommands(
  agentId: string,
  database: MirrorDatabase,
  capabilities: Capability[],
): string[] {
  const agent = database.agents[agentId];
  const commands = recordedCommands(agentId, database.agents, capabilities);
  return commands.length
    ? commands
    : [reproduceCommand(agentId, agent, agent.current_version)];
}

/** The runs in these capabilities' reproduce lines that name this framework's runner, once each. */
export function recordedCommands(
  agentId: string,
  agents: Record<string, Agent>,
  capabilities: Capability[],
): string[] {
  const runner = runnerId(agentId, agents);
  const commands = new Set<string>();
  for (const capability of capabilities) {
    for (const command of (capability.reproduce ?? "").split(" && ")) {
      const named = command.match(/--(?:frameworks|renderers)\s+(\S+)/)?.[1];
      if (runner && named?.split(",").includes(runner)) {
        commands.add(command.trim());
      }
    }
  }
  return [...commands];
}

/**
 * The notes behind a framework's unreachable ('x') cells, one entry per distinct text.
 * An adapter that cannot open a session gives the same reason on every capability, so
 * the page states each reason once with the capabilities it covers.
 */
export function unreachableNotes(
  agentId: string,
  database: MirrorDatabase,
  capabilities: Capability[],
): {text: string; capabilityIds: string[]}[] {
  const agent = database.agents[agentId];
  const byText = new Map<string, string[]>();
  for (const capability of capabilities) {
    const support = parseSupport(capability.stats[agentId]?.[agent.current_version]);
    if (support.code !== "x") {
      continue;
    }
    for (const noteRef of support.noteRefs) {
      const text = capability.notes_by_num?.[String(noteRef)];
      if (text) {
        byText.set(text, [...(byText.get(text) ?? []), capability.id]);
      }
    }
  }
  return [...byText].map(([text, capabilityIds]) => ({text, capabilityIds}));
}
