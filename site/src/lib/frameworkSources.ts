import manifestsJson from "../../../src/mcp_mirror/runner-manifests.json";
import defaultsJson from "../../../data/framework_defaults.json";

import type {FrameworkDefaults, RunnerManifests} from "./frameworks";

/** The runtime of every runner, which decides a framework's language group. */
export const RUNNER_MANIFESTS = manifestsJson as RunnerManifests;


/** The frameworks the default view shows, chosen in data/framework_defaults.json. */
export const FRAMEWORK_DEFAULTS = defaultsJson as FrameworkDefaults;
