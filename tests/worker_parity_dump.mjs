// Runs the Worker's catalog port on the built data for every case in
// worker_parity_cases.json and prints the answers; tests/test_worker_parity.py compares
// them with src/mcp_mirror/catalog.py's. Usage: node worker_parity_dump.mjs <data dir>.
import {readFileSync} from "node:fs";
import path from "node:path";

import {Catalog} from "../worker/catalog.ts";

const dataDir = process.argv[2];
const catalog = new Catalog(async (file) => JSON.parse(readFileSync(path.join(dataDir, file), "utf8")));
const cases = JSON.parse(readFileSync(new URL("./worker_parity_cases.json", import.meta.url), "utf8"));

const call = ({tool, arguments: args}) => {
  const revision = args.revision ?? "default";
  const frameworks = args.frameworks ?? null;
  switch (tool) {
    case "list_snapshots":
      return catalog.listSnapshots();
    case "search_capabilities":
      return catalog.searchCapabilities(args.query, revision);
    case "lookup_capability":
      return catalog.lookupCapability(args.capability_id, revision, frameworks);
    case "inspect_tool_definition":
      return catalog.inspectToolDefinition(args.tool, revision, frameworks);
    case "inspect_tool_result":
      return catalog.inspectToolResult(args.result, revision, frameworks);
    default:
      throw new Error(`unknown tool ${tool}`);
  }
};

const answers = [];
for (const item of cases) {
  try {
    answers.push({ok: true, result: await call(item)});
  } catch (error) {
    answers.push({ok: false, error: error.message});
  }
}
process.stdout.write(JSON.stringify(answers));
