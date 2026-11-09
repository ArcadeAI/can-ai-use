// Ship each published harness's wire captures with the site, so every harness cell can link
// to the exact capture it was derived from without the repository being public. Preview
// harnesses are skipped: their captures stay private with their cells.
import {cpSync, existsSync, readFileSync} from "node:fs";
import path from "node:path";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const harnesses = JSON.parse(readFileSync(path.join(root, "data/harnesses/harnesses.json"), "utf8"));
const rows = harnesses.harnesses ?? harnesses.agents ?? {};
let copied = 0;
for (const [id, harness] of Object.entries(rows)) {
  if ((harness.visibility ?? "published") === "preview") continue;
  // The row names its capture (measurements/wire/<id>/session.json, or wire-live for a
  // harness measured against its live service); the directory it sits in is what ships.
  const wire = (harness.capture ?? "").split("/")[1] || "wire";
  const source = path.join(root, "data/harnesses/measurements", wire, id);
  if (!existsSync(source)) continue;
  cpSync(source, path.join(root, "docs/data/harnesses/measurements", wire, id), {recursive: true});
  copied += 1;
}
console.log(`copied wire captures for ${copied} published harness(es)`);
