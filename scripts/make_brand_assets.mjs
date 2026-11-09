// Favicon, touch icons and the social card, rendered from HTML with the site's own palette.
// The card's matrix is drawn from real published cells, so it never shows invented data.
//   node scripts/make_brand_assets.mjs   (after npm run build:data)
import {readFileSync, writeFileSync} from "node:fs";
import path from "node:path";
import {chromium} from "playwright-core";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const out = (name) => path.join(root, "site/public", name);
const C = {bg: "#002b36", bg2: "#073642", fg: "#fdf6e3", muted: "#93a1a1", dim: "#586e75",
  y: "#859900", a: "#b58900", n: "#dc322f", x: "#586e75", u: "#073642", link: "#e4a72c"};

const favicon = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<rect width="64" height="64" rx="14" fill="${C.bg}"/>
<rect x="12" y="12" width="18" height="18" rx="4" fill="${C.y}"/>
<rect x="34" y="12" width="18" height="18" rx="4" fill="${C.a}"/>
<rect x="12" y="34" width="18" height="18" rx="4" fill="${C.n}"/>
<rect x="34" y="34" width="18" height="18" rx="4" fill="${C.x}"/>
</svg>`;
writeFileSync(out("favicon.svg"), favicon + "\n");

// Real cells: a few capabilities across the published frameworks of the default snapshot.
const index = JSON.parse(readFileSync(path.join(root, "data/fulldata/specs.json"), "utf8"));
const legacy = index.specs.find((s) => s.measured_frameworks === s.total_frameworks) ?? index.specs[0];
const db = JSON.parse(readFileSync(path.join(root, "data/fulldata", legacy.file), "utf8"));
const agents = Object.entries(db.agents).filter(([, a]) => (a.visibility ?? "published") !== "preview");
const caps = ["tool-name", "enum", "destructive-hint", "read-only-hint", "tool-execution-errors", "structured-content", "tool-image-content", "prompts-list"]
  .filter((id) => db.data[id] && db.data[id].measurement_state !== "not_measured");
const code = (cap, [id, agent]) => (Object.values(cap.stats[id] ?? {})[0] ?? "u").split(" ")[0];
const rows = caps.map((id) => {
  const cap = db.data[id];
  const cells = agents.map((a) => `<div class="c" style="background:${C[code(cap, a)] ?? C.u}"></div>`).join("");
  return `<div class="r"><div class="l">${cap.title.replaceAll("`", "")}</div>${cells}</div>`;
}).join("");
const heads = agents.map(([, a]) => `<div class="h">${a.abbr ?? a.name.slice(0, 3)}</div>`).join("");

const card = `<!doctype html><html><head><style>
*{box-sizing:border-box;margin:0}
body{width:1200px;height:630px;background:${C.bg};color:${C.fg};font-family:"Open Sans",Helvetica,Arial,sans-serif;
  display:flex;padding:72px 76px;gap:56px;position:relative;overflow:hidden}
body:before{content:"";position:absolute;inset:0;background-image:linear-gradient(${C.bg2} 1px,transparent 1px),linear-gradient(90deg,${C.bg2} 1px,transparent 1px);background-size:40px 40px;opacity:.55}
.left,.right{position:relative}
.left{flex:1.1;display:flex;flex-direction:column}
h1{font-family:Georgia,"Times New Roman",serif;font-weight:400;font-size:104px;line-height:1;letter-spacing:-1px}
p{margin-top:30px;font-size:31px;line-height:1.35;color:${C.muted};max-width:560px}
.foot{margin-top:auto;display:flex;gap:22px;align-items:baseline;font-size:25px}
.url{color:${C.link};font-weight:600}.by{color:${C.dim}}
.right{flex:.9;align-self:center;background:rgba(7,54,66,.85);border:1px solid ${C.dim};border-radius:14px;padding:22px 24px}
.r{display:flex;align-items:center;gap:6px;margin-top:6px}.r:first-child{margin-top:0}
.l{width:210px;font-family:"Inconsolata",Menlo,monospace;font-size:19px;color:${C.muted};white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.h{width:42px;text-align:center;font-size:15px;color:${C.dim};font-weight:600}
.c{width:42px;height:34px;border-radius:5px}
.legend{display:flex;gap:16px;margin-top:16px;font-size:15px;color:${C.dim}}
.legend span:before{content:"";display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px;background:var(--k)}
</style></head><body>
<div class="left"><h1>Can AI use</h1>
<p>Measured MCP support: what each agent framework and coding harness actually passes to the model.</p>
<div class="foot"><span class="url">canaiuse.com</span><span class="by">Maintained by Arcade.dev</span></div></div>
<div class="right"><div class="r"><div class="l"></div>${heads}</div>${rows}
<div class="legend"><span style="--k:${C.y}">present</span><span style="--k:${C.a}">changed</span><span style="--k:${C.n}">dropped</span><span style="--k:${C.x}">n/a</span></div></div>
</body></html>`;

const browser = await chromium.launch({channel: "chrome"}).catch(() => chromium.launch());
const page = await browser.newPage({viewport: {width: 1200, height: 630}, deviceScaleFactor: 1});
await page.setContent(card, {waitUntil: "load"});
await page.screenshot({path: out("og.png")});
for (const [name, size] of [["favicon-32.png", 32], ["apple-touch-icon.png", 180], ["icon-192.png", 192], ["icon-512.png", 512]]) {
  const icon = await browser.newPage({viewport: {width: size, height: size}, deviceScaleFactor: 1});
  await icon.setContent(`<html><body style="margin:0;background:transparent">${favicon.replace("<svg ", `<svg width="${size}" height="${size}" `)}</body></html>`);
  await icon.screenshot({path: out(name), omitBackground: true});
}
await browser.close();
writeFileSync(out("site.webmanifest"), JSON.stringify({
  name: "Can AI use", short_name: "Can AI use", start_url: "/", display: "standalone",
  background_color: C.bg, theme_color: C.bg,
  icons: [{src: "/icon-192.png", sizes: "192x192", type: "image/png"}, {src: "/icon-512.png", sizes: "512x512", type: "image/png"}],
}, null, 2) + "\n");
console.log(`wrote favicon.svg, PNG icons, site.webmanifest and og.png (${caps.length} real rows x ${agents.length} frameworks from ${legacy.id})`);
