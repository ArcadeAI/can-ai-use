import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";

import worker from "../worker/index.ts";
import {HOURLY_LIMIT, MAX_FEEDBACK_BYTES} from "../worker/feedback.ts";
import {MAX_BODY_BYTES, TOOLS} from "../worker/mcp.ts";

// The Worker's assets binding, stood in for by the built site in docs/.
const assets = {
  async fetch(request) {
    const {pathname} = new URL(request.url);
    try {
      const file = pathname === "/" ? "/index.html" : pathname;
      const body = readFileSync(new URL(`../docs${file}`, import.meta.url));
      return new Response(body, {status: 200, headers: {"Content-Type": "application/json"}});
    } catch {
      return new Response("not found", {status: 404});
    }
  },
};
const env = {ASSETS: assets};

const post = (body, headers = {}) =>
  worker.fetch(
    new Request("https://canaiuse.com/mcp", {
      method: "POST",
      headers: {"Content-Type": "application/json", Accept: "application/json, text/event-stream", ...headers},
      body: typeof body === "string" ? body : JSON.stringify(body),
    }),
    env,
  );
const rpc = (id, method, params) => post({jsonrpc: "2.0", id, method, params});

test("initialize negotiates a supported revision and falls back to the newest one", async () => {
  for (const [asked, answered] of [
    ["2025-11-25", "2025-11-25"],
    ["2025-06-18", "2025-06-18"],
    ["2026-07-28", "2025-11-25"],
    ["2024-11-05", "2025-11-25"],
  ]) {
    const response = await rpc(1, "initialize", {protocolVersion: asked, capabilities: {}, clientInfo: {name: "t", version: "0"}});
    assert.equal(response.status, 200);
    assert.equal(response.headers.get("content-type"), "application/json");
    assert.equal(response.headers.get("access-control-allow-origin"), "*");
    const body = await response.json();
    assert.equal(body.id, 1);
    assert.equal(body.result.protocolVersion, answered);
    assert.deepEqual(body.result.capabilities, {tools: {}});
    assert.equal(body.result.serverInfo.name, "canaiuse");
  }
});

test("notifications are accepted without a body and ping answers", async () => {
  const notified = await post({jsonrpc: "2.0", method: "notifications/initialized"});
  assert.equal(notified.status, 202);
  assert.equal(await notified.text(), "");
  const pinged = await rpc("p", "ping", {});
  assert.deepEqual(await pinged.json(), {jsonrpc: "2.0", id: "p", result: {}});
});

test("tools/list names the five catalog tools", async () => {
  const body = await (await rpc(2, "tools/list", {})).json();
  assert.deepEqual(
    body.result.tools.map((tool) => tool.name),
    ["list_snapshots", "search_capabilities", "lookup_capability", "inspect_tool_definition", "inspect_tool_result"],
  );
  assert.equal(body.result.tools.length, TOOLS.length);
  for (const tool of body.result.tools) assert.equal(tool.annotations.readOnlyHint, true);
});

test("tools/call returns structuredContent and a matching JSON text block", async () => {
  const body = await (await rpc(3, "tools/call", {name: "lookup_capability", arguments: {capability_id: "destructive-hint", frameworks: ["langchain"]}})).json();
  const {content, structuredContent, isError} = body.result;
  assert.equal(isError, undefined);
  assert.equal(structuredContent.revision, "2025-11-25");
  assert.equal(structuredContent.observations.length, 1);
  assert.equal(structuredContent.observations[0].framework_id, "langchain");
  assert.deepEqual(JSON.parse(content[0].text), structuredContent);
});

test("a catalog rejection is a tool error, a bad argument is a protocol error", async () => {
  const unknown = await (await rpc(4, "tools/call", {name: "lookup_capability", arguments: {capability_id: "no-such-thing", revision: "latest"}})).json();
  assert.equal(unknown.result.isError, true);
  assert.match(unknown.result.content[0].text, /unknown capability 'no-such-thing'/);

  const badRevision = await (await rpc(5, "tools/call", {name: "search_capabilities", arguments: {query: "hint", revision: "1999-01-01"}})).json();
  assert.equal(badRevision.result.isError, true);
  assert.match(badRevision.result.content[0].text, /unknown MCP revision '1999-01-01'; available: 2025-11-25, 2026-07-28/);

  const badType = await (await rpc(6, "tools/call", {name: "inspect_tool_definition", arguments: {tool: "not an object"}})).json();
  assert.equal(badType.error.code, -32602);
  const noTool = await (await rpc(7, "tools/call", {name: "drop_tables", arguments: {}})).json();
  assert.equal(noTool.error.code, -32602);
  const noMethod = await (await rpc(8, "resources/list", {})).json();
  assert.equal(noMethod.error.code, -32601);
});

test("GET, DELETE and OPTIONS follow the stateless transport rules", async () => {
  for (const method of ["GET", "DELETE"]) {
    const response = await worker.fetch(new Request("https://canaiuse.com/mcp", {method}), env);
    assert.equal(response.status, 405);
    assert.equal(response.headers.get("allow"), "POST, OPTIONS");
  }
  const preflight = await worker.fetch(new Request("https://canaiuse.com/mcp", {method: "OPTIONS"}), env);
  assert.equal(preflight.status, 204);
  assert.match(preflight.headers.get("access-control-allow-headers"), /MCP-Protocol-Version/);
});

test("non-JSON, batches, oversized bodies and unknown protocol headers are rejected", async () => {
  const notJson = await post("<html>");
  assert.equal(notJson.status, 400);
  assert.equal((await notJson.json()).error.code, -32700);

  const batch = await post([{jsonrpc: "2.0", id: 1, method: "ping"}]);
  assert.equal(batch.status, 400);
  assert.equal((await batch.json()).error.code, -32600);

  const big = await post(JSON.stringify({jsonrpc: "2.0", id: 1, method: "ping", params: {pad: "x".repeat(MAX_BODY_BYTES)}}));
  assert.equal(big.status, 413);

  const stale = await rpc(1, "ping", {}).then(() => post({jsonrpc: "2.0", id: 1, method: "ping"}, {"MCP-Protocol-Version": "2024-11-05"}));
  assert.equal(stale.status, 400);
});

test("every other URL goes to the assets binding untouched", async () => {
  for (const path of ["/", "/data/specs.json", "/favicon.svg"]) {
    const response = await worker.fetch(new Request(`https://canaiuse.com${path}`), env);
    assert.equal(response.status, 200, path);
  }
});

test("/harnesses redirects to the agents page the site routes by hash", async () => {
  for (const path of ["/harnesses", "/harnesses/"]) {
    const response = await worker.fetch(new Request(`https://canaiuse.com${path}`), env);
    assert.equal(response.status, 301, path);
    assert.equal(response.headers.get("location"), "https://canaiuse.com/#/agents", path);
  }
});

// D1 stood in for by an array: the handler's two statements, a count and an insert.
function fakeDb() {
  const rows = [];
  return {
    rows,
    prepare(query) {
      return {
        bind(...values) {
          return {
            async first() {
              return {n: rows.filter((row) => row.client === values[0]).length};
            },
            async run() {
              const [kind, subject, about, revision, page, message, contact, client] = values;
              rows.push({kind, subject, about, revision, page, message, contact, client});
            },
          };
        },
      };
    },
  };
}

const sendFeedback = (db, body, headers = {}) =>
  worker.fetch(
    new Request("https://canaiuse.com/feedback", {
      method: "POST",
      headers: {"Content-Type": "application/json", Origin: "https://canaiuse.com", "CF-Connecting-IP": "203.0.113.9", ...headers},
      body: typeof body === "string" ? body : JSON.stringify(body),
    }),
    {...env, FEEDBACK: db},
  );
const message = {kind: "capability", subject: "tool-output-schema", about: "#3 · Mastra 2.2.0", revision: "2025-11-25", page: "https://canaiuse.com/#/cap/tool-output-schema", message: "Mastra keeps outputSchema", contact: "me@example.com"};

test("feedback is stored privately, without the sender's IP", async () => {
  const db = fakeDb();
  const response = await sendFeedback(db, message);
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), {ok: true});
  assert.equal(db.rows.length, 1);
  const [row] = db.rows;
  assert.equal(row.subject, "tool-output-schema");
  assert.equal(row.about, "#3 · Mastra 2.2.0");
  assert.equal(row.contact, "me@example.com");
  assert.match(row.client, /^[0-9a-f]{32}$/);
  assert.ok(!JSON.stringify(row).includes("203.0.113.9"));
});

test("feedback takes a framework or agent page as its subject", async () => {
  const db = fakeDb();
  for (const [kind, subject] of [["framework", "pydantic-ai"], ["agent", "claude_code"]]) {
    assert.equal((await sendFeedback(db, {...message, kind, subject})).status, 200);
  }
  assert.deepEqual(db.rows.map((row) => [row.kind, row.subject]), [["framework", "pydantic-ai"], ["agent", "claude_code"]]);
});

test("feedback refuses what it should and drops bots quietly", async () => {
  const db = fakeDb();
  const get = await worker.fetch(new Request("https://canaiuse.com/feedback"), {...env, FEEDBACK: db});
  assert.equal(get.status, 405);
  assert.equal((await sendFeedback(db, message, {Origin: "https://evil.example"})).status, 403);
  assert.equal((await sendFeedback(db, "x", {"Content-Type": "text/plain"})).status, 415);
  assert.equal((await sendFeedback(db, "{nope")).status, 400);
  assert.equal((await sendFeedback(db, {...message, subject: "../etc"})).status, 400);
  assert.equal((await sendFeedback(db, {...message, kind: "other"})).status, 400);
  assert.equal((await sendFeedback(db, {...message, message: "   "})).status, 400);
  assert.equal((await sendFeedback(db, {...message, message: "x".repeat(MAX_FEEDBACK_BYTES)})).status, 413);
  assert.equal((await sendFeedback(db, {...message, website: "http://spam"})).status, 200);
  assert.equal(db.rows.length, 0);
  const off = await worker.fetch(
    new Request("https://canaiuse.com/feedback", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(message)}),
    env,
  );
  assert.equal(off.status, 503);
});

test("feedback rate-limits one sender per hour", async () => {
  const db = fakeDb();
  for (let i = 0; i < HOURLY_LIMIT; i += 1) assert.equal((await sendFeedback(db, message)).status, 200);
  assert.equal((await sendFeedback(db, message)).status, 429);
  assert.equal((await sendFeedback(db, message, {"CF-Connecting-IP": "198.51.100.4"})).status, 200);
});
