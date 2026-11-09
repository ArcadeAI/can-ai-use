// Node worker that drives Pi's REAL MCP client (@earendil-works/pi-mcp McpClient with its
// StdioTransport or StreamableHttpTransport) so mcp-mirror can capture what the client
// hands its caller, never a reimplementation (decision D1).
//
// pi-mcp is a standalone client with its own JSON-RPC core; it does not wrap the
// official MCP SDK. It has no agent loop of its own. Tool definitions are the Tool
// objects McpClient.listTools returns. A tool result is captured at two boundaries:
// the CallToolResult McpClient.callTool returns, and the LlmContent[] the package's
// own toLlmContent turns it into for a model. No model runs.
//
// Usage: node _pi_worker.mjs '<json-config>'
//   stdio: {"transport":"stdio","command":"python","args":["server.py"]}
//   http : {"transport":"http","url":"https://host/mcp","headers":{...}}
// pi-mcp has no HTTP+SSE transport (its README lists legacy HTTP+SSE as outside the core).

import { McpClient, StdioTransport, StreamableHttpTransport, toLlmContent } from '@earendil-works/pi-mcp';

const CLIENT = { name: 'mcp-mirror', version: '0.1.0' };

const errorText = (err) => {
  const own = String((err && err.message) || err);
  const cause = err && err.cause ? `: ${String(err.cause.message || err.cause)}` : '';
  return (own + cause).slice(0, 500);
};
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const stamp = () => new Date().toISOString().replace('Z', '000+00:00');

async function attempt(fn) {
  try {
    return { ok: true, ...(await fn()) };
  } catch (err) {
    return { ok: false, error: errorText(err) };
  }
}

function buildTransport(cfg) {
  if (cfg.transport === 'http') {
    const options = { url: cfg.url };
    if (cfg.headers && Object.keys(cfg.headers).length) options.headers = cfg.headers;
    return new StreamableHttpTransport(options);
  }
  if (cfg.transport === 'sse') {
    throw new Error('pi-mcp has no HTTP+SSE transport; its README lists legacy HTTP+SSE as outside the core');
  }
  // Pi's StdioTransport inherits the parent environment by default; stderr is piped so
  // the fixture's log lines do not mix into this worker's JSON on stdout.
  return new StdioTransport({ command: cfg.command, args: cfg.args || [], stderr: 'pipe' });
}

async function open(cfg, options = {}) {
  const client = new McpClient({ ...CLIENT, ...options });
  await client.connect(buildTransport(cfg));
  return client;
}

// McpClient.protocolVersion is a public getter on the version the server selected in
// its initialize result. Fail closed if it is missing rather than copying the direct
// source connection's version by assumption.
function negotiatedVersion(client) {
  const version = client.protocolVersion;
  if (typeof version !== 'string' || !version) {
    throw new Error('pi worker: negotiated MCP protocol version was not observable');
  }
  return version;
}

async function call(client, tool, args, options) {
  try {
    const result = await client.callTool(tool, args || {}, options);
    return { tool, ok: true, returned: { direct: result, llm: toLlmContent(result) } };
  } catch (err) {
    return { tool, ok: false, threw: true, error: errorText(err), code: err && err.code };
  }
}

async function sha256Hex(bytes) {
  const { createHash } = await import('node:crypto');
  return createHash('sha256').update(bytes).digest('hex');
}

// Resources through McpClient.listResources / readResource / listResourceTemplates, in
// the shape src/mcp_mirror/probe.py documents. McpClient has no prompts method.
async function probeSurface(client) {
  const out = {};
  out['resources.list'] = await attempt(async () => {
    const listed = await client.listResources();
    return { entries: listed.map((r) => ({
      uri: r.uri, name: r.name ?? null, title: r.title ?? null, description: r.description ?? null, mimeType: r.mimeType ?? null,
    })) };
  });
  const entries = out['resources.list'].ok ? out['resources.list'].entries : [];
  const textUri = entries.find((e) => (e.mimeType || '').startsWith('text/'))?.uri;
  const blobUri = entries.find((e) => e.mimeType && !e.mimeType.startsWith('text/'))?.uri;
  const read = async (uri) => {
    const result = await client.readResource(uri);
    const first = (result.contents || [])[0] || {};
    if (first.blob !== undefined) {
      const bytes = Buffer.from(first.blob, 'base64');
      return { uri: first.uri ?? null, mimeType: first.mimeType ?? null, blob_sha256: await sha256Hex(bytes), blob_bytes: bytes.length };
    }
    return { uri: first.uri ?? null, mimeType: first.mimeType ?? null, text: first.text ?? null };
  };
  out['resources.read.text'] = textUri ? await attempt(() => read(textUri)) : { ok: false, error: 'no text resource listed' };
  out['resources.read.blob'] = blobUri ? await attempt(() => read(blobUri)) : { ok: false, error: 'no binary resource listed' };
  out['resources.templates'] = await attempt(async () => {
    const listed = await client.listResourceTemplates();
    return { entries: listed.map((t) => ({
      uriTemplate: t.uriTemplate, name: t.name ?? null, description: t.description ?? null, mimeType: t.mimeType ?? null,
    })) };
  });
  const noPrompts = 'McpClient has no prompts method (pi-mcp 1.1.0 dist/client.d.ts); only the generic request()';
  out['prompts.list'] = { ok: false, unsupported: noPrompts };
  out['prompts.get'] = { ok: false, unsupported: noPrompts };
  return out;
}

function textOf(result) {
  const block = (result?.content || []).find((b) => typeof b?.text === 'string');
  return block ? block.text : JSON.stringify(result ?? null);
}

// Server utilities, one question per worker run (each run is its own recorded fixture
// process), in the shape src/mcp_mirror/probe.py documents. A typed McpClient method or
// option is route "adapter". McpClient.request(method, params) is a generic JSON-RPC
// call where the caller names the method and reads the raw result, reported as
// route "manual".
async function probeUtility(cfg, feature, spec) {
  const unsupported = (reason) => ({ ok: false, unsupported: reason });
  const connected = async (fn) => {
    const client = await open(cfg);
    try { return await fn(client); } finally { await client.close().catch(() => {}); }
  };
  const logListener = (client, messages) => client.onNotification('notifications/message', (params) => {
    messages.push({ level: params?.level ?? null, data: params?.data ?? null });
  });
  switch (feature) {
    case 'tools.list':
      return attempt(() => connected(async (client) => ({
        route: 'adapter', api: 'McpClient.listTools (follows nextCursor)', names: (await client.listTools()).map((t) => t.name),
      })));
    case 'progress':
      return attempt(() => connected(async (client) => {
        const events = [];
        const result = await client.callTool('slow_with_progress', {}, {
          onProgress: (p) => events.push({ progress: p.progress ?? null, total: p.total ?? null, message: p.message ?? null }),
        });
        return { route: 'adapter', api: 'McpClient.callTool(name, args, { onProgress })', events, text: textOf(result) };
      }));
    case 'logging':
      return attempt(() => connected(async (client) => {
        const messages = [];
        logListener(client, messages);
        const result = await client.callTool('emit_logs', {});
        await sleep(300);
        return { route: 'adapter', api: "McpClient.onNotification('notifications/message')", messages, text: textOf(result) };
      }));
    case 'set_level':
      return attempt(() => connected(async (client) => {
        await client.request('logging/setLevel', { level: spec.set_level });
        const result = await client.callTool('emit_logs', {});
        return { route: 'manual', api: "McpClient.request('logging/setLevel'); McpClient has no setLevel method", level: spec.set_level, text: textOf(result) };
      }));
    case 'completion':
      return attempt(() => connected(async (client) => {
        const complete = async (ref, name, value) => {
          const result = await client.request('completion/complete', { ref, argument: { name, value } });
          return result?.completion?.values || [];
        };
        const [prompt, promptArg, promptValue] = spec.completion_prompt;
        const [uri, resourceArg, resourceValue] = spec.completion_resource;
        return {
          route: 'manual',
          api: "McpClient.request('completion/complete'); McpClient has no completion method",
          prompt: await complete({ type: 'ref/prompt', name: prompt }, promptArg, promptValue),
          resource: await complete({ type: 'ref/resource', uri }, resourceArg, resourceValue),
        };
      }));
    case 'cancellation':
      return attempt(() => connected(async (client) => {
        const controller = new AbortController();
        const started = Date.now();
        setTimeout(() => controller.abort('mcp-mirror cancel'), spec.cancel_after_s * 1000);
        let outcome;
        try {
          const result = await client.callTool('cancellable_wait', { seconds: spec.cancel_wait_s }, { signal: controller.signal });
          outcome = `returned: ${textOf(result)}`;
        } catch (err) {
          outcome = `threw: ${errorText(err).slice(0, 200)}`;
        }
        const abandoned = (Date.now() - started) / 1000;
        await sleep(300);
        return { route: 'adapter', mechanism: 'McpClient.callTool(name, args, { signal }) aborted', abandoned_after_s: abandoned, outcome };
      }));
    case 'tasks':
      return attempt(() => connected(async (client) => {
        const result = await client.callTool('task_capable', {});
        return { route: 'adapter', text: textOf(result) };
      }));
    case 'caching':
      return attempt(() => connected(async (client) => {
        const marks = [];
        const list = async (label) => {
          const started = stamp();
          const listed = await client.listTools();
          marks.push({ call: label, started, finished: stamp(), count: listed.length });
        };
        await list('first');
        await list('within_ttl');
        await sleep(spec.cache_pause_s * 1000);
        await list('after_ttl');
        return { route: 'adapter', api: 'McpClient.listTools (no cache)', marks };
      }));
    default:
      return unsupported(`unknown utility ${feature}`);
  }
}

// Catalogue changes against fixtures/notify_server.py, in the shape
// src/mcp_mirror/notify.py documents. McpClient.onNotification is the README's documented
// route for tool-list-change notifications; it hears any method by name. There is no
// prompts or subscribe method.
async function notifyProbe(client, negotiated) {
  const heard = { tools: 0, resources: 0 };
  client.onNotification('notifications/tools/list_changed', () => { heard.tools += 1; });
  client.onNotification('notifications/resources/list_changed', () => { heard.resources += 1; });
  const lists = {
    tools: async () => (await client.listTools()).map((t) => t.name),
    resources: async () => (await client.listResources()).map((r) => r.uri),
  };
  const changes = [
    ['tools.list_changed', 'add_tool', 'tools', 'SENTINEL_ADDED_TOOL'],
    ['resources.list_changed', 'add_resource', 'resources', 'fixtures/added-'],
  ];
  const out = { negotiated };
  for (const [key, tool, kind, sentinel] of changes) {
    out[key] = await attempt(async () => {
      const before = (await lists[kind]()).sort();
      const said = textOf(await client.callTool(tool, {}));
      for (let waited = 0; waited < 3000 && !heard[kind]; waited += 100) await sleep(100);
      const after = (await lists[kind]()).sort();
      return {
        hook: `McpClient.onNotification('notifications/${kind}/list_changed')`,
        notified: heard[kind] > 0, before, after, sentinel, server_said: said,
      };
    });
  }
  out['prompts.list_changed'] = { ok: false, unsupported: 'McpClient has no prompts method to list the change with' };
  // McpClient has no subscribe method. Its generic request() and onNotification are
  // public, so a caller can subscribe by naming the wire methods: route "manual".
  out['resources.subscribe'] = await attempt(async () => {
    const watched = 'file:///fixtures/watched.txt';
    let updates = 0;
    client.onNotification('notifications/resources/updated', (params) => { if (params?.uri === watched) updates += 1; });
    const api = "McpClient.request('resources/subscribe' | 'resources/unsubscribe') + onNotification('notifications/resources/updated'); McpClient has no subscribe method";
    const answer = { route: 'manual', api, subscribed: false, notified: false, unsubscribed: false, notified_after_unsubscribe: false };
    try { await client.request('resources/subscribe', { uri: watched }); answer.subscribed = true; } catch (err) { answer.subscribe_error = errorText(err); return answer; }
    answer.server_said = textOf(await client.callTool('update_resource', {}));
    for (let waited = 0; waited < 3000 && !updates; waited += 100) await sleep(100);
    answer.notified = updates > 0;
    try { await client.request('resources/unsubscribe', { uri: watched }); answer.unsubscribed = true; } catch (err) { answer.unsubscribe_error = errorText(err); }
    const before = updates;
    await client.callTool('update_resource', {});
    await sleep(1500);
    answer.notified_after_unsubscribe = updates > before;
    return answer;
  });
  if (negotiated && negotiated >= '2026-07-28') {
    out['subscriptions.listen'] = { ok: false, unsupported: 'McpClient never opens subscriptions/listen' };
  }
  return out;
}

async function main() {
  const raw = process.argv[2];
  if (!raw) {
    process.stderr.write('pi worker: missing config argument\n');
    process.exit(2);
  }
  const cfg = JSON.parse(raw);

  if (cfg.utility) {
    // Each utility opens its own client.
    const probe = await open(cfg);
    let negotiatedMcpSpecVersion;
    try { negotiatedMcpSpecVersion = negotiatedVersion(probe); } finally { await probe.close().catch(() => {}); }
    const utility = await probeUtility(cfg, cfg.utility, cfg.utilitySpec || {});
    process.stdout.write(JSON.stringify({ tools: [], results: [], negotiatedMcpSpecVersion, utility }));
    return;
  }

  // The one client-feature hook pi-mcp documents is the `roots` option (README;
  // client.d.ts McpClientOptions.roots), which also declares the roots capability.
  // Sampling and elicitation requests get McpClient's default: method not found.
  const seen = [];
  const options = cfg.roots ? {
    roots: () => { seen.push({ hook: 'roots' }); return cfg.roots; },
  } : {};
  const client = await open(cfg, options);
  try {
    const negotiatedMcpSpecVersion = negotiatedVersion(client);
    // listTools returns the server's Tool objects as parsed JSON, after checking only
    // that each has a string name and an object inputSchema.
    const tools = await client.listTools();

    const results = [];
    for (const c of cfg.calls || []) results.push(await call(client, c.tool, c.arguments));

    const probe = cfg.probe ? await probeSurface(client) : undefined;
    const notify = cfg.notify ? await notifyProbe(client, negotiatedMcpSpecVersion) : undefined;
    process.stdout.write(JSON.stringify({ tools, results, negotiatedMcpSpecVersion, probe, notify, seen }));
  } finally {
    await client.close().catch(() => {});
  }
}

main().catch((err) => {
  process.stderr.write(String((err && err.stack) || err) + '\n');
  process.exit(1);
});
