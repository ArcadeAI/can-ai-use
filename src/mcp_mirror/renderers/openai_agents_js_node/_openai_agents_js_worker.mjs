// Node worker that drives the OpenAI Agents SDK's REAL MCP servers (@openai/agents
// MCPServerStdio, MCPServerStreamableHttp, MCPServerSSE; agents-core src/mcp.ts) so
// mcp-mirror can capture what the SDK hands its model, never a reimplementation
// (decision D1).
//
// Every capture goes through a real Agent turn: an Agent is built with the server in
// `mcpServers`, and Runner.run drives it against a scripted Model. The Agent loads the
// MCP tools itself (getAllMcpTools -> mcpToFunctionTool), so the tool definitions are
// the `tools` of the ModelRequest the Model receives, and a tool result is the
// `function_call_result` item in the next ModelRequest's input. No OpenAI provider
// model runs and no provider request is serialized.
//
// Usage: node _openai_agents_js_worker.mjs '<json-config>'
//   stdio: {"transport":"stdio","command":"python","args":["server.py"]}
//   http : {"transport":"http","url":"https://host/mcp","headers":{...}}
//   sse  : {"transport":"sse","url":"https://host/sse"}

import { Agent, Runner, Usage, MCPServerStdio, MCPServerStreamableHttp, MCPServerSSE } from '@openai/agents';

// The SDK's default is 5 seconds, and a cold fixture interpreter can take longer than
// that to answer the `server/discover` probe. On stdio the v2 client then falls back
// to the legacy initialize handshake, so the default makes the negotiated revision
// depend on disk caches. 20 seconds is the documented option set to what the Python
// Agents SDK row uses.
const SESSION_TIMEOUT_S = 20;

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

function buildServer(cfg, options = {}) {
  const common = { clientSessionTimeoutSeconds: SESSION_TIMEOUT_S, name: 'src', ...options };
  if (cfg.transport === 'http') {
    const params = { url: cfg.url, ...common };
    if (cfg.headers && Object.keys(cfg.headers).length) params.requestInit = { headers: cfg.headers };
    return new MCPServerStreamableHttp(params);
  }
  if (cfg.transport === 'sse') {
    const params = { url: cfg.url, ...common };
    if (cfg.headers && Object.keys(cfg.headers).length) params.requestInit = { headers: cfg.headers };
    return new MCPServerSSE(params);
  }
  return new MCPServerStdio({ command: cfg.command, args: cfg.args || [], ...common });
}

// The SDK exposes no negotiated protocol version. The v2 Client it drives records it,
// and that Client sits on the node shim's `session` field, which the SDK's typings mark
// protected. Read it there as evidence only and fail closed if the seam moves, rather
// than copying the direct source connection's version by assumption.
function negotiatedVersion(server) {
  const session = server.underlying && server.underlying.session;
  const version = session && typeof session.getNegotiatedProtocolVersion === 'function'
    ? session.getNegotiatedProtocolVersion()
    : undefined;
  if (typeof version !== 'string' || !version) {
    throw new Error('openai_agents_js worker: negotiated MCP protocol version was not observable');
  }
  return version;
}

// A Model that answers from a script. The first turn optionally asks for one tool
// call; every later turn ends the run. It records each ModelRequest it is handed.
function scriptedModel(call) {
  const requests = [];
  let turn = 0;
  const model = {
    async getResponse(request) {
      requests.push(JSON.parse(JSON.stringify({ tools: request.tools, input: request.input })));
      turn += 1;
      if (turn === 1 && call) {
        return {
          usage: new Usage(),
          output: [{
            type: 'function_call', callId: 'mcp-mirror-call', name: call.tool,
            arguments: JSON.stringify(call.arguments || {}), status: 'completed',
          }],
        };
      }
      return {
        usage: new Usage(),
        output: [{ type: 'message', role: 'assistant', status: 'completed', content: [{ type: 'output_text', text: 'done' }] }],
      };
    },
    async *getStreamedResponse() {
      throw new Error('the scripted model only answers non-streamed runs');
    },
  };
  return { model, requests };
}

async function agentTurn(server, call, runOptions = {}) {
  const { model, requests } = scriptedModel(call);
  const agent = new Agent({ name: 'mcp-mirror', instructions: 'mcp-mirror probe', model, mcpServers: [server] });
  await new Runner({ tracingDisabled: true }).run(agent, 'mcp-mirror', runOptions);
  return requests;
}

function resultItem(requests) {
  const input = (requests[1] && requests[1].input) || [];
  return Array.isArray(input) ? input.find((item) => item && item.type === 'function_call_result') : undefined;
}

// MCPServer.callTool returns the content array. The node shim attaches the result's
// `_meta`, `structuredContent` and `isError` to that array (attachCallToolResultMetadata);
// report them when they are there so the capture shows what a caller can read.
function directResult(content) {
  const out = { content: Array.isArray(content) ? [...content] : content };
  for (const key of ['structuredContent', 'isError', '_meta']) {
    if (content && content[key] !== undefined) out[key] = content[key];
  }
  return out;
}

async function sha256Hex(bytes) {
  const { createHash } = await import('node:crypto');
  return createHash('sha256').update(bytes).digest('hex');
}

// Resources through MCPServer.listResources / readResource / listResourceTemplates, in
// the shape src/mcp_mirror/probe.py documents. The SDK has no prompts API.
async function probeSurface(server) {
  const out = {};
  out['resources.list'] = await attempt(async () => {
    const listed = await server.listResources();
    return { entries: (listed.resources || []).map((r) => ({
      uri: r.uri, name: r.name, title: r.title ?? null, description: r.description ?? null, mimeType: r.mimeType ?? null,
    })) };
  });
  const entries = out['resources.list'].ok ? out['resources.list'].entries : [];
  const textUri = entries.find((e) => (e.mimeType || '').startsWith('text/'))?.uri;
  const blobUri = entries.find((e) => e.mimeType && !e.mimeType.startsWith('text/'))?.uri;
  const read = async (uri) => {
    const result = await server.readResource(uri);
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
    const listed = await server.listResourceTemplates();
    return { entries: (listed.resourceTemplates || []).map((t) => ({
      uriTemplate: t.uriTemplate, name: t.name, description: t.description ?? null, mimeType: t.mimeType ?? null,
    })) };
  });
  const noPrompts = 'the MCPServer interface (agents-core src/mcp.ts) has no prompts method';
  out['prompts.list'] = { ok: false, unsupported: noPrompts };
  out['prompts.get'] = { ok: false, unsupported: noPrompts };
  return out;
}

function textOf(returned) {
  if (typeof returned === 'string') return returned;
  const list = Array.isArray(returned) ? returned : (returned?.content || []);
  const block = list.find((b) => typeof b?.text === 'string');
  return block ? block.text : JSON.stringify(returned ?? null);
}

// Server utilities, one question per worker run (each run is its own recorded fixture
// process), in the shape src/mcp_mirror/probe.py documents. The SDK takes no message
// handler, so no notification the server sends reaches the caller; the Client that
// could answer set_level or completion is the protected `session` field.
async function probeUtility(cfg, feature, spec) {
  const unsupported = (reason) => ({ ok: false, unsupported: reason });
  const connected = async (fn, options) => {
    const server = buildServer(cfg, options);
    await server.connect();
    try { return await fn(server); } finally { await server.close().catch(() => {}); }
  };
  switch (feature) {
    case 'tools.list':
      return attempt(() => connected(async (server) => {
        const requests = await agentTurn(server, null);
        return { route: 'adapter', api: 'Agent({ mcpServers }) tools in the ModelRequest', names: (requests[0].tools || []).map((t) => t.name) };
      }));
    case 'progress':
      // toolMetaResolver is the documented way to put `_meta` on a tool call, so the
      // token can be sent; nothing in the SDK receives notifications/progress.
      return attempt(() => connected(async (server) => {
        const requests = await agentTurn(server, { tool: 'slow_with_progress', arguments: {} });
        return {
          route: 'manual', api: 'MCPServerStdio({ toolMetaResolver: () => ({ progressToken }) }); no hook receives notifications/progress',
          events: [], text: textOf(resultItem(requests)?.output),
        };
      }, { toolMetaResolver: () => ({ progressToken: 'mcp-mirror-progress' }) }));
    case 'logging':
      // The tool is still called, so the wire shows what the server sent.
      try {
        await connected((server) => agentTurn(server, { tool: 'emit_logs', arguments: {} }));
      } catch { /* the answer below stands either way */ }
      return unsupported('MCPServer options take no message or logging handler; notifications/message reaches nothing');
    case 'set_level':
      return unsupported('MCPServer has no logging/setLevel method; the SDK Client.setLoggingLevel sits behind the protected session field');
    case 'completion':
      return unsupported('MCPServer has no completion method; the SDK Client.complete sits behind the protected session field');
    case 'cancellation':
      return attempt(() => connected(async (server) => {
        const controller = new AbortController();
        const started = Date.now();
        setTimeout(() => controller.abort('mcp-mirror cancel'), spec.cancel_after_s * 1000);
        let outcome;
        try {
          const requests = await agentTurn(server, { tool: 'cancellable_wait', arguments: { seconds: spec.cancel_wait_s } }, { signal: controller.signal });
          outcome = `returned: ${textOf(resultItem(requests)?.output)}`;
        } catch (err) {
          outcome = `threw: ${errorText(err).slice(0, 200)}`;
        }
        const abandoned = (Date.now() - started) / 1000;
        await sleep(300);
        return { route: 'adapter', mechanism: 'Runner.run(agent, input, { signal }) aborting the tool call', abandoned_after_s: abandoned, outcome };
      }));
    case 'tasks':
      return attempt(() => connected(async (server) => {
        const requests = await agentTurn(server, { tool: 'task_capable', arguments: {} });
        return { route: 'adapter', text: textOf(resultItem(requests)?.output) };
      }));
    case 'caching':
      return attempt(() => connected(async (server) => {
        const marks = [];
        const list = async (call) => {
          const started = stamp();
          const listed = await server.listTools();
          marks.push({ call, started, finished: stamp(), count: listed.length });
        };
        await list('first');
        await list('within_ttl');
        await sleep(spec.cache_pause_s * 1000);
        await list('after_ttl');
        return { route: 'adapter', api: 'MCPServer.listTools (cacheToolsList default false)', marks };
      }));
    default:
      return unsupported(`unknown utility ${feature}`);
  }
}

// Catalogue changes against fixtures/notify_server.py, in the shape
// src/mcp_mirror/notify.py documents. The SDK offers no hook for any notification and
// no subscribe API, so each change is read only from the next listing.
async function notifyProbe(server, negotiated) {
  const call = async (name) => textOf(await server.callTool(name, {}));
  const lists = {
    tools: async () => (await server.listTools()).map((t) => t.name),
    resources: async () => ((await server.listResources()).resources || []).map((r) => r.uri),
  };
  const changes = [
    ['tools.list_changed', 'add_tool', 'tools', 'SENTINEL_ADDED_TOOL'],
    ['resources.list_changed', 'add_resource', 'resources', 'fixtures/added-'],
  ];
  const out = { negotiated };
  for (const [key, tool, kind, sentinel] of changes) {
    out[key] = await attempt(async () => {
      const before = (await lists[kind]()).sort();
      const said = await call(tool);
      await sleep(1000);
      const after = (await lists[kind]()).sort();
      return { hook: null, notified: false, before, after, sentinel, server_said: said };
    });
  }
  out['prompts.list_changed'] = { ok: false, unsupported: 'the MCPServer interface has no prompts method and no notification hook' };
  out['resources.subscribe'] = { ok: false, unsupported: 'the MCPServer interface has no resources/subscribe method' };
  if (negotiated && negotiated >= '2026-07-28') {
    out['subscriptions.listen'] = { ok: false, unsupported: 'the SDK never opens subscriptions/listen and exposes no notification hook' };
  }
  return out;
}

async function main() {
  const raw = process.argv[2];
  if (!raw) {
    process.stderr.write('openai_agents_js worker: missing config argument\n');
    process.exit(2);
  }
  const cfg = JSON.parse(raw);

  if (cfg.utility) {
    // Each utility builds its own server with the options it needs.
    const probe = buildServer(cfg);
    await probe.connect();
    let negotiatedMcpSpecVersion;
    try { negotiatedMcpSpecVersion = negotiatedVersion(probe); } finally { await probe.close().catch(() => {}); }
    const utility = await probeUtility(cfg, cfg.utility, cfg.utilitySpec || {});
    process.stdout.write(JSON.stringify({ tools: [], results: [], negotiatedMcpSpecVersion, utility }));
    return;
  }

  // `useStructuredContent` is a documented per-server option, default off. With it on,
  // the SDK hands the model JSON.stringify(structuredContent) instead of the content
  // blocks whenever the result is not an error.
  const server = buildServer(cfg, cfg.useStructuredContent ? { useStructuredContent: true } : {});
  await server.connect();
  try {
    const negotiatedMcpSpecVersion = negotiatedVersion(server);
    const definitions = await agentTurn(server, null);
    const tools = (definitions[0].tools || []).map((t) => ({
      name: t.name,
      description: t.description ?? null,
      parameters: t.parameters || {},
      strict: t.strict ?? null,
      type: t.type ?? null,
    }));

    const results = [];
    for (const call of cfg.calls || []) {
      if (cfg.boundary === 'direct') {
        try {
          results.push({ tool: call.tool, ok: true, returned: directResult(await server.callTool(call.tool, call.arguments || {})) });
        } catch (err) {
          results.push({ tool: call.tool, ok: false, threw: true, error: errorText(err) });
        }
        continue;
      }
      try {
        const requests = await agentTurn(server, call);
        const item = resultItem(requests);
        if (!item) {
          results.push({ tool: call.tool, ok: false, error: 'the next ModelRequest carried no function_call_result' });
        } else {
          results.push({ tool: call.tool, ok: true, returned: item });
        }
      } catch (err) {
        results.push({ tool: call.tool, ok: false, threw: true, error: errorText(err) });
      }
    }

    const probe = cfg.probe ? await probeSurface(server) : undefined;
    const notify = cfg.notify ? await notifyProbe(server, negotiatedMcpSpecVersion) : undefined;
    process.stdout.write(JSON.stringify({ tools, results, negotiatedMcpSpecVersion, probe, notify }));
  } finally {
    await server.close().catch(() => {});
  }
}

main().catch((err) => {
  process.stderr.write(String((err && err.stack) || err) + '\n');
  process.exit(1);
});
