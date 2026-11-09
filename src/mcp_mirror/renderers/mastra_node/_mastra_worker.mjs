// Node worker that drives Mastra's REAL MCP client (@mastra/mcp) so mcp-mirror can
// capture the tool actions returned by MCPClient.listTools, never a reimplementation
// (decision D1). It does not invoke an AI SDK provider or capture a provider request.
//
// Usage: node _mastra_worker.mjs '<json-server-config>'
//   stdio: {"transport":"stdio","command":"python","args":["server.py"]}
//   http : {"transport":"http","url":"https://host/mcp","headers":{...}}
// Emits a JSON array of {name, description, parameters, outputSchema, annotations} on stdout.

import { MCPClient } from '@mastra/mcp';

// Mastra wraps a tool's schema in a Standard-Schema JsonSchemaWrapper. Extract the
// JSON Schema represented by that wrapper at the declared capture boundary.
function extractJsonSchema(inputSchema) {
  if (!inputSchema || typeof inputSchema !== 'object') return {};
  if (typeof inputSchema.getSchema === 'function') {
    try {
      const s = inputSchema.getSchema();
      if (s && typeof s === 'object') return s;
    } catch { /* fall through */ }
  }
  const std = inputSchema['~standard'];
  if (std && std.jsonSchema && typeof std.jsonSchema === 'object' && Object.keys(std.jsonSchema).length) {
    return std.jsonSchema;
  }
  if (inputSchema.type || inputSchema.properties) return inputSchema;
  return {};
}

// A tool's outputSchema is a Standard JSON Schema wrapper whose `jsonSchema.output`
// renders the schema the action declares for its result. Null when the tool has none.
function extractOutputSchema(outputSchema) {
  if (!outputSchema || typeof outputSchema !== 'object') return null;
  const std = outputSchema['~standard'];
  const render = std && std.jsonSchema && std.jsonSchema.output;
  if (typeof render === 'function') {
    try {
      const s = render({ target: 'draft-2020-12' });
      if (s && typeof s === 'object') return s;
    } catch { /* fall through */ }
  }
  const s = extractJsonSchema(outputSchema);
  return Object.keys(s).length ? s : null;
}

function retainedMetadata(tool) {
  const out = {};
  const annotations = tool && tool.mcp && tool.mcp.annotations;
  if (annotations && typeof annotations === 'object' && Object.keys(annotations).length) {
    out.annotations = annotations;
  }
  if (tool && typeof tool.title === 'string' && tool.title) {
    out.title = tool.title;
  }
  return out;
}

async function attempt(fn) {
  try {
    return { ok: true, ...(await fn()) };
  } catch (err) {
    return { ok: false, error: String((err && err.message) || err).slice(0, 300) };
  }
}

async function sha256Hex(bytes) {
  const { createHash } = await import('node:crypto');
  return createHash('sha256').update(bytes).digest('hex');
}

// Resources and prompts through MCPClient.resources / MCPClient.prompts, in the
// shape src/mcp_mirror/probe.py documents so the cell generator can compare it with
// what the server sent.
async function probeSurface(mcp) {
  const out = {};
  out['resources.list'] = await attempt(async () => {
    const byServer = await mcp.resources.list();
    const entries = (byServer.src || []).map((r) => ({
      uri: r.uri, name: r.name, title: r.title ?? null, description: r.description ?? null, mimeType: r.mimeType ?? null,
    }));
    return { entries };
  });
  const entries = out['resources.list'].ok ? out['resources.list'].entries : [];
  const textUri = entries.find((e) => (e.mimeType || '').startsWith('text/'))?.uri;
  const blobUri = entries.find((e) => e.mimeType && !e.mimeType.startsWith('text/'))?.uri;
  const read = async (uri) => {
    const result = await mcp.resources.read('src', uri);
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
    const byServer = await mcp.resources.templates();
    return { entries: (byServer.src || []).map((t) => ({ uriTemplate: t.uriTemplate, name: t.name, description: t.description ?? null, mimeType: t.mimeType ?? null })) };
  });
  out['prompts.list'] = await attempt(async () => {
    const byServer = await mcp.prompts.list();
    return { entries: (byServer.src || []).map((p) => ({
      name: p.name, title: p.title ?? null, description: p.description ?? null,
      arguments: (p.arguments || []).map((a) => ({ name: a.name, description: a.description ?? null, required: a.required ?? null })),
    })) };
  });
  const prompts = out['prompts.list'].ok ? out['prompts.list'].entries : [];
  out['prompts.get'] = prompts.length ? await attempt(async () => {
    const first = prompts[0];
    const args = Object.fromEntries(first.arguments.filter((a) => a.required).map((a) => [a.name, 'x']));
    const result = await mcp.prompts.get({ serverName: 'src', name: first.name, args });
    return {
      description: result.description ?? null,
      messages: (result.messages || []).map((m) => ({ role: m.role ?? null, type: m.content?.type ?? null, text: m.content?.text ?? null })),
    };
  }) : { ok: false, error: 'no prompt listed' };
  return out;
}

// Server utilities through Mastra's public surface, one question per worker run (each
// run is its own recorded fixture process), in the shape src/mcp_mirror/probe.py
// documents. A utility Mastra does not wrap is reported unsupported even when the SDK
// client underneath has it: that client is only reachable through the private
// `mcpClientsById` field, which is not a route a Mastra user is given.
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const stamp = () => new Date().toISOString().replace('Z', '000+00:00');

function textOf(returned) {
  if (typeof returned === 'string') return returned;
  const block = (returned?.content || []).find((b) => typeof b?.text === 'string');
  return block ? block.text : JSON.stringify(returned ?? null);
}

async function probeUtility(mcp, tools, feature, serverLogs, spec) {
  const tool = (name) => tools[`src_${name}`] ?? tools[name];
  const unsupported = (reason) => ({ ok: false, unsupported: reason });
  switch (feature) {
    case 'tools.list':
      return { ok: true, route: 'adapter', api: 'MCPClient.listTools', names: Object.keys(tools).map((k) => k.replace(/^src_/, '')) };
    case 'progress':
      return attempt(async () => {
        const events = [];
        await mcp.progress.onUpdate('src', (p) => events.push({ progress: p.progress ?? null, total: p.total ?? null, message: p.message ?? null }));
        const returned = await tool('slow_with_progress').execute({}, { runId: 'mcp-mirror-progress' });
        return { route: 'adapter', api: 'enableProgressTracking + MCPClient.progress.onUpdate', events, text: textOf(returned) };
      });
    case 'logging':
      return attempt(async () => {
        const returned = await tool('emit_logs').execute({});
        await sleep(200);
        return { route: 'adapter', api: 'server option logger (enableServerLogs, serverLogLevel)', messages: serverLogs.map((m) => ({ level: m.level ?? null, data: m.data ?? m.details ?? null })), text: textOf(returned) };
      });
    case 'set_level':
      return unsupported('@mastra/mcp has no logging/setLevel API; the SDK Client.setLoggingLevel sits behind the private mcpClientsById field');
    case 'completion':
      return unsupported('@mastra/mcp has no completion API; the SDK Client.complete sits behind the private mcpClientsById field');
    case 'cancellation':
      return attempt(async () => {
        const controller = new AbortController();
        const started = Date.now();
        setTimeout(() => controller.abort('mcp-mirror cancel'), spec.cancel_after_s * 1000);
        let outcome;
        try {
          outcome = `returned: ${textOf(await tool('cancellable_wait').execute({ seconds: spec.cancel_wait_s }, { abortSignal: controller.signal }))}`;
        } catch (err) {
          outcome = `threw: ${String((err && err.message) || err).slice(0, 200)}`;
        }
        const abandoned = (Date.now() - started) / 1000;
        await sleep(300);
        return { route: 'adapter', mechanism: 'execute(args, { abortSignal })', abandoned_after_s: abandoned, outcome };
      });
    case 'tasks':
      return attempt(async () => ({ route: 'adapter', text: textOf(await tool('task_capable').execute({})) }));
    case 'caching':
      return attempt(async () => {
        const marks = [];
        const list = async (call) => {
          const started = stamp();
          const listed = await mcp.listTools();
          marks.push({ call, started, finished: stamp(), count: Object.keys(listed).length });
        };
        await list('first');
        await list('within_ttl');
        await sleep(spec.cache_pause_s * 1000);
        await list('after_ttl');
        return { route: 'adapter', api: 'MCPClient.listTools', marks };
      });
    default:
      return unsupported(`unknown utility ${feature}`);
  }
}

// Catalogue changes against fixtures/notify_server.py through MCPClient's own hooks
// (tools/resources/prompts.onListChanged, resources.onUpdated) and its subscribe API,
// in the shape src/mcp_mirror/notify.py documents. At 2026-07-28 those hooks ride the
// client's one managed subscriptions/listen stream, so the listen row is read from
// what they delivered.
const WATCHED_URI = 'file:///fixtures/watched.txt';
const errorText = (err) => {
  const own = String((err && err.message) || err);
  const cause = err && err.cause ? `: ${String(err.cause.message || err.cause)}` : '';
  return (own + cause).slice(0, 500);
};

async function notifyProbe(mcp, negotiated) {
  const seen = {};
  const note = (kind) => { seen[kind] = (seen[kind] || 0) + 1; };
  const count = (kind) => seen[kind] || 0;
  const wait = async (kind, since, ms = 3000) => {
    const end = Date.now() + ms;
    while (count(kind) <= since && Date.now() < end) await sleep(50);
    return count(kind) > since;
  };
  const hookErrors = {};
  const register = async (kind, fn) => {
    try { await fn(); } catch (err) { hookErrors[kind] = errorText(err); }
  };
  await register('tools', () => mcp.tools.onListChanged('src', () => note('tools')));
  await register('resources', () => mcp.resources.onListChanged('src', () => note('resources')));
  await register('prompts', () => mcp.prompts.onListChanged('src', () => note('prompts')));
  await register('updated', () => mcp.resources.onUpdated('src', () => note('updated')));
  const hooks = {
    tools: "MCPClient.tools.onListChanged('src', fn)",
    resources: "MCPClient.resources.onListChanged('src', fn)",
    prompts: "MCPClient.prompts.onListChanged('src', fn)",
  };
  const call = async (name) => {
    const tools = await mcp.listTools();
    const returned = await tools[`src_${name}`].execute({});
    return (returned && returned.content && returned.content[0] && returned.content[0].text) ?? null;
  };
  const lists = {
    tools: async () => Object.keys(await mcp.listTools()),
    resources: async () => ((await mcp.resources.list()).src || []).map((r) => r.uri),
    prompts: async () => ((await mcp.prompts.list()).src || []).map((p) => p.name),
  };
  const changes = [
    ['tools.list_changed', 'add_tool', 'tools', 'SENTINEL_ADDED_TOOL'],
    ['resources.list_changed', 'add_resource', 'resources', 'fixtures/added-'],
    ['prompts.list_changed', 'add_prompt', 'prompts', 'SENTINEL_ADDED_PROMPT'],
  ];
  const out = { negotiated };
  for (const [key, tool, kind, sentinel] of changes) {
    out[key] = await attempt(async () => {
      const before = (await lists[kind]()).sort();
      const since = count(kind);
      const said = await call(tool);
      const notified = await wait(kind, since);
      const after = (await lists[kind]()).sort();
      return { hook: hooks[kind], hook_error: hookErrors[kind] ?? null, notified, before, after, sentinel, server_said: said };
    });
  }
  out['resources.subscribe'] = await attempt(async () => {
    const entry = {
      route: 'adapter', api: "MCPClient.resources.subscribe / unsubscribe('src', uri) with resources.onUpdated",
      hook_error: hookErrors.updated ?? null, subscribed: false, subscribe_error: null, notified: false,
      unsubscribed: false, unsubscribe_error: null, notified_after_unsubscribe: null, server_said: [],
    };
    try { await mcp.resources.subscribe('src', WATCHED_URI); entry.subscribed = true; } catch (err) { entry.subscribe_error = errorText(err); }
    let since = count('updated');
    entry.server_said.push(await call('update_resource'));
    entry.notified = await wait('updated', since);
    if (!entry.subscribed) return entry;
    try { await mcp.resources.unsubscribe('src', WATCHED_URI); entry.unsubscribed = true; } catch (err) { entry.unsubscribe_error = errorText(err); return entry; }
    since = count('updated');
    entry.server_said.push(await call('update_resource'));
    entry.notified_after_unsubscribe = await wait('updated', since, 1500);
    return entry;
  });
  if (negotiated && negotiated >= '2026-07-28') {
    const names = { tools: 'ToolsListChanged', resources: 'ResourcesListChanged', prompts: 'PromptsListChanged', updated: 'ResourceUpdated' };
    const errors = Object.values(hookErrors);
    out['subscriptions.listen'] = {
      ok: true, route: 'adapter',
      api: 'MCPClient managed subscriptions/listen stream, opened by the onListChanged hooks and resources.subscribe',
      opened: errors.length === 0, open_error: errors[0] ?? null, honored: null,
      events: Object.keys(names).filter((kind) => count(kind) > 0).map((kind) => names[kind]),
    };
  }
  return out;
}

async function main() {
  const raw = process.argv[2];
  if (!raw) {
    process.stderr.write('mastra worker: missing server config argument\n');
    process.exit(2);
  }
  const cfg = JSON.parse(raw);

  let serverConfig;
  if (cfg.transport === 'http') {
    serverConfig = { url: new URL(cfg.url) };
    if (cfg.headers && Object.keys(cfg.headers).length) {
      serverConfig.requestInit = { headers: cfg.headers };
    }
  } else {
    serverConfig = { command: cfg.command, args: cfg.args || [] };
  }
  // `onToolError` is a documented per-server option typed 'throw' | 'return', default
  // 'throw'. It decides whether an `isError` result reaches the tool action as a
  // thrown MastraError or as the CallToolResult envelope, so the caller selects it
  // to capture the result boundary it asked for.
  if (cfg.onToolError) {
    serverConfig.onToolError = cfg.onToolError;
  }
  // Client-features probe: `inputRequests` is @mastra/mcp's one hook for a request the
  // server makes of the client, typed to answer elicitations. Widening
  // `capabilities.elicitation` is its documented way to accept url mode as well. Every
  // call it receives is recorded, so the probe can tell which requests reached it.
  const inputRequestsSeen = [];
  if (cfg.inputRequests) {
    serverConfig.capabilities = { elicitation: { form: {}, url: {} } };
    serverConfig.inputRequests = async ({ key, params }) => {
      const mode = params?.mode ?? (params?.url ? 'url' : 'form');
      inputRequestsSeen.push({ hook: 'inputRequests', key, mode });
      if (mode === 'url') return { action: 'accept' };
      return { action: 'accept', content: { name: cfg.inputRequests.elicited } };
    };
  }
  // The utilities probe opts in to every documented per-server switch Mastra has for
  // them, so a gap is Mastra's and not a default left off.
  const serverLogs = [];
  if (cfg.utility) {
    Object.assign(serverConfig, {
      enableProgressTracking: true,
      enableServerLogs: true,
      serverLogLevel: 'debug',
      logger: (m) => { if (String(m.message || '').includes('[MCP SERVER LOG]')) serverLogs.push({ level: m.level, ...(m.details || {}) }); },
    });
  }

  const mcp = new MCPClient({
    id: `mcp-mirror-${Date.now()}-${Math.random().toString(36).slice(2)}`,
    servers: { src: serverConfig },
  });

  try {
    const tools = await mcp.listTools();
    const out = [];
    for (const [name, tool] of Object.entries(tools)) {
      out.push({
        name, // Mastra namespaces this captured action as `<server>_<tool>`
        description: tool && tool.description != null ? String(tool.description) : null,
        parameters: extractJsonSchema(tool && tool.inputSchema),
        // A declared field of the tool action, beside inputSchema.
        outputSchema: extractOutputSchema(tool && tool.outputSchema),
        annotations: {}, // the declared tool fields (id, description, inputSchema) carry none
        // @mastra/mcp 2.x keeps the server's annotations under `tool.mcp.annotations`
        // and mirrors the title onto `tool.title`. Neither is part of the tool
        // definition an agent is handed, so they are reported as retained elsewhere.
        retained: retainedMetadata(tool),
      });
    }

    // Result boundary: `execute` is what a Mastra agent calls, so its return value is
    // the last thing between the tool and the model. Captured only when asked for, so
    // the definition-only path stays a pure read.
    const results = [];
    for (const call of cfg.calls || []) {
      // Mastra namespaces tools as `<server>_<tool>`; accept either form.
      const entry =
        tools[call.tool] ?? tools[`src_${call.tool}`] ??
        Object.entries(tools).find(([k]) => k.endsWith(`_${call.tool}`))?.[1];
      if (!entry || typeof entry.execute !== 'function') {
        results.push({ tool: call.tool, ok: false, error: 'no executable tool by that name' });
        continue;
      }
      try {
        // Mastra's action takes the arguments object directly. Some versions instead
        // expect it under `context`, and a mismatch comes back as a *returned*
        // validation error rather than a throw, so detect that shape and retry.
        const args = call.arguments || {};
        let returned = await entry.execute(args);
        if (returned && returned.error === true && returned.validationErrors) {
          returned = await entry.execute({ context: args });
        }
        results.push({ tool: call.tool, ok: true, returned });
      } catch (err) {
        // A tool that reported `isError` may surface here as a thrown error instead,
        // which is a different contract for the agent rather than a worker failure.
        results.push({
          tool: call.tool,
          ok: false,
          threw: true,
          error: String((err && err.message) || err),
        });
      }
    }
    // @mastra/mcp does not expose the negotiated version publicly. Its pinned MCP
    // client stores the initialize result here; fail closed if that evidence seam
    // moves rather than copying the direct source connection's version by assumption.
    // `getServerProtocolVersions` is public API from @mastra/mcp 2.x; older releases
    // only exposed the negotiated version on the underlying SDK client.
    let negotiatedMcpSpecVersion;
    if (typeof mcp.getServerProtocolVersions === 'function') {
      negotiatedMcpSpecVersion = (await mcp.getServerProtocolVersions())?.src;
    }
    if (!negotiatedMcpSpecVersion) {
      negotiatedMcpSpecVersion =
        mcp.mcpClientsById.get('src')?.client?._negotiatedProtocolVersion;
    }
    if (typeof negotiatedMcpSpecVersion !== 'string' || !negotiatedMcpSpecVersion) {
      throw new Error('mastra worker: negotiated MCP protocol version was not observable');
    }
    let probe;
    if (cfg.probe) {
      probe = await probeSurface(mcp);
    }
    let utility;
    if (cfg.utility) {
      utility = await probeUtility(mcp, tools, cfg.utility, serverLogs, cfg.utilitySpec || {});
    }
    let notify;
    if (cfg.notify) {
      notify = await notifyProbe(mcp, negotiatedMcpSpecVersion);
    }
    process.stdout.write(JSON.stringify({
      tools: out,
      results,
      negotiatedMcpSpecVersion,
      probe,
      notify,
      inputRequestsSeen,
      utility,
    }));
  } finally {
    try {
      await mcp.disconnect();
    } catch { /* best effort */ }
  }
}

main().catch((err) => {
  process.stderr.write(String((err && err.stack) || err) + '\n');
  process.exit(1);
});
