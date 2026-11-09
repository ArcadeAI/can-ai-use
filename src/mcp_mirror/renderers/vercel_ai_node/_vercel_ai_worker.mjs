// Node worker that drives the Vercel AI SDK's REAL MCP client (@ai-sdk/mcp) so
// mcp-mirror can capture the tools createMCPClient hands generateText / streamText,
// never a reimplementation (decision D1). It does not invoke a model provider or
// capture a provider request.
//
// Usage: node _vercel_ai_worker.mjs '<json-config>'
//   stdio: {"transport":"stdio","command":"python","args":["server.py"]}
//   http : {"transport":"http","url":"https://host/mcp","headers":{...}}
//   sse  : {"transport":"sse","url":"http://host/sse"}   (the deprecated HTTP+SSE transport)
// Optional keys: "calls" [{tool, arguments}], "boundary" ("model_output" | "execute"),
// "probe" (true), "elicitation" ({elicited}), "utility" (a probe.py UTILITY_FEATURES
// key, with "utilitySpec"), "notify" (true). Emits one JSON object on stdout.

import { createMCPClient, ElicitationRequestSchema } from '@ai-sdk/mcp';
import { Experimental_StdioMCPTransport } from '@ai-sdk/mcp/mcp-stdio';

// `client.tools()` wraps each server schema with the AI SDK's `jsonSchema()` helper.
// The wrapper exposes the JSON Schema the provider adapter will send on `.jsonSchema`,
// which may be a value or, for lazily built schemas, a promise of one.
async function extractJsonSchema(inputSchema) {
  if (!inputSchema || typeof inputSchema !== 'object') return {};
  const schema = await inputSchema.jsonSchema;
  if (schema && typeof schema === 'object') return schema;
  return {};
}

// What the tool object keeps beside the definition an AI SDK provider serializes
// (description, inputSchema). `metadata` and `_meta` are app-side fields: the
// provider adapters do not send them to the model.
function retainedMetadata(tool) {
  const out = {};
  const annotations = tool && tool.metadata && tool.metadata.annotations;
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

// Resources and prompts through the client's own listResources / readResource /
// listResourceTemplates / experimental_listPrompts / experimental_getPrompt, in the
// shape src/mcp_mirror/probe.py documents.
async function probeSurface(client) {
  const out = {};
  out['resources.list'] = await attempt(async () => {
    const listed = await client.listResources();
    const entries = (listed.resources || []).map((r) => ({
      uri: r.uri, name: r.name, title: r.title ?? null, description: r.description ?? null, mimeType: r.mimeType ?? null,
    }));
    return { entries };
  });
  const entries = out['resources.list'].ok ? out['resources.list'].entries : [];
  const textUri = entries.find((e) => (e.mimeType || '').startsWith('text/'))?.uri;
  const blobUri = entries.find((e) => e.mimeType && !e.mimeType.startsWith('text/'))?.uri;
  const read = async (uri) => {
    const result = await client.readResource({ uri });
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
    return { entries: (listed.resourceTemplates || []).map((t) => ({ uriTemplate: t.uriTemplate, name: t.name, description: t.description ?? null, mimeType: t.mimeType ?? null })) };
  });
  out['prompts.list'] = await attempt(async () => {
    const listed = await client.experimental_listPrompts();
    return { entries: (listed.prompts || []).map((p) => ({
      name: p.name, title: p.title ?? null, description: p.description ?? null,
      arguments: (p.arguments || []).map((a) => ({ name: a.name, description: a.description ?? null, required: a.required ?? null })),
    })) };
  });
  const prompts = out['prompts.list'].ok ? out['prompts.list'].entries : [];
  out['prompts.get'] = prompts.length ? await attempt(async () => {
    const first = prompts[0];
    const args = Object.fromEntries(first.arguments.filter((a) => a.required).map((a) => [a.name, 'x']));
    const result = await client.experimental_getPrompt({ name: first.name, arguments: args });
    return {
      description: result.description ?? null,
      messages: (result.messages || []).map((m) => ({ role: m.role ?? null, type: m.content?.type ?? null, text: m.content?.text ?? null })),
    };
  }) : { ok: false, error: 'no prompt listed' };
  return out;
}

// Server utilities through the client's public surface, one question per worker run
// (each run is its own recorded fixture process), in the shape src/mcp_mirror/probe.py
// documents. MCPClient has no progress, logging or setLevel API: RequestOptions takes
// only `signal`, `timeout` and `maxTotalTimeout`, and every server notification is
// handed to `onUncaughtError` as "Unsupported message type". So those questions call
// the fixture tool through `execute` the way an agent would and report what reached
// the caller; the wire records what was sent.
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const stamp = () => new Date().toISOString().replace('Z', '000+00:00');

function textOf(returned) {
  if (typeof returned === 'string') return returned;
  const block = (returned?.content || []).find((b) => typeof b?.text === 'string');
  return block ? block.text : JSON.stringify(returned ?? null);
}

async function probeUtility(client, tools, feature, spec, uncaught) {
  const run = (name, args = {}, options = {}) => tools[name].execute(args, { toolCallId: `mcp-mirror-${name}`, messages: [], ...options });
  const unsupported = (reason) => ({ ok: false, unsupported: reason });
  switch (feature) {
    case 'tools.list':
      return { ok: true, route: 'adapter', api: 'MCPClient.tools', names: Object.keys(tools) };
    case 'progress':
      return attempt(async () => {
        const returned = await run('slow_with_progress');
        return {
          route: 'adapter', api: 'tool execute (RequestOptions has no progress callback)', events: [],
          text: textOf(returned), uncaught_notifications: uncaught.length,
        };
      });
    case 'logging':
      return attempt(async () => {
        const before = uncaught.length;
        const returned = await run('emit_logs');
        await sleep(200);
        return {
          route: 'adapter', api: 'tool execute (no logging handler; notifications reach onUncaughtError without their payload)',
          messages: [], text: textOf(returned), uncaught_notifications: uncaught.length - before,
        };
      });
    case 'set_level':
      return unsupported('MCPClient has no logging/setLevel API, and createMCPClient offers no route to send one');
    case 'completion':
      return attempt(async () => {
        const [promptName, promptArg, promptValue] = spec.completion_prompt;
        const [resourceUri, resourceArg, resourceValue] = spec.completion_resource;
        const prompt = await client.complete({ ref: { type: 'ref/prompt', name: promptName }, argument: { name: promptArg, value: promptValue } });
        const resource = await client.complete({ ref: { type: 'ref/resource', uri: resourceUri }, argument: { name: resourceArg, value: resourceValue } });
        return {
          route: 'adapter', api: 'MCPClient.complete',
          prompt: prompt?.completion?.values ?? [], resource: resource?.completion?.values ?? [],
        };
      });
    case 'cancellation':
      return attempt(async () => {
        const controller = new AbortController();
        const started = Date.now();
        setTimeout(() => controller.abort('mcp-mirror cancel'), spec.cancel_after_s * 1000);
        let outcome;
        try {
          outcome = `returned: ${textOf(await run('cancellable_wait', { seconds: spec.cancel_wait_s }, { abortSignal: controller.signal }))}`;
        } catch (err) {
          outcome = `threw: ${String((err && err.message) || err).slice(0, 200)}`;
        }
        const abandoned = (Date.now() - started) / 1000;
        // Long enough for the server to finish the call it was never told to stop.
        await sleep((spec.cancel_wait_s - spec.cancel_after_s) * 1000 + 500);
        return { route: 'adapter', mechanism: 'execute(args, { abortSignal })', abandoned_after_s: abandoned, outcome };
      });
    case 'tasks':
      return attempt(async () => ({ route: 'adapter', text: textOf(await run('task_capable')) }));
    case 'caching':
      return attempt(async () => {
        const marks = [];
        const list = async (call) => {
          const started = stamp();
          const listed = await client.tools();
          marks.push({ call, started, finished: stamp(), count: Object.keys(listed).length });
        };
        await list('first');
        await list('within_ttl');
        await sleep(spec.cache_pause_s * 1000);
        await list('after_ttl');
        return { route: 'adapter', api: 'MCPClient.tools', marks };
      });
    default:
      return unsupported(`unknown utility ${feature}`);
  }
}

// Catalogue changes against fixtures/notify_server.py, in the shape
// src/mcp_mirror/notify.py documents. MCPClient registers no notification handler of
// any kind: a server notification reaches only `onUncaughtError`, as an
// "Unsupported message type" error that does not say which notification it was. So
// `hook` is null, the count of those errors is kept beside it, and each change is
// checked by listing again through the client's own methods, which refetch every call.
const WATCHED_URI = 'file:///fixtures/watched.txt';

async function notifyProbe(client, negotiated, uncaught) {
  const call = async (name) => {
    const tools = await client.tools();
    return textOf(await tools[name].execute({}, { toolCallId: `mcp-mirror-${name}`, messages: [] }));
  };
  const lists = {
    tools: async () => Object.keys(await client.tools()),
    resources: async () => ((await client.listResources()).resources || []).map((r) => r.uri),
    prompts: async () => ((await client.experimental_listPrompts()).prompts || []).map((p) => p.name),
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
      const since = uncaught.length;
      const said = await call(tool);
      await sleep(1000);
      const after = (await lists[kind]()).sort();
      return {
        hook: null, notified: false, uncaught_notifications: uncaught.length - since,
        before, after, sentinel, server_said: said,
      };
    });
  }
  out['resources.subscribe'] = {
    ok: false,
    unsupported: `MCPClient has no resources/subscribe or unsubscribe API and no handler for notifications/resources/updated (${WATCHED_URI} not watched)`,
  };
  if (negotiated && negotiated >= '2026-07-28') {
    out['subscriptions.listen'] = {
      ok: false,
      unsupported: 'MCPClient has no subscriptions/listen API; it opens no stream for server notifications',
    };
  }
  return out;
}

// Result boundaries. `execute` resolves with the server's CallToolResult: that is the
// `output` generateText records in a step's toolResults. Before the next model call
// the AI SDK hands that output to the tool's own `toModelOutput`, which @ai-sdk/mcp
// sets to its MCP converter; its return value is what the provider adapter turns into
// the tool-result message the model reads.
async function invoke(tool, name, args, boundary) {
  const toolCallId = `mcp-mirror-${name}`;
  const options = { toolCallId, messages: [] };
  const output = await tool.execute(args, options);
  if (boundary === 'execute') return output;
  if (typeof tool.toModelOutput !== 'function') {
    throw new Error('the tool carries no toModelOutput');
  }
  return await tool.toModelOutput({ toolCallId, input: args, output });
}

async function main() {
  const raw = process.argv[2];
  if (!raw) {
    process.stderr.write('vercel_ai worker: missing config argument\n');
    process.exit(2);
  }
  const cfg = JSON.parse(raw);

  const headers = cfg.headers && Object.keys(cfg.headers).length ? { headers: cfg.headers } : {};
  const transport = cfg.transport === 'http' || cfg.transport === 'sse'
    ? { type: cfg.transport, url: cfg.url, ...headers }
    : new Experimental_StdioMCPTransport({ command: cfg.command, args: cfg.args || [] });

  // A server notification is not an error, but `onUncaughtError` is the only place
  // MCPClient hands one over, so the probes count what arrives there.
  const uncaught = [];
  const options = {
    transport,
    name: 'mcp-mirror',
    onUncaughtError: (err) => uncaught.push(String((err && err.message) || err).slice(0, 200)),
  };
  // Client-features probe: elicitation is the one server request MCPClient answers,
  // through `onElicitationRequest`, and the documented way to accept it is declaring
  // `capabilities.elicitation`. Form and URL modes are both declared so a gap is the
  // client's and not a capability left off. Every request the handler gets is recorded.
  const handlerSaw = [];
  if (cfg.elicitation) {
    options.capabilities = { elicitation: { form: {}, url: {} } };
  }
  const client = await createMCPClient(options);
  if (cfg.elicitation) {
    client.onElicitationRequest(ElicitationRequestSchema, async (request) => {
      const params = request.params || {};
      const mode = params.mode ?? (params.url ? 'url' : 'form');
      handlerSaw.push({ hook: 'onElicitationRequest', mode });
      if (mode === 'url') return { action: 'accept' };
      return { action: 'accept', content: { name: cfg.elicitation.elicited } };
    });
  }
  try {
    const tools = await client.tools();
    const out = [];
    for (const [name, tool] of Object.entries(tools)) {
      out.push({
        name,
        description: tool && tool.description != null ? String(tool.description) : null,
        parameters: await extractJsonSchema(tool && tool.inputSchema),
        annotations: {}, // the provider-bound fields (description, inputSchema) carry none
        retained: retainedMetadata(tool),
      });
    }

    const results = [];
    for (const call of cfg.calls || []) {
      const tool = tools[call.tool];
      if (!tool || typeof tool.execute !== 'function') {
        results.push({ tool: call.tool, ok: false, error: 'no executable tool by that name' });
        continue;
      }
      try {
        const returned = await invoke(tool, call.tool, call.arguments || {}, cfg.boundary || 'model_output');
        results.push({ tool: call.tool, ok: true, returned });
      } catch (err) {
        results.push({
          tool: call.tool,
          ok: false,
          threw: true,
          error: String((err && err.message) || err),
        });
      }
    }

    // `initializeResult` is public API: the result the client initialized with,
    // whose protocolVersion is the revision it negotiated with this server.
    const negotiatedMcpSpecVersion = client.initializeResult && client.initializeResult.protocolVersion;
    if (typeof negotiatedMcpSpecVersion !== 'string' || !negotiatedMcpSpecVersion) {
      throw new Error('vercel_ai worker: negotiated MCP protocol version was not observable');
    }
    let probe;
    if (cfg.probe) {
      probe = await probeSurface(client);
    }
    let utility;
    if (cfg.utility) {
      utility = await probeUtility(client, tools, cfg.utility, cfg.utilitySpec || {}, uncaught);
    }
    let notify;
    if (cfg.notify) {
      notify = await notifyProbe(client, negotiatedMcpSpecVersion, uncaught);
    }
    process.stdout.write(JSON.stringify({
      tools: out, results, negotiatedMcpSpecVersion, probe, utility, notify, handlerSaw,
    }));
  } finally {
    // Bounded: when the server command is a wrapper (the wire recorder), closing the
    // stdio transport can wait on a child that outlives it. The answer is already
    // written by then, so the worker does not wait more than two seconds for it.
    await Promise.race([
      client.close().catch(() => { /* best effort */ }),
      new Promise((resolve) => setTimeout(resolve, 2000).unref()),
    ]);
  }
}

main().then(
  () => process.exit(0),
  (err) => {
    process.stderr.write(String((err && err.stack) || err) + '\n');
    process.exit(1);
  },
);
