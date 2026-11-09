// Node worker that drives LangChain.js's REAL MCP adapter (@langchain/mcp-adapters
// MCPAdapter) so mcp-mirror can capture the tools it hands a chat model, never a
// reimplementation (decision D1). Tool definitions are taken through @langchain/core's
// convertToOpenAITool, the provider-shaped conversion bindTools uses. No chat model
// runs and no provider request is serialized.
//
// Usage: node _langchain_js_worker.mjs '<json-config>'
//   stdio: {"transport":"stdio","command":"python","args":["server.py"]}
//   http : {"transport":"http","url":"https://host/mcp","headers":{...}}
//   sse  : {"transport":"sse","url":"http://host/sse"}   (the deprecated HTTP+SSE transport)
// Optional keys: "calls" [{tool, arguments}], "boundary" ("agent" | "direct" | "graph"),
// "probe" (true), "utility" (a probe.py UTILITY_FEATURES key, with "utilitySpec"),
// "notify" (true), "elicitation" ({elicited}), "legacy" (true: the server option
// mode "legacy"). Emits one JSON object on stdout.

import { MCPAdapter, createMCPElicitationResume } from '@langchain/mcp-adapters';
import { convertToOpenAITool } from '@langchain/core/utils/function_calling';
import { AIMessage } from '@langchain/core/messages';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { Command, END, MemorySaver, MessagesAnnotation, START, StateGraph } from '@langchain/langgraph';

const SERVER = 'src';
// MCPAdapter prefixes every tool name with its server name and one underscore by default.
const PREFIX = `${SERVER}_`;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const stamp = () => new Date().toISOString().replace('Z', '000+00:00');
const message = (err) => {
  const own = String((err && err.message) || err);
  const cause = err && err.cause && err.cause.message && !own.includes(err.cause.message) ? `: ${err.cause.message}` : '';
  return own + cause;
};
const sourceName = (name) => (name.startsWith(PREFIX) ? name.slice(PREFIX.length) : name);

async function attempt(fn) {
  try {
    return { ok: true, ...(await fn()) };
  } catch (err) {
    return { ok: false, error: message(err).slice(0, 300) };
  }
}

async function sha256Hex(bytes) {
  const { createHash } = await import('node:crypto');
  return createHash('sha256').update(bytes).digest('hex');
}

function textOf(value) {
  if (typeof value === 'string') return value;
  const content = value && typeof value === 'object' && 'content' in value ? value.content : value;
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const block = content.find((b) => typeof b === 'string' || typeof b?.text === 'string');
    if (block) return typeof block === 'string' ? block : block.text;
  }
  return JSON.stringify(value ?? null);
}

// What a LangChain message holds, as plain JSON.
function messageJson(msg) {
  return {
    type: typeof msg?.getType === 'function' ? msg.getType() : null,
    status: msg?.status ?? null,
    name: msg?.name ?? null,
    tool_call_id: msg?.tool_call_id ?? null,
    content: msg?.content ?? null,
    artifact: msg?.artifact ?? null,
  };
}

const toolCall = (tool, args) => ({ name: tool.name, args, id: `mcp-mirror-${sourceName(tool.name)}`, type: 'tool_call' });

// A one-node LangGraph graph that runs a tool call through ToolNode, compiled with a
// checkpointer: the adapter's documented home for a tool that asks for input. At
// 2026-07-28 the adapter raises a LangGraph interrupt for each elicitation round; the
// graph is resumed with createMCPElicitationResume, as the adapter's docs show.
async function invokeInGraph(tool, args, answer) {
  const graph = new StateGraph(MessagesAnnotation)
    .addNode('tools', new ToolNode([tool]))
    .addEdge(START, 'tools')
    .addEdge('tools', END)
    .compile({ checkpointer: new MemorySaver() });
  const config = { configurable: { thread_id: `mcp-mirror-${sourceName(tool.name)}` } };
  let state = await graph.invoke({ messages: [new AIMessage({ content: '', tool_calls: [toolCall(tool, args)] })] }, config);
  for (let round = 0; round < 4; round += 1) {
    const snapshot = await graph.getState(config);
    const pending = (snapshot.tasks || []).flatMap((task) => task.interrupts || []);
    if (!pending.length) break;
    const interrupt = pending[0];
    const responses = Object.fromEntries(
      Object.entries(interrupt.value?.requests || {}).map(([key, request]) => [key, answer(request)]),
    );
    state = await graph.invoke(new Command({ resume: createMCPElicitationResume(interrupt, responses) }), config);
  }
  const messages = state.messages || [];
  return { kind: 'tool_message', value: messageJson(messages[messages.length - 1]) };
}

// Result boundaries. `direct` is StructuredTool.invoke with a plain argument object,
// which hands back the tool's content and discards the artifact half of the
// content_and_artifact tuple. `agent` is LangGraph's prebuilt ToolNode running the
// tool call an AIMessage carries, which is how createReactAgent and createAgent run
// tools; it returns the ToolMessage the next model call reads.
async function invoke(tool, args, boundary, answer) {
  if (boundary === 'direct') {
    return { kind: 'direct', value: await tool.invoke(args) };
  }
  if (boundary === 'graph') return invokeInGraph(tool, args, answer);
  const node = new ToolNode([tool]);
  const state = await node.invoke({ messages: [new AIMessage({ content: '', tool_calls: [toolCall(tool, args)] })] });
  const messages = Array.isArray(state) ? state : state.messages;
  return { kind: 'tool_message', value: messageJson(messages[messages.length - 1]) };
}

// Resources through MCPAdapter's listResources / readResource / listResourceTemplates,
// in the shape src/mcp_mirror/probe.py documents. The adapter has no prompt API.
async function probeSurface(adapter) {
  const out = {};
  out['resources.list'] = await attempt(async () => {
    const listed = (await adapter.listResources(SERVER))[SERVER] || [];
    return {
      entries: listed.map((r) => ({
        uri: r.uri, name: r.name ?? null, title: r.title ?? null, description: r.description ?? null, mimeType: r.mimeType ?? null,
      })),
    };
  });
  const entries = out['resources.list'].ok ? out['resources.list'].entries : [];
  const textUri = entries.find((e) => (e.mimeType || '').startsWith('text/'))?.uri;
  const blobUri = entries.find((e) => e.mimeType && !e.mimeType.startsWith('text/'))?.uri;
  const read = async (uri) => {
    const first = (await adapter.readResource(SERVER, uri))[0] || {};
    if (first.blob !== undefined) {
      const bytes = Buffer.from(first.blob, 'base64');
      return { uri: first.uri ?? null, mimeType: first.mimeType ?? null, blob_sha256: await sha256Hex(bytes), blob_bytes: bytes.length };
    }
    return { uri: first.uri ?? null, mimeType: first.mimeType ?? null, text: first.text ?? null };
  };
  out['resources.read.text'] = textUri ? await attempt(() => read(textUri)) : { ok: false, error: 'no text resource listed' };
  out['resources.read.blob'] = blobUri ? await attempt(() => read(blobUri)) : { ok: false, error: 'no binary resource listed' };
  out['resources.templates'] = await attempt(async () => {
    const listed = (await adapter.listResourceTemplates(SERVER))[SERVER] || [];
    return { entries: listed.map((t) => ({ uriTemplate: t.uriTemplate, name: t.name ?? null, description: t.description ?? null, mimeType: t.mimeType ?? null })) };
  });
  const noPrompts = 'MCPAdapter has no prompt API (no list, get or loader for prompts)';
  out['prompts.list'] = { ok: false, unsupported: noPrompts };
  out['prompts.get'] = { ok: false, unsupported: noPrompts };
  return out;
}

// Server utilities, one question per worker run (each run is its own recorded fixture
// process), in the shape src/mcp_mirror/probe.py documents. `hooks` holds what the
// server's onProgress and onMessage callbacks received.
async function probeUtility(adapter, era, tools, feature, spec, hooks) {
  const run = (name, args = {}, config = undefined) => tools[name].invoke(args, config);
  const unsupported = (reason) => ({ ok: false, unsupported: reason });
  switch (feature) {
    case 'tools.list':
      return { ok: true, route: 'adapter', api: 'MCPAdapter.listTools', names: Object.keys(tools) };
    case 'progress':
      return attempt(async () => {
        const text = textOf(await run('slow_with_progress'));
        return { route: 'adapter', api: 'server option onProgress', events: [...hooks.progress], text };
      });
    case 'logging':
      return attempt(async () => {
        const text = textOf(await run('emit_logs'));
        await sleep(200);
        return { route: 'adapter', api: era === 'modern' ? 'server options onMessage + logLevel' : 'server option onMessage', messages: [...hooks.messages], text };
      });
    case 'set_level':
      return attempt(async () => {
        // Two documented routes, one per era: setLoggingLevel for a legacy session,
        // and the per-server logLevel option (already set for this run) that a
        // modern tool request carries in its _meta. setLoggingLevel throws on modern.
        let api = 'server option logLevel';
        if (era !== 'modern') {
          await adapter.setLoggingLevel(SERVER, spec.set_level);
          api = 'MCPAdapter.setLoggingLevel';
        }
        const text = textOf(await run('emit_logs'));
        await sleep(200);
        return { route: 'adapter', api, level: spec.set_level, messages: [...hooks.messages], text };
      });
    case 'completion':
      return attempt(async () => {
        // MCPAdapter has no completion method; getClient hands out the MCP SDK Client
        // it connected, which has one.
        const session = await adapter.getClient(SERVER);
        const [promptName, promptArg, promptValue] = spec.completion_prompt;
        const [resourceUri, resourceArg, resourceValue] = spec.completion_resource;
        const prompt = await session.complete({ ref: { type: 'ref/prompt', name: promptName }, argument: { name: promptArg, value: promptValue } });
        const resource = await session.complete({ ref: { type: 'ref/resource', uri: resourceUri }, argument: { name: resourceArg, value: resourceValue } });
        return {
          route: 'session', api: 'MCPAdapter.getClient().complete',
          prompt: prompt?.completion?.values ?? [], resource: resource?.completion?.values ?? [],
        };
      });
    case 'cancellation':
      return attempt(async () => {
        const controller = new AbortController();
        const started = Date.now();
        setTimeout(() => controller.abort(new Error('mcp-mirror cancel')), spec.cancel_after_s * 1000);
        let outcome;
        try {
          outcome = `returned: ${textOf(await run('cancellable_wait', { seconds: spec.cancel_wait_s }, { signal: controller.signal }))}`;
        } catch (err) {
          outcome = `threw: ${message(err).slice(0, 200)}`;
        }
        const abandoned = (Date.now() - started) / 1000;
        // Long enough for the server to finish a call it was never told to stop.
        await sleep((spec.cancel_wait_s - spec.cancel_after_s) * 1000 + 500);
        return { route: 'adapter', mechanism: 'StructuredTool.invoke(args, { signal })', abandoned_after_s: abandoned, outcome };
      });
    case 'tasks':
      return attempt(async () => ({ route: 'adapter', text: textOf(await run('task_capable')) }));
    case 'caching':
      return attempt(async () => {
        const marks = [];
        const list = async (call) => {
          const started = stamp();
          const listed = await adapter.listTools(SERVER);
          marks.push({ call, started, finished: stamp(), count: listed.length });
        };
        await list('first');
        await list('within_ttl');
        await sleep(spec.cache_pause_s * 1000);
        await list('after_ttl');
        return { route: 'adapter', api: 'MCPAdapter.listTools', marks };
      });
    default:
      return unsupported(`unknown utility ${feature}`);
  }
}

// Catalogue changes against fixtures/notify_server.py, in the shape
// src/mcp_mirror/notify.py documents. The server's own hooks (onToolsListChanged,
// onResourcesListChanged, onPromptsListChanged, onResourcesUpdated) record what
// arrives; each change is then checked by listing again through the adapter. The
// adapter's only subscribe route is the per-server resourceSubscriptions option,
// applied when it connects: resources/subscribe on a legacy session, a
// subscriptions/listen stream on a modern one. It keeps no handle to undo either.
const WATCHED_URI = 'file:///fixtures/watched.txt';
const WAIT_MS = 3000;

async function waitFor(hooks, kind, since, ms) {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    if (hooks.notified[kind] > since) return true;
    await sleep(50);
  }
  return hooks.notified[kind] > since;
}

async function notifyProbe(adapter, negotiatedVersion, hooks, subscribeError) {
  const call = async (name) => {
    const tools = Object.fromEntries((await adapter.listTools(SERVER)).map((t) => [sourceName(t.name), t]));
    return textOf(await tools[name].invoke({}));
  };
  const lists = {
    tools: async () => (await adapter.listTools(SERVER)).map((t) => sourceName(t.name)),
    resources: async () => ((await adapter.listResources(SERVER))[SERVER] || []).map((r) => r.uri),
  };
  const hookNames = {
    tools: 'server option onToolsListChanged',
    resources: 'server option onResourcesListChanged',
    prompts: 'server option onPromptsListChanged',
  };
  const changes = [
    ['tools.list_changed', 'add_tool', 'tools', 'SENTINEL_ADDED_TOOL'],
    ['resources.list_changed', 'add_resource', 'resources', 'fixtures/added-'],
    ['prompts.list_changed', 'add_prompt', 'prompts', 'SENTINEL_ADDED_PROMPT'],
  ];
  const out = { negotiated: negotiatedVersion };
  for (const [key, tool, kind, sentinel] of changes) {
    if (!lists[kind]) {
      // Still fire the change, so the prompts hook is exercised for the listen row.
      const since = hooks.notified[kind];
      await attempt(async () => ({ said: await call(tool) }));
      await waitFor(hooks, kind, since, WAIT_MS);
      out[key] = { ok: false, unsupported: 'MCPAdapter has no prompt listing API' };
      continue;
    }
    out[key] = await attempt(async () => {
      const before = (await lists[kind]()).sort();
      const since = hooks.notified[kind];
      const said = await call(tool);
      const notified = await waitFor(hooks, kind, since, WAIT_MS);
      const after = (await lists[kind]()).sort();
      return { hook: hookNames[kind], notified, before, after, sentinel, server_said: said };
    });
  }
  out['resources.subscribe'] = await attempt(async () => {
    const result = {
      route: 'adapter', api: 'server option resourceSubscriptions + onResourcesUpdated',
      subscribed: !subscribeError, subscribe_error: subscribeError, notified: false, unsubscribed: false,
      unsubscribe_error: null, notified_after_unsubscribe: null, server_said: [],
    };
    const since = hooks.notified.updated;
    result.server_said.push(await call('update_resource'));
    result.notified = await waitFor(hooks, 'updated', since, WAIT_MS);
    if (result.subscribed) {
      result.unsubscribe_error = 'MCPAdapter has no unsubscribe method, and resourceSubscriptions is fixed for the connection';
    }
    return result;
  });
  if (negotiatedVersion && negotiatedVersion >= '2026-07-28') {
    // On a modern session the adapter opens one subscriptions/listen stream when it
    // connects, filtered to the hooks configured; what those hooks heard is the row.
    const names = { tools: 'ToolsListChanged', resources: 'ResourcesListChanged', prompts: 'PromptsListChanged', updated: 'ResourceUpdated' };
    out['subscriptions.listen'] = {
      ok: true, route: 'adapter',
      api: 'MCPAdapter subscriptions/listen stream, opened at connect by the list-changed hooks and resourceSubscriptions',
      opened: !subscribeError, open_error: subscribeError, honored: null,
      events: Object.keys(names).filter((kind) => hooks.notified[kind] > 0).map((kind) => names[kind]),
    };
  }
  return out;
}

function connection(cfg, hooks, extra) {
  const headers = cfg.headers && Object.keys(cfg.headers).length ? { headers: cfg.headers } : {};
  const base = cfg.transport === 'http' || cfg.transport === 'sse'
    ? { transport: cfg.transport, url: cfg.url, ...headers }
    : { transport: 'stdio', command: cfg.command, args: cfg.args || [] };
  // Every callback the adapter documents for server messages, registered once so each
  // probe can read what reached the application. In 2.0 they are server options.
  return {
    ...base,
    onProgress: (progress) => {
      hooks.progress.push({ progress: progress?.progress ?? null, total: progress?.total ?? null, message: progress?.message ?? null });
    },
    onMessage: (log) => {
      hooks.messages.push({ level: log?.level ?? null, data: log?.data ?? null });
    },
    onToolsListChanged: () => { hooks.notified.tools += 1; },
    onResourcesListChanged: () => { hooks.notified.resources += 1; },
    onPromptsListChanged: () => { hooks.notified.prompts += 1; },
    onResourcesUpdated: () => { hooks.notified.updated += 1; },
    ...extra,
  };
}

async function closeBounded(adapter) {
  // Bounded: when the server command is a wrapper (the wire recorder), closing the
  // stdio transport can wait on a child that outlives it.
  await Promise.race([
    adapter.close().catch(() => { /* best effort */ }),
    new Promise((resolve) => setTimeout(resolve, 2000).unref()),
  ]);
}

async function main() {
  const raw = process.argv[2];
  if (!raw) {
    process.stderr.write('langchain_js worker: missing config argument\n');
    process.exit(2);
  }
  const cfg = JSON.parse(raw);

  const hooks = { progress: [], messages: [], notified: { tools: 0, resources: 0, prompts: 0, updated: 0 } };
  const handlerSaw = [];
  const extra = {};
  if (cfg.legacy) extra.mode = 'legacy';
  if (cfg.utility === 'set_level' && !cfg.legacy && cfg.transport !== 'sse') extra.logLevel = cfg.utilitySpec?.set_level;
  // A 2026-07-28 server logs only at a level the request names in _meta, and the
  // logLevel option is the adapter's documented way to name one. The adapter adds it
  // to modern tool requests only, so a legacy session is unchanged.
  if (cfg.utility === 'logging' && !cfg.legacy && cfg.transport !== 'sse') extra.logLevel = 'debug';
  if (cfg.elicitation && cfg.legacy) {
    // The legacy-session hook: answers elicitation/create requests on the session.
    extra.onElicitation = async (params) => {
      handlerSaw.push({ hook: 'onElicitation', mode: params?.mode ?? 'form' });
      return params?.mode === 'url' ? { action: 'accept' } : { action: 'accept', content: { name: cfg.elicitation.elicited } };
    };
  }
  // A modern question arrives as a LangGraph interrupt instead; this answers it.
  const answer = (request) => {
    handlerSaw.push({ hook: 'LangGraph interrupt + createMCPElicitationResume', mode: request?.mode ?? 'form' });
    return request?.mode === 'url' ? { action: 'accept' } : { action: 'accept', content: { name: cfg.elicitation?.elicited } };
  };
  const build = (more) => new MCPAdapter({ servers: { [SERVER]: connection(cfg, hooks, { ...extra, ...more }) } });

  let adapter = build(cfg.notify ? { resourceSubscriptions: [WATCHED_URI] } : {});
  let subscribeError = null;
  let loaded;
  try {
    loaded = await adapter.listTools(SERVER);
  } catch (err) {
    if (!cfg.notify) throw err;
    // The subscription is applied at connect, so a server that refuses it fails the
    // connection; record that and measure the rest without it.
    subscribeError = message(err).slice(0, 500);
    await closeBounded(adapter);
    adapter = build({});
    loaded = await adapter.listTools(SERVER);
  }
  try {
    const tools = Object.fromEntries(loaded.map((t) => [sourceName(t.name), t]));
    const out = [];
    for (const tool of loaded) {
      // The OpenAI-compatible function tool bindTools builds for a provider.
      const converted = convertToOpenAITool(tool);
      const fn = converted.function || {};
      out.push({
        name: fn.name ?? tool.name,
        description: fn.description ?? null,
        parameters: fn.parameters ?? {},
        annotations: {}, // the provider-shaped tool carries none
        // The adapter keeps the server's annotations on the tool's app-side
        // `metadata`, which convertToOpenAITool does not read.
        retained: { annotations: tool.metadata?.annotations ?? null },
      });
    }

    const session = await adapter.getClient(SERVER);
    const negotiatedMcpSpecVersion = session?.getNegotiatedProtocolVersion();
    const protocolEra = session?.getProtocolEra() ?? null;
    if (typeof negotiatedMcpSpecVersion !== 'string' || !negotiatedMcpSpecVersion) {
      throw new Error('langchain_js worker: negotiated MCP protocol version was not observable');
    }

    const results = [];
    for (const call of cfg.calls || []) {
      // Callers may pass the model-facing name (src_x) or the server's own (x).
      const tool = tools[sourceName(call.tool)];
      if (!tool) {
        results.push({ tool: call.tool, ok: false, error: 'no tool by that name' });
        continue;
      }
      try {
        const returned = await invoke(tool, call.arguments || {}, cfg.boundary || 'agent', answer);
        results.push({ tool: call.tool, ok: true, returned });
      } catch (err) {
        results.push({ tool: call.tool, ok: false, threw: true, error: message(err) });
      }
    }

    const probe = cfg.probe ? await probeSurface(adapter) : undefined;
    const utility = cfg.utility ? await probeUtility(adapter, protocolEra, tools, cfg.utility, cfg.utilitySpec || {}, hooks) : undefined;
    const notify = cfg.notify ? await notifyProbe(adapter, negotiatedMcpSpecVersion, hooks, subscribeError) : undefined;
    process.stdout.write(JSON.stringify({
      tools: out, results, negotiatedMcpSpecVersion, protocolEra, probe, utility, notify, handlerSaw,
    }));
  } finally {
    await closeBounded(adapter);
  }
}

main().then(
  () => process.exit(0),
  (err) => {
    process.stderr.write(String((err && err.stack) || err) + '\n');
    process.exit(1);
  },
);
