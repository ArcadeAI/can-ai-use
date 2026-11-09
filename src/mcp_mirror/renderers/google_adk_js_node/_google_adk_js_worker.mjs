// Node worker that drives Google ADK (TypeScript)'s REAL MCP adapter (@google/adk
// MCPToolset and MCPSessionManager; adk-js core/src/tools/mcp) so mcp-mirror can
// capture what ADK hands its model, never a reimplementation (decision D1).
//
// Every capture goes through a real agent turn: an LlmAgent is built with the
// MCPToolset in `tools`, and an InMemoryRunner drives it against a scripted BaseLlm.
// ADK loads the MCP tools itself (MCPToolset.getTools -> MCPTool), and each MCPTool adds
// its FunctionDeclaration to the request in BaseTool.processLlmRequest. The tool
// definitions are the `functionDeclarations` in the LlmRequest.config.tools the model
// receives, and a tool result is the `functionResponse` part in the next LlmRequest's
// contents. No Gemini model runs and no provider request is serialized.
//
// Usage: node _google_adk_js_worker.mjs '<json-config>'
//   stdio: {"transport":"stdio","command":"python","args":["server.py"]}
//   http : {"transport":"http","url":"https://host/mcp","headers":{...}}

import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { BaseLlm, InMemoryRunner, LlmAgent } from '@google/adk';
import { MCPSessionManager, MCPToolset } from '@google/adk/tools/mcp';

// ADK loads its optional peer @modelcontextprotocol/sdk through require(), so the
// Client it builds is the CommonJS build. Load the same module instance here.
const require = createRequire(import.meta.url);
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const SDK_TYPES = require('@modelcontextprotocol/sdk/types.js');

// stdout carries this worker's JSON answer and nothing else. ADK's logger (MCP
// transport errors included) and anything else that writes there go to stderr.
const writeAnswer = process.stdout.write.bind(process.stdout);
process.stdout.write = process.stderr.write.bind(process.stderr);
for (const method of ['log', 'info', 'debug', 'warn']) console[method] = console.error;

const APP = 'mcp-mirror';
const USER = 'mcp-mirror';
const CALL_ID = 'mcp-mirror-call';
const WATCHED_URI = 'file:///fixtures/watched.txt';

const errorText = (err) => {
  const own = String((err && err.message) || err);
  const cause = err && err.cause ? `: ${String(err.cause.message || err.cause)}` : '';
  return (own + cause).slice(0, 500);
};
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const stamp = () => new Date().toISOString().replace('Z', '000+00:00');
const clone = (value) => (value === undefined ? undefined : JSON.parse(JSON.stringify(value)));

async function attempt(fn) {
  try {
    return { ok: true, ...(await fn()) };
  } catch (err) {
    return { ok: false, error: errorText(err) };
  }
}

// ADK exposes no negotiated protocol version, and the SDK Client stores none it can be
// read back from on stdio. Every MCPSessionManager.createSession runs Client.connect,
// which sends `initialize` through Client.request; record the protocolVersion of each
// answer as evidence only. The worker fails closed if none or more than one is seen,
// rather than copying the direct source connection's version by assumption.
const negotiated = new Set();
const originalRequest = Client.prototype.request;
Client.prototype.request = async function request(req, ...rest) {
  const result = await originalRequest.call(this, req, ...rest);
  if (req && req.method === 'initialize' && result && typeof result.protocolVersion === 'string') {
    negotiated.add(result.protocolVersion);
  }
  return result;
};

function negotiatedVersion() {
  if (negotiated.size !== 1) {
    throw new Error(`google_adk_js worker: expected one negotiated MCP protocol version, observed ${JSON.stringify([...negotiated])}`);
  }
  return [...negotiated][0];
}

function connectionParams(cfg) {
  if (cfg.transport === 'http') {
    const params = { type: 'StreamableHTTPConnectionParams', url: cfg.url };
    if (cfg.headers && Object.keys(cfg.headers).length) params.header = cfg.headers;
    return params;
  }
  if (cfg.transport === 'sse') {
    throw new Error('MCPConnectionParams has no SSE variant; ADK (TypeScript) offers stdio and Streamable HTTP only');
  }
  return { type: 'StdioConnectionParams', serverParams: { command: cfg.command, args: cfg.args || [] } };
}

// A BaseLlm that answers from a script. The first turn optionally asks for one tool
// call; every later turn ends the run. It records each LlmRequest it is handed.
class ScriptedLlm extends BaseLlm {
  constructor(call) {
    super({ model: 'mcp-mirror-scripted' });
    this.call = call;
    this.requests = [];
  }

  async *generateContentAsync(llmRequest) {
    const declarations = ((llmRequest.config && llmRequest.config.tools) || [])
      .flatMap((tool) => (tool && tool.functionDeclarations) || []);
    const responses = (llmRequest.contents || [])
      .flatMap((content) => content.parts || [])
      .filter((part) => part.functionResponse !== undefined)
      .map((part) => part.functionResponse);
    this.requests.push(clone({ declarations, responses }));
    const part = this.call && this.requests.length === 1
      ? { functionCall: { id: CALL_ID, name: this.call.tool, args: this.call.arguments || {} } }
      : { text: 'done' };
    yield { content: { role: 'model', parts: [part] } };
  }

  async connect() {
    throw new Error('the scripted model only answers generateContentAsync');
  }
}

// One LlmAgent run. Returns the requests the model saw and, when a tool ran, the value
// MCPTool.runAsync returned as the agent's afterToolCallback received it.
async function agentTurn(toolset, call, runOptions = {}) {
  const model = new ScriptedLlm(call);
  const captured = {};
  const agent = new LlmAgent({
    name: 'mcp_mirror',
    model,
    instruction: 'mcp-mirror probe',
    tools: [toolset],
    afterToolCallback: ({ response }) => {
      captured.direct = clone(response);
      return undefined;
    },
  });
  const runner = new InMemoryRunner({ agent, appName: APP });
  const session = await runner.sessionService.createSession({ appName: APP, userId: USER });
  const events = runner.runAsync({
    userId: USER,
    sessionId: session.id,
    newMessage: { role: 'user', parts: [{ text: 'mcp-mirror' }] },
    ...runOptions,
  });
  for await (const _event of events) { /* drain */ }
  return { requests: model.requests, direct: captured.direct };
}

function agentResult(run) {
  const second = run.requests[1];
  const responses = (second && second.responses) || [];
  return responses.find((r) => r && r.id === CALL_ID) || responses[responses.length - 1];
}

function textOf(returned) {
  if (typeof returned === 'string') return returned;
  const list = Array.isArray(returned) ? returned : (returned?.content || returned?.response?.content || []);
  const block = Array.isArray(list) ? list.find((b) => typeof b?.text === 'string') : undefined;
  return block ? block.text : JSON.stringify(returned ?? null);
}

// A raw SDK Client from MCPSessionManager.createSession: the session manager MCPToolset
// and MCPTool are built on, exported beside them. Answers reached here are route
// "session".
async function onSession(cfg, body) {
  const manager = new MCPSessionManager(connectionParams(cfg));
  const client = await manager.createSession();
  try {
    return await body(client);
  } finally {
    await manager.closeSession(client).catch(() => {});
  }
}

// Resources through MCPToolset.listResources / getResourceInfo / readResource, in the
// shape src/mcp_mirror/probe.py documents. readResource takes the resource's name, not
// its URI. MCPToolset has no template or prompt method.
async function probeSurface(toolset) {
  const out = {};
  out['resources.list'] = await attempt(async () => {
    const names = await toolset.listResources();
    const entries = [];
    for (const name of names) {
      const r = await toolset.getResourceInfo(name);
      entries.push({ uri: r.uri, name: r.name, title: r.title ?? null, description: r.description ?? null, mimeType: r.mimeType ?? null });
    }
    return { entries };
  });
  const entries = out['resources.list'].ok ? out['resources.list'].entries : [];
  const textName = entries.find((e) => (e.mimeType || '').startsWith('text/'))?.name;
  const blobName = entries.find((e) => !(e.mimeType || '').startsWith('text/'))?.name;
  const read = async (name) => {
    const contents = await toolset.readResource(name);
    const first = (contents || [])[0] || {};
    if (first.blob !== undefined) {
      const bytes = Buffer.from(first.blob, 'base64');
      return { uri: first.uri ?? null, mimeType: first.mimeType ?? null, blob_sha256: createHash('sha256').update(bytes).digest('hex'), blob_bytes: bytes.length };
    }
    return { uri: first.uri ?? null, mimeType: first.mimeType ?? null, text: first.text ?? null };
  };
  out['resources.read.text'] = textName ? await attempt(() => read(textName)) : { ok: false, error: 'no text resource listed' };
  out['resources.read.blob'] = blobName ? await attempt(() => read(blobName)) : { ok: false, error: 'no binary resource listed' };
  for (const [feature, method] of [
    ['resources.templates', 'listResourceTemplates'],
    ['prompts.list', 'listPrompts'],
    ['prompts.get', 'getPrompt'],
  ]) {
    out[feature] = typeof toolset[method] === 'function'
      ? { ok: false, error: `MCPToolset.${method} exists but is not probed` }
      : { ok: false, unsupported: `MCPToolset has no ${method} method` };
  }
  return out;
}

// Server utilities, one question per worker run (each run is its own recorded fixture
// process), in the shape src/mcp_mirror/probe.py documents. MCPToolset takes only a
// tool filter and a prefix: no progress, logging or message hook, no log level and no
// completion API. The Client that MCPSessionManager.createSession returns is the public
// session route, so those answers carry route "session".
async function probeUtility(cfg, feature, spec) {
  const toolset = () => new MCPToolset(connectionParams(cfg));
  const withToolset = async (fn) => {
    const ts = toolset();
    try { return await fn(ts); } finally { await ts.close().catch(() => {}); }
  };
  const callTool = (client, name, args = {}, options) =>
    client.callTool({ name, arguments: args }, undefined, options);
  switch (feature) {
    case 'tools.list':
      return attempt(() => withToolset(async (ts) => {
        const run = await agentTurn(ts, null);
        return { route: 'adapter', api: 'LlmAgent({ tools: [MCPToolset] }) functionDeclarations in the LlmRequest', names: run.requests[0].declarations.map((d) => d.name) };
      }));
    case 'progress':
      return attempt(() => onSession(cfg, async (client) => {
        const events = [];
        const result = await callTool(client, 'slow_with_progress', {}, {
          onprogress: (p) => events.push({ progress: p.progress, total: p.total ?? null, message: p.message ?? null }),
        });
        return { route: 'session', api: 'MCPSessionManager.createSession().callTool(params, undefined, { onprogress })', events, text: textOf(result) };
      }));
    case 'logging':
      return attempt(() => onSession(cfg, async (client) => {
        const messages = [];
        client.setNotificationHandler(SDK_TYPES.LoggingMessageNotificationSchema, (n) => {
          messages.push({ level: n.params.level, data: n.params.data });
        });
        const result = await callTool(client, 'emit_logs');
        await sleep(300);
        return { route: 'session', api: 'MCPSessionManager.createSession().setNotificationHandler(LoggingMessageNotificationSchema)', messages, text: textOf(result) };
      }));
    case 'set_level':
      return attempt(() => onSession(cfg, async (client) => {
        await client.setLoggingLevel(spec.set_level);
        const result = await callTool(client, 'emit_logs');
        return { route: 'session', api: 'MCPSessionManager.createSession().setLoggingLevel', level: spec.set_level, text: textOf(result) };
      }));
    case 'completion':
      return attempt(() => onSession(cfg, async (client) => {
        const [promptName, promptArg, promptValue] = spec.completion_prompt;
        const [uri, resourceArg, resourceValue] = spec.completion_resource;
        const prompt = await client.complete({ ref: { type: 'ref/prompt', name: promptName }, argument: { name: promptArg, value: promptValue } });
        const resource = await client.complete({ ref: { type: 'ref/resource', uri }, argument: { name: resourceArg, value: resourceValue } });
        return { route: 'session', api: 'MCPSessionManager.createSession().complete', prompt: prompt.completion.values, resource: resource.completion.values };
      }));
    case 'cancellation':
      // MCPTool.runAsync hands toolContext.abortSignal to Client.callTool, and
      // Runner.runAsync takes the abortSignal that context carries.
      return attempt(() => withToolset(async (ts) => {
        const controller = new AbortController();
        const started = Date.now();
        setTimeout(() => controller.abort('mcp-mirror cancel'), spec.cancel_after_s * 1000);
        let outcome;
        try {
          const run = await agentTurn(ts, { tool: 'cancellable_wait', arguments: { seconds: spec.cancel_wait_s } }, { abortSignal: controller.signal });
          const item = agentResult(run);
          outcome = item ? `returned: ${textOf(item.response)}` : 'cancelled: the run ended with no functionResponse';
        } catch (err) {
          outcome = `threw: ${errorText(err).slice(0, 200)}`;
        }
        const abandoned = (Date.now() - started) / 1000;
        await sleep(300);
        return { route: 'adapter', mechanism: 'Runner.runAsync({ abortSignal }) aborting the MCPTool call', abandoned_after_s: abandoned, outcome };
      }));
    case 'tasks':
      return attempt(() => withToolset(async (ts) => {
        const run = await agentTurn(ts, { tool: 'task_capable', arguments: {} });
        return { route: 'adapter', text: textOf(run.direct) };
      }));
    case 'caching':
      return attempt(() => withToolset(async (ts) => {
        const marks = [];
        for (const call of ['first', 'within_ttl', 'after_ttl']) {
          if (call === 'after_ttl') await sleep(spec.cache_pause_s * 1000);
          const started = stamp();
          const listed = await ts.getTools();
          marks.push({ call, started, finished: stamp(), count: listed.length });
        }
        return { route: 'adapter', api: 'MCPToolset.getTools (no cache option)', marks };
      }));
    default:
      return { ok: false, unsupported: `unknown utility ${feature}` };
  }
}

// Catalogue changes against fixtures/notify_server.py, in the shape
// src/mcp_mirror/notify.py documents. MCPToolset takes no message handler, so each
// change is read only from its next listing. MCPToolset has no prompt or subscribe
// API; the subscription runs on a Client from MCPSessionManager.createSession, with the
// update fired on that same Client.
async function notifyProbe(cfg, toolset) {
  const call = async (name) => textOf((await agentTurn(toolset, { tool: name, arguments: {} })).direct);
  const lists = {
    tools: async () => (await toolset.getTools()).map((t) => t.name),
    resources: async () => {
      const uris = [];
      for (const name of await toolset.listResources()) uris.push((await toolset.getResourceInfo(name)).uri);
      return uris;
    },
  };
  await lists.tools();
  const negotiated = negotiatedVersion();
  const out = { negotiated };
  for (const [key, tool, kind, sentinel] of [
    ['tools.list_changed', 'add_tool', 'tools', 'SENTINEL_ADDED_TOOL'],
    ['resources.list_changed', 'add_resource', 'resources', 'fixtures/added-'],
  ]) {
    out[key] = await attempt(async () => {
      const before = (await lists[kind]()).sort();
      const said = await call(tool);
      await sleep(1000);
      const after = (await lists[kind]()).sort();
      return { hook: null, notified: false, before, after, sentinel, server_said: said };
    });
  }
  out['prompts.list_changed'] = { ok: false, unsupported: 'MCPToolset has no prompt listing API' };
  out['resources.subscribe'] = await attempt(() => onSession(cfg, async (client) => {
    let updates = 0;
    client.setNotificationHandler(SDK_TYPES.ResourceUpdatedNotificationSchema, () => { updates += 1; });
    const waitFor = async (since, ms) => {
      const deadline = Date.now() + ms;
      while (updates <= since && Date.now() < deadline) await sleep(50);
      return updates > since;
    };
    const fire = async () => textOf(await client.callTool({ name: 'update_resource', arguments: {} }));
    const result = {
      route: 'session',
      api: 'MCPSessionManager.createSession().subscribeResource / unsubscribeResource, with setNotificationHandler(ResourceUpdatedNotificationSchema) (MCPToolset takes no message handler)',
      subscribed: false, subscribe_error: null, notified: false,
      unsubscribed: false, unsubscribe_error: null, notified_after_unsubscribe: null, server_said: [],
    };
    try {
      await client.subscribeResource({ uri: WATCHED_URI });
      result.subscribed = true;
    } catch (err) {
      result.subscribe_error = errorText(err);
    }
    let since = updates;
    result.server_said.push(await fire());
    result.notified = await waitFor(since, 3000);
    if (!result.subscribed) return result;
    try {
      await client.unsubscribeResource({ uri: WATCHED_URI });
      result.unsubscribed = true;
    } catch (err) {
      result.unsubscribe_error = errorText(err);
      return result;
    }
    since = updates;
    result.server_said.push(await fire());
    result.notified_after_unsubscribe = await waitFor(since, 1500);
    return result;
  }));
  if (negotiated >= '2026-07-28') {
    out['subscriptions.listen'] = { ok: false, unsupported: 'neither MCPToolset nor the SDK v1 Client opens subscriptions/listen' };
  }
  return out;
}

async function main() {
  const raw = process.argv[2];
  if (!raw) {
    process.stderr.write('google_adk_js worker: missing config argument\n');
    process.exit(2);
  }
  const cfg = JSON.parse(raw);
  if (cfg.utility) {
    const utility = await probeUtility(cfg, cfg.utility, cfg.utilitySpec || {});
    writeAnswer(JSON.stringify({ tools: [], results: [], negotiatedMcpSpecVersion: negotiatedVersion(), utility }));
    return;
  }
  const toolset = new MCPToolset(connectionParams(cfg));
  try {
    const definitions = await agentTurn(toolset, null);
    const tools = clone(definitions.requests[0].declarations);
    const results = [];
    for (const call of cfg.calls || []) {
      try {
        const run = await agentTurn(toolset, call);
        const item = agentResult(run);
        if (!item) {
          results.push({ tool: call.tool, ok: false, error: 'the next LlmRequest carried no functionResponse' });
        } else {
          results.push({ tool: call.tool, ok: true, agent: item, direct: run.direct ?? null });
        }
      } catch (err) {
        results.push({ tool: call.tool, ok: false, threw: true, error: errorText(err) });
      }
    }
    const probe = cfg.probe ? await probeSurface(toolset) : undefined;
    const notify = cfg.notify ? await notifyProbe(cfg, toolset) : undefined;
    writeAnswer(JSON.stringify({ tools, results, negotiatedMcpSpecVersion: negotiatedVersion(), probe, notify }));
  } finally {
    await toolset.close().catch(() => {});
  }
}

main().catch((err) => {
  process.stderr.write(String((err && err.stack) || err) + '\n');
  process.exit(1);
});
