// Node worker that drives the REAL GitHub Copilot SDK (@github/copilot-sdk) so
// mcp-mirror can capture what its MCP integration hands a model, never a
// reimplementation (decision D1). The SDK starts its bundled Copilot runtime as a child
// process; the runtime is the MCP client. A session is created with the fixture as its
// only MCP server and a BYOK OpenAI-compatible provider, and a CopilotRequestHandler
// subclass answers every model request the runtime sends, so the captured object is
// the serialized chat/completions request body and no network call is made.
//
// The runtime runs with HOME and COPILOT_HOME in a fresh temporary directory and
// enableConfigDiscovery off, so no operator config, MCP server list or login is read.
//
// Usage: node _copilot_sdk_worker.mjs '<json-config>'
//   stdio: {"transport":"stdio","command":"python","args":["server.py"]}
//   http : {"transport":"http","url":"https://host/mcp","headers":{...}}
//   sse  : {"transport":"sse","url":"http://host/sse"}   (the deprecated HTTP+SSE transport)
// Optional keys: "calls" [{tool, arguments}], "boundary" ("agent" | "direct"),
// "probe" (true), "utility" (a probe.py UTILITY_FEATURES key, with "utilitySpec"),
// "notify" (true), "elicitation" ({elicited}), "sampling" ({sampled, afterTool}).
// Emits one JSON object on stdout.

import { CopilotClient, CopilotRequestHandler, approveAll } from '@github/copilot-sdk';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';

const SERVER = 'src';
// The runtime names each MCP tool `<server>-<tool>` in the model request.
const PREFIX = `${SERVER}-`;
const MODEL = 'gpt-4.1';
const CALL_ID = 'call_mcp_mirror';
const TURN_TIMEOUT_MS = 60000;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const stamp = () => new Date().toISOString().replace('Z', '000+00:00');
const message = (err) => {
  const own = String((err && err.message) || err);
  const cause = err && err.cause && err.cause.message && !own.includes(err.cause.message) ? `: ${err.cause.message}` : '';
  return own + cause;
};
const sourceName = (name) => (name.startsWith(PREFIX) ? name.slice(PREFIX.length) : name);
const modelName = (name) => (name.startsWith(PREFIX) ? name : PREFIX + name);

async function attempt(fn) {
  try {
    return { ok: true, ...(await fn()) };
  } catch (err) {
    return { ok: false, error: message(err).slice(0, 300) };
  }
}

function completion(msg) {
  return new Response(JSON.stringify({
    id: 'mcp-mirror', object: 'chat.completion', created: 0, model: MODEL,
    choices: [{ index: 0, finish_reason: msg.tool_calls ? 'tool_calls' : 'stop', message: msg }],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  }), { status: 200, headers: { 'content-type': 'application/json' } });
}

// The stub model. Every request the runtime sends lands here; `plan` decides the reply.
// A request that offers none of the session's own tools is a server's sampling
// request the runtime forwarded to the session model.
class StubModel extends CopilotRequestHandler {
  constructor(state) {
    super();
    this.state = state;
  }

  async sendRequest(request, ctx) {
    const body = JSON.parse((await request.text()) || '{}');
    const state = this.state;
    state.requests.push({ url: String(ctx?.url ?? ''), body });
    const tools = (body.tools || []).map((t) => t.function?.name ?? t.name);
    const agentTurn = tools.some((name) => name === 'bash' || name === 'view' || name?.startsWith(PREFIX));
    if (!agentTurn) {
      state.samplingRequests.push(body);
      return completion(samplingReply(body, state.sampling));
    }
    const messages = body.messages || [];
    if (state.callAll && !messages.some((m) => m.role === 'tool')) {
      // Ask for every MCP tool at once, so the runtime's approval step sees each.
      return completion({
        role: 'assistant', content: null,
        tool_calls: tools.filter((name) => name?.startsWith(PREFIX)).map((name, i) => ({ id: `${CALL_ID}_all_${i}`, type: 'function', function: { name, arguments: '{}' } })),
      });
    }
    const answered = messages.findIndex((m) => (m.tool_calls || []).some((c) => c.id === state.callId));
    if (state.pendingCall && answered < 0) {
      const call = state.pendingCall;
      return completion({
        role: 'assistant', content: null,
        tool_calls: [{ id: state.callId, type: 'function', function: { name: modelName(call.tool), arguments: JSON.stringify(call.arguments || {}) } }],
      });
    }
    if (answered >= 0 && !state.answered) {
      // Everything the runtime appended after the model's tool call: the tool message
      // and any message it added for the same result (images ride a user message).
      state.answered = messages.slice(answered + 1);
    }
    return completion({ role: 'assistant', content: 'done' });
  }
}

// The sampling stub, in OpenAI chat form: call the first offered tool once, then
// answer in text naming whether a tool result came back (client_probe.sampling_reply).
function samplingReply(body, spec) {
  const messages = body.messages || [];
  const afterTool = messages.some((m) => m.role === 'tool');
  const tools = body.tools || [];
  if (tools.length && !afterTool) {
    return {
      role: 'assistant', content: null,
      tool_calls: [{ id: 'SENTINEL_TOOL_USE_ID', type: 'function', function: { name: tools[0].function?.name ?? tools[0].name, arguments: JSON.stringify({ word: 'mirror' }) } }],
    };
  }
  return { role: 'assistant', content: afterTool ? spec?.afterTool ?? 'SENTINEL_SAMPLED_AFTER_TOOL' : spec?.sampled ?? 'SENTINEL_SAMPLED' };
}

function serverConfig(cfg) {
  if (cfg.transport === 'http' || cfg.transport === 'sse') {
    const headers = cfg.headers && Object.keys(cfg.headers).length ? { headers: cfg.headers } : {};
    return { type: cfg.transport, url: cfg.url, ...headers, tools: ['*'] };
  }
  return { type: 'stdio', command: cfg.command, args: cfg.args || [], tools: ['*'], env: { ...process.env } };
}

// The JSON-RPC frames the runtime exchanged with the server, read through the
// session's own diagnostics API at trace level.
async function frames(session) {
  const out = [];
  let cursor;
  for (let page = 0; page < 20; page += 1) {
    const read = await session.rpc.diagnostics.read({ sources: ['mcp'], max: 500, ...(cursor ? { cursor } : {}) });
    const entries = read.entries || [];
    for (const entry of entries) {
      const details = entry.details || {};
      if (details.kind !== 'protocol' || typeof details.data !== 'string') continue;
      try {
        out.push({ direction: details.direction, frame: JSON.parse(details.data) });
      } catch {
        // A frame the runtime truncated is not evidence of anything.
      }
    }
    cursor = read.nextCursor ?? read.cursor;
    if (!entries.length || !cursor || read.hasMore === false) break;
  }
  return out;
}

// The revision the runtime settled on: the `initialize` result's protocolVersion on a
// handshake-era session, or, when server/discover succeeded, the protocolVersion the
// runtime then stamps on each request's _meta.
function negotiated(seen) {
  const byId = new Map();
  for (const { direction, frame } of seen) {
    if (direction === 'client-to-server' && frame.method && frame.id !== undefined) byId.set(frame.id, frame.method);
  }
  for (const { direction, frame } of seen) {
    if (direction !== 'server-to-client' || frame.id === undefined) continue;
    if (byId.get(frame.id) === 'initialize' && frame.result?.protocolVersion) return frame.result.protocolVersion;
  }
  const discovered = seen.some(({ direction, frame }) => direction === 'server-to-client' && byId.get(frame.id) === 'server/discover' && frame.result);
  if (discovered) {
    for (const { direction, frame } of seen) {
      const version = frame.params?._meta?.['io.modelcontextprotocol/protocolVersion'];
      if (direction === 'client-to-server' && frame.method && frame.method !== 'server/discover' && version) return version;
    }
  }
  return null;
}

// The content the model reads for one tool call, as message parts: the tool
// message's text, then the parts of any message the runtime added for it.
function answeredParts(answered) {
  const parts = [];
  for (const msg of answered || []) {
    const content = msg.content;
    if (typeof content === 'string') {
      if (msg.role === 'tool' || content) parts.push({ role: msg.role, type: 'text', text: content });
      continue;
    }
    for (const part of content || []) {
      if (part.type === 'text') parts.push({ role: msg.role, type: 'text', text: part.text });
      else if (part.type === 'image_url') {
        const url = String(part.image_url?.url ?? '');
        const match = /^data:([^;]+);base64,(.*)$/s.exec(url);
        parts.push(match ? { role: msg.role, type: 'image', mimeType: match[1], data: match[2] } : { role: msg.role, type: 'image', url });
      } else parts.push({ role: msg.role, ...part });
    }
  }
  return parts;
}

class Run {
  constructor(cfg) {
    this.cfg = cfg;
    this.home = mkdtempSync(join(tmpdir(), 'mcp-mirror-copilot-'));
    this.state = { requests: [], samplingRequests: [], sampling: cfg.sampling, pendingCall: null, callId: CALL_ID, answered: null };
    this.events = [];
    this.handlerSaw = [];
    this.permissions = [];
  }

  async open(extra = {}) {
    // Nothing credential-like from the caller reaches the runtime: no GitHub token, and
    // no session tokens of the tools that launched this worker.
    const passthrough = Object.fromEntries(Object.entries(process.env).filter(
      ([name]) => !/(TOKEN|SECRET|PASSWORD|API_KEY|_KEY$|CREDENTIAL|^GH_|^GITHUB_|^COPILOT_)/i.test(name)));
    const env = { ...passthrough, HOME: this.home, COPILOT_HOME: this.home, USERPROFILE: this.home };
    this.client = new CopilotClient({
      baseDirectory: this.home, useLoggedInUser: false, requestHandler: new StubModel(this.state), env, logLevel: 'error',
    });
    await this.client.start();
    const config = {
      model: MODEL,
      provider: { type: 'openai', baseUrl: 'http://127.0.0.1:9/v1', apiKey: 'unused' },
      // gpt-4.1 reads images; a BYOK model's capabilities default to none.
      modelCapabilities: { supports: { vision: true } },
      onPermissionRequest: this.cfg.retained ? this.recordPermission.bind(this) : approveAll,
      enableConfigDiscovery: false,
      skipCustomInstructions: true,
      diagnostics: { sources: { mcp: { level: 'trace' } } },
      mcpServers: { [SERVER]: serverConfig(this.cfg) },
      ...extra,
    };
    if (this.cfg.elicitation) {
      const elicited = this.cfg.elicitation.elicited;
      config.onElicitationRequest = async (context) => {
        this.handlerSaw.push({ hook: 'onElicitationRequest', mode: context?.mode ?? 'form', source: context?.elicitationSource ?? null });
        return context?.mode === 'url' ? { action: 'accept' } : { action: 'accept', content: { name: elicited } };
      };
    }
    if (this.cfg.auth?.clientMetadataUrl) config.authClientIdMetadataUrl = this.cfg.auth.clientMetadataUrl;
    this.session = await this.client.createSession(config);
    this.session.on((event) => this.events.push(event));
    if (this.cfg.sampling) this.session.on((event) => { if (event.type === 'sampling.requested') this.sample(event.data); });
    return this.session;
  }

  // A server's sampling request reaches the host as a sampling.requested event. The
  // host runs it on the session model with rpc.mcp.executeSampling and resolves the
  // pending request with rpc.ui.handlePendingSampling.
  async sample(data) {
    const seen = { hook: 'sampling.requested', requestId: data?.requestId ?? null, serverName: data?.serverName ?? null };
    this.handlerSaw.push(seen);
    try {
      // The event carries only identifiers; the CreateMessage params are the inbound
      // sampling/createMessage frame with the same JSON-RPC id.
      const frame = (await frames(this.session)).find((f) => f.frame?.method === 'sampling/createMessage' && f.frame?.id === data.mcpRequestId);
      seen.params_from_frame = Boolean(frame);
      const executed = await this.session.rpc.mcp.executeSampling({
        requestId: `${data.requestId}_exec`, serverName: data.serverName, mcpRequestId: data.mcpRequestId, request: frame?.frame?.params ?? {},
      });
      seen.action = executed?.action ?? null;
      seen.error = executed?.error ?? null;
      const resolved = await this.session.rpc.ui.handlePendingSampling({
        requestId: data.requestId, ...(executed?.action === 'success' ? { response: executed.result } : {}),
      });
      seen.resolved = resolved ?? null;
    } catch (error) {
      seen.error = String(error?.message ?? error);
    }
  }

  // The approval step, when render asks what the runtime keeps of each tool: record
  // the request and refuse it, so no tool that asks for approval runs.
  recordPermission(request) {
    this.permissions.push(request);
    return { kind: 'reject' };
  }

  // One agent turn. With `call`, the stub model asks for that tool first.
  async turn(call = null, prompt = 'Use the tools as needed.') {
    this.state.pendingCall = call;
    this.state.callId = `${CALL_ID}_${this.state.requests.length}`;
    this.state.answered = null;
    await this.session.sendAndWait({ prompt }, TURN_TIMEOUT_MS);
    this.state.pendingCall = null;
    return this.state.answered;
  }

  async close() {
    await Promise.race([
      (async () => {
        await this.session?.disconnect().catch(() => {});
        await this.client?.stop().catch(() => {});
      })(),
      sleep(5000),
    ]);
    rmSync(this.home, { recursive: true, force: true });
  }
}

function completedFor(events, callId) {
  return events.find((e) => e.type === 'tool.execution_complete' && e.data?.toolCallId === callId)?.data ?? null;
}

// Result boundaries. `agent` is the runtime's own tool loop: the stub model calls the
// tool, the runtime runs it and serializes the result into the next model request,
// which is captured. `direct` is session.rpc.tools.execute, the SDK's documented way
// to run an offered tool outside a model turn; it returns the runtime's ToolResult.
async function invoke(run, call, boundary) {
  if (boundary === 'direct') {
    const value = await run.session.rpc.tools.execute({ name: modelName(call.tool), arguments: call.arguments || {}, toolCallId: `${CALL_ID}_direct` });
    return { kind: 'tool_result', value };
  }
  const answered = await run.turn(call);
  const callId = run.state.callId;
  return {
    kind: 'provider_request',
    parts: answeredParts(answered),
    messages: answered,
    execution: completedFor(run.events, callId),
  };
}

// Sign in to an HTTP server through the runtime's own OAuth flow. The runtime does
// discovery, registration and PKCE, returns the authorization URL and listens for the
// callback itself; the host's part is the browser leg, done here as one request whose
// redirect is then delivered to the runtime's callback listener.
async function authenticate(run, auth) {
  const login = await run.session.rpc.mcp.oauth.login({
    serverName: SERVER,
    clientName: auth.clientName,
    ...(auth.clientId ? { clientId: auth.clientId, ...(auth.clientSecret ? { clientSecret: auth.clientSecret, publicClient: false } : { publicClient: true }) } : {}),
  });
  const out = { hook: 'CopilotSession.rpc.mcp.oauth.login', status: login.status ?? null, authorizationUrl: Boolean(login.authorizationUrl) };
  if (!login.authorizationUrl) return out;
  const response = await fetch(login.authorizationUrl, { redirect: 'manual' });
  const location = response.headers.get('location');
  out.redirected = Boolean(location);
  if (location) {
    const callback = await fetch(location).catch((err) => ({ status: String(err?.message || err) }));
    out.callback = callback.status;
  }
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    const servers = (await run.session.rpc.mcp.list().catch(() => ({}))).servers || [];
    out.serverStatus = servers.find((s) => s.name === SERVER)?.status ?? null;
    if (out.serverStatus === 'connected') break;
    await sleep(250);
  }
  return out;
}

async function probeSurface(session) {
  const out = {};
  out['resources.list'] = await attempt(async () => {
    const listed = (await session.rpc.mcp.resources.list({ serverName: SERVER })).resources || [];
    return { entries: listed.map((r) => ({ uri: r.uri, name: r.name ?? null, title: r.title ?? null, description: r.description ?? null, mimeType: r.mimeType ?? null })) };
  });
  const entries = out['resources.list'].ok ? out['resources.list'].entries : [];
  const textUri = entries.find((e) => (e.mimeType || '').startsWith('text/'))?.uri;
  const blobUri = entries.find((e) => e.mimeType && !e.mimeType.startsWith('text/'))?.uri;
  const read = async (uri) => {
    const first = ((await session.rpc.mcp.resources.read({ serverName: SERVER, uri })).contents || [])[0] || {};
    if (first.blob !== undefined) {
      const bytes = Buffer.from(first.blob, 'base64');
      return { uri: first.uri ?? null, mimeType: first.mimeType ?? null, blob_sha256: createHash('sha256').update(bytes).digest('hex'), blob_bytes: bytes.length };
    }
    return { uri: first.uri ?? null, mimeType: first.mimeType ?? null, text: first.text ?? null };
  };
  out['resources.read.text'] = textUri ? await attempt(() => read(textUri)) : { ok: false, error: 'no text resource listed' };
  out['resources.read.blob'] = blobUri ? await attempt(() => read(blobUri)) : { ok: false, error: 'no binary resource listed' };
  out['resources.templates'] = await attempt(async () => {
    const listed = (await session.rpc.mcp.resources.listTemplates({ serverName: SERVER })).resourceTemplates || [];
    return { entries: listed.map((t) => ({ uriTemplate: t.uriTemplate, name: t.name ?? null, description: t.description ?? null, mimeType: t.mimeType ?? null })) };
  });
  // session.rpc.mcp.prompts arrived in SDK 1.0.19 (marked experimental).
  out['prompts.list'] = await attempt(async () => {
    const listed = (await session.rpc.mcp.prompts.list({ serverName: SERVER })).prompts || [];
    return { entries: listed.map((p) => ({
      name: p.name, title: p.title ?? null, description: p.description ?? null,
      arguments: (p.arguments || []).map((a) => ({ name: a.name, description: a.description ?? null, required: a.required ?? null })),
    })) };
  });
  const prompts = out['prompts.list'].ok ? out['prompts.list'].entries : [];
  out['prompts.get'] = prompts.length ? await attempt(async () => {
    const first = prompts[0];
    const args = Object.fromEntries(first.arguments.filter((a) => a.required).map((a) => [a.name, 'x']));
    const result = await session.rpc.mcp.prompts.get({ serverName: SERVER, promptName: first.name, arguments: args });
    return {
      description: result.description ?? null,
      messages: (result.messages || []).map((m) => ({ role: m.role ?? null, type: m.content?.type ?? null, text: m.content?.text ?? null })),
    };
  }) : { ok: false, error: 'no prompt listed' };
  return out;
}

function toolText(result) {
  if (!result) return null;
  if (typeof result.content === 'string') return result.content;
  if (typeof result.textResultForLlm === 'string') return result.textResultForLlm;
  return JSON.stringify(result);
}

async function agentText(run, tool, args = {}) {
  const answered = await run.turn({ tool, arguments: args });
  const text = answeredParts(answered).filter((p) => p.type === 'text').map((p) => p.text).join('\n');
  return text || toolText(completedFor(run.events, run.state.callId)?.result);
}

async function probeUtility(run, feature, spec) {
  const session = run.session;
  const unsupported = (reason) => ({ ok: false, unsupported: reason });
  switch (feature) {
    case 'tools.list':
      return attempt(async () => {
        const listed = (await session.rpc.mcp.listTools({ serverName: SERVER })).tools || [];
        return { route: 'adapter', api: 'session.rpc.mcp.listTools', names: listed.map((t) => t.name) };
      });
    case 'progress':
      return attempt(async () => {
        const text = await agentText(run, 'slow_with_progress');
        const events = run.events
          .filter((e) => e.type === 'tool.execution_progress')
          .map((e) => ({ progress: e.data?.progress ?? null, total: e.data?.total ?? null, message: e.data?.progressMessage ?? e.data?.message ?? null }));
        return { route: 'adapter', api: 'session event tool.execution_progress', events, text };
      });
    case 'logging':
      return attempt(async () => {
        const text = await agentText(run, 'emit_logs');
        await sleep(300);
        // Server log notifications as the runtime surfaces them: diagnostics entries
        // of their own, not the raw protocol frames.
        const read = await session.rpc.diagnostics.read({ sources: ['mcp'], max: 500 });
        const messages = (read.entries || [])
          .filter((e) => e.details?.kind && e.details.kind !== 'protocol' && /log|notification/i.test(`${e.details.kind} ${e.message ?? ''}`))
          .map((e) => ({ level: e.level ?? null, data: e.details?.data ?? e.message ?? null }));
        return { route: 'adapter', api: 'session.rpc.diagnostics.read (non-protocol entries)', messages, text };
      });
    case 'set_level':
      return unsupported('the Copilot SDK has no API that sends logging/setLevel or names a log level for an MCP server');
    case 'completion':
      return unsupported('the Copilot SDK has no API that sends completion/complete to an MCP server');
    case 'cancellation':
      return attempt(async () => {
        const started = Date.now();
        setTimeout(() => { session.abort().catch(() => {}); }, spec.cancel_after_s * 1000);
        let outcome;
        try {
          const text = await agentText(run, 'cancellable_wait', { seconds: spec.cancel_wait_s });
          outcome = `returned: ${text}`;
        } catch (err) {
          outcome = `threw: ${message(err).slice(0, 200)}`;
        }
        const abandoned = (Date.now() - started) / 1000;
        await sleep((spec.cancel_wait_s - spec.cancel_after_s) * 1000 + 500);
        return { route: 'adapter', mechanism: 'CopilotSession.abort() during the tool call', abandoned_after_s: abandoned, outcome };
      });
    case 'tasks':
      return attempt(async () => ({ route: 'adapter', text: await agentText(run, 'task_capable') }));
    case 'caching':
      return attempt(async () => {
        const marks = [];
        const list = async (call) => {
          const started = stamp();
          const listed = (await session.rpc.mcp.listTools({ serverName: SERVER })).tools || [];
          marks.push({ call, started, finished: stamp(), count: listed.length });
        };
        await list('first');
        await list('within_ttl');
        await sleep(spec.cache_pause_s * 1000);
        await list('after_ttl');
        return { route: 'adapter', api: 'session.rpc.mcp.listTools', marks };
      });
    default:
      return unsupported(`unknown utility ${feature}`);
  }
}

// Catalogue changes against fixtures/notify_server.py, in the shape
// src/mcp_mirror/notify.py documents. The session emits mcp.tools.list_changed,
// mcp.resources.list_changed and mcp.prompts.list_changed events; each change is then
// checked by listing again through the session. The SDK has no subscribe API.
const WAIT_MS = 3000;

async function waitForEvent(run, type, since, ms) {
  const count = () => run.events.filter((e) => e.type === type).length;
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    if (count() > since) return true;
    await sleep(50);
  }
  return count() > since;
}

async function notifyProbe(run, negotiatedVersion) {
  const session = run.session;
  const lists = {
    tools: async () => ((await session.rpc.mcp.listTools({ serverName: SERVER })).tools || []).map((t) => t.name),
    resources: async () => ((await session.rpc.mcp.resources.list({ serverName: SERVER })).resources || []).map((r) => r.uri),
    prompts: async () => ((await session.rpc.mcp.prompts.list({ serverName: SERVER })).prompts || []).map((p) => p.name),
  };
  const changes = [
    ['tools.list_changed', 'add_tool', 'tools', 'SENTINEL_ADDED_TOOL'],
    ['resources.list_changed', 'add_resource', 'resources', 'fixtures/added-'],
    ['prompts.list_changed', 'add_prompt', 'prompts', 'SENTINEL_ADDED_PROMPT'],
  ];
  const out = { negotiated: negotiatedVersion };
  const seen = (type) => run.events.filter((e) => e.type === type).length;
  for (const [key, tool, kind, sentinel] of changes) {
    const type = `mcp.${kind}.list_changed`;
    out[key] = await attempt(async () => {
      const before = (await lists[kind]()).sort();
      const since = seen(type);
      const said = await agentText(run, tool);
      const notified = await waitForEvent(run, type, since, WAIT_MS);
      const after = (await lists[kind]()).sort();
      return { hook: `session event ${type}`, notified, before, after, sentinel, server_said: said };
    });
  }
  out['resources.subscribe'] = { ok: false, unsupported: 'the Copilot SDK has no API that subscribes to an MCP resource' };
  if (negotiatedVersion && negotiatedVersion >= '2026-07-28') {
    out['subscriptions.listen'] = { ok: false, unsupported: 'the Copilot SDK has no API that opens subscriptions/listen' };
  }
  return out;
}

async function main() {
  const raw = process.argv[2];
  if (!raw) {
    process.stderr.write('copilot_sdk worker: missing config argument\n');
    process.exit(2);
  }
  const cfg = JSON.parse(raw);
  const run = new Run(cfg);
  try {
    await run.open();
    const auth = cfg.auth ? await authenticate(run, cfg.auth) : undefined;
    // The first turn: the request the runtime sends a model, with the MCP tools in it.
    await run.turn(null, 'List what you can do.');
    const first = run.state.requests.find((r) => (r.body.tools || []).length) || run.state.requests[0];
    const tools = [];
    for (const tool of first?.body?.tools || []) {
      const fn = tool.function || tool;
      if (!String(fn.name || '').startsWith(PREFIX)) continue; // the runtime's built-in tools
      tools.push({ name: fn.name, description: fn.description ?? null, parameters: fn.parameters ?? {}, strict: fn.strict ?? null, type: tool.type ?? null });
    }
    // What the runtime keeps of each tool outside the request: its approval step. A
    // permission request names the tool's title and a readOnly flag; a tool the runtime
    // ran without asking (under a handler that refuses everything) it treated as
    // read-only.
    const retained = {};
    if (cfg.retained && tools.length) {
      run.state.callAll = true;
      await run.turn(null, 'Call every tool.');
      run.state.callAll = false;
      for (const event of run.events.filter((e) => e.type === 'tool.execution_start' && e.data?.mcpServerName === SERVER)) {
        const name = event.data.toolName;
        const asked = run.permissions.find((p) => p.toolCallId === event.data.toolCallId);
        retained[name] = {
          title: asked?.toolTitle ?? event.data.toolTitle ?? null,
          readOnly: asked ? asked.readOnly : true,
          approval: asked ? 'requested' : 'not requested',
        };
      }
    }
    const builtins = (first?.body?.tools || []).map((t) => (t.function || t).name).filter((n) => !String(n || '').startsWith(PREFIX));

    const results = [];
    for (const call of cfg.calls || []) {
      try {
        const returned = await invoke(run, { tool: sourceName(call.tool), arguments: call.arguments || {} }, cfg.boundary || 'agent');
        results.push({ tool: call.tool, ok: true, returned });
      } catch (err) {
        results.push({ tool: call.tool, ok: false, threw: true, error: message(err) });
      }
    }
    const probe = cfg.probe ? await probeSurface(run.session) : undefined;
    const utility = cfg.utility ? await probeUtility(run, cfg.utility, cfg.utilitySpec || {}) : undefined;

    const seen = await frames(run.session);
    const negotiatedMcpSpecVersion = negotiated(seen);
    if (!negotiatedMcpSpecVersion) {
      throw new Error('copilot_sdk worker: negotiated MCP protocol version was not observable in the session diagnostics');
    }
    const notify = cfg.notify ? await notifyProbe(run, negotiatedMcpSpecVersion) : undefined;
    const handshake = seen
      .filter(({ frame }) => ['server/discover', 'initialize'].includes(frame.method))
      .map(({ frame }) => ({ method: frame.method, protocolVersion: frame.params?.protocolVersion ?? frame.params?._meta?.['io.modelcontextprotocol/protocolVersion'] ?? null }));
    process.stdout.write(JSON.stringify({
      tools, builtins, retained, results, negotiatedMcpSpecVersion, handshake, probe, utility, notify, auth,
      handlerSaw: run.handlerSaw,
      sampling: run.state.samplingRequests.map((b) => ({ messages: b.messages, tools: (b.tools || []).map((t) => t.function?.name ?? t.name) })),
      requestUrl: first?.url ?? null,
    }));
  } finally {
    await run.close();
  }
}

main().then(
  () => process.exit(0),
  (err) => {
    process.stderr.write(String((err && err.stack) || err) + '\n');
    process.exit(1);
  },
);
