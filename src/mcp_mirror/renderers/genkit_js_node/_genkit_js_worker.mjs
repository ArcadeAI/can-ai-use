// Node worker that drives Genkit's REAL MCP adapter (@genkit-ai/mcp createMcpHost and
// the GenkitMcpClient it builds) so mcp-mirror can capture what Genkit hands its
// model, never a reimplementation (decision D1).
//
// Every capture goes through a real generate call: createMcpHost connects the fixture,
// GenkitMcpHost.getActiveTools(ai) turns each MCP tool into a Genkit ToolAction, and
// ai.generate({ model, tools }) runs them against a scripted model registered with
// ai.defineModel. The tool definitions are the ToolDefinitions in the GenerateRequest
// the model action receives, and a tool result is the toolResponse part in the next
// GenerateRequest's messages. No model provider runs and no provider request is
// serialized.
//
// Usage: node _genkit_js_worker.mjs '<json-config>'
//   stdio: {"transport":"stdio","command":"python","args":["server.py"]}
//   http : {"transport":"http","url":"https://host/mcp","headers":{...}}

import { createHash } from 'node:crypto';
import { genkit } from 'genkit';
import { createMcpHost } from '@genkit-ai/mcp';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import * as SDK_TYPES from '@modelcontextprotocol/sdk/types.js';

// stdout carries this worker's JSON answer and nothing else. Genkit's logger and
// anything else that writes there go to stderr.
const writeAnswer = process.stdout.write.bind(process.stdout);
process.stdout.write = process.stderr.write.bind(process.stderr);
for (const method of ['log', 'info', 'debug', 'warn']) console[method] = console.error;

const HOST = 'mcp-mirror';
const SERVER = 'fixture';
const MODEL = 'mcp-mirror/scripted';
const CALL_REF = 'mcp-mirror-call';
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

// Genkit exposes no negotiated protocol version, and the SDK Client stores none it can
// be read back from on stdio. GenkitMcpClient._connect runs Client.connect, which sends
// `initialize` through Client.request; record the protocolVersion of each answer as
// evidence only. @genkit-ai/mcp imports the SDK's ESM build, the same module instance
// imported here. The worker fails closed if none or more than one is seen, rather than
// copying the direct source connection's version by assumption.
const negotiated = new Set();
const originalRequest = Client.prototype.request;
Client.prototype.request = async function request(req, ...rest) {
  const result = await originalRequest.call(this, req, ...rest);
  if (req && req.method === 'initialize' && result && typeof result.protocolVersion === 'string') {
    negotiated.add(result.protocolVersion);
  }
  return result;
};

// processResult hands back structuredContent alone when the server sent one, and
// otherwise the joined text, JSON-parsed when it looks like JSON, so the output alone
// cannot say which it was. Record, as evidence only, whether the CallToolResult behind
// the most recent tools/call carried structuredContent.
let lastCallHadStructured = null;
const originalCallTool = Client.prototype.callTool;
Client.prototype.callTool = async function callTool(...args) {
  const result = await originalCallTool.apply(this, args);
  lastCallHadStructured = !!result && result.structuredContent !== undefined;
  return result;
};

function negotiatedVersion() {
  if (negotiated.size !== 1) {
    throw new Error(`genkit_js worker: expected one negotiated MCP protocol version, observed ${JSON.stringify([...negotiated])}`);
  }
  return [...negotiated][0];
}

// McpServerConfig: { command, args } for stdio, { url, requestInit } for Streamable
// HTTP, or a custom { transport }. There is no SSE variant.
function serverConfig(cfg, extra = {}) {
  if (cfg.transport === 'http') {
    const config = { url: cfg.url, ...extra };
    if (cfg.headers && Object.keys(cfg.headers).length) config.requestInit = { headers: cfg.headers };
    return config;
  }
  if (cfg.transport === 'sse') {
    throw new Error('McpServerConfig has no SSE variant; @genkit-ai/mcp offers stdio and Streamable HTTP only');
  }
  return { command: cfg.command, args: cfg.args || [], ...extra };
}

// One Genkit instance with a scripted model. The first turn optionally asks for one
// tool call; every later turn ends the run. It records each GenerateRequest it is
// handed.
function scriptedGenkit() {
  const ai = genkit({});
  const script = { call: null, requests: [] };
  ai.defineModel(
    { name: MODEL, supports: { tools: true, multiturn: true, media: true, systemRole: true } },
    async (request) => {
      const responses = (request.messages || [])
        .flatMap((message) => message.content || [])
        .filter((part) => part.toolResponse !== undefined)
        .map((part) => part.toolResponse);
      script.requests.push(clone({ tools: request.tools || [], responses }));
      const part = script.call && script.requests.length === 1
        ? { toolRequest: { ref: CALL_REF, name: script.call.tool, input: script.call.arguments || {} } }
        : { text: 'done' };
      return { message: { role: 'model', content: [part] }, finishReason: 'stop' };
    },
  );
  return { ai, script };
}

function newHost(cfg, options = {}) {
  const { extra, ...hostOptions } = options;
  return createMcpHost({ name: HOST, mcpServers: { [SERVER]: serverConfig(cfg, extra) }, ...hostOptions });
}

// One generate run. Returns the requests the model saw.
async function generateTurn(ai, script, tools, call, options = {}) {
  script.call = call;
  script.requests = [];
  await ai.generate({ model: MODEL, prompt: 'mcp-mirror', tools, ...options });
  return script.requests;
}

function toolResponse(requests) {
  const second = requests[1];
  const responses = (second && second.responses) || [];
  return responses.find((r) => r && r.ref === CALL_REF) || responses[responses.length - 1];
}

function textOf(returned) {
  if (typeof returned === 'string') return returned;
  if (returned && typeof returned === 'object' && 'output' in returned && !('content' in returned && Array.isArray(returned.content) && returned.content.some((b) => typeof b?.text === 'string'))) {
    return textOf(returned.output);
  }
  const list = Array.isArray(returned) ? returned : (returned?.content || []);
  const block = Array.isArray(list) ? list.find((b) => typeof b?.text === 'string') : undefined;
  return block ? block.text : JSON.stringify(returned ?? null);
}

function toolByName(tools, name) {
  return tools.find((t) => t.__action.name === `${SERVER}/${name}` || t.__action.name === name);
}

// The SDK Client GenkitMcpClient exposes as its public `mcpClient` getter ("the
// underlying MCP SDK client"). Answers reached here are route "session".
async function sessionOf(host) {
  await host.ready();
  const client = host.getClient(SERVER);
  await client.ready();
  if (!client.mcpClient) throw new Error('GenkitMcpClient.mcpClient is undefined after ready()');
  return client.mcpClient;
}

// Resources through GenkitMcpHost.getActiveResources(ai) and prompts through
// getActivePrompts(ai), in the shape src/mcp_mirror/probe.py documents. A resource is a
// DynamicResourceAction named `<server>/<resource name>`; reading it runs the action
// with { uri }. A prompt is an ExecutablePrompt; render(input) builds its messages.
async function probeSurface(ai, host) {
  const out = {};
  let actions = [];
  const listed = await attempt(async () => {
    actions = await host.getActiveResources(ai);
    return {};
  });
  const metadataOf = (action) => action.__action.metadata || {};
  const unprefixed = (name) => (name || '').replace(new RegExp(`^${SERVER}/`), '');
  const fixed = actions.filter((a) => metadataOf(a).resource?.uri);
  const templated = actions.filter((a) => metadataOf(a).resource?.template);
  out['resources.list'] = listed.ok
    ? { ok: true, api: 'GenkitMcpHost.getActiveResources(ai)', entries: fixed.map((a) => ({
      uri: metadataOf(a).resource.uri, name: unprefixed(a.__action.name), title: null,
      description: a.__action.description ?? null, mimeType: null,
    })) }
    : listed;
  out['resources.templates'] = listed.ok
    ? { ok: true, api: 'GenkitMcpHost.getActiveResources(ai)', entries: templated.map((a) => ({
      uriTemplate: metadataOf(a).resource.template, name: unprefixed(a.__action.name),
      description: a.__action.description ?? null, mimeType: null,
    })) }
    : listed;
  const session = await sessionOf(host);
  // The fixture's MIME types decide which resource is text and which is binary; the
  // Genkit action does not carry one.
  const sourceMime = {};
  try {
    for (const r of (await session.listResources()).resources) sourceMime[r.uri] = r.mimeType || '';
  } catch { /* the read still runs on the first two resources */ }
  const textAction = fixed.find((a) => (sourceMime[metadataOf(a).resource.uri] || '').startsWith('text/'));
  const blobAction = fixed.find((a) => !(sourceMime[metadataOf(a).resource.uri] || 'text/').startsWith('text/'));
  const read = async (action) => {
    const uri = metadataOf(action).resource.uri;
    const result = await action({ uri });
    const first = (result.content || [])[0] || {};
    if (first.media) {
      const match = /^data:([^;]*);base64,(.*)$/s.exec(first.media.url || '');
      const bytes = Buffer.from(match ? match[2] : '', 'base64');
      return { uri: null, mimeType: first.media.contentType ?? null, blob_sha256: createHash('sha256').update(bytes).digest('hex'), blob_bytes: bytes.length, part: Object.keys(first) };
    }
    return { uri: null, mimeType: null, text: first.text ?? null, part: Object.keys(first) };
  };
  out['resources.read.text'] = textAction ? await attempt(() => read(textAction)) : { ok: false, error: 'no text resource listed' };
  out['resources.read.blob'] = blobAction ? await attempt(() => read(blobAction)) : { ok: false, error: 'no binary resource listed' };
  let prompts = [];
  out['prompts.list'] = await attempt(async () => {
    prompts = await host.getActivePrompts(ai);
    return { api: 'GenkitMcpHost.getActivePrompts(ai)', entries: prompts.map((p) => ({
      name: p.ref.name, title: null, description: p.ref.metadata?.description ?? null,
      arguments: (p.ref.metadata?.arguments || []).map((a) => ({ name: a.name, description: a.description ?? null, required: a.required ?? null })),
    })) };
  });
  out['prompts.get'] = prompts.length ? await attempt(async () => {
    const first = prompts[0];
    const args = Object.fromEntries((first.ref.metadata?.arguments || []).filter((a) => a.required).map((a) => [a.name, 'x']));
    const rendered = await first.render(args);
    return {
      api: 'ExecutablePrompt.render(input)',
      description: null,
      messages: (rendered.messages || []).map((m) => {
        const part = (m.content || [])[0] || {};
        return { role: m.role ?? null, type: part.text !== undefined ? 'text' : (Object.keys(part)[0] ?? null), text: part.text ?? null };
      }),
    };
  }) : { ok: false, error: 'no prompt listed' };
  return out;
}

// Server utilities, one question per worker run (each run is its own recorded fixture
// process), in the shape src/mcp_mirror/probe.py documents. @genkit-ai/mcp has no
// progress, logging or message hook, no log level and no completion API. The SDK
// Client behind GenkitMcpClient.mcpClient is the public session route, so those
// answers carry route "session".
async function probeUtility(cfg, feature, spec) {
  const host = newHost(cfg);
  const { ai, script } = scriptedGenkit();
  const callTool = (client, name, args = {}, options) =>
    client.callTool({ name, arguments: args }, undefined, options);
  try {
    switch (feature) {
      case 'tools.list':
        return await attempt(async () => {
          const tools = await host.getActiveTools(ai);
          const requests = await generateTurn(ai, script, tools, null);
          return { route: 'adapter', api: 'ai.generate({ tools: await host.getActiveTools(ai) }) ToolDefinitions in the GenerateRequest', names: requests[0].tools.map((d) => d.name) };
        });
      case 'progress':
        return await attempt(async () => {
          const client = await sessionOf(host);
          const events = [];
          const result = await callTool(client, 'slow_with_progress', {}, {
            onprogress: (p) => events.push({ progress: p.progress, total: p.total ?? null, message: p.message ?? null }),
          });
          return { route: 'session', api: 'GenkitMcpClient.mcpClient.callTool(params, undefined, { onprogress })', events, text: textOf(result) };
        });
      case 'logging':
        return await attempt(async () => {
          const client = await sessionOf(host);
          const messages = [];
          client.setNotificationHandler(SDK_TYPES.LoggingMessageNotificationSchema, (n) => {
            messages.push({ level: n.params.level, data: n.params.data });
          });
          const result = await callTool(client, 'emit_logs');
          await sleep(300);
          return { route: 'session', api: 'GenkitMcpClient.mcpClient.setNotificationHandler(LoggingMessageNotificationSchema)', messages, text: textOf(result) };
        });
      case 'set_level':
        return await attempt(async () => {
          const client = await sessionOf(host);
          await client.setLoggingLevel(spec.set_level);
          const result = await callTool(client, 'emit_logs');
          return { route: 'session', api: 'GenkitMcpClient.mcpClient.setLoggingLevel', level: spec.set_level, text: textOf(result) };
        });
      case 'completion':
        return await attempt(async () => {
          const client = await sessionOf(host);
          const [promptName, promptArg, promptValue] = spec.completion_prompt;
          const [uri, resourceArg, resourceValue] = spec.completion_resource;
          const prompt = await client.complete({ ref: { type: 'ref/prompt', name: promptName }, argument: { name: promptArg, value: promptValue } });
          const resource = await client.complete({ ref: { type: 'ref/resource', uri }, argument: { name: resourceArg, value: resourceValue } });
          return { route: 'session', api: 'GenkitMcpClient.mcpClient.complete', prompt: prompt.completion.values, resource: resource.completion.values };
        });
      case 'cancellation':
        // ai.generate takes an abortSignal, but the MCP tool action calls
        // Client.callTool with no signal, so aborting the generate cannot reach the
        // request already on the wire.
        return await attempt(async () => {
          const tools = await host.getActiveTools(ai);
          const controller = new AbortController();
          const started = Date.now();
          setTimeout(() => controller.abort('mcp-mirror cancel'), spec.cancel_after_s * 1000);
          let outcome;
          try {
            const requests = await generateTurn(ai, script, tools, { tool: 'cancellable_wait', arguments: { seconds: spec.cancel_wait_s } }, { abortSignal: controller.signal });
            const item = toolResponse(requests);
            outcome = item ? `returned: ${textOf(item.output)}` : 'cancelled: the run ended with no toolResponse';
          } catch (err) {
            outcome = `threw: ${errorText(err).slice(0, 200)}`;
          }
          const abandoned = (Date.now() - started) / 1000;
          await sleep(300);
          return { route: 'adapter', mechanism: 'ai.generate({ abortSignal }) aborting the MCP tool call', abandoned_after_s: abandoned, outcome };
        });
      case 'tasks':
        return await attempt(async () => {
          const tools = await host.getActiveTools(ai);
          const requests = await generateTurn(ai, script, tools, { tool: 'task_capable', arguments: {} });
          const item = toolResponse(requests);
          return { route: 'adapter', text: textOf(item ? item.output : null) };
        });
      case 'caching':
        return await attempt(async () => {
          const marks = [];
          for (const call of ['first', 'within_ttl', 'after_ttl']) {
            if (call === 'after_ttl') await sleep(spec.cache_pause_s * 1000);
            const started = stamp();
            const listed = await host.getActiveTools(ai);
            marks.push({ call, started, finished: stamp(), count: listed.length });
          }
          return { route: 'adapter', api: 'GenkitMcpHost.getActiveTools(ai) (no cache option; defineMcpHost caches only through a registry DAP)', marks };
        });
      default:
        return { ok: false, unsupported: `unknown utility ${feature}` };
    }
  } finally {
    await host.close().catch(() => {});
  }
}

// Catalogue changes against fixtures/notify_server.py, in the shape
// src/mcp_mirror/notify.py documents. GenkitMcpClient registers no notification
// handler, so each change is read only from the next listing on the same session.
// Subscriptions run on the SDK Client behind GenkitMcpClient.mcpClient, with the update
// fired on that same Client.
async function notifyProbe(ai, script, host) {
  const call = async (name) => {
    const requests = await generateTurn(ai, script, await host.getActiveTools(ai), { tool: name, arguments: {} });
    const item = toolResponse(requests);
    return textOf(item ? item.output : null);
  };
  const lists = {
    tools: async () => (await host.getActiveTools(ai)).map((t) => t.__action.name.replace(`${SERVER}/`, '')),
    resources: async () => (await host.getActiveResources(ai))
      .map((a) => a.__action.metadata?.resource?.uri).filter(Boolean),
    prompts: async () => (await host.getActivePrompts(ai)).map((p) => p.ref.name),
  };
  await lists.tools();
  const version = negotiatedVersion();
  const out = { negotiated: version };
  for (const [key, tool, kind, sentinel] of [
    ['tools.list_changed', 'add_tool', 'tools', 'SENTINEL_ADDED_TOOL'],
    ['resources.list_changed', 'add_resource', 'resources', 'fixtures/added-'],
    ['prompts.list_changed', 'add_prompt', 'prompts', 'SENTINEL_ADDED_PROMPT'],
  ]) {
    out[key] = await attempt(async () => {
      const before = (await lists[kind]()).sort();
      const said = await call(tool);
      await sleep(1000);
      const after = (await lists[kind]()).sort();
      return { hook: null, notified: false, before, after, sentinel, server_said: said };
    });
  }
  out['resources.subscribe'] = await attempt(async () => {
    const client = await sessionOf(host);
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
      api: 'GenkitMcpClient.mcpClient.subscribeResource / unsubscribeResource, with setNotificationHandler(ResourceUpdatedNotificationSchema) (@genkit-ai/mcp registers no notification handler)',
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
  });
  if (version >= '2026-07-28') {
    out['subscriptions.listen'] = { ok: false, unsupported: 'neither @genkit-ai/mcp nor the SDK v1 Client opens subscriptions/listen' };
  }
  return out;
}

async function main() {
  const raw = process.argv[2];
  if (!raw) {
    process.stderr.write('genkit_js worker: missing config argument\n');
    process.exit(2);
  }
  const cfg = JSON.parse(raw);
  if (cfg.utility) {
    const utility = await probeUtility(cfg, cfg.utility, cfg.utilitySpec || {});
    writeAnswer(JSON.stringify({ tools: [], results: [], negotiatedMcpSpecVersion: negotiatedVersion(), utility }));
    return;
  }
  // createMcpHost options a caller sets: multipart turns each tool into a tool.v2
  // action whose media reach the model as toolResponse content; roots is the list the
  // client answers roots/list with.
  const hostOptions = {};
  if (cfg.multipart) hostOptions.multipart = true;
  if (cfg.roots) hostOptions.roots = cfg.roots;
  const host = newHost(cfg, hostOptions);
  const { ai, script } = scriptedGenkit();
  try {
    const actions = await host.getActiveTools(ai);
    if (!actions.length && host.getClient(SERVER)?._server?.error) {
      throw new Error(`GenkitMcpClient could not connect: ${host.getClient(SERVER)._server.error}`);
    }
    const definitions = await generateTurn(ai, script, actions, null);
    const tools = definitions[0].tools.map((definition) => {
      const action = toolByName(actions, definition.name);
      return { definition, actionName: action ? action.__action.name : null, actionMetadata: action ? clone(action.__action.metadata) : null };
    });
    const results = [];
    for (const call of cfg.calls || []) {
      const entry = { tool: call.tool };
      lastCallHadStructured = null;
      if (cfg.rootsChange) {
        await host.getClient(SERVER).updateRoots(cfg.rootsChange);
        await sleep(300);
      }
      // One boundary per run, so the fixture sees each call once. "direct" runs the
      // ToolAction getActiveTools returned; "agent" runs it inside ai.generate.
      if (cfg.boundary === 'direct') {
        const action = toolByName(actions, call.tool);
        try {
          if (!action) throw new Error(`no ToolAction named ${SERVER}/${call.tool}`);
          Object.assign(entry, { ok: true, direct: clone(await action(call.arguments || {})) ?? null });
        } catch (err) {
          Object.assign(entry, { ok: false, threw: true, error: errorText(err) });
        }
      } else {
        try {
          const requests = await generateTurn(ai, script, actions, call);
          const item = toolResponse(requests);
          if (!item) Object.assign(entry, { ok: false, error: 'the next GenerateRequest carried no toolResponse' });
          else Object.assign(entry, { ok: true, agent: item });
        } catch (err) {
          Object.assign(entry, { ok: false, threw: true, error: errorText(err) });
        }
      }
      entry.structuredFromServer = lastCallHadStructured;
      results.push(entry);
    }
    const probe = cfg.probe ? await probeSurface(ai, host) : undefined;
    const notify = cfg.notify ? await notifyProbe(ai, script, host) : undefined;
    writeAnswer(JSON.stringify({ tools, results, negotiatedMcpSpecVersion: negotiatedVersion(), probe, notify }));
  } finally {
    await host.close().catch(() => {});
  }
}

main().catch((err) => {
  process.stderr.write(String((err && err.stack) || err) + '\n');
  process.exit(1);
});
