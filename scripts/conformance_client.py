"""Drive one framework's real MCP adapter as a client under the MCP conformance suite.

    python scripts/conformance_client.py <renderer> <url>

``npx @modelcontextprotocol/conformance client --command ...`` starts a scenario server,
appends its URL to the command, and grades what the server saw. This is that command:
it connects the adapter to the scenario server, lists the tools, and calls one. Nothing
here judges the result; the suite writes the checks, ``scripts/run_conformance.py``
collects them, and ``src/mcp_mirror/cells.py`` derives the cells.

The adapter is the one every other instrument drives. A Python renderer runs in its
pinned managed environment (``runner-manifests.json``) through the same
``render``/``render_result`` calls the definition scan and the result capture make.
Mastra runs through its Node worker.

Authorization scenarios (``MCP_CONFORMANCE_SCENARIO`` starting ``auth/``) need a client
that can complete an OAuth flow, and no adapter does that unaided: each takes an auth
object through its own documented hook. So for those scenarios the adapter gets the
OAuth implementation its stack ships, attached the way a user would attach it:

* Python adapters: the MCP Python SDK's ``OAuthClientProvider``, through LangChain's
  connection ``auth``, Pydantic AI's fastmcp transport ``auth``, the OpenAI Agents
  SDK's ``MCPServerStreamableHttp`` params ``auth``, and the CrewAI adapter's server
  params dict, which ``mcpadapt`` forwards to the SDK's ``streamablehttp_client``.
* Mastra: ``MCPOAuthClientProvider`` and ``MCPClient.authenticate``, which is Mastra's
  own documented flow. It never performs dynamic client registration by design.
* OpenAI Agents SDK (JS): an OAuthClientProvider through ``MCPServerStreamableHttp``'s
  documented ``authProvider`` option, with the code redeemed by the client package's
  exported ``auth()``.
* Pi: the package's own ``McpOAuthProvider`` through ``adaptOAuthProvider`` as
  ``StreamableHttpTransport``'s ``authProvider``, with the code redeemed by
  ``authorizeMcp``, as ``@earendil-works/pi-mcp/oauth`` documents.
* Google ADK (TypeScript): an OAuthClientProvider as ``StreamableHTTPConnectionParams``
  ``transportOptions.authProvider``, with the code redeemed by the SDK's exported
  ``auth()``.
* LangChain.js: an OAuthClientProvider as the MCPAdapter server's documented
  ``authProvider``, with the code redeemed by the client package's exported ``auth()``.
* Vercel AI SDK: an ``OAuthClientProvider`` as the HTTP transport's ``authProvider``,
  then ``auth()`` with the authorization code, as the ``@ai-sdk/mcp`` docs describe.
* Spring AI: none. The starter has no OAuth hook, so the adapter runs unauthenticated.
* GitHub Copilot SDK: ``CopilotSession.rpc.mcp.oauth.login``, the runtime's own OAuth
  flow, with ``SessionConfig.authClientIdMetadataUrl`` for a URL-based client_id.
* CrewAI (native MCP): none. ``MCPServerHTTP`` takes headers only, so the adapter runs
  unauthenticated.
* Microsoft Agent Framework (.NET): ``ClientOAuthOptions`` as
  ``HttpClientTransportOptions.OAuth``, the MCP C# SDK's own OAuth flow, with an
  ``AuthorizationCallbackHandler`` doing the browser leg.

The client identifies itself the way a well-configured host would: the pre-registered
credentials when the scenario hands them over in ``MCP_CONFORMANCE_CONTEXT``, and
otherwise the suite's Client ID Metadata Document URL, which the SDKs use only when
the authorization server advertises support for it. The browser leg of the
authorization-code flow is a plain HTTP request to the authorization URL; the
conformance authorization server approves without a login page.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
IN_RUNNER = "--in-runner"
# The URL the suite's auth/basic-cimd scenario expects as a URL-based client_id.
CLIENT_METADATA_URL = "https://conformance-test.local/client-metadata.json"
REDIRECT_URI = "http://localhost:3000/callback"
CLIENT_NAME = "mcp-mirror conformance client"


def _scenario() -> str:
    return os.environ.get("MCP_CONFORMANCE_SCENARIO", "")


def _context() -> dict:
    raw = os.environ.get("MCP_CONFORMANCE_CONTEXT") or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _arguments(schema: dict) -> dict:
    """Arguments that satisfy a tool's required parameters, so the call reaches the server."""

    samples = {"number": 1, "integer": 1, "boolean": True, "string": "mcp-mirror", "array": [], "object": {}}
    properties = schema.get("properties") or {}
    out: dict[str, Any] = {}
    for name in schema.get("required") or []:
        spec = properties.get(name) or {}
        kind = spec.get("type")
        if isinstance(kind, list):
            kind = next((k for k in kind if k != "null"), "string")
        if spec.get("enum"):
            out[name] = spec["enum"][0]
        elif "default" in spec:
            out[name] = spec["default"]
        else:
            out[name] = samples.get(kind or "string", "mcp-mirror")
    return out


# ---------------------------------------------------------------------------
# OAuth for the Python adapters: the MCP Python SDK's own provider
# ---------------------------------------------------------------------------


def _authorize(url: str) -> dict[str, str]:
    """Visit the authorization URL as the browser would and return the redirect's query."""

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return None

    opener = urllib.request.build_opener(NoRedirect)
    try:
        response = opener.open(url, timeout=30)
        location = response.headers.get("Location", "")
    except urllib.error.HTTPError as redirect:
        location = redirect.headers.get("Location", "")
    query = urllib.parse.parse_qs(urllib.parse.urlparse(location).query)
    return {key: values[0] for key, values in query.items()}


def _oauth_provider(server_url: str):
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata

    try:  # 2.x hands the callback's answer back as a model; 1.x as (code, state)
        from mcp.shared.auth import AuthorizationCodeResult
    except ImportError:
        AuthorizationCodeResult = None  # noqa: N806

    context = _context()
    metadata = OAuthClientMetadata(
        client_name=CLIENT_NAME,
        redirect_uris=[REDIRECT_URI],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )

    class Storage:
        def __init__(self) -> None:
            self.tokens = None
            self.client_info = None
            if context.get("client_id"):
                self.client_info = OAuthClientInformationFull(
                    client_id=context["client_id"],
                    client_secret=context.get("client_secret"),
                    redirect_uris=[REDIRECT_URI],
                    token_endpoint_auth_method="client_secret_basic"
                    if context.get("client_secret")
                    else "none",
                )

        async def get_tokens(self):
            return self.tokens

        async def set_tokens(self, tokens) -> None:
            self.tokens = tokens

        async def get_client_info(self):
            return self.client_info

        async def set_client_info(self, client_info) -> None:
            self.client_info = client_info

    redirected: dict[str, str] = {}

    async def redirect_handler(authorization_url: str) -> None:
        redirected.clear()
        redirected.update(_authorize(authorization_url))

    async def callback_handler():
        code, state = redirected.get("code", ""), redirected.get("state")
        if AuthorizationCodeResult is not None:
            return AuthorizationCodeResult(code=code, state=state, iss=redirected.get("iss"))
        return code, state

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=metadata,
        storage=Storage(),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
        client_metadata_url=None if context.get("client_id") else CLIENT_METADATA_URL,
    )


def _attach_auth(renderer_id: str, renderer: Any, auth: Any) -> str:
    """Hand ``auth`` to the adapter through its own documented hook; return the hook's name.

    The renderers build the adapter's connection from a ``ServerHandle`` that carries
    headers only, so the connection builder is wrapped here and everything downstream
    of it (session, listing, invocation) is still the renderer's own code.
    """

    if renderer_id == "langchain":
        from mcp_mirror.renderers import langchain_renderer

        build = langchain_renderer._connection

        def connection(handle):
            return {**build(handle), "auth": auth}

        langchain_renderer._connection = connection
        return "langchain_mcp_adapters StreamableHttpConnection['auth']"
    if renderer_id == "pydantic_ai":
        from fastmcp.client.transports import StreamableHttpTransport
        from pydantic_ai.mcp import MCPToolset

        renderer._build_server = lambda handle: MCPToolset(
            StreamableHttpTransport(url=handle.url, headers=dict(handle.headers) or None, auth=auth),
            max_retries=0,
        )
        return "fastmcp StreamableHttpTransport(auth=...) inside pydantic_ai MCPToolset"
    if renderer_id == "openai_agents":
        from agents.mcp import MCPServerStreamableHttp

        renderer._build_server = lambda handle: MCPServerStreamableHttp(
            params={"url": handle.url, "headers": dict(handle.headers), "auth": auth},
            client_session_timeout_seconds=20,
        )
        return "agents.mcp.MCPServerStreamableHttp params['auth']"
    if renderer_id == "crewai":
        build = renderer._build_params
        renderer._build_params = lambda handle: {**build(handle), "auth": auth}
        return "crewai_tools MCPServerAdapter server params['auth'] (forwarded by mcpadapt)"
    if renderer_id == "google_adk":
        from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams

        # ADK's own default factory, with the auth object the SDK's client takes.
        default = StreamableHTTPConnectionParams.model_fields["httpx_client_factory"].default

        def factory(headers=None, timeout=None, auth=None):  # noqa: ANN001
            return default(headers=headers, timeout=timeout, auth=auth or _auth)

        _auth = auth
        renderer._build_toolset = lambda handle: McpToolset(
            connection_params=StreamableHTTPConnectionParams(
                url=handle.url, headers=dict(handle.headers) or None, timeout=30, httpx_client_factory=factory
            )
        )
        return "google.adk StreamableHTTPConnectionParams(httpx_client_factory=...) with auth"
    if renderer_id == "ms_agent_framework":
        import httpx

        # MCPStreamableHTTPTool takes the httpx client it sends through; auth rides on it.
        build = renderer._build_tool
        renderer._build_tool = lambda handle, cls=None, **options: build(
            handle, cls, http_client=httpx.AsyncClient(auth=auth, follow_redirects=True), **options
        )
        return "agent_framework MCPStreamableHTTPTool(http_client=httpx.AsyncClient(auth=...))"
    if renderer_id == "autogen":
        # StreamableHttpServerParams declares url, headers, timeouts and
        # terminate_on_close, and create_mcp_server_session passes exactly those to the
        # SDK's streamablehttp_client, so AutoGen has no auth hook. The adapter runs
        # unauthenticated and the suite grades what it did.
        return None
    if renderer_id == "strands":
        build = renderer._build_client
        renderer._build_client = lambda handle, **options: build(handle, auth_provider=auth, **options)
        return "strands.tools.mcp.MCPClient(url=..., auth_provider=...)"
    if renderer_id == "semantic_kernel":
        from semantic_kernel.connectors.mcp import MCPStreamableHttpPlugin

        # Extra constructor keywords go to mcp's streamablehttp_client, which takes auth.
        build = renderer._build_plugin

        def plugin(handle, **options):
            if handle.transport != "http":
                return build(handle, **options)
            options.setdefault("request_timeout", 30)
            return MCPStreamableHttpPlugin(
                name="mcp", url=handle.url, headers=dict(handle.headers) or None, auth=auth, **options
            )

        renderer._build_plugin = plugin
        return "semantic_kernel MCPStreamableHttpPlugin(auth=...) forwarded to streamablehttp_client"
    if renderer_id == "llamaindex":
        build = renderer._client
        renderer._client = lambda handle, **kwargs: build(handle, auth=auth, **kwargs)
        return "llama_index.tools.mcp BasicMCPClient(auth=...)"
    if renderer_id == "agno":
        # MCPTools takes static headers or a header_provider and builds its fastmcp
        # transport with auth=None, so Agno has no OAuth hook. The adapter runs
        # unauthenticated and the suite grades what it did.
        return None
    if renderer_id == "haystack":
        # StreamableHttpServerInfo and SSEServerInfo take a static token or headers,
        # and the client builds its httpx client with no auth, so mcp-haystack has no
        # OAuth hook. The adapter runs unauthenticated and the suite grades what it did.
        return None
    if renderer_id == "smolagents":
        # MCPClient hands its server params dict to mcpadapt, which passes every key
        # but "transport" to the SDK's streamablehttp_client, the same route as CrewAI.
        build = renderer._params
        renderer._params = lambda handle: {**build(handle), "auth": auth}
        return "smolagents.MCPClient server params['auth'] (forwarded by mcpadapt)"
    if renderer_id == "dspy":
        # dspy.Tool.from_mcp_tool takes the caller's session, so auth rides on the
        # caller's mcp.Client transport: streamable_http_client(http_client=...(auth=...)).
        build = renderer._client
        renderer._client = lambda handle, **options: build(handle, auth=auth, **options)
        return "the caller's mcp.Client over streamable_http_client(http_client=create_mcp_http_client(auth=...)), handed to dspy.Tool.from_mcp_tool"
    if renderer_id == "spring_ai":
        # spring-ai-starter-mcp-client builds its HTTP transports with no OAuth client;
        # its only request hook, McpSyncHttpClientRequestCustomizer, can set a static
        # header but cannot run a flow. Spring AI points OAuth users at the separate
        # community module mcp-client-security, which is not part of the starter. The
        # adapter runs unauthenticated and the suite grades what it did.
        return None
    if renderer_id == "beeai":
        # MCPTool.from_client takes the transport the caller builds, so auth rides on
        # the SDK's streamablehttp_client(auth=...) handed to it.
        build = renderer._transport
        renderer._transport = lambda handle, **_: build(handle, auth=auth)
        return "beeai MCPTool.from_client(streamablehttp_client(url, auth=...))"
    if renderer_id == "copilot_sdk":
        # The Copilot runtime runs MCP OAuth itself: session.rpc.mcp.oauth.login does
        # discovery, registration and PKCE, returns the authorization URL and listens
        # for the callback. The worker does the browser leg; the Python provider is
        # unused. A suite-supplied client goes in as the login's static client, and the
        # CIMD URL as SessionConfig.authClientIdMetadataUrl.
        context = _context()
        renderer._auth = {
            "clientName": CLIENT_NAME,
            **({"clientId": context["client_id"]} if context.get("client_id") else {"clientMetadataUrl": CLIENT_METADATA_URL}),
            **({"clientSecret": context["client_secret"]} if context.get("client_secret") else {}),
        }
        return "CopilotSession.rpc.mcp.oauth.login (the runtime's own OAuth flow), SessionConfig.authClientIdMetadataUrl"
    if renderer_id == "crewai_native":
        # crewai.mcp.MCPServerHTTP takes url and headers only, and its HTTPTransport
        # calls streamablehttp_client(url, headers=...) with no auth, so CrewAI's native
        # MCP client has no OAuth hook. The adapter runs unauthenticated.
        return None
    if renderer_id == "google_adk_java":
        # StreamableHttpServerParameters and SseServerParameters take only static
        # headers, and McpTool.runAsync carries a TODO for auth; ADK for Java has no
        # OAuth hook for MCP. The adapter runs unauthenticated and the suite grades what
        # it did.
        return None
    if renderer_id == "langchain4j":
        # StreamableHttpMcpTransport takes static customHeaders or an McpHeadersSupplier
        # called per request, and langchain4j-mcp ships no OAuth client, so it has no
        # hook that can run a flow. The adapter runs unauthenticated and the suite
        # grades what it did.
        return None
    if renderer_id == "ms_agent_framework_dotnet":
        # The Agent Framework connects through the MCP C# SDK's HttpClientTransport,
        # whose OAuth option runs discovery, registration and PKCE itself. The worker's
        # AuthorizationCallbackHandler does the browser leg; the Python provider is
        # unused. A suite-supplied client goes in as ClientId/ClientSecret, otherwise the
        # CIMD URL as ClientMetadataDocumentUri.
        context = _context()
        renderer._auth = {
            "clientName": CLIENT_NAME,
            "redirectUri": REDIRECT_URI,
            **({"clientId": context["client_id"]} if context.get("client_id") else {"clientMetadataUrl": CLIENT_METADATA_URL}),
            **({"clientSecret": context["client_secret"]} if context.get("client_secret") else {}),
        }
        return "HttpClientTransportOptions.OAuth (ClientOAuthOptions, the MCP C# SDK's own OAuth flow)"
    if renderer_id == "google_adk_go":
        # mcptoolset.Config.Transport takes the caller's *mcp.StreamableClientTransport,
        # whose OAuthHandler field ADK's own Config doc names. The worker sets the
        # go-sdk's auth.AuthorizationCodeHandler there and does the browser leg; the
        # Python provider is unused. A suite-supplied client is preregistered, else the
        # handler tries the CIMD URL and falls back to dynamic registration.
        context = _context()
        renderer._auth = {
            "clientName": CLIENT_NAME,
            **({"clientId": context["client_id"]} if context.get("client_id") else {"clientMetadataUrl": CLIENT_METADATA_URL}),
            **({"clientSecret": context["client_secret"]} if context.get("client_secret") else {}),
        }
        return (
            "mcptoolset.Config.Transport = mcp.StreamableClientTransport{OAuthHandler: "
            "go-sdk auth.AuthorizationCodeHandler}"
        )
    if renderer_id == "koog":
        # McpToolRegistryProvider.streamableHttp takes a url and an optional ktor
        # HttpClient, and neither Koog's agents-mcp nor the Kotlin MCP SDK 0.11.1 client
        # has an OAuth flow (no discovery, registration or PKCE). The adapter runs
        # unauthenticated and the suite grades what it did.
        return None
    raise SystemExit(f"{renderer_id}: no documented auth hook is wired for this adapter")


def run_in_runner(renderer_id: str, url: str) -> int:
    """Inside the adapter's pinned environment: list the tools and call one."""

    from mcp_mirror.renderers import available_renderers
    from mcp_mirror.source import parse_server

    renderer = available_renderers().get(renderer_id)
    if renderer is None:
        print(f"{renderer_id}: renderer is not installed in this environment", file=sys.stderr)
        return 2
    hook = None
    if _scenario().startswith("auth/"):
        hook = _attach_auth(renderer_id, renderer, _oauth_provider(url))
    handle = parse_server(url)
    reps = renderer.render(handle)
    summary: dict[str, Any] = {
        "renderer": renderer_id,
        "versions": renderer.versions(),
        "auth_hook": hook,
        "tools": [rep.name for rep in reps],
    }
    if reps:
        tool = reps[0]
        arguments = _arguments(tool.params)
        result = renderer.render_result(handle, tool.name, arguments)
        summary["called"] = {"tool": tool.name, "arguments": arguments, "result": result.model_dump(mode="json")}
    print(json.dumps(summary, default=str)[:4000])
    return 0


# ---------------------------------------------------------------------------
# Mastra: the renderer's Node worker, or Mastra's own OAuth flow for auth scenarios
# ---------------------------------------------------------------------------

MASTRA_NODE = ROOT / "src" / "mcp_mirror" / "renderers" / "mastra_node"

# Mastra's documented OAuth flow: configure MCPOAuthClientProvider on the server, and
# when the server's auth state reads `needs-auth`, call MCPClient.authenticate, which
# listens on the loopback redirect URL and hands the authorization URL to
# `onRedirectToAuthorization`. Visiting that URL is the browser's job; here it is a
# fetch that follows the authorization server's redirect to the loopback listener.
MASTRA_AUTH_DRIVER = r"""
import { MCPClient, MCPOAuthClientProvider } from '@mastra/mcp';

const [url, port, clientName, metadataUrl] = process.argv.slice(1);
const context = JSON.parse(process.env.MCP_CONFORMANCE_CONTEXT || '{}');
const redirectUrl = `http://127.0.0.1:${port}/callback`;
const identity = context.client_id
  ? { clientInformation: { client_id: context.client_id, client_secret: context.client_secret } }
  : { clientMetadataUrl: metadataUrl };
const provider = new MCPOAuthClientProvider({
  redirectUrl,
  clientMetadata: {
    client_name: clientName,
    redirect_uris: [redirectUrl],
    grant_types: ['authorization_code', 'refresh_token'],
    response_types: ['code'],
    ...(context.client_secret ? { token_endpoint_auth_method: 'client_secret_basic' } : {}),
  },
  ...identity,
  onRedirectToAuthorization: (authorizationUrl) => {
    fetch(authorizationUrl).catch((err) => process.stderr.write(`authorization visit failed: ${err}\n`));
  },
});
const mcp = new MCPClient({
  id: `mcp-mirror-conformance-${Date.now()}`,
  servers: { src: { url: new URL(url), authProvider: provider } },
});

async function withAuth(fn) {
  try {
    const value = await fn();
    if (mcp.getServerAuthState('src') !== 'needs-auth') return value;
  } catch (err) {
    if (mcp.getServerAuthState('src') !== 'needs-auth') throw err;
  }
  await mcp.authenticate('src', { timeoutMs: 20000 });
  return fn();
}

try {
  const tools = await withAuth(() => mcp.listTools());
  const names = Object.keys(tools);
  const summary = { renderer: 'mastra', auth_hook: 'MCPOAuthClientProvider + MCPClient.authenticate', tools: names };
  if (names.length) {
    const schema = tools[names[0]].inputSchema;
    const json = (schema && typeof schema.getSchema === 'function' && schema.getSchema()) || {};
    const args = Object.fromEntries((json.required || []).map((k) => [k, (json.properties?.[k]?.type === 'number' || json.properties?.[k]?.type === 'integer') ? 1 : 'mcp-mirror']));
    const returned = await withAuth(() => tools[names[0]].execute(args));
    summary.called = { tool: names[0], arguments: args, returned };
  }
  process.stdout.write(JSON.stringify(summary).slice(0, 4000) + '\n');
} finally {
  await mcp.disconnect().catch(() => {});
}
"""


def run_mastra(url: str) -> int:
    if not (MASTRA_NODE / "node_modules" / "@mastra" / "mcp").exists():
        subprocess.run(["npm", "ci", "--prefix", str(MASTRA_NODE)], check=True, stdout=sys.stderr)
    if _scenario().startswith("auth/"):
        port = str(random.randint(40000, 59000))
        proc = subprocess.run(
            ["node", "--input-type=module", "-e", MASTRA_AUTH_DRIVER, url, port, CLIENT_NAME, CLIENT_METADATA_URL],
            cwd=MASTRA_NODE,
            timeout=110,
        )
        return proc.returncode
    sys.path.insert(0, str(ROOT / "src"))
    return run_in_runner("mastra", url)


# ---------------------------------------------------------------------------
# Vercel AI SDK: the renderer's Node worker, or the AI SDK's own OAuth flow for auth
# ---------------------------------------------------------------------------

VERCEL_AI_NODE = ROOT / "src" / "mcp_mirror" / "renderers" / "vercel_ai_node"

# The AI SDK's documented OAuth flow: pass an OAuthClientProvider as the HTTP
# transport's `authProvider`. When the server answers 401, the transport runs the
# discovery, registration and PKCE steps through the provider, hands the authorization
# URL to `redirectToAuthorization`, and throws UnauthorizedError. The app then finishes
# with `auth(provider, {serverUrl, authorizationCode})` and connects again. Visiting
# the authorization URL is the browser's job; here it is a fetch that follows the
# authorization server's redirect to a loopback listener that records the code.
VERCEL_AI_AUTH_DRIVER = r"""
import { createServer } from 'node:http';
import { createMCPClient, auth, UnauthorizedError } from '@ai-sdk/mcp';

const [url, port, clientName] = process.argv.slice(1);
const context = JSON.parse(process.env.MCP_CONFORMANCE_CONTEXT || '{}');
const redirectUrl = `http://127.0.0.1:${port}/callback`;

let callback;
let arrived = new Promise((resolve) => { callback = resolve; });
const listener = createServer((req, res) => {
  const query = new URL(req.url, redirectUrl).searchParams;
  res.end('ok');
  callback({ code: query.get('code'), state: query.get('state'), iss: query.get('iss') });
});
await new Promise((resolve) => listener.listen(Number(port), '127.0.0.1', resolve));

const store = {
  client: context.client_id ? { client_id: context.client_id, client_secret: context.client_secret } : undefined,
};
const provider = {
  get redirectUrl() { return redirectUrl; },
  get clientMetadata() {
    return {
      client_name: clientName,
      redirect_uris: [redirectUrl],
      grant_types: ['authorization_code', 'refresh_token'],
      response_types: ['code'],
      token_endpoint_auth_method: context.client_secret ? 'client_secret_basic' : 'none',
    };
  },
  clientInformation: () => store.client,
  saveClientInformation: (info) => { store.client = info; },
  tokens: () => store.tokens,
  saveTokens: (tokens) => { store.tokens = tokens; },
  saveCodeVerifier: (verifier) => { store.verifier = verifier; },
  codeVerifier: () => store.verifier,
  redirectToAuthorization: (authorizationUrl) => {
    fetch(authorizationUrl).catch((err) => process.stderr.write(`authorization visit failed: ${err}\n`));
  },
};

async function session() {
  const client = await createMCPClient({ transport: { type: 'http', url, authProvider: provider } });
  try {
    const tools = await client.tools();
    const names = Object.keys(tools);
    const summary = { renderer: 'vercel_ai', auth_hook: 'OAuthClientProvider as transport authProvider + auth()', tools: names };
    if (names.length) {
      const schema = (await tools[names[0]].inputSchema.jsonSchema) || {};
      const args = Object.fromEntries((schema.required || []).map((k) => [k, (schema.properties?.[k]?.type === 'number' || schema.properties?.[k]?.type === 'integer') ? 1 : 'mcp-mirror']));
      const returned = await tools[names[0]].execute(args, { toolCallId: 'conformance', messages: [] });
      summary.called = { tool: names[0], arguments: args, returned };
    }
    return summary;
  } finally {
    await client.close().catch(() => {});
  }
}

try {
  let summary;
  for (let attempt = 0; attempt < 3 && !summary; attempt++) {
    try {
      summary = await session();
    } catch (err) {
      if (!(err instanceof UnauthorizedError) && err?.name !== 'UnauthorizedError') throw err;
      const { code, state, iss } = await Promise.race([
        arrived,
        new Promise((_, reject) => setTimeout(() => reject(new Error('no authorization code arrived')), 20000)),
      ]);
      arrived = new Promise((resolve) => { callback = resolve; });
      await auth(provider, { serverUrl: url, authorizationCode: code, callbackState: state ?? undefined, callbackIssuer: iss ?? undefined });
    }
  }
  if (!summary) throw new Error('still unauthorized after three authorization attempts');
  process.stdout.write(JSON.stringify(summary).slice(0, 4000) + '\n');
} finally {
  listener.close();
}
"""


def run_vercel_ai(url: str) -> int:
    if not (VERCEL_AI_NODE / "node_modules" / "@ai-sdk" / "mcp").exists():
        subprocess.run(["npm", "ci", "--prefix", str(VERCEL_AI_NODE)], check=True, stdout=sys.stderr)
    if _scenario().startswith("auth/"):
        port = str(random.randint(40000, 59000))
        proc = subprocess.run(
            ["node", "--input-type=module", "-e", VERCEL_AI_AUTH_DRIVER, url, port, CLIENT_NAME],
            cwd=VERCEL_AI_NODE,
            timeout=110,
        )
        return proc.returncode
    sys.path.insert(0, str(ROOT / "src"))
    return run_in_runner("vercel_ai", url)

# ---------------------------------------------------------------------------
# OpenAI Agents SDK (JS): the renderer's Node worker, or the SDK's authProvider hook
# ---------------------------------------------------------------------------

OPENAI_AGENTS_JS_NODE = ROOT / "src" / "mcp_mirror" / "renderers" / "openai_agents_js_node"

# MCPServerStreamableHttp documents `authProvider`, an OAuthClientProvider from
# @modelcontextprotocol/client, as its auth hook. The SDK keeps the transport private,
# so the transport's finishAuth is not a route a user has. What a user can do is what
# this does: answer the redirect, redeem the code with the client package's exported
# `auth()` into the same provider, and connect again with the tokens it now holds.
OPENAI_AGENTS_JS_AUTH_DRIVER = r"""
import { MCPServerStreamableHttp } from '@openai/agents';
import { auth } from '@modelcontextprotocol/client';

const [url, redirectUrl, clientName, metadataUrl] = process.argv.slice(1);
const context = JSON.parse(process.env.MCP_CONFORMANCE_CONTEXT || '{}');
const store = { tokens: undefined, verifier: undefined, code: undefined, iss: undefined,
  client: context.client_id ? { client_id: context.client_id, client_secret: context.client_secret } : undefined };
const provider = {
  get redirectUrl() { return redirectUrl; },
  ...(context.client_id ? {} : { clientMetadataUrl: metadataUrl }),
  get clientMetadata() {
    return {
      client_name: clientName, redirect_uris: [redirectUrl],
      grant_types: ['authorization_code', 'refresh_token'], response_types: ['code'],
      token_endpoint_auth_method: context.client_secret ? 'client_secret_basic' : 'none',
    };
  },
  clientInformation: () => store.client,
  saveClientInformation: (info) => { store.client = info; },
  tokens: () => store.tokens,
  saveTokens: (tokens) => { store.tokens = tokens; },
  saveCodeVerifier: (verifier) => { store.verifier = verifier; },
  codeVerifier: () => store.verifier,
  // Kept so the code is redeemed against the authorization server it was issued by.
  saveDiscoveryState: (state) => { store.discovery = state; },
  discoveryState: () => store.discovery,
  // The browser leg: follow the authorization URL once and keep the redirect's query.
  redirectToAuthorization: async (authorizationUrl) => {
    const response = await fetch(authorizationUrl, { redirect: 'manual' });
    const location = new URL(response.headers.get('location') || '', redirectUrl);
    store.code = location.searchParams.get('code') || undefined;
    store.iss = location.searchParams.get('iss') || undefined;
  },
};

const open = async () => {
  const server = new MCPServerStreamableHttp({ url, authProvider: provider, clientSessionTimeoutSeconds: 20 });
  await server.connect();
  return server;
};
let server;
try {
  server = await open();
} catch (err) {
  if (!store.code) throw err;
  await auth(provider, { serverUrl: url, authorizationCode: store.code, iss: store.iss });
  server = await open();
}
try {
  const tools = await server.listTools();
  const summary = { renderer: 'openai_agents_js', auth_hook: 'MCPServerStreamableHttp({ authProvider }) + auth() from @modelcontextprotocol/client', tools: tools.map((t) => t.name) };
  if (tools.length) {
    const schema = tools[0].inputSchema || {};
    const args = Object.fromEntries((schema.required || []).map((k) => [k, (schema.properties?.[k]?.type === 'number' || schema.properties?.[k]?.type === 'integer') ? 1 : 'mcp-mirror']));
    summary.called = { tool: tools[0].name, arguments: args, returned: await server.callTool(tools[0].name, args) };
  }
  process.stdout.write(JSON.stringify(summary).slice(0, 4000) + '\n');
} finally {
  await server.close().catch(() => {});
}
"""


def run_openai_agents_js(url: str) -> int:
    if not (OPENAI_AGENTS_JS_NODE / "node_modules" / "@openai" / "agents").exists():
        subprocess.run(["npm", "ci", "--prefix", str(OPENAI_AGENTS_JS_NODE)], check=True, stdout=sys.stderr)
    if _scenario().startswith("auth/"):
        proc = subprocess.run(
            ["node", "--input-type=module", "-e", OPENAI_AGENTS_JS_AUTH_DRIVER, url, REDIRECT_URI, CLIENT_NAME, CLIENT_METADATA_URL],
            cwd=OPENAI_AGENTS_JS_NODE,
            timeout=110,
        )
        return proc.returncode
    sys.path.insert(0, str(ROOT / "src"))
    return run_in_runner("openai_agents_js", url)

# ---------------------------------------------------------------------------
# Pi: the renderer's Node worker, or pi-mcp's own OAuth provider
# ---------------------------------------------------------------------------

PI_NODE = ROOT / "src" / "mcp_mirror" / "renderers" / "pi_node"

# @earendil-works/pi-mcp/oauth ships McpOAuthProvider, the default stateful provider,
# and adaptOAuthProvider to hand it to StreamableHttpTransport. After a 401 the adapter
# runs authorizeMcp, which calls onRedirect and throws
# McpOAuthAuthorizationRequiredError. The documented next step is authorizeMcp with the
# code, then a fresh connection. McpOAuthProvider takes no client metadata document
# URL, so without pre-registered credentials it registers dynamically.
PI_AUTH_DRIVER = r"""
import { McpClient, StreamableHttpTransport } from '@earendil-works/pi-mcp';
import { McpOAuthProvider, McpOAuthAuthorizationRequiredError, adaptOAuthProvider, authorizeMcp } from '@earendil-works/pi-mcp/oauth';

const [url, redirectUrl, clientName] = process.argv.slice(1);
const context = JSON.parse(process.env.MCP_CONFORMANCE_CONTEXT || '{}');
const pending = { code: undefined };
const provider = new McpOAuthProvider({
  serverUrl: url,
  redirectUrl,
  clientMetadata: {
    client_name: clientName,
    grant_types: ['authorization_code', 'refresh_token'], response_types: ['code'],
    token_endpoint_auth_method: context.client_secret ? 'client_secret_basic' : 'none',
  },
  ...(context.client_id ? { clientId: context.client_id, clientSecret: context.client_secret } : {}),
  // The browser leg: follow the authorization URL once and keep the redirect's code.
  onRedirect: async (authorizationUrl) => {
    const response = await fetch(authorizationUrl, { redirect: 'manual' });
    const location = new URL(response.headers.get('location') || '', redirectUrl);
    pending.code = location.searchParams.get('code') || undefined;
  },
});

const session = async () => {
  const client = new McpClient({ name: 'mcp-mirror', version: '0.1.0' });
  await client.connect(new StreamableHttpTransport({ url, authProvider: adaptOAuthProvider(provider) }));
  try {
    const tools = await client.listTools();
    const summary = { renderer: 'pi', auth_hook: 'StreamableHttpTransport({ authProvider: adaptOAuthProvider(new McpOAuthProvider(...)) }) + authorizeMcp', tools: tools.map((t) => t.name) };
    if (tools.length) {
      const schema = tools[0].inputSchema || {};
      const args = Object.fromEntries((schema.required || []).map((k) => [k, (schema.properties?.[k]?.type === 'number' || schema.properties?.[k]?.type === 'integer') ? 1 : 'mcp-mirror']));
      summary.called = { tool: tools[0].name, arguments: args, returned: await client.callTool(tools[0].name, args) };
    }
    return summary;
  } finally {
    await client.close().catch(() => {});
  }
};

// A step-up scope challenge mid-session asks for authorization once more. Stop after
// three authorizations in all, the limit the scope-retry-limit scenario expects.
let summary;
for (let round = 0; ; round += 1) {
  try {
    summary = await session();
    break;
  } catch (err) {
    if (!(err instanceof McpOAuthAuthorizationRequiredError) || !pending.code || round >= 2) throw err;
    const code = pending.code;
    pending.code = undefined;
    await authorizeMcp(provider, { serverUrl: url, authorizationCode: code });
  }
}
process.stdout.write(JSON.stringify(summary).slice(0, 4000) + '\n');
"""


def run_pi(url: str) -> int:
    if not (PI_NODE / "node_modules" / "@earendil-works" / "pi-mcp").exists():
        subprocess.run(["npm", "ci", "--prefix", str(PI_NODE)], check=True, stdout=sys.stderr)
    if _scenario().startswith("auth/"):
        proc = subprocess.run(
            ["node", "--input-type=module", "-e", PI_AUTH_DRIVER, url, REDIRECT_URI, CLIENT_NAME],
            cwd=PI_NODE,
            timeout=110,
        )
        return proc.returncode
    sys.path.insert(0, str(ROOT / "src"))
    return run_in_runner("pi", url)



# ---------------------------------------------------------------------------
# Google ADK (TypeScript): the renderer's Node worker, or the transport's authProvider
# ---------------------------------------------------------------------------

GOOGLE_ADK_JS_NODE = ROOT / "src" / "mcp_mirror" / "renderers" / "google_adk_js_node"

# StreamableHTTPConnectionParams takes `transportOptions`, which MCPSessionManager hands
# to the SDK's StreamableHTTPClientTransport, so its `authProvider` (an
# OAuthClientProvider from @modelcontextprotocol/sdk) is the hook a user has. ADK builds
# a new transport per operation and keeps none, so finishAuth is not a route; the code
# is redeemed with the SDK's exported `auth()` into the same provider, and the toolset
# is used again with the tokens it now holds.
GOOGLE_ADK_JS_AUTH_DRIVER = r"""
import { createRequire } from 'node:module';
import { MCPToolset } from '@google/adk/tools/mcp';

const require = createRequire(import.meta.url);
const { auth } = require('@modelcontextprotocol/sdk/client/auth.js');
for (const method of ['log', 'info', 'debug', 'warn']) console[method] = console.error;

const [url, redirectUrl, clientName, metadataUrl] = process.argv.slice(1);
const context = JSON.parse(process.env.MCP_CONFORMANCE_CONTEXT || '{}');
const store = { tokens: undefined, verifier: undefined, code: undefined,
  client: context.client_id ? { client_id: context.client_id, client_secret: context.client_secret } : undefined };
const provider = {
  get redirectUrl() { return redirectUrl; },
  ...(context.client_id ? {} : { clientMetadataUrl: metadataUrl }),
  get clientMetadata() {
    return {
      client_name: clientName, redirect_uris: [redirectUrl],
      grant_types: ['authorization_code', 'refresh_token'], response_types: ['code'],
      token_endpoint_auth_method: context.client_secret ? 'client_secret_basic' : 'none',
    };
  },
  clientInformation: () => store.client,
  saveClientInformation: (info) => { store.client = info; },
  tokens: () => store.tokens,
  saveTokens: (tokens) => { store.tokens = tokens; },
  saveCodeVerifier: (verifier) => { store.verifier = verifier; },
  codeVerifier: () => store.verifier,
  saveDiscoveryState: (state) => { store.discovery = state; },
  discoveryState: () => store.discovery,
  // The browser leg: follow the authorization URL once and keep the redirect's code.
  redirectToAuthorization: async (authorizationUrl) => {
    const response = await fetch(authorizationUrl, { redirect: 'manual' });
    const location = new URL(response.headers.get('location') || '', redirectUrl);
    store.code = location.searchParams.get('code') || undefined;
  },
};

const toolset = new MCPToolset({ type: 'StreamableHTTPConnectionParams', url, transportOptions: { authProvider: provider } });
let tools;
try {
  tools = await toolset.getTools();
} catch (err) {
  if (!store.code) throw err;
  await auth(provider, { serverUrl: url, authorizationCode: store.code });
  tools = await toolset.getTools();
}
try {
  const summary = { renderer: 'google_adk_js', auth_hook: 'StreamableHTTPConnectionParams({ transportOptions: { authProvider } }) + auth() from @modelcontextprotocol/sdk', tools: tools.map((t) => t.name) };
  if (tools.length) {
    const schema = tools[0].mcpTool?.inputSchema || {};
    const args = Object.fromEntries((schema.required || []).map((k) => [k, (schema.properties?.[k]?.type === 'number' || schema.properties?.[k]?.type === 'integer') ? 1 : 'mcp-mirror']));
    summary.called = { tool: tools[0].name, arguments: args, returned: await tools[0].runAsync({ args, toolContext: {} }) };
  }
  process.stdout.write(JSON.stringify(summary).slice(0, 4000) + '\n');
} finally {
  await toolset.close().catch(() => {});
}
"""


def run_google_adk_js(url: str) -> int:
    if not (GOOGLE_ADK_JS_NODE / "node_modules" / "@google" / "adk").exists():
        subprocess.run(["npm", "ci", "--prefix", str(GOOGLE_ADK_JS_NODE)], check=True, stdout=sys.stderr)
    if _scenario().startswith("auth/"):
        proc = subprocess.run(
            ["node", "--input-type=module", "-e", GOOGLE_ADK_JS_AUTH_DRIVER, url, REDIRECT_URI, CLIENT_NAME, CLIENT_METADATA_URL],
            cwd=GOOGLE_ADK_JS_NODE,
            timeout=110,
        )
        return proc.returncode
    sys.path.insert(0, str(ROOT / "src"))
    return run_in_runner("google_adk_js", url)


# ---------------------------------------------------------------------------
# LangChain.js: the renderer's Node worker, or the connection's authProvider option
# ---------------------------------------------------------------------------

LANGCHAIN_JS_NODE = ROOT / "src" / "mcp_mirror" / "renderers" / "langchain_js_node"

# MCPAdapter documents `authProvider`, an OAuthClientProvider from
# @modelcontextprotocol/client, on an HTTP server, and hands it to the SDK transport
# it builds. The transport stays inside the client, so this does what a user can do:
# answer the redirect, redeem the code with the SDK's exported `auth()` into the same
# provider, and build the client again with the tokens it now holds.
LANGCHAIN_JS_AUTH_DRIVER = r"""
import { MCPAdapter } from '@langchain/mcp-adapters';
import { auth } from '@modelcontextprotocol/client';

const [url, redirectUrl, clientName, metadataUrl] = process.argv.slice(1);
const context = JSON.parse(process.env.MCP_CONFORMANCE_CONTEXT || '{}');
const store = { tokens: undefined, verifier: undefined, code: undefined, iss: undefined,
  client: context.client_id ? { client_id: context.client_id, client_secret: context.client_secret } : undefined };
const provider = {
  get redirectUrl() { return redirectUrl; },
  ...(context.client_id ? {} : { clientMetadataUrl: metadataUrl }),
  get clientMetadata() {
    return {
      client_name: clientName, redirect_uris: [redirectUrl],
      grant_types: ['authorization_code', 'refresh_token'], response_types: ['code'],
      token_endpoint_auth_method: context.client_secret ? 'client_secret_basic' : 'none',
    };
  },
  clientInformation: () => store.client,
  saveClientInformation: (info) => { store.client = info; },
  tokens: () => store.tokens,
  saveTokens: (tokens) => { store.tokens = tokens; },
  saveCodeVerifier: (verifier) => { store.verifier = verifier; },
  codeVerifier: () => store.verifier,
  // Kept so the code is redeemed against the authorization server it was issued by.
  saveDiscoveryState: (state) => { store.discovery = state; },
  discoveryState: () => store.discovery,
  // The browser leg: follow the authorization URL once and keep the redirect's query.
  redirectToAuthorization: async (authorizationUrl) => {
    const response = await fetch(authorizationUrl, { redirect: 'manual' });
    const location = new URL(response.headers.get('location') || '', redirectUrl);
    store.code = location.searchParams.get('code') || undefined;
    store.iss = location.searchParams.get('iss') || undefined;
  },
};

const open = async () => {
  const client = new MCPAdapter({ servers: { src: { transport: 'http', url, authProvider: provider } } });
  try {
    return { client, tools: await client.listTools('src') };
  } catch (err) {
    await client.close().catch(() => {});
    throw err;
  }
};
let opened;
try {
  opened = await open();
} catch (err) {
  if (!store.code) throw err;
  await auth(provider, { serverUrl: url, authorizationCode: store.code, iss: store.iss });
  opened = await open();
}
const { client, tools } = opened;
try {
  const summary = { renderer: 'langchain_js', auth_hook: 'MCPAdapter server authProvider + auth() from @modelcontextprotocol/client', tools: tools.map((t) => t.name) };
  if (tools.length) {
    // The adapter hands the server's JSON Schema to the tool as `schema`.
    const json = tools[0].schema && typeof tools[0].schema === 'object' ? tools[0].schema : {};
    const args = Object.fromEntries((json.required || []).map((k) => [k, (json.properties?.[k]?.type === 'number' || json.properties?.[k]?.type === 'integer') ? 1 : 'mcp-mirror']));
    let returned;
    try { returned = await tools[0].invoke(args); } catch (err) { returned = { threw: String(err?.message || err) }; }
    summary.called = { tool: tools[0].name, arguments: args, returned };
  }
  process.stdout.write(JSON.stringify(summary).slice(0, 4000) + '\n');
} finally {
  await client.close().catch(() => {});
}
"""


def run_langchain_js(url: str) -> int:
    if not (LANGCHAIN_JS_NODE / "node_modules" / "@langchain" / "mcp-adapters").exists():
        subprocess.run(["npm", "ci", "--prefix", str(LANGCHAIN_JS_NODE)], check=True, stdout=sys.stderr)
    if _scenario().startswith("auth/"):
        proc = subprocess.run(
            ["node", "--input-type=module", "-e", LANGCHAIN_JS_AUTH_DRIVER, url, REDIRECT_URI, CLIENT_NAME, CLIENT_METADATA_URL],
            cwd=LANGCHAIN_JS_NODE,
            timeout=110,
        )
        return proc.returncode
    sys.path.insert(0, str(ROOT / "src"))
    return run_in_runner("langchain_js", url)


COPILOT_SDK_NODE = ROOT / "src" / "mcp_mirror" / "renderers" / "copilot_sdk_node"


def run_copilot_sdk(url: str) -> int:
    if not (COPILOT_SDK_NODE / "node_modules" / "@github" / "copilot-sdk").exists():
        subprocess.run(["npm", "ci", "--prefix", str(COPILOT_SDK_NODE)], check=True, stdout=sys.stderr)
    sys.path.insert(0, str(ROOT / "src"))
    return run_in_runner("copilot_sdk", url)


# ---------------------------------------------------------------------------
# Genkit (JS): the renderer's Node worker, or the transport's authProvider
# ---------------------------------------------------------------------------

GENKIT_JS_NODE = ROOT / "src" / "mcp_mirror" / "renderers" / "genkit_js_node"

# A Genkit HTTP server config is `{ url, ...rest }`, and transportFrom spreads `rest`
# into the SDK's StreamableHTTPClientTransport options, so `authProvider` (an
# OAuthClientProvider from @modelcontextprotocol/sdk) is the hook a user has. A failed
# connect is caught and kept as the client's error, so the code is redeemed with the
# SDK's exported `auth()` into the same provider and GenkitMcpClient.restart()
# reconnects with the tokens it now holds.
GENKIT_JS_AUTH_DRIVER = r"""
import { createRequire } from 'node:module';
import { genkit } from 'genkit';
import { createMcpClient } from '@genkit-ai/mcp';

const require = createRequire(import.meta.url);
const { auth } = require('@modelcontextprotocol/sdk/client/auth.js');
for (const method of ['log', 'info', 'debug', 'warn']) console[method] = console.error;

const [url, redirectUrl, clientName, metadataUrl] = process.argv.slice(1);
const context = JSON.parse(process.env.MCP_CONFORMANCE_CONTEXT || '{}');
const store = { tokens: undefined, verifier: undefined, code: undefined,
  client: context.client_id ? { client_id: context.client_id, client_secret: context.client_secret } : undefined };
const provider = {
  get redirectUrl() { return redirectUrl; },
  ...(context.client_id ? {} : { clientMetadataUrl: metadataUrl }),
  get clientMetadata() {
    return {
      client_name: clientName, redirect_uris: [redirectUrl],
      grant_types: ['authorization_code', 'refresh_token'], response_types: ['code'],
      token_endpoint_auth_method: context.client_secret ? 'client_secret_basic' : 'none',
    };
  },
  clientInformation: () => store.client,
  saveClientInformation: (info) => { store.client = info; },
  tokens: () => store.tokens,
  saveTokens: (tokens) => { store.tokens = tokens; },
  saveCodeVerifier: (verifier) => { store.verifier = verifier; },
  codeVerifier: () => store.verifier,
  saveDiscoveryState: (state) => { store.discovery = state; },
  discoveryState: () => store.discovery,
  // The browser leg: follow the authorization URL once and keep the redirect's code.
  redirectToAuthorization: async (authorizationUrl) => {
    const response = await fetch(authorizationUrl, { redirect: 'manual' });
    const location = new URL(response.headers.get('location') || '', redirectUrl);
    store.code = location.searchParams.get('code') || undefined;
  },
};

const ai = genkit({});
const client = createMcpClient({ name: 'mcp-mirror', serverName: 'fixture', mcpServer: { url, authProvider: provider } });
// The 401 can come at connect, which GenkitMcpClient keeps as its error, or on a later
// request, which rejects. Either way the code the redirect left is redeemed and the
// client reconnects through restart(), the only reconnect it offers.
const redeem = async () => {
  const code = store.code;
  store.code = undefined;
  await auth(provider, { serverUrl: url, authorizationCode: code });
  await client.restart();
};
const withAuth = async (fn) => {
  try {
    return await fn();
  } catch (err) {
    if (!store.code) throw err;
    await redeem();
    return await fn();
  }
};
await client.ready();
if (client._server?.error) {
  if (!store.code) throw new Error(client._server.error);
  await redeem();
  if (client._server?.error) throw new Error(client._server.error);
}
try {
  const listed = async () => {
    const found = await client.getActiveTools(ai);
    if (!found.length && store.code) throw new Error('tools/list needed authorization');
    return found;
  };
  const tools = await withAuth(listed);
  const summary = { renderer: 'genkit_js', auth_hook: 'createMcpClient({ mcpServer: { url, authProvider } }) + auth() from @modelcontextprotocol/sdk + GenkitMcpClient.restart()', tools: tools.map((t) => t.__action.name) };
  if (tools.length) {
    const schema = tools[0].__action.inputJsonSchema || {};
    const args = Object.fromEntries((schema.required || []).map((k) => [k, (schema.properties?.[k]?.type === 'number' || schema.properties?.[k]?.type === 'integer') ? 1 : 'mcp-mirror']));
    const name = tools[0].__action.name;
    const returned = await withAuth(async () => {
      const current = (await client.getActiveTools(ai)).find((t) => t.__action.name === name) || tools[0];
      return current(args);
    });
    summary.called = { tool: name, arguments: args, returned };
  }
  process.stdout.write(JSON.stringify(summary).slice(0, 4000) + '\n');
} finally {
  await client.disable().catch(() => {});
}
"""


def run_genkit_js(url: str) -> int:
    if not (GENKIT_JS_NODE / "node_modules" / "@genkit-ai" / "mcp").exists():
        subprocess.run(["npm", "ci", "--prefix", str(GENKIT_JS_NODE)], check=True, stdout=sys.stderr)
    if _scenario().startswith("auth/"):
        proc = subprocess.run(
            ["node", "--input-type=module", "-e", GENKIT_JS_AUTH_DRIVER, url, REDIRECT_URI, CLIENT_NAME, CLIENT_METADATA_URL],
            cwd=GENKIT_JS_NODE,
            timeout=110,
        )
        return proc.returncode
    sys.path.insert(0, str(ROOT / "src"))
    return run_in_runner("genkit_js", url)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    if argv and argv[0] == IN_RUNNER:
        return run_in_runner(argv[1], argv[2])
    if len(argv) != 2:
        print(__doc__.splitlines()[2].strip(), file=sys.stderr)
        return 2
    sys.path.insert(0, str(ROOT / "src"))
    from mcp_mirror.cli import _managed_import_root
    from mcp_mirror.renderers import canonical_id
    from mcp_mirror.runner_manifests import load_runner_manifests

    renderer_id, url = canonical_id(argv[0]), argv[1]
    manifest = load_runner_manifests().get(renderer_id)
    if manifest is None:
        print(f"unknown renderer {renderer_id!r}", file=sys.stderr)
        return 2
    if renderer_id == "vercel_ai":
        return run_vercel_ai(url)
    if renderer_id == "openai_agents_js":
        return run_openai_agents_js(url)
    if renderer_id == "pi":
        return run_pi(url)
    if renderer_id == "google_adk_js":
        return run_google_adk_js(url)
    if renderer_id == "langchain_js":
        return run_langchain_js(url)
    if renderer_id == "genkit_js":
        return run_genkit_js(url)
    if renderer_id == "copilot_sdk":
        return run_copilot_sdk(url)
    if manifest.runtime == "node":
        return run_mastra(url)
    uv = shutil.which("uv")
    if not uv:
        print("managed Python runners require uv: https://docs.astral.sh/uv/", file=sys.stderr)
        return 2
    command = [uv, "run", "--isolated", "--no-project", "--python", manifest.python or "3.11"]
    packages = manifest.packages
    if manifest.runtime == "java":
        # The Java worker jar and its pinned JDK come from the managed cache; the
        # renderer that drives them runs on the manifest's host packages.
        from mcp_mirror.runner_manifests import install_java_runner

        command.extend(["--env-file", str(install_java_runner(manifest) / "runner.env")])
        packages = manifest.host_packages
    if manifest.runtime == "dotnet":
        # Likewise the published .NET worker and its pinned SDK.
        from mcp_mirror.runner_manifests import install_dotnet_runner

        command.extend(["--env-file", str(install_dotnet_runner(manifest) / "runner.env")])
        packages = manifest.host_packages
    if manifest.runtime == "go":
        # Likewise the Go worker binary, built by the managed cache's pinned Go.
        from mcp_mirror.runner_manifests import install_go_runner

        command.extend(["--env-file", str(install_go_runner(manifest) / "runner.env")])
        packages = manifest.host_packages
    for package in packages:
        command.extend(["--with", package])
    command.extend(["python", str(Path(__file__).resolve()), IN_RUNNER, renderer_id, url])
    environment = {**os.environ, "PYTHONPATH": _managed_import_root()}
    return subprocess.run(command, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
