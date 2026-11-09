// canaiuse.com/mcp: the catalog as a stateless MCP server over Streamable HTTP.
//
// One POST carries one JSON-RPC message and gets one JSON response; there are no
// sessions and no SSE streams, so a GET is answered with 405 as the transport allows.
// Every answer is public static data, so there is no authentication.

import {Catalog} from "./catalog.ts";

type Json = Record<string, unknown>;

// Newest first. A client asking for a revision this server does not know is answered
// with the newest one it supports, as the lifecycle spec requires.
export const PROTOCOL_VERSIONS = ["2025-11-25", "2025-06-18"];
export const MAX_BODY_BYTES = 256 * 1024;

const SERVER_INFO = {name: "canaiuse", title: "Can AI use", version: "0.1.0"};
const INSTRUCTIONS =
  "Answers how agent frameworks handle each MCP feature, from canaiuse.com's measured " +
  "support tables. Every observation names the framework version it was measured at and " +
  "the data date; say both when you report a result. Codes: y present at the capture " +
  "boundary, a changed or retained elsewhere, n dropped, x cannot negotiate the revision, " +
  "u not measured.";

const REVISION = {
  type: "string",
  default: "default",
  description:
    "MCP revision: 'default' (what canaiuse.com opens on), 'latest', or an exact revision such as '2026-07-28'",
};
const FRAMEWORKS = {
  type: ["array", "null"],
  items: {type: "string"},
  default: null,
  description: "Framework ids to include, e.g. ['langchain', 'mastra']; omit for every framework",
};
const READ_ONLY = {readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false};

export const TOOLS = [
  {
    name: "list_snapshots",
    description:
      "List the MCP revisions canaiuse.com measures, with each framework's measured version, " +
      "that version's release date, and the date the data was last updated.",
    inputSchema: {type: "object", properties: {}},
    annotations: READ_ONLY,
  },
  {
    name: "search_capabilities",
    description:
      "Search the MCP features canaiuse.com measures (tool annotations, results, resources, " +
      "prompts, client features, transports, auth) and return their ids.",
    inputSchema: {
      type: "object",
      properties: {
        query: {type: "string", description: "Words to search for, e.g. 'destructive', 'outputSchema', 'sampling'"},
        revision: REVISION,
      },
      required: ["query"],
    },
    annotations: READ_ONLY,
  },
  {
    name: "lookup_capability",
    description:
      "How each framework handles one MCP feature: support code, the mechanism behind it, " +
      "the framework version measured and where the capture was taken.",
    inputSchema: {
      type: "object",
      properties: {
        capability_id: {type: "string", description: "A capability id from search_capabilities, e.g. 'destructive-hint'"},
        revision: REVISION,
        frameworks: FRAMEWORKS,
      },
      required: ["capability_id"],
    },
    annotations: READ_ONLY,
  },
  {
    name: "inspect_tool_definition",
    description:
      "Find which MCP features one of your tool definitions relies on, and how each framework handles each of them.",
    inputSchema: {
      type: "object",
      properties: {
        tool: {
          type: "object",
          additionalProperties: true,
          description: "An MCP tool definition as tools/list returns it (name, description, inputSchema, ...)",
        },
        revision: REVISION,
        frameworks: FRAMEWORKS,
      },
      required: ["tool"],
    },
    annotations: READ_ONLY,
  },
  {
    name: "inspect_tool_result",
    description:
      "Find which MCP features one of your tool results relies on, and how each framework handles each of them.",
    inputSchema: {
      type: "object",
      properties: {
        result: {
          type: "object",
          additionalProperties: true,
          description: "An MCP CallToolResult (content, structuredContent, isError, ...)",
        },
        revision: REVISION,
        frameworks: FRAMEWORKS,
      },
      required: ["result"],
    },
    annotations: READ_ONLY,
  },
];

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Accept, Authorization, MCP-Protocol-Version, Mcp-Session-Id",
  "Access-Control-Expose-Headers": "MCP-Protocol-Version",
  "Access-Control-Max-Age": "86400",
};

class RpcError extends Error {
  code: number;
  status: number;
  constructor(code: number, message: string, status = 200) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

const PARSE_ERROR = -32700;
const INVALID_REQUEST = -32600;
const METHOD_NOT_FOUND = -32601;
const INVALID_PARAMS = -32602;

const isObject = (value: unknown): value is Json => typeof value === "object" && value !== null && !Array.isArray(value);

const json = (body: unknown, status = 200, headers: Record<string, string> = {}): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: {"Content-Type": "application/json", "Cache-Control": "no-store", ...CORS_HEADERS, ...headers},
  });

const rpcError = (id: unknown, code: number, message: string, status = 200): Response =>
  json({jsonrpc: "2.0", id: id ?? null, error: {code, message}}, status);

// Argument checks mirror the stdio server's signature: wrong types are a protocol
// error (-32602); a valid call that the catalog rejects is a tool error (isError).
const argString = (args: Json, key: string, fallback: string | null): string | null => {
  const value = args[key];
  if (value === undefined || value === null) return fallback;
  if (typeof value !== "string") throw new RpcError(INVALID_PARAMS, `${key} must be a string`);
  return value;
};
const argObject = (args: Json, key: string): Json => {
  const value = args[key];
  if (!isObject(value)) throw new RpcError(INVALID_PARAMS, `${key} must be an object`);
  return value;
};
const argFrameworks = (args: Json): string[] | null => {
  const value = args.frameworks;
  if (value === undefined || value === null) return null;
  if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) {
    throw new RpcError(INVALID_PARAMS, "frameworks must be a list of framework ids");
  }
  return value as string[];
};

const callTool = async (catalog: Catalog, name: string, args: Json): Promise<Json> => {
  switch (name) {
    case "list_snapshots":
      return catalog.listSnapshots();
    case "search_capabilities":
      return catalog.searchCapabilities(argString(args, "query", null) ?? missing("query"), argString(args, "revision", "default"));
    case "lookup_capability":
      return catalog.lookupCapability(
        argString(args, "capability_id", null) ?? missing("capability_id"),
        argString(args, "revision", "default"),
        argFrameworks(args),
      );
    case "inspect_tool_definition":
      return catalog.inspectToolDefinition(argObject(args, "tool"), argString(args, "revision", "default"), argFrameworks(args));
    case "inspect_tool_result":
      return catalog.inspectToolResult(argObject(args, "result"), argString(args, "revision", "default"), argFrameworks(args));
    default:
      throw new RpcError(INVALID_PARAMS, `unknown tool: ${name}`);
  }
};

const missing = (key: string): never => {
  throw new RpcError(INVALID_PARAMS, `${key} is required`);
};

const handleRequest = async (catalog: Catalog, method: string, params: Json): Promise<unknown> => {
  switch (method) {
    case "initialize": {
      const requested = params.protocolVersion;
      const protocolVersion =
        typeof requested === "string" && PROTOCOL_VERSIONS.includes(requested) ? requested : PROTOCOL_VERSIONS[0];
      return {protocolVersion, capabilities: {tools: {}}, serverInfo: SERVER_INFO, instructions: INSTRUCTIONS};
    }
    case "ping":
      return {};
    case "tools/list":
      return {tools: TOOLS};
    case "tools/call": {
      const name = params.name;
      if (typeof name !== "string") throw new RpcError(INVALID_PARAMS, "name must be a string");
      const args = params.arguments ?? {};
      if (!isObject(args)) throw new RpcError(INVALID_PARAMS, "arguments must be an object");
      try {
        const data = await callTool(catalog, name, args);
        return {content: [{type: "text", text: JSON.stringify(data, null, 2)}], structuredContent: data};
      } catch (error) {
        if (error instanceof RpcError) throw error;
        const message = error instanceof Error ? error.message : String(error);
        return {content: [{type: "text", text: `Error executing tool ${name}: ${message}`}], isError: true};
      }
    }
    default:
      throw new RpcError(METHOD_NOT_FOUND, `Method not found: ${method}`);
  }
};

const readMessage = async (request: Request): Promise<unknown> => {
  const declared = Number(request.headers.get("Content-Length") ?? 0);
  if (declared > MAX_BODY_BYTES) throw new RpcError(INVALID_REQUEST, `request body exceeds ${MAX_BODY_BYTES} bytes`, 413);
  const body = await request.arrayBuffer();
  if (body.byteLength > MAX_BODY_BYTES) throw new RpcError(INVALID_REQUEST, `request body exceeds ${MAX_BODY_BYTES} bytes`, 413);
  try {
    return JSON.parse(new TextDecoder().decode(body));
  } catch {
    throw new RpcError(PARSE_ERROR, "Parse error: request body is not JSON", 400);
  }
};

export async function handleMcp(request: Request, catalog: Catalog): Promise<Response> {
  if (request.method === "OPTIONS") return new Response(null, {status: 204, headers: CORS_HEADERS});
  if (request.method !== "POST") {
    return json(
      {error: "canaiuse.com/mcp is a stateless MCP server: POST one JSON-RPC message with Content-Type: application/json."},
      405,
      {Allow: "POST, OPTIONS"},
    );
  }
  const declaredVersion = request.headers.get("MCP-Protocol-Version");
  if (declaredVersion !== null && !PROTOCOL_VERSIONS.includes(declaredVersion)) {
    return rpcError(null, INVALID_REQUEST, `Unsupported MCP-Protocol-Version: ${declaredVersion}`, 400);
  }

  let message: unknown;
  try {
    message = await readMessage(request);
  } catch (error) {
    const {code, message: text, status} = error as RpcError;
    return rpcError(null, code, text, status);
  }
  if (!isObject(message) || message.jsonrpc !== "2.0" || typeof message.method !== "string") {
    return rpcError(isObject(message) ? message.id : null, INVALID_REQUEST, "Invalid Request: expected one JSON-RPC 2.0 request", 400);
  }
  const params = isObject(message.params) ? message.params : {};
  if (!("id" in message) || message.id === null) {
    // A notification (notifications/initialized, notifications/cancelled, ...) needs no answer.
    return new Response(null, {status: 202, headers: CORS_HEADERS});
  }
  try {
    return json({jsonrpc: "2.0", id: message.id, result: await handleRequest(catalog, message.method, params)});
  } catch (error) {
    if (error instanceof RpcError) return rpcError(message.id, error.code, error.message, error.status);
    const text = error instanceof Error ? error.message : String(error);
    return rpcError(message.id, -32603, `Internal error: ${text}`, 500);
  }
}
