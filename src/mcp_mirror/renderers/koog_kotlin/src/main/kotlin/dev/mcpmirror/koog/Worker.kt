@file:OptIn(ai.koog.agents.core.annotation.InternalAgentsApi::class)

package dev.mcpmirror.koog

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.features.eventHandler.feature.EventHandler
import ai.koog.agents.mcp.DefaultMcpToolDescriptorParser
import ai.koog.agents.mcp.McpTool
import ai.koog.agents.mcp.McpToolRegistryProvider
import ai.koog.agents.mcp.defaultStdioTransport
import ai.koog.agents.mcp.metadata.McpServerInfo
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.clients.openai.OpenAIClientSettings
import ai.koog.prompt.executor.clients.openai.OpenAILLMClient
import ai.koog.prompt.executor.clients.openai.OpenAIModels
import ai.koog.prompt.executor.llms.MultiLLMPromptExecutor
import ai.koog.prompt.llm.LLMProvider
import ai.koog.serialization.kotlinx.toKotlinxJsonElement
import com.sun.net.httpserver.HttpServer
import io.ktor.client.HttpClient
import io.ktor.client.plugins.sse.SSE
import io.ktor.client.request.header
import io.modelcontextprotocol.kotlin.sdk.client.Client
import io.modelcontextprotocol.kotlin.sdk.client.ClientOptions
import io.modelcontextprotocol.kotlin.sdk.client.mcpStreamableHttpTransport
import io.modelcontextprotocol.kotlin.sdk.client.SseClientTransport
import io.modelcontextprotocol.kotlin.sdk.shared.Transport
import io.modelcontextprotocol.kotlin.sdk.types.ClientCapabilities
import io.modelcontextprotocol.kotlin.sdk.types.CompleteRequest
import io.modelcontextprotocol.kotlin.sdk.types.CompleteRequestParams
import io.modelcontextprotocol.kotlin.sdk.types.CreateMessageRequest
import io.modelcontextprotocol.kotlin.sdk.types.CreateMessageResult
import io.modelcontextprotocol.kotlin.sdk.types.ElicitRequestURLParams
import io.modelcontextprotocol.kotlin.sdk.types.ElicitResult
import io.modelcontextprotocol.kotlin.sdk.types.GetPromptRequest
import io.modelcontextprotocol.kotlin.sdk.types.GetPromptRequestParams
import io.modelcontextprotocol.kotlin.sdk.types.Implementation
import io.modelcontextprotocol.kotlin.sdk.types.ListResourceTemplatesRequest
import io.modelcontextprotocol.kotlin.sdk.types.LoggingLevel
import io.modelcontextprotocol.kotlin.sdk.types.LoggingMessageNotification
import io.modelcontextprotocol.kotlin.sdk.types.McpJson
import io.modelcontextprotocol.kotlin.sdk.types.Method
import io.modelcontextprotocol.kotlin.sdk.types.PromptReference
import io.modelcontextprotocol.kotlin.sdk.types.ReadResourceRequest
import io.modelcontextprotocol.kotlin.sdk.types.ReadResourceRequestParams
import io.modelcontextprotocol.kotlin.sdk.types.ResourceTemplateReference
import io.modelcontextprotocol.kotlin.sdk.types.ResourceUpdatedNotification
import io.modelcontextprotocol.kotlin.sdk.types.Role
import io.modelcontextprotocol.kotlin.sdk.types.StopReason
import io.modelcontextprotocol.kotlin.sdk.types.SubscribeRequest
import io.modelcontextprotocol.kotlin.sdk.types.SubscribeRequestParams
import io.modelcontextprotocol.kotlin.sdk.types.TextContent
import io.modelcontextprotocol.kotlin.sdk.types.UnsubscribeRequest
import io.modelcontextprotocol.kotlin.sdk.types.UnsubscribeRequestParams
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.encodeToJsonElement
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.longOrNull
import java.io.File
import java.net.InetSocketAddress
import java.net.URI
import java.security.MessageDigest
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.util.Base64
import java.util.Collections
import kotlin.system.exitProcess

/*
 * mcp-mirror's Koog worker.
 *
 * `java -jar worker.jar <config.json> <out.json>`. The config names one MCP server; the worker
 * opens it with the Kotlin MCP SDK transport Koog's own entrypoints use, has Koog's
 * McpToolRegistryProvider.fromClient build the ToolRegistry (McpToolRegistryProvider lists the
 * tools, DefaultMcpToolDescriptorParser turns each inputSchema into a ToolDescriptor and McpTool
 * wraps it), and runs a real Koog AIAgent with that registry. The agent's prompt executor is
 * Koog's own OpenAILLMClient pointed at a fake chat-completions endpoint on 127.0.0.1, so the
 * tool definitions are the `tools` array of the HTTP request Koog serialized for the provider,
 * and a tool result is the tool-role message in the next request. No model runs. Output goes to
 * a file so nothing the stdio server or a logger prints can corrupt it.
 */

const val CALL_ID = "mcp-mirror-call"
const val WATCHED_URI = "file:///fixtures/watched.txt"
const val SAMPLED = "SENTINEL_SAMPLED"
const val ELICITED = "SENTINEL_ELICITED"
const val ROOT_URI = "file:///tmp/mcp-mirror-root"
const val CHANGED_ROOT_URI = "file:///tmp/mcp-mirror-root-changed"

val STAMP: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSSSS'+00:00'").withZone(ZoneOffset.UTC)

// ---------------------------------------------------------------------------
// JSON helpers: config in, plain values out
// ---------------------------------------------------------------------------

fun plain(element: JsonElement?): Any? = when (element) {
    null, is JsonNull -> null
    is JsonObject -> element.mapValues { plain(it.value) }
    is JsonArray -> element.map { plain(it) }
    is JsonPrimitive -> when {
        element.isString -> element.content
        element.booleanOrNull != null -> element.booleanOrNull
        element.longOrNull != null -> element.longOrNull
        else -> element.doubleOrNull
    }
}

fun json(value: Any?): JsonElement = when (value) {
    null -> JsonNull
    is JsonElement -> value
    is Map<*, *> -> JsonObject(value.entries.associate { it.key.toString() to json(it.value) })
    is List<*> -> JsonArray(value.map { json(it) })
    is Boolean -> JsonPrimitive(value)
    is Number -> JsonPrimitive(value)
    else -> JsonPrimitive(value.toString())
}

/** An MCP SDK value as plain JSON, through the SDK's own serializer. */
inline fun <reified T> mcp(value: T): Any? = plain(McpJson.encodeToJsonElement(value))

fun describe(error: Throwable): String {
    val text = StringBuilder("${error::class.simpleName}: ${error.message}")
    var cause = error.cause
    var depth = 0
    while (cause != null && cause !== error && depth < 3) {
        text.append(" <- ${cause::class.simpleName}: ${cause.message}")
        cause = cause.cause
        depth++
    }
    return text.toString().take(1500)
}

fun stamp(): String = STAMP.format(Instant.now())

class Unsupported(message: String) : Exception(message)

/** {"ok": true, ...} or the failure, the way src/mcp_mirror/probe.py records it. */
suspend fun attempt(body: suspend () -> Map<String, Any?>): Map<String, Any?> = try {
    linkedMapOf<String, Any?>("ok" to true) + body()
} catch (error: Unsupported) {
    linkedMapOf("ok" to false, "unsupported" to error.message)
} catch (error: Throwable) {
    linkedMapOf("ok" to false, "error" to describe(error))
}

fun unsupported(reason: String): Map<String, Any?> = linkedMapOf("ok" to false, "unsupported" to reason)

// ---------------------------------------------------------------------------
// The fake chat-completions endpoint Koog's OpenAILLMClient talks to
// ---------------------------------------------------------------------------

class FakeOpenAI {
    private val server: HttpServer = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
    val requests: MutableList<Map<String, Any?>> = Collections.synchronizedList(mutableListOf())
    @Volatile var call: Map<String, Any?>? = null
    @Volatile private var turns = 0

    init {
        server.createContext("/") { exchange ->
            val body = exchange.requestBody.readBytes().decodeToString()
            @Suppress("UNCHECKED_CAST")
            val request = runCatching { plain(McpJson.parseToJsonElement(body)) as Map<String, Any?> }.getOrElse { mapOf("raw" to body) }
            requests.add(mapOf("path" to exchange.requestURI.path, "body" to request))
            val first = turns++ == 0
            val pending = call
            val message: Map<String, Any?> = if (first && pending != null) {
                mapOf(
                    "role" to "assistant",
                    "content" to null,
                    "tool_calls" to listOf(
                        mapOf(
                            "id" to CALL_ID,
                            "type" to "function",
                            "function" to mapOf(
                                "name" to pending["tool"],
                                "arguments" to json(pending["arguments"] ?: emptyMap<String, Any?>()).toString(),
                            ),
                        ),
                    ),
                )
            } else {
                mapOf("role" to "assistant", "content" to "done")
            }
            val reply = mapOf(
                "id" to "chatcmpl-mcp-mirror-$turns",
                "object" to "chat.completion",
                "created" to Instant.now().epochSecond,
                "model" to (request["model"] ?: "gpt-4o"),
                "choices" to listOf(
                    mapOf(
                        "index" to 0,
                        "message" to message,
                        "finish_reason" to if (message.containsKey("tool_calls")) "tool_calls" else "stop",
                    ),
                ),
                "usage" to mapOf("prompt_tokens" to 1, "completion_tokens" to 1, "total_tokens" to 2),
            )
            val bytes = json(reply).toString().encodeToByteArray()
            exchange.responseHeaders.add("Content-Type", "application/json")
            exchange.sendResponseHeaders(200, bytes.size.toLong())
            exchange.responseBody.use { it.write(bytes) }
        }
        server.start()
    }

    val baseUrl: String get() = "http://127.0.0.1:${server.address.port}"

    fun begin(next: Map<String, Any?>?) {
        call = next
        turns = 0
        requests.clear()
    }

    fun stop() = server.stop(0)
}

/** What one agent run showed the model, and what Koog's tool pipeline produced. */
class Turn(
    val requests: List<Map<String, Any?>>,
    val direct: JsonElement?,
    val failed: String?,
) {
    @Suppress("UNCHECKED_CAST")
    fun tools(): List<Map<String, Any?>> {
        val body = requests.firstOrNull()?.get("body") as? Map<String, Any?> ?: return emptyList()
        return (body["tools"] as? List<Map<String, Any?>>).orEmpty()
    }

    fun toolNames(): List<String> = tools().mapNotNull { (it["function"] as? Map<*, *>)?.get("name")?.toString() }

    /** The tool-role message the second provider request carries for our call. */
    @Suppress("UNCHECKED_CAST")
    fun toolMessage(): Map<String, Any?>? {
        val body = requests.getOrNull(1)?.get("body") as? Map<String, Any?> ?: return null
        val messages = (body["messages"] as? List<Map<String, Any?>>).orEmpty().filter { it["role"] == "tool" }
        return messages.firstOrNull { it["tool_call_id"] == CALL_ID } ?: messages.lastOrNull()
    }

    /** The first text the tool answer carries: a text block of the CallToolResult, else the tool message. */
    fun text(): String? {
        val content = (direct as? JsonObject)?.get("content")
        if (content is JsonArray) {
            for (block in content) {
                val text = (block as? JsonObject)?.get("text")
                if (text is JsonPrimitive) return text.contentOrNull
            }
        }
        return failed ?: toolMessage()?.get("content")?.toString()
    }
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

class Connection(val transport: Transport, val info: McpServerInfo, val close: () -> Unit)

/** Child processes get only an environment with nothing that looks like a credential in it. */
val SECRET_NAME = Regex("(?i)(token|secret|key|passw|credential|auth|cookie|session)")

@Suppress("UNCHECKED_CAST")
fun connection(cfg: Map<String, Any?>): Connection {
    val headers = (cfg["headers"] as? Map<String, Any?>).orEmpty().mapValues { it.value.toString() }
    return when (val transport = cfg["transport"]?.toString() ?: "stdio") {
        "stdio" -> {
            val argv = listOf(cfg["command"].toString()) + (cfg["args"] as? List<Any?>).orEmpty().map { it.toString() }
            val builder = ProcessBuilder(argv).redirectError(ProcessBuilder.Redirect.DISCARD)
            builder.environment().keys.removeIf { SECRET_NAME.containsMatchIn(it) }
            val process = builder.start()
            Connection(
                McpToolRegistryProvider.defaultStdioTransport(process),
                McpServerInfo(command = argv.joinToString(" ")),
            ) { process.destroy() }
        }
        "http" -> {
            // McpToolRegistryProvider.streamableHttp builds HttpClient { install(SSE) } and
            // mcpStreamableHttpTransport(url); this adds only the configured headers.
            val url = cfg["url"].toString()
            val client = HttpClient { install(SSE) }
            Connection(
                client.mcpStreamableHttpTransport(url) { headers.forEach { (k, v) -> header(k, v) } },
                McpServerInfo(url = url),
            ) { client.close() }
        }
        "sse" -> {
            val url = cfg["url"].toString()
            val client = HttpClient { install(SSE) }
            // McpToolRegistryProvider.defaultSseTransport takes no headers, so a server that
            // needs them gets the same SseClientTransport with a request builder.
            val sse = if (headers.isEmpty()) {
                McpToolRegistryProvider.defaultSseTransport(url, client)
            } else {
                SseClientTransport(client, url) { headers.forEach { (k, v) -> header(k, v) } }
            }
            Connection(sse, McpServerInfo(url = url)) { client.close() }
        }
        else -> throw IllegalArgumentException("unknown transport $transport")
    }
}

/** What the client-feature handlers were handed, in call order. */
object Hooks {
    val seen: MutableList<Map<String, Any?>> = Collections.synchronizedList(mutableListOf())
    fun note(hook: String, facts: Map<String, Any?> = emptyMap()) {
        seen.add(linkedMapOf<String, Any?>("hook" to hook) + facts)
    }
    fun drain(): List<Map<String, Any?>> = synchronized(seen) { seen.toList().also { seen.clear() } }
}

/**
 * The Kotlin SDK Client the registry is built from. McpToolRegistryProvider's own entrypoints
 * (fromTransport, fromProcess, fromSseUrl, streamableHttp) all build
 * Client(Implementation("mcp-client-cli", "1.0.0")) with default ClientOptions; this is that client
 * unless the client-features probe asks for hooks, which an application can only add by building
 * its own Client and handing it to fromClient.
 */
fun client(hooks: Boolean): Client {
    val info = Implementation(McpToolRegistryProvider.DEFAULT_MCP_CLIENT_NAME, McpToolRegistryProvider.DEFAULT_MCP_CLIENT_VERSION)
    if (!hooks) return Client(info)
    val capabilities = ClientCapabilities(
        sampling = JsonObject(emptyMap()),
        roots = ClientCapabilities.Roots(listChanged = true),
        elicitation = JsonObject(mapOf("form" to JsonObject(emptyMap()), "url" to JsonObject(emptyMap()))),
    )
    val client = Client(info, ClientOptions(capabilities = capabilities))
    client.setRequestHandler<CreateMessageRequest>(Method.Defined.SamplingCreateMessage) { request, _ ->
        Hooks.note("Client.setRequestHandler<CreateMessageRequest>", mapOf("messages" to request.params.messages.size))
        CreateMessageResult(role = Role.Assistant, content = TextContent(SAMPLED), model = "mcp-mirror-stub", stopReason = StopReason.EndTurn)
    }
    client.setElicitationHandler { request ->
        val url = request.params is ElicitRequestURLParams
        Hooks.note("Client.setElicitationHandler", mapOf("mode" to if (url) "url" else "form"))
        if (url) ElicitResult(ElicitResult.Action.Accept) else ElicitResult(ElicitResult.Action.Accept, JsonObject(mapOf("name" to JsonPrimitive(ELICITED))))
    }
    client.addRoot(ROOT_URI, "mcp-mirror")
    return client
}

class Run(val cfg: Map<String, Any?>, val out: MutableMap<String, Any?>) {
    val fake = FakeOpenAI()
    val executor = MultiLLMPromptExecutor(
        mapOf(LLMProvider.OpenAI to OpenAILLMClient("mcp-mirror-no-key", OpenAIClientSettings(baseUrl = fake.baseUrl))),
    )
    lateinit var client: Client
    lateinit var conn: Connection
    lateinit var registry: ToolRegistry
    @Volatile var negotiated: String? = null

    @Suppress("UNCHECKED_CAST")
    suspend fun run() {
        val mode = cfg["mode"]?.toString() ?: "tools"
        out["tools"] = emptyList<Any?>()
        conn = connection(cfg)
        client = client(hooks = mode == "client")
        // AbstractTransport chains onMessage callbacks, so this sees the initialize result
        // before the Client does. Client.connect reads it and keeps no protocolVersion.
        conn.transport.onMessage { message ->
            if (negotiated == null) {
                val result = (McpJson.encodeToJsonElement(message) as? JsonObject)?.get("result") as? JsonObject
                (result?.get("protocolVersion") as? JsonPrimitive)?.contentOrNull?.let { negotiated = it }
            }
        }
        client.connect(conn.transport)
        registry = McpToolRegistryProvider.fromClient(client, conn.info)
        out["negotiatedMcpSpecVersion"] = negotiated
        out["clientInfo"] = mapOf("name" to McpToolRegistryProvider.DEFAULT_MCP_CLIENT_NAME, "version" to McpToolRegistryProvider.DEFAULT_MCP_CLIENT_VERSION)
        val definitions = turn(null)
        out["providerPath"] = definitions.requests.firstOrNull()?.get("path")
        when (mode) {
            "tools" -> {
                out["tools"] = definitions.tools()
                out["retained"] = retained()
                out["dropped"] = dropped()
                out["results"] = (cfg["calls"] as? List<Map<String, Any?>>).orEmpty().map { result(it) }
            }
            "probe" -> out["probe"] = probe()
            "notify" -> out["notify"] = notifyProbe(definitions)
            "utility" -> out["utility"] = utility(cfg["utility"].toString(), (cfg["utilitySpec"] as? Map<String, Any?>).orEmpty())
            "client" -> out["client"] = clientFeatures((cfg["clientCalls"] as? List<Any?>).orEmpty().map { it.toString() })
            else -> throw IllegalArgumentException("unknown mode $mode")
        }
        if (mode != "tools") out["tools"] = definitions.tools()
    }

    fun close() {
        runCatching { runBlocking { client.close() } }
        runCatching { conn.close() }
        fake.stop()
    }

    /** What each McpTool keeps beside its descriptor: Koog's metadata map. */
    fun retained(): Map<String, Any?> = registry.tools.filterIsInstance<McpTool>().associate { tool ->
        tool.name to mapOf("metadata" to tool.metadata, "descriptor" to tool.descriptor.toString())
    }

    /**
     * Listed tools the registry left out. fromClient wraps McpToolDescriptorParser.parse in a
     * catch-all and only logs the Throwable, so the same parser is run again here to record why.
     */
    suspend fun dropped(): List<Map<String, Any?>> {
        val kept = registry.tools.map { it.name }.toSet()
        return client.listTools().tools.filter { it.name !in kept }.map { tool ->
            val reason = runCatching { DefaultMcpToolDescriptorParser.parse(tool) }.exceptionOrNull()
            linkedMapOf("name" to tool.name, "parser_error" to reason?.let { describe(it) })
        }
    }

    /** One AIAgent run (singleRunStrategy) against the fake provider. */
    suspend fun turn(call: Map<String, Any?>?): Turn {
        fake.begin(call)
        var direct: JsonElement? = null
        var failed: String? = null
        val agent = AIAgent(
            promptExecutor = executor,
            agentConfig = AIAgentConfig(
                prompt = prompt("mcp-mirror") { system("mcp-mirror probe") },
                model = OpenAIModels.Chat.GPT4o,
                maxAgentIterations = 10,
            ),
            toolRegistry = registry,
        ) {
            install(EventHandler) {
                onToolCallCompleted { event -> direct = event.toolResult?.toKotlinxJsonElement() }
                onToolCallFailed { event -> failed = event.message }
                onToolValidationFailed { event -> failed = event.message }
            }
        }
        agent.run("mcp-mirror")
        return Turn(fake.requests.toList(), direct, failed)
    }

    suspend fun turn(tool: String, arguments: Map<String, Any?> = emptyMap()): Turn =
        turn(mapOf("tool" to tool, "arguments" to arguments))

    suspend fun agentText(tool: String, arguments: Map<String, Any?> = emptyMap()): String? = turn(tool, arguments).text()

    @Suppress("UNCHECKED_CAST")
    suspend fun result(call: Map<String, Any?>): Map<String, Any?> {
        val tool = call["tool"].toString()
        val entry = linkedMapOf<String, Any?>("tool" to tool)
        try {
            val run = turn(tool, (call["arguments"] as? Map<String, Any?>).orEmpty())
            val message = run.toolMessage()
            if (message == null) {
                entry["ok"] = false
                entry["error"] = "the next provider request carried no tool message"
            } else {
                entry["ok"] = true
                entry["agent"] = message
                entry["direct"] = plain(run.direct)
                entry["failed"] = run.failed
            }
        } catch (error: Throwable) {
            entry["ok"] = false
            entry["threw"] = true
            entry["exception"] = error::class.qualifiedName
            entry["error"] = describe(error)
        }
        return entry
    }

    // -----------------------------------------------------------------
    // Resources and prompts. Koog wraps only tools; the rest is reached on the Kotlin SDK
    // Client the registry was built from.
    // -----------------------------------------------------------------

    @Suppress("UNCHECKED_CAST")
    suspend fun probe(): Map<String, Any?> {
        val probe = linkedMapOf<String, Any?>()
        val listed = mutableListOf<Map<String, Any?>>()
        probe["resources.list"] = attempt {
            listed.addAll((mcp(client.listResources()) as Map<String, Any?>)["resources"] as List<Map<String, Any?>>)
            mapOf("route" to "session", "entries" to listed.map { r ->
                linkedMapOf("uri" to r["uri"], "name" to r["name"], "title" to r["title"], "description" to r["description"], "mimeType" to r["mimeType"])
            })
        }
        val textUri = listed.firstOrNull { (it["mimeType"]?.toString() ?: "").startsWith("text/") }?.get("uri")?.toString()
        val blobUri = listed.firstOrNull { val m = it["mimeType"]?.toString() ?: ""; m.isNotEmpty() && !m.startsWith("text/") }?.get("uri")?.toString()
        probe["resources.read.text"] = if (textUri == null) mapOf("ok" to false, "error" to "no text resource listed") else attempt { read(textUri) }
        probe["resources.read.blob"] = if (blobUri == null) mapOf("ok" to false, "error" to "no binary resource listed") else attempt { read(blobUri) }
        probe["resources.templates"] = attempt {
            val templates = (mcp(client.listResourceTemplates(ListResourceTemplatesRequest())) as Map<String, Any?>)["resourceTemplates"] as List<Map<String, Any?>>
            mapOf("route" to "session", "entries" to templates.map { t ->
                linkedMapOf("uriTemplate" to t["uriTemplate"], "name" to t["name"], "description" to t["description"], "mimeType" to t["mimeType"])
            })
        }
        val prompts = mutableListOf<Map<String, Any?>>()
        probe["prompts.list"] = attempt {
            prompts.addAll((mcp(client.listPrompts()) as Map<String, Any?>)["prompts"] as List<Map<String, Any?>>)
            mapOf("route" to "session", "entries" to prompts.map { p ->
                linkedMapOf(
                    "name" to p["name"], "title" to p["title"], "description" to p["description"],
                    "arguments" to (p["arguments"] as? List<Map<String, Any?>>).orEmpty().map { a ->
                        linkedMapOf("name" to a["name"], "description" to a["description"], "required" to a["required"])
                    },
                )
            })
        }
        probe["prompts.get"] = if (prompts.isEmpty()) mapOf("ok" to false, "error" to "no prompt listed") else attempt {
            val first = prompts[0]
            val arguments = (first["arguments"] as? List<Map<String, Any?>>).orEmpty()
                .filter { it["required"] == true }.associate { it["name"].toString() to "x" }
            val result = mcp(client.getPrompt(GetPromptRequest(GetPromptRequestParams(first["name"].toString(), arguments)))) as Map<String, Any?>
            linkedMapOf(
                "route" to "session",
                "description" to result["description"],
                "messages" to (result["messages"] as List<Map<String, Any?>>).map { m ->
                    val content = m["content"] as? Map<String, Any?>
                    linkedMapOf("role" to m["role"], "type" to content?.get("type"), "text" to content?.get("text"))
                },
            )
        }
        return probe
    }

    @Suppress("UNCHECKED_CAST")
    suspend fun read(uri: String): Map<String, Any?> {
        val result = mcp(client.readResource(ReadResourceRequest(ReadResourceRequestParams(uri)))) as Map<String, Any?>
        val first = (result["contents"] as List<Map<String, Any?>>)[0]
        val answer = linkedMapOf<String, Any?>("route" to "session", "uri" to first["uri"], "mimeType" to first["mimeType"])
        val blob = first["blob"]
        if (blob is String) {
            val bytes = Base64.getDecoder().decode(blob)
            answer["blob_sha256"] = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
            answer["blob_bytes"] = bytes.size
        } else {
            answer["text"] = first["text"]
        }
        return answer
    }

    // -----------------------------------------------------------------
    // Catalogue changes against fixtures/notify_server.py. Koog registers no notification
    // handler on the client, and the ToolRegistry fromClient returns is a fixed list.
    // -----------------------------------------------------------------

    suspend fun callRaw(tool: String): String? {
        val result = client.callTool(tool, emptyMap())
        return result.content.filterIsInstance<TextContent>().firstOrNull()?.text
    }

    @Suppress("UNCHECKED_CAST")
    suspend fun listing(kind: String): List<String> = when (kind) {
        "tools" -> turn(null).toolNames()
        "resources" -> ((mcp(client.listResources()) as Map<String, Any?>)["resources"] as List<Map<String, Any?>>).map { it["uri"].toString() }
        else -> ((mcp(client.listPrompts()) as Map<String, Any?>)["prompts"] as List<Map<String, Any?>>).map { it["name"].toString() }
    }.sorted()

    suspend fun notifyProbe(definitions: Turn): Map<String, Any?> {
        val notify = linkedMapOf<String, Any?>("negotiated" to negotiated)
        val changes = listOf(
            listOf("tools.list_changed", "add_tool", "tools", "SENTINEL_ADDED_TOOL"),
            listOf("resources.list_changed", "add_resource", "resources", "fixtures/added-"),
            listOf("prompts.list_changed", "add_prompt", "prompts", "SENTINEL_ADDED_PROMPT"),
        )
        for ((key, tool, kind, sentinel) in changes) {
            notify[key] = attempt {
                val before = listing(kind)
                val said = if (kind == "tools") agentText(tool) else callRaw(tool)
                delay(1000)
                val after = listing(kind)
                linkedMapOf(
                    "hook" to null, "notified" to false, "before" to before, "after" to after,
                    "sentinel" to sentinel, "server_said" to said,
                )
            }
        }
        notify["resources.subscribe"] = attempt {
            val updates = Collections.synchronizedList(mutableListOf<String>())
            client.setNotificationHandler<ResourceUpdatedNotification>(Method.Defined.NotificationsResourcesUpdated) { notification ->
                updates.add(notification.params.uri)
                CompletableDeferred(Unit)
            }
            val answer = linkedMapOf<String, Any?>(
                "route" to "underlying",
                "api" to "Client.subscribeResource / unsubscribeResource and Client.setNotificationHandler" +
                    "(notifications/resources/updated) on the Kotlin SDK Client handed to McpToolRegistryProvider.fromClient; " +
                    "Koog itself registers no handler",
                "subscribed" to false, "subscribe_error" to null, "notified" to false, "unsubscribed" to false,
                "unsubscribe_error" to null, "notified_after_unsubscribe" to null,
            )
            val said = mutableListOf<String?>()
            answer["server_said"] = said
            try {
                client.subscribeResource(SubscribeRequest(SubscribeRequestParams(WATCHED_URI)))
                answer["subscribed"] = true
            } catch (error: Throwable) {
                answer["subscribe_error"] = describe(error)
            }
            said.add(callRaw("update_resource"))
            delay(1000)
            answer["notified"] = updates.isNotEmpty()
            if (answer["subscribed"] != true) return@attempt answer
            val heard = updates.size
            try {
                client.unsubscribeResource(UnsubscribeRequest(UnsubscribeRequestParams(WATCHED_URI)))
                answer["unsubscribed"] = true
            } catch (error: Throwable) {
                answer["unsubscribe_error"] = describe(error)
                return@attempt answer
            }
            said.add(callRaw("update_resource"))
            delay(1000)
            answer["notified_after_unsubscribe"] = updates.size > heard
            answer
        }
        if ((negotiated ?: "") >= "2026-07-28") {
            notify["subscriptions.listen"] = unsupported("neither Koog nor the Kotlin MCP SDK 0.11.1 client opens subscriptions/listen")
        }
        return notify
    }

    // -----------------------------------------------------------------
    // Server utilities, one question per worker run
    // -----------------------------------------------------------------

    @Suppress("UNCHECKED_CAST")
    suspend fun utility(feature: String, spec: Map<String, Any?>): Map<String, Any?> = when (feature) {
        "tools.list" -> attempt {
            mapOf(
                "route" to "adapter",
                "api" to "McpToolRegistryProvider.fromClient ToolRegistry, as the tools array of the provider request",
                "names" to turn(null).toolNames(),
            )
        }
        "progress" -> unsupported(
            "McpTool.execute calls Client.callTool(name, arguments) with no RequestOptions.onProgress, so no " +
                "progressToken is sent, and McpToolRegistryProvider takes no progress callback",
        )
        "logging" -> attempt {
            val messages = Collections.synchronizedList(mutableListOf<Map<String, Any?>>())
            client.setNotificationHandler<LoggingMessageNotification>(Method.Defined.NotificationsMessage) { notification ->
                messages.add(mapOf("level" to mcp(notification.params.level), "data" to plain(notification.params.data)))
                CompletableDeferred(Unit)
            }
            val text = agentText("emit_logs")
            delay(300)
            linkedMapOf(
                "route" to "underlying",
                "api" to "Client.setNotificationHandler(notifications/message) on the Kotlin SDK Client handed to " +
                    "McpToolRegistryProvider.fromClient; Koog registers no logging handler",
                "messages" to messages.toList(),
                "text" to text,
            )
        }
        "set_level" -> attempt {
            val level = spec["set_level"]?.toString() ?: "warning"
            client.setLoggingLevel(LoggingLevel.entries.first { mcp(it) == level })
            linkedMapOf("route" to "session", "api" to "Client.setLoggingLevel", "level" to level, "text" to agentText("emit_logs"))
        }
        "completion" -> attempt {
            val p = spec["completion_prompt"] as List<Any?>
            val r = spec["completion_resource"] as List<Any?>
            val prompt = client.complete(
                CompleteRequest(CompleteRequestParams(CompleteRequestParams.Argument(p[1].toString(), p[2].toString()), PromptReference(p[0].toString()))),
            )
            val resource = client.complete(
                CompleteRequest(CompleteRequestParams(CompleteRequestParams.Argument(r[1].toString(), r[2].toString()), ResourceTemplateReference(r[0].toString()))),
            )
            mapOf("route" to "session", "api" to "Client.complete", "prompt" to prompt.completion.values, "resource" to resource.completion.values)
        }
        "cancellation" -> attempt {
            val after = (spec["cancel_after_s"] as? Number)?.toDouble() ?: 1.0
            val wait = spec["cancel_wait_s"] ?: 4
            val started = System.nanoTime()
            val outcome = coroutineScope {
                val run = async { agentText("cancellable_wait", mapOf("seconds" to wait)) }
                val text = withTimeoutOrNull((after * 1000).toLong()) { run.await() }
                if (text != null) {
                    "returned: $text"
                } else {
                    run.cancel()
                    "abandoned: Job.cancel() on the coroutine running AIAgent.run"
                }
            }
            val abandoned = (System.nanoTime() - started) / 1e9
            delay(300)
            linkedMapOf(
                "route" to "adapter",
                "mechanism" to "cancelling the coroutine running AIAgent.run; McpTool.execute suspends in Client.callTool, " +
                    "which sends notifications/cancelled only when its own request timeout fires",
                "abandoned_after_s" to abandoned,
                "outcome" to outcome.take(200),
            )
        }
        "tasks" -> attempt { mapOf("route" to "adapter", "text" to agentText("task_capable").toString()) }
        "caching" -> attempt {
            val pause = (spec["cache_pause_s"] as? Number)?.toDouble() ?: 2.0
            val marks = mutableListOf<Map<String, Any?>>()
            for (mark in listOf("first", "within_ttl", "after_ttl")) {
                if (mark == "after_ttl") delay((pause * 1000).toLong())
                val startedAt = stamp()
                val count = McpToolRegistryProvider.fromClient(client, conn.info).tools.size
                marks.add(linkedMapOf("call" to mark, "started" to startedAt, "finished" to stamp(), "count" to count))
            }
            mapOf(
                "route" to "adapter",
                "api" to "McpToolRegistryProvider.fromClient (lists on every call; the ToolRegistry it returns is fixed, no cache option)",
                "marks" to marks,
            )
        }
        else -> unsupported("unknown utility $feature")
    }

    // -----------------------------------------------------------------
    // Client features against fixtures/client_features_server.py
    // -----------------------------------------------------------------

    suspend fun clientFeatures(tools: List<String>): Map<String, Any?> {
        val answers = linkedMapOf<String, Any?>()
        Hooks.drain()
        out["clientCapabilities"] = mcp(ClientCapabilities(
            sampling = JsonObject(emptyMap()),
            roots = ClientCapabilities.Roots(listChanged = true),
            elicitation = JsonObject(mapOf("form" to JsonObject(emptyMap()), "url" to JsonObject(emptyMap()))),
        ))
        for (tool in tools) {
            val entry = linkedMapOf<String, Any?>()
            if (tool == "roots_changes_seen") {
                try {
                    client.removeRoot(ROOT_URI)
                    client.addRoot(CHANGED_ROOT_URI, "mcp-mirror-changed")
                    client.sendRootsListChanged()
                    delay(300)
                } catch (error: Throwable) {
                    entry["root_change_error"] = describe(error)
                }
            }
            entry.putAll(result(mapOf("tool" to tool, "arguments" to emptyMap<String, Any?>())))
            entry["handler_saw"] = Hooks.drain()
            answers[tool] = entry
        }
        return answers
    }
}

fun main(args: Array<String>) {
    @Suppress("UNCHECKED_CAST")
    val cfg = plain(McpJson.parseToJsonElement(File(args[0]).readText())) as Map<String, Any?>
    val out = linkedMapOf<String, Any?>()
    var status = 0
    val run = Run(cfg, out)
    try {
        runBlocking { run.run() }
    } catch (error: Throwable) {
        out["fatal"] = describe(error)
        status = 2
    } finally {
        runCatching { run.close() }
    }
    File(args[1]).writeText(json(out).toString())
    // The stdio transport and ktor leave non-daemon threads behind.
    exitProcess(status)
}
