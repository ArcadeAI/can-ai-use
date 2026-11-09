package dev.mcpmirror.langchain4j;

import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import com.fasterxml.jackson.databind.ObjectMapper;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.internal.JsonSchemaElementUtils;
import dev.langchain4j.invocation.InvocationContext;
import dev.langchain4j.mcp.McpToolProvider;
import dev.langchain4j.mcp.client.DefaultMcpClient;
import dev.langchain4j.mcp.client.McpBlobResourceContents;
import dev.langchain4j.mcp.client.McpCallContext;
import dev.langchain4j.mcp.client.McpClientListener;
import dev.langchain4j.mcp.client.McpGetPromptResult;
import dev.langchain4j.mcp.client.McpPrompt;
import dev.langchain4j.mcp.client.McpPromptArgument;
import dev.langchain4j.mcp.client.McpPromptMessage;
import dev.langchain4j.mcp.client.McpResource;
import dev.langchain4j.mcp.client.McpResourceContents;
import dev.langchain4j.mcp.client.McpResourceTemplate;
import dev.langchain4j.mcp.client.McpRoot;
import dev.langchain4j.mcp.client.McpTextContent;
import dev.langchain4j.mcp.client.McpTextResourceContents;
import dev.langchain4j.mcp.client.logging.McpLogMessage;
import dev.langchain4j.mcp.client.progress.McpProgressNotification;
import dev.langchain4j.mcp.client.transport.McpTransport;
import dev.langchain4j.mcp.client.transport.http.StreamableHttpMcpTransport;
import dev.langchain4j.mcp.client.transport.stdio.StdioMcpTransport;
import dev.langchain4j.mcp.protocol.McpClientMessage;
import dev.langchain4j.mcp.protocol.McpClientRequest;
import dev.langchain4j.mcp.protocol.McpInitializeResult;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.tool.ToolExecutionResult;
import dev.langchain4j.service.tool.ToolExecutor;
import dev.langchain4j.service.tool.ToolProviderRequest;
import dev.langchain4j.service.tool.ToolProviderResult;

/**
 * mcp-mirror's LangChain4j worker.
 *
 * {@code java -jar worker.jar <config.json> <out.json>}. The config names one MCP
 * server; the worker connects a langchain4j-mcp {@link DefaultMcpClient} to it, wraps it
 * in an {@link McpToolProvider}, and runs an {@link AiServices} agent over a scripted
 * {@link ChatModel}, reporting what the model was handed. Output goes to a file so
 * nothing the stdio server prints can corrupt it.
 */
public final class Worker {

	static final ObjectMapper JSON = new ObjectMapper();

	static final DateTimeFormatter STAMP = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSSSS'+00:00'")
		.withZone(ZoneOffset.UTC);

	static final String PROTOCOL_META_KEY = "io.modelcontextprotocol/protocolVersion";

	static final String WATCHED_URI = "file:///fixtures/watched.txt";

	static final String ROOT_URI = "file:///tmp/mcp-mirror-root";

	static final String CHANGED_ROOT_URI = "file:///tmp/mcp-mirror-root-changed";

	static final String CALL_ID = "call_mirror_1";

	private Worker() {
	}

	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws Exception {
		Map<String, Object> cfg = JSON.readValue(Path.of(args[0]).toFile(), Map.class);
		Path outPath = Path.of(args[1]);
		Map<String, Object> out = new LinkedHashMap<>();
		int status = 0;
		try (Run run = new Run(cfg, out)) {
			run.run();
		}
		catch (Throwable error) {
			out.put("fatal", describe(error));
			status = 2;
		}
		Files.writeString(outPath, JSON.writeValueAsString(out));
		// The stdio transport leaves non-daemon reader threads behind.
		System.exit(status);
	}

	static String describe(Throwable error) {
		String message = error.getMessage();
		return error.getClass().getSimpleName() + ": " + (message == null ? "" : message);
	}

	static String stamp() {
		return STAMP.format(Instant.now());
	}

	static void sleep(long millis) {
		try {
			Thread.sleep(millis);
		}
		catch (InterruptedException ignored) {
			Thread.currentThread().interrupt();
		}
	}

	interface Body {

		Map<String, Object> get() throws Exception;

	}

	/** {"ok": true, ...} or the failure, the way src/mcp_mirror/probe.py records it. */
	static Map<String, Object> attempt(Body body) {
		Map<String, Object> out = new LinkedHashMap<>();
		try {
			Map<String, Object> answer = body.get();
			out.put("ok", true);
			out.putAll(answer);
		}
		catch (UnsupportedOperationException error) {
			out.put("ok", false);
			out.put("unsupported", describe(error));
		}
		catch (Throwable error) {
			out.put("ok", false);
			out.put("error", describe(error));
		}
		return out;
	}

	static Map<String, Object> unsupported(String reason) {
		Map<String, Object> out = new LinkedHashMap<>();
		out.put("ok", false);
		out.put("unsupported", reason);
		return out;
	}

	static List<Map<String, Object>> contents(List<Content> contents) {
		List<Map<String, Object>> blocks = new ArrayList<>();
		for (Content content : contents == null ? List.<Content>of() : contents) {
			Map<String, Object> block = new LinkedHashMap<>();
			block.put("type", content.type() == null ? null : content.type().name().toLowerCase());
			if (content instanceof TextContent text) {
				block.put("text", text.text());
			}
			blocks.add(block);
		}
		return blocks;
	}

	/** What reached the client's hooks, in call order, and how often each kind arrived. */
	static final class Hooks implements McpClientListener {

		final List<Map<String, Object>> seen = Collections.synchronizedList(new ArrayList<>());

		final Map<String, Integer> counts = Collections.synchronizedMap(new LinkedHashMap<>());

		final List<String> events = Collections.synchronizedList(new ArrayList<>());

		final List<Map<String, Object>> progress = Collections.synchronizedList(new ArrayList<>());

		final List<Map<String, Object>> logs = Collections.synchronizedList(new ArrayList<>());

		final List<String> advertised = Collections.synchronizedList(new ArrayList<>());

		void note(String hook, Map<String, Object> facts) {
			Map<String, Object> entry = new LinkedHashMap<>();
			entry.put("hook", hook);
			entry.putAll(facts);
			seen.add(entry);
		}

		void count(String kind, String event) {
			counts.merge(kind, 1, Integer::sum);
			if (!events.contains(event)) {
				events.add(event);
			}
		}

		int seen(String kind) {
			return counts.getOrDefault(kind, 0);
		}

		List<Map<String, Object>> drain() {
			synchronized (seen) {
				List<Map<String, Object>> out = new ArrayList<>(seen);
				seen.clear();
				return out;
			}
		}

		/**
		 * The McpMetaSupplier: called for every request with the message the client is
		 * about to send, after a modern client has stamped its protocol version into
		 * {@code _meta}. It adds nothing.
		 */
		Map<String, Object> meta(McpCallContext context) {
			McpClientMessage message = context == null ? null : context.message();
			if (message instanceof McpClientRequest request && request.getParams() != null
					&& request.getParams().getMeta() != null
					&& request.getParams().getMeta().get(PROTOCOL_META_KEY) instanceof String version) {
				advertised.add(version);
			}
			return Map.of();
		}

		void log(McpLogMessage message) {
			Map<String, Object> entry = new LinkedHashMap<>();
			entry.put("level", message.level() == null ? null : message.level().name().toLowerCase());
			entry.put("data", message.dataAsObject());
			logs.add(entry);
		}

		void progress(McpProgressNotification notification) {
			Map<String, Object> entry = new LinkedHashMap<>();
			entry.put("progress", notification.progress());
			entry.put("total", notification.total());
			entry.put("message", notification.message());
			progress.add(entry);
		}

		@Override
		public void onNotificationToolsListChanged() {
			count("tools", "ToolsListChanged");
		}

		@Override
		public void onNotificationResourcesListChanged() {
			count("resources", "ResourcesListChanged");
		}

		@Override
		public void onNotificationPromptsListChanged() {
			count("prompts", "PromptsListChanged");
		}

		@Override
		public void onNotificationResourceUpdated(String uri) {
			count("updated", "ResourceUpdated");
		}

		@Override
		public void onServerRootsList() {
			note("DefaultMcpClient.Builder.roots", Map.of("request", "roots/list"));
		}

	}

	/**
	 * A ChatModel that runs no model. Its first answer asks for one tool call, if the
	 * run names one; every later answer is plain text. It keeps the first request's
	 * tool specifications and the tool result message AiServices sends back.
	 */
	static final class ScriptedModel implements ChatModel {

		final String tool;

		final String arguments;

		List<ToolSpecification> offered;

		ToolExecutionResultMessage answer;

		ScriptedModel(String tool, String arguments) {
			this.tool = tool;
			this.arguments = arguments;
		}

		@Override
		public ChatResponse doChat(ChatRequest request) {
			if (offered == null) {
				offered = request.toolSpecifications() == null ? List.of() : request.toolSpecifications();
			}
			for (ChatMessage message : request.messages()) {
				if (message instanceof ToolExecutionResultMessage result) {
					answer = result;
				}
			}
			if (tool != null && answer == null) {
				ToolExecutionRequest call = ToolExecutionRequest.builder()
					.id(CALL_ID)
					.name(tool)
					.arguments(arguments)
					.build();
				return ChatResponse.builder().aiMessage(AiMessage.from(call)).build();
			}
			return ChatResponse.builder().aiMessage(AiMessage.from("mcp-mirror done")).build();
		}

	}

	interface Assistant {

		String chat(String message);

	}

	static final class Run implements AutoCloseable {

		final Map<String, Object> cfg;

		final Map<String, Object> out;

		final Hooks hooks = new Hooks();

		final DefaultMcpClient client;

		final McpToolProvider provider;

		@SuppressWarnings("unchecked")
		Run(Map<String, Object> cfg, Map<String, Object> out) {
			this.cfg = cfg;
			this.out = out;
			boolean withHooks = Boolean.TRUE.equals(cfg.get("hooks"));
			DefaultMcpClient.Builder builder = DefaultMcpClient.builder()
				.key("src")
				.transport(transport(cfg))
				.listener(hooks)
				.metaSupplier(hooks::meta);
			if (cfg.get("toolExecutionTimeoutMs") instanceof Number timeout) {
				builder.toolExecutionTimeout(Duration.ofMillis(timeout.longValue()));
			}
			if (withHooks) {
				// What an application registers to hear the server: log messages, progress,
				// resource updates and the roots it answers roots/list with.
				builder.logHandler(hooks::log)
					.progressHandler(hooks::progress)
					.roots(List.of(new McpRoot("mcp-mirror", ROOT_URI)))
					.onResourceUpdated((mcp, uri) -> hooks.note("DefaultMcpClient.Builder.onResourceUpdated",
							Map.of("uri", uri)));
			}
			this.client = builder.build();
			this.provider = McpToolProvider.builder().mcpClients(client).build();
		}

		static McpTransport transport(Map<String, Object> cfg) {
			String kind = String.valueOf(cfg.getOrDefault("transport", "stdio"));
			if (kind.equals("stdio")) {
				List<String> command = new ArrayList<>();
				command.add(String.valueOf(cfg.get("command")));
				for (Object arg : (List<?>) cfg.getOrDefault("args", List.of())) {
					command.add(String.valueOf(arg));
				}
				return StdioMcpTransport.builder().command(command).logEvents(false).build();
			}
			if (kind.equals("http")) {
				Map<String, String> headers = new LinkedHashMap<>();
				if (cfg.get("headers") instanceof Map<?, ?> map) {
					map.forEach((k, v) -> headers.put(String.valueOf(k), String.valueOf(v)));
				}
				return StreamableHttpMcpTransport.builder()
					.url(String.valueOf(cfg.get("url")))
					.customHeaders(headers)
					.timeout(Duration.ofSeconds(60))
					.build();
			}
			throw new UnsupportedOperationException(
					"langchain4j-mcp ships stdio, Streamable HTTP and WebSocket transports; it has no " + kind
							+ " transport");
		}

		@Override
		public void close() {
			try {
				client.close();
			}
			catch (Exception ignored) {
				// the worker exits next
			}
		}

		/**
		 * The protocol revision the client speaks. A modern client stamps it into the
		 * {@code _meta} of every request, which the McpMetaSupplier sees; a legacy client
		 * keeps the server's initialize answer in a private field with no getter.
		 */
		String negotiated() throws Exception {
			if (client.isModernProtocol()) {
				client.listTools();
				synchronized (hooks.advertised) {
					return hooks.advertised.isEmpty() ? null : hooks.advertised.get(hooks.advertised.size() - 1);
				}
			}
			Field field = DefaultMcpClient.class.getDeclaredField("initializeResult");
			field.setAccessible(true);
			McpInitializeResult init = (McpInitializeResult) field.get(client);
			return init == null || init.getResult() == null ? null : init.getResult().getProtocolVersion();
		}

		@SuppressWarnings("unchecked")
		void run() throws Exception {
			out.put("negotiatedMcpSpecVersion", negotiated());
			out.put("modernProtocol", client.isModernProtocol());
			out.put("clientInfo", Map.of("name", "langchain4j"));
			String mode = String.valueOf(cfg.getOrDefault("mode", "tools"));
			out.put("tools", List.of());
			switch (mode) {
				case "tools" -> {
					out.put("tools", definitions());
					List<Map<String, Object>> results = new ArrayList<>();
					for (Object call : (List<Object>) cfg.getOrDefault("calls", List.of())) {
						Map<String, Object> c = (Map<String, Object>) call;
						String tool = String.valueOf(c.get("tool"));
						Map<String, Object> arguments = (Map<String, Object>) c.getOrDefault("arguments", Map.of());
						Map<String, Object> entry = "direct".equals(cfg.get("boundary")) ? direct(tool, arguments)
								: agent(tool, arguments);
						entry.put("tool", tool);
						results.add(entry);
					}
					out.put("results", results);
				}
				case "probe" -> out.put("probe", probe());
				case "notify" -> out.put("notify", notifyProbe());
				case "utility" -> out.put("utility",
						utility(String.valueOf(cfg.get("utility")), (Map<String, Object>) cfg.getOrDefault("utilitySpec", Map.of())));
				case "client" -> out.put("client", clientFeatures((List<Object>) cfg.getOrDefault("clientCalls", List.of())));
				default -> throw new IllegalArgumentException("unknown mode " + mode);
			}
		}

		ToolProviderResult provided() {
			return provider.provideTools(new ToolProviderRequest("mcp-mirror", UserMessage.from("mcp-mirror probe")));
		}

		Assistant assistant(ScriptedModel model) {
			return AiServices.builder(Assistant.class).chatModel(model).toolProvider(provider).build();
		}

		/** The ToolSpecifications AiServices hands the ChatModel in its first ChatRequest. */
		List<Map<String, Object>> definitions() {
			ScriptedModel model = new ScriptedModel(null, null);
			assistant(model).chat("mcp-mirror probe");
			List<Map<String, Object>> tools = new ArrayList<>();
			for (ToolSpecification spec : model.offered) {
				Map<String, Object> entry = new LinkedHashMap<>();
				entry.put("name", spec.name());
				entry.put("description", spec.description());
				entry.put("parameters", spec.parameters() == null ? null : JsonSchemaElementUtils.toMap(spec.parameters()));
				entry.put("metadata", spec.metadata());
				tools.add(entry);
			}
			return tools;
		}

		/**
		 * One scripted assistant turn through AiServices: the model asks for the tool, the
		 * service runs it through McpToolProvider's executor, and the answer is the
		 * ToolExecutionResultMessage the next ChatRequest carries.
		 */
		Map<String, Object> agent(String tool, Map<String, Object> arguments) {
			Map<String, Object> entry = new LinkedHashMap<>();
			try {
				ScriptedModel model = new ScriptedModel(tool, JSON.writeValueAsString(arguments));
				assistant(model).chat("mcp-mirror probe");
				ToolExecutionResultMessage message = model.answer;
				if (message == null) {
					throw new IllegalStateException("AiServices sent the model no tool result for " + tool);
				}
				Map<String, Object> returned = new LinkedHashMap<>();
				returned.put("id", message.id());
				returned.put("toolName", message.toolName());
				returned.put("text", message.hasSingleText() ? message.text() : null);
				returned.put("contents", contents(message.contents()));
				returned.put("isError", message.isError());
				returned.put("attributes", message.attributes());
				entry.put("ok", true);
				entry.put("returned", returned);
			}
			catch (Throwable error) {
				entry.put("ok", false);
				entry.put("threw", true);
				entry.put("exception", error.getClass().getName());
				entry.put("error", describe(error));
			}
			return entry;
		}

		/** The ToolExecutor McpToolProvider hands out, called the way AiServices calls it. */
		Map<String, Object> direct(String tool, Map<String, Object> arguments) {
			Map<String, Object> entry = new LinkedHashMap<>();
			try {
				ToolExecutor executor = provided().toolExecutorByName(tool);
				if (executor == null) {
					throw new IllegalStateException("McpToolProvider exposes no executor for MCP tool " + tool);
				}
				ToolExecutionRequest call = ToolExecutionRequest.builder()
					.id(CALL_ID)
					.name(tool)
					.arguments(JSON.writeValueAsString(arguments))
					.build();
				ToolExecutionResult result = executor.executeWithContext(call,
						InvocationContext.builder().chatMemoryId("mcp-mirror").build());
				Map<String, Object> returned = new LinkedHashMap<>();
				returned.put("resultText", result.resultText());
				returned.put("result", result.result());
				returned.put("contents", contents(result.resultContents()));
				returned.put("isError", result.isError());
				returned.put("attributes", result.attributes());
				entry.put("ok", true);
				entry.put("returned", returned);
			}
			catch (Throwable error) {
				entry.put("ok", false);
				entry.put("threw", true);
				entry.put("exception", error.getClass().getName());
				entry.put("error", describe(error));
			}
			return entry;
		}

		String agentText(String tool, Map<String, Object> arguments) {
			Map<String, Object> entry = agent(tool, arguments);
			if (!Boolean.TRUE.equals(entry.get("ok"))) {
				throw new IllegalStateException(String.valueOf(entry.get("error")));
			}
			Map<?, ?> returned = (Map<?, ?>) entry.get("returned");
			Object text = returned.get("text");
			return text == null ? String.valueOf(returned.get("contents")) : String.valueOf(text);
		}

		// -----------------------------------------------------------------
		// Resources and prompts, through the McpClient
		// -----------------------------------------------------------------

		Map<String, Object> probe() {
			Map<String, Object> probe = new LinkedHashMap<>();
			List<McpResource> listed = new ArrayList<>();
			probe.put("resources.list", attempt(() -> {
				listed.addAll(client.listResources());
				List<Map<String, Object>> entries = new ArrayList<>();
				for (McpResource r : listed) {
					Map<String, Object> e = new LinkedHashMap<>();
					e.put("uri", r.uri());
					e.put("name", r.name());
					// McpResource has no title field.
					e.put("title", null);
					e.put("description", r.description());
					e.put("mimeType", r.mimeType());
					entries.add(e);
				}
				return Map.of("entries", entries);
			}));
			String textUri = null;
			String blobUri = null;
			for (McpResource r : listed) {
				String mime = r.mimeType() == null ? "" : r.mimeType();
				if (textUri == null && mime.startsWith("text/")) {
					textUri = r.uri();
				}
				if (blobUri == null && !mime.isEmpty() && !mime.startsWith("text/")) {
					blobUri = r.uri();
				}
			}
			probe.put("resources.read.text", textUri == null ? Map.of("ok", false, "error", "no text resource listed")
					: attempt(read(textUri)));
			probe.put("resources.read.blob", blobUri == null ? Map.of("ok", false, "error", "no binary resource listed")
					: attempt(read(blobUri)));
			probe.put("resources.templates", attempt(() -> {
				List<Map<String, Object>> entries = new ArrayList<>();
				for (McpResourceTemplate t : client.listResourceTemplates()) {
					Map<String, Object> e = new LinkedHashMap<>();
					e.put("uriTemplate", t.uriTemplate());
					e.put("name", t.name());
					e.put("description", t.description());
					e.put("mimeType", t.mimeType());
					entries.add(e);
				}
				return Map.of("entries", entries);
			}));
			List<McpPrompt> prompts = new ArrayList<>();
			probe.put("prompts.list", attempt(() -> {
				prompts.addAll(client.listPrompts());
				List<Map<String, Object>> entries = new ArrayList<>();
				for (McpPrompt p : prompts) {
					Map<String, Object> e = new LinkedHashMap<>();
					e.put("name", p.name());
					// McpPrompt has no title field.
					e.put("title", null);
					e.put("description", p.description());
					List<Map<String, Object>> arguments = new ArrayList<>();
					for (McpPromptArgument a : p.arguments() == null ? List.<McpPromptArgument>of() : p.arguments()) {
						Map<String, Object> arg = new LinkedHashMap<>();
						arg.put("name", a.name());
						arg.put("description", a.description());
						arg.put("required", a.required());
						arguments.add(arg);
					}
					e.put("arguments", arguments);
					entries.add(e);
				}
				return Map.of("entries", entries);
			}));
			probe.put("prompts.get", prompts.isEmpty() ? Map.of("ok", false, "error", "no prompt listed") : attempt(() -> {
				McpPrompt first = prompts.get(0);
				Map<String, Object> arguments = new LinkedHashMap<>();
				if (first.arguments() != null) {
					for (McpPromptArgument a : first.arguments()) {
						if (a.required()) {
							arguments.put(a.name(), "x");
						}
					}
				}
				McpGetPromptResult result = client.getPrompt(first.name(), arguments);
				List<Map<String, Object>> messages = new ArrayList<>();
				for (McpPromptMessage m : result.messages()) {
					Map<String, Object> e = new LinkedHashMap<>();
					e.put("role", m.role() == null ? null : m.role().name().toLowerCase());
					e.put("type", m.content() == null || m.content().type() == null ? null
							: m.content().type().name().toLowerCase());
					e.put("text", m.content() instanceof McpTextContent text ? text.text() : null);
					messages.add(e);
				}
				Map<String, Object> answer = new LinkedHashMap<>();
				answer.put("description", result.description());
				answer.put("messages", messages);
				return answer;
			}));
			return probe;
		}

		Body read(String uri) {
			return () -> {
				McpResourceContents first = client.readResource(uri).contents().get(0);
				Map<String, Object> answer = new LinkedHashMap<>();
				if (first instanceof McpBlobResourceContents blob) {
					answer.put("uri", blob.uri());
					answer.put("mimeType", blob.mimeType());
					byte[] bytes = Base64.getDecoder().decode(blob.blob());
					answer.put("blob_sha256",
							HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)));
					answer.put("blob_bytes", bytes.length);
				}
				else if (first instanceof McpTextResourceContents text) {
					answer.put("uri", text.uri());
					answer.put("mimeType", text.mimeType());
					answer.put("text", text.text());
				}
				return answer;
			};
		}

		// -----------------------------------------------------------------
		// Catalogue changes against fixtures/notify_server.py
		// -----------------------------------------------------------------

		boolean waitFor(String kind, int since, long millis) {
			long deadline = System.currentTimeMillis() + millis;
			while (System.currentTimeMillis() < deadline) {
				if (hooks.seen(kind) > since) {
					return true;
				}
				sleep(50);
			}
			return hooks.seen(kind) > since;
		}

		/** A fixture mutation, called on the client so it never goes through a model. */
		String callRaw(String tool) {
			return client.executeTool(ToolExecutionRequest.builder().id(CALL_ID).name(tool).arguments("{}").build())
				.resultText();
		}

		List<String> listing(String kind) {
			List<String> names = new ArrayList<>();
			switch (kind) {
				case "tools" -> provided().tools().keySet().forEach(spec -> names.add(spec.name()));
				case "resources" -> client.listResources().forEach(r -> names.add(r.uri()));
				default -> client.listPrompts().forEach(p -> names.add(p.name()));
			}
			names.sort(null);
			return names;
		}

		Map<String, Object> notifyProbe() {
			Map<String, Object> notify = new LinkedHashMap<>();
			notify.put("negotiated", out.get("negotiatedMcpSpecVersion"));
			String[][] changes = { { "tools.list_changed", "add_tool", "tools", "SENTINEL_ADDED_TOOL",
					"McpClientListener.onNotificationToolsListChanged; McpToolProvider re-lists through DefaultMcpClient.listTools" },
					{ "resources.list_changed", "add_resource", "resources", "fixtures/added-",
							"McpClientListener.onNotificationResourcesListChanged" },
					{ "prompts.list_changed", "add_prompt", "prompts", "SENTINEL_ADDED_PROMPT",
							"McpClientListener.onNotificationPromptsListChanged" } };
			for (String[] change : changes) {
				notify.put(change[0], attempt(() -> {
					List<String> before = listing(change[2]);
					int since = hooks.seen(change[2]);
					String said = callRaw(change[1]);
					boolean notified = waitFor(change[2], since, 3000);
					List<String> after = listing(change[2]);
					Map<String, Object> answer = new LinkedHashMap<>();
					answer.put("hook", change[4]);
					answer.put("notified", notified);
					answer.put("before", before);
					answer.put("after", after);
					answer.put("sentinel", change[3]);
					answer.put("server_said", said);
					return answer;
				}));
			}
			boolean modern = client.isModernProtocol();
			notify.put("resources.subscribe", attempt(() -> {
				Map<String, Object> answer = new LinkedHashMap<>();
				answer.put("route", "adapter");
				answer.put("api", modern
						? "McpClient.subscribeToResources / unsubscribeFromResources(subscriptionId); updates through McpClientListener.onNotificationResourceUpdated"
						: "McpClient.subscribeToResource / unsubscribeFromResource; updates through McpClientListener.onNotificationResourceUpdated");
				answer.put("subscribed", false);
				answer.put("subscribe_error", null);
				answer.put("notified", false);
				answer.put("unsubscribed", false);
				answer.put("unsubscribe_error", null);
				answer.put("notified_after_unsubscribe", null);
				List<String> said = new ArrayList<>();
				answer.put("server_said", said);
				long subscription = -1;
				try {
					if (modern) {
						subscription = client.subscribeToResources(List.of(WATCHED_URI));
					}
					else {
						client.subscribeToResource(WATCHED_URI);
					}
					answer.put("subscribed", true);
				}
				catch (Throwable error) {
					answer.put("subscribe_error", describe(error));
				}
				int since = hooks.seen("updated");
				said.add(callRaw("update_resource"));
				answer.put("notified", waitFor("updated", since, 3000));
				if (!Boolean.TRUE.equals(answer.get("subscribed"))) {
					return answer;
				}
				try {
					if (modern) {
						client.unsubscribeFromResources(subscription);
					}
					else {
						client.unsubscribeFromResource(WATCHED_URI);
					}
					answer.put("unsubscribed", true);
				}
				catch (Throwable error) {
					answer.put("unsubscribe_error", describe(error));
					return answer;
				}
				since = hooks.seen("updated");
				said.add(callRaw("update_resource"));
				answer.put("notified_after_unsubscribe", waitFor("updated", since, 1500));
				return answer;
			}));
			if (modern) {
				Map<String, Object> listen = new LinkedHashMap<>();
				listen.put("ok", true);
				listen.put("route", "adapter");
				listen.put("api",
						"DefaultMcpClient subscriptions/listen stream, opened at connect by subscribeToToolListChanges, "
								+ "subscribeToResourceListChanges and subscribeToPromptListChanges (all default true)");
				List<String> events = new ArrayList<>(hooks.events);
				listen.put("opened", !events.isEmpty());
				listen.put("open_error", null);
				listen.put("honored", null);
				listen.put("events", events);
				notify.put("subscriptions.listen", listen);
			}
			return notify;
		}

		// -----------------------------------------------------------------
		// Server utilities, one question per worker run
		// -----------------------------------------------------------------

		Map<String, Object> utility(String feature, Map<String, Object> spec) {
			return switch (feature) {
				case "tools.list" -> attempt(() -> {
					List<String> names = new ArrayList<>();
					for (Map<String, Object> tool : definitions()) {
						names.add(String.valueOf(tool.get("name")));
					}
					return Map.of("route", "adapter", "api",
							"McpToolProvider.provideTools, as AiServices hands the ChatModel its ToolSpecifications",
							"names", names);
				});
				case "progress" -> attempt(() -> {
					String text = agentText("slow_with_progress", Map.of());
					sleep(300);
					Map<String, Object> answer = new LinkedHashMap<>();
					answer.put("route", "adapter");
					answer.put("api",
							"DefaultMcpClient.Builder.progressHandler(McpProgressHandler); the client then sends a progressToken with each tools/call");
					answer.put("events", new ArrayList<>(hooks.progress));
					answer.put("text", text);
					return answer;
				});
				case "logging" -> attempt(() -> {
					String text = agentText("emit_logs", Map.of());
					sleep(300);
					Map<String, Object> answer = new LinkedHashMap<>();
					answer.put("route", "adapter");
					answer.put("api", "DefaultMcpClient.Builder.logHandler(McpLogMessageHandler)");
					answer.put("messages", new ArrayList<>(hooks.logs));
					answer.put("text", text);
					return answer;
				});
				case "set_level" -> unsupported(
						"McpClient has no logging/setLevel method; DefaultMcpClient never sends one");
				case "completion" -> unsupported(
						"McpClient has no completion/complete method; DefaultMcpClient never sends one");
				case "cancellation" -> attempt(() -> {
					double after = ((Number) spec.getOrDefault("cancel_after_s", 1.0)).doubleValue();
					Object wait = spec.getOrDefault("cancel_wait_s", 4);
					// The run's client was built with toolExecutionTimeout = cancel_after_s.
					ExecutorService executor = Executors.newSingleThreadExecutor();
					long started = System.nanoTime();
					Future<String> call = executor.submit(() -> agentText("cancellable_wait", Map.of("seconds", wait)));
					String outcome;
					try {
						outcome = "returned: " + call.get((long) (after * 1000) + 10_000, TimeUnit.MILLISECONDS);
					}
					catch (TimeoutException timeout) {
						call.cancel(true);
						outcome = "abandoned: the tool call outlived toolExecutionTimeout by 10s";
					}
					double abandoned = (System.nanoTime() - started) / 1e9;
					sleep(300);
					executor.shutdownNow();
					Map<String, Object> answer = new LinkedHashMap<>();
					answer.put("route", "adapter");
					answer.put("mechanism",
							"DefaultMcpClient.Builder.toolExecutionTimeout: when it expires the client sends notifications/cancelled "
									+ "and answers the tool call with toolExecutionTimeoutErrorMessage");
					answer.put("abandoned_after_s", abandoned);
					answer.put("outcome", outcome.length() > 200 ? outcome.substring(0, 200) : outcome);
					return answer;
				});
				case "tasks" -> attempt(() -> Map.of("route", "adapter", "text",
						String.valueOf(agentText("task_capable", Map.of()))));
				case "caching" -> attempt(() -> {
					List<Map<String, Object>> marks = new ArrayList<>();
					double pause = ((Number) spec.getOrDefault("cache_pause_s", 2.0)).doubleValue();
					for (String mark : List.of("first", "within_ttl", "after_ttl")) {
						if (mark.equals("after_ttl")) {
							sleep((long) (pause * 1000));
						}
						String startedAt = stamp();
						int count = provided().tools().size();
						Map<String, Object> entry = new LinkedHashMap<>();
						entry.put("call", mark);
						entry.put("started", startedAt);
						entry.put("finished", stamp());
						entry.put("count", count);
						marks.add(entry);
					}
					return Map.of("route", "adapter", "api",
							"McpToolProvider.provideTools over DefaultMcpClient.listTools (cacheToolList, default true, "
									+ "until a tools/list_changed notification)",
							"marks", marks);
				});
				default -> unsupported("unknown utility " + feature);
			};
		}

		// -----------------------------------------------------------------
		// Client features against fixtures/client_features_server.py
		// -----------------------------------------------------------------

		Map<String, Object> clientFeatures(List<Object> tools) {
			Map<String, Object> answers = new LinkedHashMap<>();
			hooks.drain();
			for (Object name : tools) {
				String tool = String.valueOf(name);
				Map<String, Object> entry = new LinkedHashMap<>();
				if (tool.equals("roots_changes_seen")) {
					try {
						client.setRoots(List.of(new McpRoot("mcp-mirror-changed", CHANGED_ROOT_URI)));
						sleep(300);
					}
					catch (Throwable error) {
						entry.put("root_change_error", describe(error));
					}
				}
				entry.putAll(agent(tool, Map.of()));
				entry.put("handler_saw", hooks.drain());
				answers.put(tool, entry);
			}
			return answers;
		}

	}

}
