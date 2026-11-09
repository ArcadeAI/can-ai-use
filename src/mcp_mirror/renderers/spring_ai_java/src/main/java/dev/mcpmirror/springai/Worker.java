package dev.mcpmirror.springai;

import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.Supplier;

import io.modelcontextprotocol.client.McpSyncClient;
import io.modelcontextprotocol.client.transport.customizer.McpSyncHttpClientRequestCustomizer;
import io.modelcontextprotocol.spec.McpSchema;
import tools.jackson.databind.json.JsonMapper;

import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.mcp.SyncMcpToolCallback;
import org.springframework.ai.mcp.SyncMcpToolCallbackProvider;
import org.springframework.ai.model.tool.DefaultToolCallingManager;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.model.tool.ToolExecutionResult;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.context.annotation.Bean;

/**
 * mcp-mirror's Spring AI worker: a command-line Spring Boot application.
 *
 * {@code java -jar worker.jar <config.json> <out.json>}. The config names one MCP
 * server; the worker turns it into spring.ai.mcp.client.* properties for a connection
 * called "src", lets spring-ai-starter-mcp-client build the McpSyncClient and the
 * SyncMcpToolCallbackProvider, and reports what they produced. Output goes to a file so
 * nothing Spring or the stdio server prints can corrupt it.
 */
@SpringBootApplication
public class Worker {

	static final JsonMapper JSON = JsonMapper.builder().build();

	static final Map<String, String> HEADERS = new LinkedHashMap<>();

	static final DateTimeFormatter STAMP = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSSSS'+00:00'")
		.withZone(ZoneOffset.UTC);

	static final String WATCHED_URI = "file:///fixtures/watched.txt";

	static final String CHANGED_ROOT_URI = "file:///tmp/mcp-mirror-root-changed";

	/** Fixed headers for an HTTP connection; the starter has no property for them. */
	@Bean
	McpSyncHttpClientRequestCustomizer mirrorHeaders() {
		return (builder, method, endpoint, body, context) -> HEADERS.forEach(builder::header);
	}

	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws Exception {
		Map<String, Object> cfg = JSON.readValue(Path.of(args[0]).toFile(), Map.class);
		Path outPath = Path.of(args[1]);
		Map<String, Object> out = new LinkedHashMap<>();
		int status = 0;
		Object headers = cfg.get("headers");
		if (headers instanceof Map<?, ?> map) {
			map.forEach((k, v) -> HEADERS.put(String.valueOf(k), String.valueOf(v)));
		}
		SpringApplication app = new SpringApplication(Worker.class);
		app.setDefaultProperties(properties(cfg));
		if (Boolean.TRUE.equals(cfg.get("hooks"))) {
			app.setAdditionalProfiles("hooks");
		}
		try (ConfigurableApplicationContext context = app.run()) {
			new Run(context, cfg, out).run();
		}
		catch (Throwable error) {
			out.put("fatal", describe(error));
			status = 2;
		}
		Files.writeString(outPath, JSON.writeValueAsString(out));
		// The stdio transport and Reactor leave non-daemon threads behind.
		System.exit(status);
	}

	static Map<String, Object> properties(Map<String, Object> cfg) {
		Map<String, Object> p = new LinkedHashMap<>();
		p.put("spring.main.web-application-type", "none");
		p.put("spring.main.banner-mode", "off");
		p.put("spring.main.log-startup-info", "false");
		p.put("logging.level.root", "OFF");
		String transport = String.valueOf(cfg.getOrDefault("transport", "stdio"));
		String prefix = "spring.ai.mcp.client.";
		if (transport.equals("stdio")) {
			String base = prefix + "stdio.connections." + Hooks.CONNECTION + ".";
			p.put(base + "command", cfg.get("command"));
			List<?> argv = (List<?>) cfg.getOrDefault("args", List.of());
			for (int i = 0; i < argv.size(); i++) {
				p.put(base + "args[" + i + "]", String.valueOf(argv.get(i)));
			}
		}
		else {
			URI uri = URI.create(String.valueOf(cfg.get("url")));
			String origin = uri.getScheme() + "://" + uri.getRawAuthority();
			String path = uri.getRawPath() == null || uri.getRawPath().isEmpty() ? "/" : uri.getRawPath();
			if (uri.getRawQuery() != null) {
				path += "?" + uri.getRawQuery();
			}
			if (transport.equals("http")) {
				String base = prefix + "streamable-http.connections." + Hooks.CONNECTION + ".";
				p.put(base + "url", origin);
				p.put(base + "endpoint", path);
			}
			else {
				String base = prefix + "sse.connections." + Hooks.CONNECTION + ".";
				p.put(base + "url", origin);
				p.put(base + "sse-endpoint", path);
			}
		}
		return p;
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

	@SuppressWarnings("unchecked")
	static Map<String, Object> tree(Object value) {
		return JSON.convertValue(value, Map.class);
	}

	/** The first text block of a tool answer the model would read, else the raw string. */
	static String textOf(Object returned) {
		if (returned == null) {
			return null;
		}
		String raw = String.valueOf(returned);
		try {
			Object parsed = JSON.readValue(raw, Object.class);
			if (parsed instanceof List<?> list) {
				for (Object item : list) {
					if (item instanceof Map<?, ?> block && block.get("text") instanceof String text) {
						return text;
					}
				}
			}
		}
		catch (Exception ignored) {
			// not JSON: the processor's message, read as is
		}
		return raw;
	}

	static final class Run {

		final ConfigurableApplicationContext context;

		final Map<String, Object> cfg;

		final Map<String, Object> out;

		final McpSyncClient client;

		final SyncMcpToolCallbackProvider provider;

		final DefaultToolCallingManager manager = DefaultToolCallingManager.builder().build();

		@SuppressWarnings("unchecked")
		Run(ConfigurableApplicationContext context, Map<String, Object> cfg, Map<String, Object> out) {
			this.context = context;
			this.cfg = cfg;
			this.out = out;
			List<McpSyncClient> clients = (List<McpSyncClient>) context.getBean("mcpSyncClients");
			this.client = clients.get(0);
			this.provider = context.getBean(SyncMcpToolCallbackProvider.class);
		}

		@SuppressWarnings("unchecked")
		void run() throws Exception {
			McpSchema.InitializeResult init = client.getCurrentInitializationResult();
			out.put("negotiatedMcpSpecVersion", init == null ? null : init.protocolVersion());
			out.put("clientInfo", tree(client.getClientInfo()));
			out.put("clientCapabilities", tree(client.getClientCapabilities()));
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
								: agent(tool, arguments, Map.of());
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

		ToolCallback[] callbacks() {
			return provider.getToolCallbacks();
		}

		/** ToolDefinitions as DefaultToolCallingManager resolves them for a chat model. */
		List<Map<String, Object>> definitions() {
			ToolCallback[] callbacks = callbacks();
			List<ToolDefinition> defs = manager
				.resolveToolDefinitions(ToolCallingChatOptions.builder().toolCallbacks(callbacks).build());
			List<Map<String, Object>> tools = new ArrayList<>();
			for (int i = 0; i < defs.size(); i++) {
				ToolDefinition def = defs.get(i);
				Map<String, Object> entry = new LinkedHashMap<>();
				entry.put("name", def.name());
				entry.put("description", def.description());
				entry.put("inputSchema", def.inputSchema());
				if (callbacks[i] instanceof SyncMcpToolCallback mcp) {
					entry.put("mcpName", mcp.getOriginalToolName());
				}
				tools.add(entry);
			}
			return tools;
		}

		/** The ToolCallback for an MCP tool name, and the name the model is shown. */
		ToolCallback callbackFor(String mcpName) {
			for (ToolCallback callback : callbacks()) {
				if (callback instanceof SyncMcpToolCallback mcp && mcp.getOriginalToolName().equals(mcpName)) {
					return callback;
				}
			}
			throw new IllegalStateException("SyncMcpToolCallbackProvider exposes no callback for MCP tool " + mcpName);
		}

		/**
		 * One scripted assistant turn through DefaultToolCallingManager.executeToolCalls,
		 * the step a ChatModel runs for each tool call; the answer is the ToolResponse the
		 * next model request carries.
		 */
		Map<String, Object> agent(String tool, Map<String, Object> arguments, Map<String, Object> toolContext) {
			Map<String, Object> entry = new LinkedHashMap<>();
			try {
				ToolCallback callback = callbackFor(tool);
				String modelName = callback.getToolDefinition().name();
				AssistantMessage assistant = AssistantMessage.builder()
					.content("")
					.toolCalls(List.of(new AssistantMessage.ToolCall("call_mirror_1", "function", modelName,
							JSON.writeValueAsString(arguments))))
					.build();
				ToolCallingChatOptions options = ToolCallingChatOptions.builder()
					.toolCallbacks(callbacks())
					.toolContext(toolContext)
					.build();
				Prompt prompt = new Prompt(List.<Message>of(new UserMessage("mcp-mirror probe")), options);
				ToolExecutionResult result = manager.executeToolCalls(prompt,
						new ChatResponse(List.of(new Generation(assistant))));
				List<Message> history = result.conversationHistory();
				ToolResponseMessage last = (ToolResponseMessage) history.get(history.size() - 1);
				ToolResponseMessage.ToolResponse response = last.getResponses().get(0);
				Map<String, Object> returned = new LinkedHashMap<>();
				returned.put("id", response.id());
				returned.put("name", response.name());
				returned.put("responseData", response.responseData());
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

		/** The adapter's own call API, SyncMcpToolCallback.call. */
		Map<String, Object> direct(String tool, Map<String, Object> arguments) {
			Map<String, Object> entry = new LinkedHashMap<>();
			try {
				String returned = callbackFor(tool).call(JSON.writeValueAsString(arguments));
				entry.put("ok", true);
				entry.put("returned", Map.of("text", returned));
			}
			catch (Throwable error) {
				entry.put("ok", false);
				entry.put("threw", true);
				entry.put("exception", error.getClass().getName());
				entry.put("error", describe(error));
			}
			return entry;
		}

		String agentText(String tool, Map<String, Object> arguments, Map<String, Object> toolContext) {
			Map<String, Object> entry = agent(tool, arguments, toolContext);
			if (!Boolean.TRUE.equals(entry.get("ok"))) {
				throw new IllegalStateException(String.valueOf(entry.get("error")));
			}
			return textOf(((Map<?, ?>) entry.get("returned")).get("responseData"));
		}

		// -----------------------------------------------------------------
		// Resources and prompts, through the McpSyncClient bean the starter publishes
		// -----------------------------------------------------------------

		Map<String, Object> probe() {
			Map<String, Object> probe = new LinkedHashMap<>();
			List<McpSchema.Resource> listed = new ArrayList<>();
			probe.put("resources.list", attempt(() -> {
				listed.addAll(client.listResources().resources());
				List<Map<String, Object>> entries = new ArrayList<>();
				for (McpSchema.Resource r : listed) {
					Map<String, Object> e = new LinkedHashMap<>();
					e.put("uri", r.uri());
					e.put("name", r.name());
					e.put("title", r.title());
					e.put("description", r.description());
					e.put("mimeType", r.mimeType());
					entries.add(e);
				}
				return Map.of("entries", entries);
			}));
			String textUri = null;
			String blobUri = null;
			for (McpSchema.Resource r : listed) {
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
				for (McpSchema.ResourceTemplate t : client.listResourceTemplates().resourceTemplates()) {
					Map<String, Object> e = new LinkedHashMap<>();
					e.put("uriTemplate", t.uriTemplate());
					e.put("name", t.name());
					e.put("description", t.description());
					e.put("mimeType", t.mimeType());
					entries.add(e);
				}
				return Map.of("entries", entries);
			}));
			List<McpSchema.Prompt> prompts = new ArrayList<>();
			probe.put("prompts.list", attempt(() -> {
				prompts.addAll(client.listPrompts().prompts());
				List<Map<String, Object>> entries = new ArrayList<>();
				for (McpSchema.Prompt p : prompts) {
					Map<String, Object> e = new LinkedHashMap<>();
					e.put("name", p.name());
					e.put("title", p.title());
					e.put("description", p.description());
					List<Map<String, Object>> arguments = new ArrayList<>();
					for (McpSchema.PromptArgument a : p.arguments() == null ? List.<McpSchema.PromptArgument>of()
							: p.arguments()) {
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
				McpSchema.Prompt first = prompts.get(0);
				Map<String, Object> arguments = new LinkedHashMap<>();
				if (first.arguments() != null) {
					for (McpSchema.PromptArgument a : first.arguments()) {
						if (Boolean.TRUE.equals(a.required())) {
							arguments.put(a.name(), "x");
						}
					}
				}
				McpSchema.GetPromptResult result = client
					.getPrompt(new McpSchema.GetPromptRequest(first.name(), arguments, null));
				List<Map<String, Object>> messages = new ArrayList<>();
				for (McpSchema.PromptMessage m : result.messages()) {
					Map<String, Object> e = new LinkedHashMap<>();
					e.put("role", m.role() == null ? null : m.role().name().toLowerCase());
					Map<String, Object> content = tree(m.content());
					e.put("type", content.get("type"));
					e.put("text", content.get("text"));
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
				McpSchema.ReadResourceResult result = client.readResource(new McpSchema.ReadResourceRequest(uri, null));
				McpSchema.ResourceContents first = result.contents().get(0);
				Map<String, Object> answer = new LinkedHashMap<>();
				answer.put("uri", first.uri());
				answer.put("mimeType", first.mimeType());
				if (first instanceof McpSchema.BlobResourceContents blob) {
					byte[] bytes = Base64.getDecoder().decode(blob.blob());
					answer.put("blob_sha256",
							HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)));
					answer.put("blob_bytes", bytes.length);
				}
				else if (first instanceof McpSchema.TextResourceContents text) {
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
				if (Hooks.seen(kind) > since) {
					return true;
				}
				sleep(50);
			}
			return Hooks.seen(kind) > since;
		}

		String callRaw(String tool) {
			McpSchema.CallToolResult result = client.callTool(new McpSchema.CallToolRequest(tool, Map.of(), null));
			for (McpSchema.Content content : result.content()) {
				if (content instanceof McpSchema.TextContent text) {
					return text.text();
				}
			}
			return null;
		}

		List<String> listing(String kind) {
			List<String> names = new ArrayList<>();
			switch (kind) {
				case "tools" -> {
					for (ToolCallback callback : callbacks()) {
						names.add(callback.getToolDefinition().name());
					}
				}
				case "resources" -> client.listResources().resources().forEach(r -> names.add(r.uri()));
				default -> client.listPrompts().prompts().forEach(p -> names.add(p.name()));
			}
			names.sort(null);
			return names;
		}

		Map<String, Object> notifyProbe() {
			Map<String, Object> notify = new LinkedHashMap<>();
			notify.put("negotiated", out.get("negotiatedMcpSpecVersion"));
			String[][] changes = { { "tools.list_changed", "add_tool", "tools", "SENTINEL_ADDED_TOOL",
					"@McpToolListChanged(clients = \"src\"); SyncMcpToolCallbackProvider re-lists on McpToolsChangedEvent" },
					{ "resources.list_changed", "add_resource", "resources", "fixtures/added-",
							"@McpResourceListChanged(clients = \"src\")" },
					{ "prompts.list_changed", "add_prompt", "prompts", "SENTINEL_ADDED_PROMPT",
							"@McpPromptListChanged(clients = \"src\")" } };
			for (String[] change : changes) {
				notify.put(change[0], attempt(() -> {
					List<String> before = listing(change[2]);
					int since = Hooks.seen(change[2]);
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
			notify.put("resources.subscribe", attempt(() -> {
				Map<String, Object> answer = new LinkedHashMap<>();
				answer.put("route", "session");
				answer.put("api",
						"McpSyncClient.subscribeResource / unsubscribeResource on the bean the starter publishes; "
								+ "updates through McpClientCustomizer<McpClient.SyncSpec> resourcesUpdateConsumer");
				answer.put("subscribed", false);
				answer.put("subscribe_error", null);
				answer.put("notified", false);
				answer.put("unsubscribed", false);
				answer.put("unsubscribe_error", null);
				answer.put("notified_after_unsubscribe", null);
				List<String> said = new ArrayList<>();
				answer.put("server_said", said);
				try {
					client.subscribeResource(new McpSchema.SubscribeRequest(WATCHED_URI, null));
					answer.put("subscribed", true);
				}
				catch (Throwable error) {
					answer.put("subscribe_error", describe(error));
				}
				int since = Hooks.seen("updated");
				said.add(callRaw("update_resource"));
				answer.put("notified", waitFor("updated", since, 3000));
				if (!Boolean.TRUE.equals(answer.get("subscribed"))) {
					return answer;
				}
				try {
					client.unsubscribeResource(new McpSchema.UnsubscribeRequest(WATCHED_URI, null));
					answer.put("unsubscribed", true);
				}
				catch (Throwable error) {
					answer.put("unsubscribe_error", describe(error));
					return answer;
				}
				since = Hooks.seen("updated");
				said.add(callRaw("update_resource"));
				answer.put("notified_after_unsubscribe", waitFor("updated", since, 1500));
				return answer;
			}));
			String negotiated = String.valueOf(out.get("negotiatedMcpSpecVersion"));
			if (negotiated.compareTo("2026-07-28") >= 0) {
				notify.put("subscriptions.listen",
						unsupported("MCP Java SDK 2.0.0 has no subscriptions/listen; Spring AI exposes none"));
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
							"SyncMcpToolCallbackProvider.getToolCallbacks via DefaultToolCallingManager.resolveToolDefinitions",
							"names", names);
				});
				case "progress" -> attempt(() -> {
					String text = agentText("slow_with_progress", Map.of(), Map.of("progressToken", "mcp-mirror-progress"));
					sleep(300);
					Map<String, Object> answer = new LinkedHashMap<>();
					answer.put("route", "adapter");
					answer.put("api",
							"ToolCallingChatOptions.toolContext(progressToken) copied into _meta by "
									+ "ToolContextToMcpMetaConverter.defaultConverter; @McpProgress(clients = \"src\")");
					answer.put("events", new ArrayList<>(Hooks.PROGRESS));
					answer.put("text", text);
					return answer;
				});
				case "logging" -> attempt(() -> {
					String text = agentText("emit_logs", Map.of(), Map.of());
					sleep(300);
					Map<String, Object> answer = new LinkedHashMap<>();
					answer.put("route", "adapter");
					answer.put("api", "@McpLogging(clients = \"src\")");
					answer.put("messages", new ArrayList<>(Hooks.LOGS));
					answer.put("text", text);
					return answer;
				});
				case "set_level" -> attempt(() -> {
					client.setLoggingLevel(McpSchema.LoggingLevel.WARNING);
					String text = agentText("emit_logs", Map.of(), Map.of());
					Map<String, Object> answer = new LinkedHashMap<>();
					answer.put("route", "session");
					answer.put("api", "McpSyncClient.setLoggingLevel on the bean the starter publishes");
					answer.put("level", "warning");
					answer.put("text", text);
					return answer;
				});
				case "completion" -> attempt(() -> {
					McpSchema.CompleteResult prompt = client.completeCompletion(new McpSchema.CompleteRequest(
							new McpSchema.PromptReference("ref/prompt", "pick_colour", null),
							new McpSchema.CompleteRequest.CompleteArgument("colour", "c"), null, null));
					McpSchema.CompleteResult resource = client.completeCompletion(new McpSchema.CompleteRequest(
							new McpSchema.ResourceReference("file:///fixtures/records/{record_id}"),
							new McpSchema.CompleteRequest.CompleteArgument("record_id", "rec_1"), null, null));
					return Map.of("route", "session", "api",
							"McpSyncClient.completeCompletion on the bean the starter publishes", "prompt",
							prompt.completion().values(), "resource", resource.completion().values());
				});
				case "cancellation" -> attempt(() -> {
					double after = ((Number) spec.getOrDefault("cancel_after_s", 1.0)).doubleValue();
					Object wait = spec.getOrDefault("cancel_wait_s", 4);
					ExecutorService executor = Executors.newSingleThreadExecutor();
					long started = System.nanoTime();
					Future<String> call = executor
						.submit(() -> agentText("cancellable_wait", Map.of("seconds", wait), Map.of()));
					String outcome;
					try {
						outcome = "returned: " + call.get((long) (after * 1000), TimeUnit.MILLISECONDS);
					}
					catch (TimeoutException timeout) {
						call.cancel(true);
						outcome = "abandoned: Future.cancel(true) interrupted the thread blocked in executeToolCalls";
					}
					double abandoned = (System.nanoTime() - started) / 1e9;
					sleep(300);
					executor.shutdownNow();
					Map<String, Object> answer = new LinkedHashMap<>();
					answer.put("route", "adapter");
					answer.put("mechanism",
							"interrupting the caller's thread (Future.cancel(true)); the synchronous ToolCallback API takes no cancellation signal");
					answer.put("abandoned_after_s", abandoned);
					answer.put("outcome", outcome.length() > 200 ? outcome.substring(0, 200) : outcome);
					return answer;
				});
				case "tasks" -> attempt(() -> Map.of("route", "adapter", "text",
						String.valueOf(agentText("task_capable", Map.of(), Map.of()))));
				case "caching" -> attempt(() -> {
					List<Map<String, Object>> marks = new ArrayList<>();
					double pause = ((Number) spec.getOrDefault("cache_pause_s", 2.0)).doubleValue();
					for (String mark : List.of("first", "within_ttl", "after_ttl")) {
						if (mark.equals("after_ttl")) {
							sleep((long) (pause * 1000));
						}
						String startedAt = stamp();
						int count = callbacks().length;
						Map<String, Object> entry = new LinkedHashMap<>();
						entry.put("call", mark);
						entry.put("started", startedAt);
						entry.put("finished", stamp());
						entry.put("count", count);
						marks.add(entry);
					}
					return Map.of("route", "adapter", "api",
							"SyncMcpToolCallbackProvider.getToolCallbacks (cached until McpToolsChangedEvent)", "marks",
							marks);
				});
				default -> unsupported("unknown utility " + feature);
			};
		}

		// -----------------------------------------------------------------
		// Client features against fixtures/client_features_server.py
		// -----------------------------------------------------------------

		Map<String, Object> clientFeatures(List<Object> tools) {
			Map<String, Object> answers = new LinkedHashMap<>();
			Hooks.drain();
			for (Object name : tools) {
				String tool = String.valueOf(name);
				Map<String, Object> entry = new LinkedHashMap<>();
				if (tool.equals("roots_changes_seen")) {
					try {
						client.removeRoot(Hooks.ROOT_URI);
						client.addRoot(new McpSchema.Root(CHANGED_ROOT_URI, "mcp-mirror-changed", null));
						sleep(300);
					}
					catch (Throwable error) {
						entry.put("root_change_error", describe(error));
					}
				}
				entry.putAll(agent(tool, Map.of(), Map.of()));
				entry.put("handler_saw", Hooks.drain());
				answers.put(tool, entry);
			}
			return answers;
		}

	}

}
