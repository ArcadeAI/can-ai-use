package dev.mcpmirror.adkjava;

import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.LoggerContext;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.adk.JsonBaseModel;
import com.google.adk.agents.LlmAgent;
import com.google.adk.models.BaseLlm;
import com.google.adk.models.BaseLlmConnection;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.adk.runner.InMemoryRunner;
import com.google.adk.sessions.Session;
import com.google.adk.tools.BaseTool;
import com.google.adk.tools.mcp.McpTool;
import com.google.adk.tools.mcp.McpToolset;
import com.google.adk.tools.mcp.SseServerParameters;
import com.google.adk.tools.mcp.StreamableHttpServerParameters;
import com.google.genai.types.Content;
import com.google.genai.types.FunctionCall;
import com.google.genai.types.FunctionDeclaration;
import com.google.genai.types.FunctionResponse;
import com.google.genai.types.Part;
import io.modelcontextprotocol.client.McpSyncClient;
import io.modelcontextprotocol.client.transport.ServerParameters;
import io.modelcontextprotocol.spec.McpSchema;
import io.reactivex.rxjava3.core.Flowable;
import org.slf4j.LoggerFactory;

/**
 * mcp-mirror's Google ADK (Java) worker.
 *
 * <p>{@code java -jar worker.jar <config.json> <out.json>}. The config names one MCP server; the
 * worker builds ADK's real {@link McpToolset} for it and drives it the way an application does:
 * an {@link LlmAgent} gets the toolset in {@code tools} and an {@link InMemoryRunner} runs it
 * against a scripted {@link BaseLlm}. ADK loads the MCP tools itself (McpToolset.getTools ->
 * McpTool) and each McpTool adds its FunctionDeclaration to the LlmRequest in
 * BaseTool.processLlmRequest. The tool definitions are the functionDeclarations in the
 * LlmRequest the model receives; a tool result is the functionResponse part in the next
 * LlmRequest. No Gemini model runs and no provider request is serialized. Output goes to a file
 * so nothing the stdio server or a logger prints can corrupt it.
 */
public final class Worker {

	static final ObjectMapper JSON = JsonBaseModel.getMapper();

	static final String APP = "mcp-mirror";

	static final String USER = "mcp-mirror";

	static final String CALL_ID = "mcp-mirror-call";

	static final String WATCHED_URI = "file:///fixtures/watched.txt";

	static final DateTimeFormatter STAMP = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSSSS'+00:00'")
		.withZone(ZoneOffset.UTC);

	/** What McpServerLogConsumer, the logging consumer ADK installs on every session, logged. */
	static final ListAppender<ILoggingEvent> SERVER_LOGS = new ListAppender<>();

	/** Warnings and errors the MCP Java SDK logged, kept as evidence for a message that never reached ADK. */
	static final ListAppender<ILoggingEvent> SDK_LOGS = new ListAppender<>();

	private Worker() {
	}

	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws Exception {
		quietLogging();
		Map<String, Object> cfg = JSON.readValue(Path.of(args[0]).toFile(), Map.class);
		Path outPath = Path.of(args[1]);
		Map<String, Object> out = new LinkedHashMap<>();
		int status = 0;
		McpToolset toolset = toolset(connection(cfg));
		try {
			new Run(toolset, cfg, out).run();
		}
		catch (Throwable error) {
			out.put("fatal", describe(error));
			status = 2;
		}
		finally {
			try {
				toolset.close();
			}
			catch (Throwable ignored) {
				// closing is best effort
			}
		}
		Files.writeString(outPath, JSON.writeValueAsString(out));
		// The stdio transport and Reactor leave non-daemon threads behind.
		System.exit(status);
	}

	/** Everything off except the logger ADK routes MCP server log messages through. */
	static void quietLogging() {
		LoggerContext context = (LoggerContext) LoggerFactory.getILoggerFactory();
		context.reset();
		context.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME).setLevel(Level.OFF);
		Logger server = context.getLogger("com.google.adk.tools.mcp.McpServerLogConsumer");
		server.setLevel(Level.ALL);
		server.setAdditive(false);
		SERVER_LOGS.setContext(context);
		SERVER_LOGS.start();
		server.addAppender(SERVER_LOGS);
		Logger sdk = context.getLogger("io.modelcontextprotocol");
		sdk.setLevel(Level.WARN);
		sdk.setAdditive(false);
		SDK_LOGS.setContext(context);
		SDK_LOGS.start();
		sdk.addAppender(SDK_LOGS);
	}

	/** McpToolset has one constructor per connection-parameter type. */
	static McpToolset toolset(Object connection) {
		if (connection instanceof ServerParameters stdio) {
			return new McpToolset(stdio);
		}
		if (connection instanceof StreamableHttpServerParameters http) {
			return new McpToolset(http);
		}
		return new McpToolset((SseServerParameters) connection);
	}

	/** The connection parameters McpToolset takes for each transport. */
	@SuppressWarnings("unchecked")
	static Object connection(Map<String, Object> cfg) {
		String transport = String.valueOf(cfg.getOrDefault("transport", "stdio"));
		Map<String, String> headers = new LinkedHashMap<>();
		if (cfg.get("headers") instanceof Map<?, ?> map) {
			map.forEach((k, v) -> headers.put(String.valueOf(k), String.valueOf(v)));
		}
		switch (transport) {
			case "stdio" -> {
				List<String> argv = new ArrayList<>();
				for (Object arg : (List<Object>) cfg.getOrDefault("args", List.of())) {
					argv.add(String.valueOf(arg));
				}
				return ServerParameters.builder(String.valueOf(cfg.get("command"))).args(argv).build();
			}
			case "http" -> {
				return StreamableHttpServerParameters.builder()
					.url(String.valueOf(cfg.get("url")))
					.headers(headers)
					.timeout(Duration.ofSeconds(30))
					.readTimeout(Duration.ofSeconds(120))
					.build();
			}
			case "sse" -> {
				URI uri = URI.create(String.valueOf(cfg.get("url")));
				String path = uri.getRawPath() == null || uri.getRawPath().isEmpty() ? "/sse" : uri.getRawPath();
				if (uri.getRawQuery() != null) {
					path += "?" + uri.getRawQuery();
				}
				return SseServerParameters.builder()
					.url(uri.getScheme() + "://" + uri.getRawAuthority())
					.sseEndpoint(path)
					.headers(new LinkedHashMap<>(headers))
					.timeout(Duration.ofSeconds(30))
					.sseReadTimeout(Duration.ofSeconds(120))
					.build();
			}
			default -> throw new IllegalArgumentException("unknown transport " + transport);
		}
	}

	static String describe(Throwable error) {
		StringBuilder text = new StringBuilder(error.getClass().getSimpleName() + ": " + error.getMessage());
		Throwable cause = error.getCause();
		for (int depth = 0; cause != null && cause != error && depth < 3; depth++) {
			text.append(" <- ").append(cause.getClass().getSimpleName()).append(": ").append(cause.getMessage());
			cause = cause.getCause();
		}
		return text.length() > 1500 ? text.substring(0, 1500) : text.toString();
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
		return value == null ? null : JSON.convertValue(value, Map.class);
	}

	/** A genai type as plain JSON, through its own serializer. */
	@SuppressWarnings("unchecked")
	static Map<String, Object> genai(com.google.genai.JsonSerializable value) throws Exception {
		return JSON.readValue(value.toJson(), Map.class);
	}

	/** The first text a tool answer carries, whatever ADK wrapped it in. */
	static String textOf(Object returned) {
		if (returned == null) {
			return null;
		}
		if (returned instanceof Map<?, ?> map) {
			if (map.get("text_output") instanceof List<?> outputs) {
				for (Object item : outputs) {
					if (item instanceof Map<?, ?> entry) {
						if (entry.get("text") instanceof String text) {
							return text;
						}
						try {
							return JSON.writeValueAsString(entry);
						}
						catch (Exception ignored) {
							return String.valueOf(entry);
						}
					}
				}
			}
			if (map.get("error") instanceof String error) {
				return error;
			}
		}
		if (returned instanceof McpSchema.CallToolResult result) {
			for (McpSchema.Content content : result.content()) {
				if (content instanceof McpSchema.TextContent text) {
					return text.text();
				}
			}
		}
		return String.valueOf(returned);
	}

	/**
	 * A BaseLlm that answers from a script. The first turn optionally asks for one tool call;
	 * every later turn ends the run. It records each LlmRequest it is handed.
	 */
	static final class ScriptedLlm extends BaseLlm {

		final Map<String, Object> call;

		final List<Map<String, Object>> requests = new ArrayList<>();

		ScriptedLlm(Map<String, Object> call) {
			super("mcp-mirror-scripted");
			this.call = call;
		}

		@Override
		@SuppressWarnings("unchecked")
		public Flowable<LlmResponse> generateContent(LlmRequest request, boolean stream) {
			return Flowable.defer(() -> {
				List<Map<String, Object>> declarations = new ArrayList<>();
				for (com.google.genai.types.Tool tool : request.config()
					.flatMap(config -> config.tools())
					.orElse(List.of())) {
					for (FunctionDeclaration declaration : tool.functionDeclarations().orElse(List.of())) {
						declarations.add(genai(declaration));
					}
				}
				List<Map<String, Object>> responses = new ArrayList<>();
				for (Content content : request.contents()) {
					for (Part part : content.parts().orElse(List.of())) {
						Optional<FunctionResponse> response = part.functionResponse();
						if (response.isPresent()) {
							responses.add(genai(response.get()));
						}
					}
				}
				Map<String, Object> seen = new LinkedHashMap<>();
				seen.put("declarations", declarations);
				seen.put("responses", responses);
				requests.add(seen);
				Part part;
				if (call != null && requests.size() == 1) {
					part = Part.builder()
						.functionCall(FunctionCall.builder()
							.id(CALL_ID)
							.name(String.valueOf(call.get("tool")))
							.args((Map<String, Object>) call.getOrDefault("arguments", Map.of()))
							.build())
						.build();
				}
				else {
					part = Part.fromText("done");
				}
				return Flowable.just(LlmResponse.builder()
					.content(Content.builder().role("model").parts(List.of(part)).build())
					.build());
			});
		}

		@Override
		public BaseLlmConnection connect(LlmRequest request) {
			throw new UnsupportedOperationException("the scripted model only answers generateContent");
		}

	}

	/** The requests one agent run showed the model, and what the tool returned to ADK. */
	record Turn(List<Map<String, Object>> requests, Object direct) {

		@SuppressWarnings("unchecked")
		Map<String, Object> functionResponse() {
			if (requests.size() < 2) {
				return null;
			}
			List<Map<String, Object>> responses = (List<Map<String, Object>>) requests.get(1).get("responses");
			for (Map<String, Object> response : responses) {
				if (CALL_ID.equals(response.get("id"))) {
					return response;
				}
			}
			return responses.isEmpty() ? null : responses.get(responses.size() - 1);
		}

	}

	static final class Run {

		final McpToolset toolset;

		final Map<String, Object> cfg;

		final Map<String, Object> out;

		Run(McpToolset toolset, Map<String, Object> cfg, Map<String, Object> out) {
			this.toolset = toolset;
			this.cfg = cfg;
			this.out = out;
		}

		@SuppressWarnings("unchecked")
		void run() throws Exception {
			String mode = String.valueOf(cfg.getOrDefault("mode", "tools"));
			out.put("tools", List.of());
			Turn definitions = turn(null);
			List<Map<String, Object>> declarations = (List<Map<String, Object>>) definitions.requests()
				.get(0)
				.get("declarations");
			McpSyncClient session = session();
			McpSchema.InitializeResult init = session.getCurrentInitializationResult();
			out.put("negotiatedMcpSpecVersion", init == null ? null : init.protocolVersion());
			out.put("clientInfo", tree(session.getClientInfo()));
			out.put("clientCapabilities", tree(session.getClientCapabilities()));
			switch (mode) {
				case "tools" -> {
					out.put("tools", declarations);
					out.put("retained", retained());
					List<Map<String, Object>> results = new ArrayList<>();
					for (Object call : (List<Object>) cfg.getOrDefault("calls", List.of())) {
						results.add(result((Map<String, Object>) call));
					}
					out.put("results", results);
				}
				case "probe" -> out.put("probe", probe(session));
				case "notify" -> out.put("notify", notifyProbe(session));
				case "utility" -> out.put("utility", utility(String.valueOf(cfg.get("utility")),
						(Map<String, Object>) cfg.getOrDefault("utilitySpec", Map.of()), session));
				default -> throw new IllegalArgumentException("unknown mode " + mode);
			}
		}

		List<BaseTool> tools() {
			return toolset.getTools(null).toList().blockingGet();
		}

		/**
		 * The McpSyncClient behind the toolset, through the public McpTool.getMcpSession.
		 * McpToolset keeps its own session private; every McpTool it hands out carries it.
		 */
		McpSyncClient session() {
			for (BaseTool tool : tools()) {
				if (tool instanceof McpTool mcp) {
					return mcp.getMcpSession();
				}
			}
			throw new IllegalStateException("McpToolset produced no McpTool to reach its session through");
		}

		/** What each McpTool keeps beside its declaration: MCP annotations and _meta. */
		Map<String, Object> retained() {
			Map<String, Object> kept = new LinkedHashMap<>();
			for (BaseTool tool : tools()) {
				if (tool instanceof McpTool mcp) {
					Map<String, Object> entry = new LinkedHashMap<>();
					entry.put("annotations", tree(mcp.annotations()));
					entry.put("meta", mcp.meta());
					kept.put(mcp.name(), entry);
				}
			}
			return kept;
		}

		/** One LlmAgent run through InMemoryRunner against the scripted model. */
		Turn turn(Map<String, Object> call) {
			ScriptedLlm model = new ScriptedLlm(call);
			Object[] direct = { null };
			LlmAgent agent = LlmAgent.builder()
				.name("mcp_mirror")
				.model(model)
				.instruction("mcp-mirror probe")
				.tools(toolset)
				.afterToolCallbackSync((context, tool, input, toolContext, response) -> {
					direct[0] = response;
					return Optional.empty();
				})
				.build();
			InMemoryRunner runner = new InMemoryRunner(agent, APP);
			Session session = runner.sessionService().createSession(APP, USER, (Map<String, Object>) null, null)
				.blockingGet();
			runner.runAsync(USER, session.id(), Content.fromParts(Part.fromText("mcp-mirror"))).blockingSubscribe();
			return new Turn(model.requests, direct[0]);
		}

		Turn turn(String tool, Map<String, Object> arguments) {
			Map<String, Object> call = new LinkedHashMap<>();
			call.put("tool", tool);
			call.put("arguments", arguments);
			return turn(call);
		}

		@SuppressWarnings("unchecked")
		Map<String, Object> result(Map<String, Object> call) {
			Map<String, Object> entry = new LinkedHashMap<>();
			String tool = String.valueOf(call.get("tool"));
			entry.put("tool", tool);
			try {
				Turn run = turn(tool, (Map<String, Object>) call.getOrDefault("arguments", Map.of()));
				Map<String, Object> response = run.functionResponse();
				if (response == null) {
					entry.put("ok", false);
					entry.put("error", "the next LlmRequest carried no functionResponse");
				}
				else {
					entry.put("ok", true);
					entry.put("agent", response);
					entry.put("direct", run.direct());
				}
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
			return textOf(turn(tool, arguments).direct());
		}

		// -----------------------------------------------------------------
		// Resources and prompts. McpToolset wraps only tools; the rest is reached on the
		// McpSyncClient McpTool.getMcpSession returns.
		// -----------------------------------------------------------------

		Map<String, Object> probe(McpSyncClient client) {
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
				return Map.of("route", "session", "entries", entries);
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
					: attempt(read(client, textUri)));
			probe.put("resources.read.blob", blobUri == null ? Map.of("ok", false, "error", "no binary resource listed")
					: attempt(read(client, blobUri)));
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
				return Map.of("route", "session", "entries", entries);
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
				return Map.of("route", "session", "entries", entries);
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
					e.put("type", content == null ? null : content.get("type"));
					e.put("text", content == null ? null : content.get("text"));
					messages.add(e);
				}
				Map<String, Object> answer = new LinkedHashMap<>();
				answer.put("route", "session");
				answer.put("description", result.description());
				answer.put("messages", messages);
				return answer;
			}));
			return probe;
		}

		Body read(McpSyncClient client, String uri) {
			return () -> {
				McpSchema.ReadResourceResult result = client.readResource(new McpSchema.ReadResourceRequest(uri, null));
				McpSchema.ResourceContents first = result.contents().get(0);
				Map<String, Object> answer = new LinkedHashMap<>();
				answer.put("route", "session");
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
		// Catalogue changes against fixtures/notify_server.py. McpSessionManager builds the
		// McpSyncClient with no change or resource-update consumers, and McpToolset takes
		// none, so every change is read only from its next listing.
		// -----------------------------------------------------------------

		String callRaw(McpSyncClient client, String tool) {
			return textOf(client.callTool(new McpSchema.CallToolRequest(tool, Map.of(), null)));
		}

		List<String> listing(McpSyncClient client, String kind) {
			List<String> names = new ArrayList<>();
			switch (kind) {
				case "tools" -> tools().forEach(t -> names.add(t.name()));
				case "resources" -> client.listResources().resources().forEach(r -> names.add(r.uri()));
				default -> client.listPrompts().prompts().forEach(p -> names.add(p.name()));
			}
			names.sort(null);
			return names;
		}

		Map<String, Object> notifyProbe(McpSyncClient client) {
			Map<String, Object> notify = new LinkedHashMap<>();
			notify.put("negotiated", out.get("negotiatedMcpSpecVersion"));
			String[][] changes = { { "tools.list_changed", "add_tool", "tools", "SENTINEL_ADDED_TOOL" },
					{ "resources.list_changed", "add_resource", "resources", "fixtures/added-" },
					{ "prompts.list_changed", "add_prompt", "prompts", "SENTINEL_ADDED_PROMPT" } };
			for (String[] change : changes) {
				notify.put(change[0], attempt(() -> {
					List<String> before = listing(client, change[2]);
					String said = change[2].equals("tools") ? agentText(change[1], Map.of()) : callRaw(client, change[1]);
					sleep(1000);
					List<String> after = listing(client, change[2]);
					Map<String, Object> answer = new LinkedHashMap<>();
					answer.put("hook", null);
					answer.put("notified", false);
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
						"McpTool.getMcpSession().subscribeResource / unsubscribeResource; McpSessionManager builds the "
								+ "client with no resourcesUpdateConsumer, so an update has nowhere to arrive");
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
				said.add(callRaw(client, "update_resource"));
				sleep(1000);
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
				said.add(callRaw(client, "update_resource"));
				answer.put("notified_after_unsubscribe", false);
				return answer;
			}));
			String negotiated = String.valueOf(out.get("negotiatedMcpSpecVersion"));
			if (negotiated.compareTo("2026-07-28") >= 0) {
				notify.put("subscriptions.listen",
						unsupported("neither McpToolset nor the MCP Java SDK 2.0.0 client opens subscriptions/listen"));
			}
			return notify;
		}

		// -----------------------------------------------------------------
		// Server utilities, one question per worker run
		// -----------------------------------------------------------------

		@SuppressWarnings("unchecked")
		Map<String, Object> utility(String feature, Map<String, Object> spec, McpSyncClient client) {
			return switch (feature) {
				case "tools.list" -> attempt(() -> {
					List<String> names = new ArrayList<>();
					for (Map<String, Object> declaration : (List<Map<String, Object>>) turn(null).requests()
						.get(0)
						.get("declarations")) {
						names.add(String.valueOf(declaration.get("name")));
					}
					return Map.of("route", "adapter", "api",
							"LlmAgent.builder().tools(McpToolset) functionDeclarations in the LlmRequest", "names", names);
				});
				case "progress" -> unsupported(
						"McpSessionManager builds the McpSyncClient with no progressConsumer, McpToolset and McpTool take "
								+ "none, and McpTool.runAsync sends no progressToken");
				case "logging" -> attempt(() -> {
					SERVER_LOGS.list.clear();
					SDK_LOGS.list.clear();
					String text = agentText("emit_logs", Map.of());
					sleep(300);
					List<Map<String, Object>> messages = new ArrayList<>();
					for (ILoggingEvent event : new ArrayList<>(SERVER_LOGS.list)) {
						Map<String, Object> message = new LinkedHashMap<>();
						message.put("level", event.getLevel().toString().toLowerCase());
						message.put("data", event.getFormattedMessage());
						messages.add(message);
					}
					Map<String, Object> answer = new LinkedHashMap<>();
					answer.put("route", "adapter");
					answer.put("api",
							"McpServerLogConsumer, the loggingConsumer McpSessionManager installs, logs each message's data "
									+ "to SLF4J logger com.google.adk.tools.mcp.McpServerLogConsumer");
					answer.put("messages", messages);
					List<String> sdkErrors = new ArrayList<>();
					for (ILoggingEvent event : new ArrayList<>(SDK_LOGS.list)) {
						String line = event.getLoggerName() + " " + event.getLevel() + ": " + event.getFormattedMessage();
						if (event.getThrowableProxy() != null) {
							line += " <- " + event.getThrowableProxy().getClassName() + ": "
									+ event.getThrowableProxy().getMessage();
						}
						sdkErrors.add(line.length() > 500 ? line.substring(0, 500) : line);
					}
					answer.put("sdk_log", sdkErrors);
					answer.put("text", text);
					return answer;
				});
				case "set_level" -> attempt(() -> {
					String level = String.valueOf(spec.getOrDefault("set_level", "warning"));
					client.setLoggingLevel(McpSchema.LoggingLevel.valueOf(level.toUpperCase()));
					String text = agentText("emit_logs", Map.of());
					Map<String, Object> answer = new LinkedHashMap<>();
					answer.put("route", "session");
					answer.put("api", "McpTool.getMcpSession().setLoggingLevel");
					answer.put("level", level);
					answer.put("text", text);
					return answer;
				});
				case "completion" -> attempt(() -> {
					List<Object> p = (List<Object>) spec.get("completion_prompt");
					List<Object> r = (List<Object>) spec.get("completion_resource");
					McpSchema.CompleteResult prompt = client.completeCompletion(new McpSchema.CompleteRequest(
							new McpSchema.PromptReference("ref/prompt", String.valueOf(p.get(0)), null),
							new McpSchema.CompleteRequest.CompleteArgument(String.valueOf(p.get(1)), String.valueOf(p.get(2))),
							null, null));
					McpSchema.CompleteResult resource = client.completeCompletion(new McpSchema.CompleteRequest(
							new McpSchema.ResourceReference(String.valueOf(r.get(0))),
							new McpSchema.CompleteRequest.CompleteArgument(String.valueOf(r.get(1)), String.valueOf(r.get(2))),
							null, null));
					return Map.of("route", "session", "api", "McpTool.getMcpSession().completeCompletion", "prompt",
							prompt.completion().values(), "resource", resource.completion().values());
				});
				case "cancellation" -> attempt(() -> {
					double after = ((Number) spec.getOrDefault("cancel_after_s", 1.0)).doubleValue();
					Object wait = spec.getOrDefault("cancel_wait_s", 4);
					ExecutorService executor = Executors.newSingleThreadExecutor();
					long started = System.nanoTime();
					Future<String> call = executor.submit(() -> agentText("cancellable_wait", Map.of("seconds", wait)));
					String outcome;
					try {
						outcome = "returned: " + call.get((long) (after * 1000), TimeUnit.MILLISECONDS);
					}
					catch (TimeoutException timeout) {
						call.cancel(true);
						outcome = "abandoned: Future.cancel(true) interrupted the thread blocked in Runner.runAsync";
					}
					double abandoned = (System.nanoTime() - started) / 1e9;
					sleep(300);
					executor.shutdownNow();
					Map<String, Object> answer = new LinkedHashMap<>();
					answer.put("route", "adapter");
					answer.put("mechanism",
							"interrupting the thread blocked on the run (Future.cancel(true)); McpTool.runAsync calls the "
									+ "blocking McpSyncClient.callTool and RunConfig carries no cancellation signal");
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
						int count = tools().size();
						Map<String, Object> entry = new LinkedHashMap<>();
						entry.put("call", mark);
						entry.put("started", startedAt);
						entry.put("finished", stamp());
						entry.put("count", count);
						marks.add(entry);
					}
					return Map.of("route", "adapter", "api", "McpToolset.getTools (lists on every call; no cache option)",
							"marks", marks);
				});
				default -> unsupported("unknown utility " + feature);
			};
		}

	}

}
