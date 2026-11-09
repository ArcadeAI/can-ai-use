package dev.mcpmirror.springai;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import io.modelcontextprotocol.client.McpClient;
import io.modelcontextprotocol.spec.McpSchema;

import org.springframework.ai.mcp.annotation.McpElicitation;
import org.springframework.ai.mcp.annotation.McpLogging;
import org.springframework.ai.mcp.annotation.McpProgress;
import org.springframework.ai.mcp.annotation.McpPromptListChanged;
import org.springframework.ai.mcp.annotation.McpResourceListChanged;
import org.springframework.ai.mcp.annotation.McpSampling;
import org.springframework.ai.mcp.annotation.McpToolListChanged;
import org.springframework.ai.mcp.annotation.spring.ClientMcpSyncHandlersRegistry;
import org.springframework.ai.mcp.customizer.McpClientCustomizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

/**
 * The hooks a Spring AI application registers to answer the server, active only under
 * the {@code hooks} profile so scan and result runs see the starter's defaults.
 *
 * Sampling, form elicitation, logging, progress and the three list_changed
 * notifications use the starter's own annotations, keyed by connection name. The
 * starter has no annotation for roots, URL elicitation or resources/updated; Spring
 * AI's documented route for those is an {@code McpClientCustomizer<McpClient.SyncSpec>}
 * bean, which hands over the MCP SDK's own builder.
 */
final class Hooks {

	static final String CONNECTION = "src";

	static final String SAMPLED = "SENTINEL_SAMPLED";

	static final String ELICITED = "SENTINEL_ELICITED";

	static final String ROOT_URI = "file:///tmp/mcp-mirror-root";

	/** What reached each hook, in call order, for the probe's handler_saw. */
	static final List<Map<String, Object>> SEEN = Collections.synchronizedList(new ArrayList<>());

	/** Notification counts by kind: tools, resources, prompts, updated. */
	static final Map<String, Integer> COUNTS = Collections.synchronizedMap(new LinkedHashMap<>());

	static final List<Map<String, Object>> PROGRESS = Collections.synchronizedList(new ArrayList<>());

	static final List<Map<String, Object>> LOGS = Collections.synchronizedList(new ArrayList<>());

	private Hooks() {
	}

	static void note(String hook, Map<String, Object> facts) {
		Map<String, Object> entry = new LinkedHashMap<>();
		entry.put("hook", hook);
		entry.putAll(facts);
		SEEN.add(entry);
	}

	static void count(String kind) {
		COUNTS.merge(kind, 1, Integer::sum);
	}

	static int seen(String kind) {
		return COUNTS.getOrDefault(kind, 0);
	}

	static List<Map<String, Object>> drain() {
		synchronized (SEEN) {
			List<Map<String, Object>> out = new ArrayList<>(SEEN);
			SEEN.clear();
			return out;
		}
	}

	@Component
	@Profile("hooks")
	static class Annotated {

		@McpSampling(clients = CONNECTION)
		public McpSchema.CreateMessageResult sampling(McpSchema.CreateMessageRequest request) {
			// McpSchema.CreateMessageRequest has no tools field in MCP Java SDK 2.0.0, so
			// the handler can only answer in text.
			note("@McpSampling", Map.of("messages", request.messages().size()));
			return McpSchema.CreateMessageResult.builder(McpSchema.Role.ASSISTANT, SAMPLED, "mcp-mirror-stub")
				.stopReason(McpSchema.CreateMessageResult.StopReason.END_TURN)
				.build();
		}

		@McpElicitation(clients = CONNECTION)
		public McpSchema.ElicitResult elicitation(McpSchema.ElicitRequest request) {
			note("@McpElicitation", Map.of("mode", request instanceof McpSchema.ElicitUrlRequest ? "url" : "form"));
			return new McpSchema.ElicitResult(McpSchema.ElicitResult.Action.ACCEPT, Map.of("name", ELICITED), null);
		}

		@McpLogging(clients = CONNECTION)
		public void logging(McpSchema.LoggingMessageNotification notification) {
			Map<String, Object> entry = new LinkedHashMap<>();
			entry.put("level", notification.level() == null ? null : notification.level().name().toLowerCase());
			entry.put("data", notification.data());
			LOGS.add(entry);
		}

		@McpProgress(clients = CONNECTION)
		public void progress(McpSchema.ProgressNotification notification) {
			Map<String, Object> entry = new LinkedHashMap<>();
			entry.put("progress", notification.progress());
			entry.put("total", notification.total());
			entry.put("message", notification.message());
			PROGRESS.add(entry);
		}

		@McpToolListChanged(clients = CONNECTION)
		public void toolsChanged(List<McpSchema.Tool> tools) {
			count("tools");
		}

		@McpResourceListChanged(clients = CONNECTION)
		public void resourcesChanged(List<McpSchema.Resource> resources) {
			count("resources");
		}

		@McpPromptListChanged(clients = CONNECTION)
		public void promptsChanged(List<McpSchema.Prompt> prompts) {
			count("prompts");
		}

	}

	@Configuration
	@Profile("hooks")
	static class Customized {

		/**
		 * Roots, URL elicitation and resources/updated through the SDK builder. Setting
		 * roots on the builder declares nothing, so the capabilities are rebuilt: what
		 * the annotation registry declared, plus roots and URL elicitation.
		 */
		@Bean
		McpClientCustomizer<McpClient.SyncSpec> mirrorSpecCustomizer(ClientMcpSyncHandlersRegistry registry) {
			return (name, spec) -> {
				McpSchema.ClientCapabilities declared = registry.getCapabilities(name);
				McpSchema.ClientCapabilities.Builder capabilities = McpSchema.ClientCapabilities.builder()
					.roots(true);
				if (declared.sampling() != null) {
					capabilities.sampling();
				}
				capabilities.elicitation(true, true);
				spec.roots(new McpSchema.Root(ROOT_URI, "mcp-mirror", null))
					.urlElicitation(request -> {
						note("McpClientCustomizer urlElicitation", Map.of("mode", "url"));
						return new McpSchema.ElicitResult(McpSchema.ElicitResult.Action.ACCEPT, null, null);
					})
					.resourcesUpdateConsumer(contents -> count("updated"))
					.capabilities(capabilities.build());
			};
		}

	}

}
