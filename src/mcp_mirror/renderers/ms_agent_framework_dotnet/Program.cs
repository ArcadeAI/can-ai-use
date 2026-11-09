// mcp-mirror's Microsoft Agent Framework (.NET) worker.
//
// `dotnet Worker.dll <config.json> <out.json>`. The config names one MCP server; the worker
// connects to it the way the Agent Framework documents for .NET: McpClient.CreateAsync over
// the transport, McpClient.ListToolsAsync for the McpClientTool list (each one an AIFunction),
// and IChatClient.AsAIAgent(tools: ...) for the agent. The agent runs against a scripted
// IChatClient. ChatClientAgent wraps it in FunctionInvokingChatClient, which puts the tools
// in ChatOptions.Tools and invokes each function call. The tool definitions are the
// AIFunctionDeclarations in ChatOptions.Tools; a tool result is the FunctionResultContent in
// the next request's messages, and the value McpClientTool returned is seen by function
// middleware registered with AIAgentBuilder.Use. No model runs and no provider request is
// serialized. Output goes to a file so nothing the stdio server prints can corrupt it.

using System.Net;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Web;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using ModelContextProtocol;
using ModelContextProtocol.Authentication;
using ModelContextProtocol.Client;
using ModelContextProtocol.Protocol;

namespace McpMirror.AgentFrameworkDotnet;

public static class Worker
{
    internal const string CallId = "mcp-mirror-call";
    internal const string WatchedUri = "file:///fixtures/watched.txt";
    internal const string Sampled = "SENTINEL_SAMPLED";
    internal const string SampledAfterTool = "SENTINEL_SAMPLED_AFTER_TOOL";
    internal const string ToolUseId = "SENTINEL_TOOL_USE_ID";
    internal const string Elicited = "SENTINEL_ELICITED";
    internal const string RootUri = "file:///tmp/mcp-mirror-root";
    internal const string ChangedRootUri = "file:///tmp/mcp-mirror-root-changed";

    internal static readonly JsonSerializerOptions Mcp = McpJsonUtilities.DefaultOptions;

    public static async Task<int> Main(string[] args)
    {
        var cfg = JsonNode.Parse(await File.ReadAllTextAsync(args[0]))!.AsObject();
        var output = new JsonObject();
        int status = 0;
        var run = new Run(cfg, output);
        try
        {
            await run.Execute();
        }
        catch (Exception error)
        {
            output["fatal"] = Describe(error);
            status = 2;
        }
        finally
        {
            await run.Close();
        }
        await File.WriteAllTextAsync(args[1], output.ToJsonString());
        return status;
    }

    internal static string Describe(Exception error)
    {
        var text = $"{error.GetType().Name}: {error.Message}";
        var cause = error.InnerException;
        for (int depth = 0; cause is not null && depth < 3; depth++)
        {
            text += $" <- {cause.GetType().Name}: {cause.Message}";
            cause = cause.InnerException;
        }
        return text.Length > 1500 ? text[..1500] : text;
    }

    internal static string Stamp() => DateTime.UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss.ffffff'+00:00'");

    internal static JsonNode? Tree<T>(T value) => value is null ? null : JsonSerializer.SerializeToNode(value, Mcp);

    internal static JsonNode? Parse(JsonElement? element) =>
        element is { } e && e.ValueKind != JsonValueKind.Undefined ? JsonNode.Parse(e.GetRawText()) : null;

    /// <summary>{"ok": true, ...} or the failure, the way src/mcp_mirror/probe.py records it.</summary>
    internal static async Task<JsonObject> Attempt(Func<Task<JsonObject>> body)
    {
        try
        {
            var answer = await body();
            var result = new JsonObject { ["ok"] = true };
            foreach (var (key, value) in answer.ToList())
            {
                answer.Remove(key);
                result[key] = value;
            }
            return result;
        }
        catch (NotSupportedException error)
        {
            return new JsonObject { ["ok"] = false, ["unsupported"] = Describe(error) };
        }
        catch (Exception error)
        {
            return new JsonObject { ["ok"] = false, ["error"] = Describe(error) };
        }
    }

    internal static JsonObject Unsupported(string reason) => new() { ["ok"] = false, ["unsupported"] = reason };

    internal static JsonObject Failed(string reason) => new() { ["ok"] = false, ["error"] = reason };

    /// <summary>One AIContent as plain JSON; binary payloads are summarized, not copied.</summary>
    internal static JsonObject ContentRep(AIContent content)
    {
        JsonObject entry = content switch
        {
            TextContent text => new JsonObject { ["type"] = "text", ["text"] = text.Text },
            DataContent data => new JsonObject
            {
                ["type"] = "data",
                ["mediaType"] = data.MediaType,
                ["name"] = data.Name,
                ["bytes"] = data.Data.Length,
                ["sha256"] = Convert.ToHexStringLower(SHA256.HashData(data.Data.Span)),
            },
            UriContent uri => new JsonObject { ["type"] = "uri", ["uri"] = uri.Uri.ToString(), ["mediaType"] = uri.MediaType },
            _ => new JsonObject
            {
                ["type"] = content.GetType().Name,
                ["json"] = JsonSerializer.SerializeToNode(content, AIJsonUtilities.DefaultOptions),
            },
        };
        entry["class"] = content.GetType().FullName;
        if (content.AdditionalProperties is { Count: > 0 } extra)
        {
            entry["additionalProperties"] = JsonSerializer.SerializeToNode(extra, AIJsonUtilities.DefaultOptions);
        }
        return entry;
    }

    /// <summary>
    /// A function result as plain JSON. McpClientTool.InvokeCoreAsync returns one AIContent,
    /// an AIContent[] or a JsonElement holding the serialized CallToolResult.
    /// </summary>
    internal static JsonNode? ValueRep(object? value) => value switch
    {
        null => null,
        JsonElement element => new JsonObject { ["kind"] = "json", ["value"] = Parse(element) },
        AIContent content => new JsonObject { ["kind"] = "content", ["item"] = ContentRep(content) },
        IEnumerable<AIContent> contents => new JsonObject
        {
            ["kind"] = "contents",
            ["items"] = new JsonArray(contents.Select(c => (JsonNode)ContentRep(c)).ToArray()),
        },
        string text => new JsonObject { ["kind"] = "string", ["value"] = text },
        _ => new JsonObject
        {
            ["kind"] = value.GetType().FullName,
            ["value"] = JsonSerializer.SerializeToNode(value, AIJsonUtilities.DefaultOptions),
        },
    };

    /// <summary>The first text a tool answer carries, whatever the adapter wrapped it in.</summary>
    internal static string? TextOf(object? value)
    {
        switch (value)
        {
            case null:
                return null;
            case TextContent text:
                return text.Text;
            case IEnumerable<AIContent> contents:
                return contents.OfType<TextContent>().Select(t => t.Text).FirstOrDefault();
            case JsonElement element:
                if (element.ValueKind == JsonValueKind.Object && element.TryGetProperty("content", out var blocks)
                    && blocks.ValueKind == JsonValueKind.Array)
                {
                    foreach (var block in blocks.EnumerateArray())
                    {
                        if (block.TryGetProperty("text", out var t) && t.ValueKind == JsonValueKind.String)
                        {
                            return t.GetString();
                        }
                    }
                }
                return element.GetRawText();
            case CallToolResult result:
                return result.Content.OfType<TextContentBlock>().Select(b => b.Text).FirstOrDefault();
            default:
                return value.ToString();
        }
    }
}

/// <summary>
/// An IChatClient that answers from a script. The first turn optionally asks for one function
/// call; every later turn ends the run. It records what each request handed it.
/// </summary>
internal sealed class ScriptedChatClient(string? tool, JsonObject? arguments) : IChatClient
{
    public List<JsonObject> Requests { get; } = [];

    public Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        var declarations = new JsonArray();
        foreach (var t in options?.Tools ?? [])
        {
            var entry = new JsonObject { ["class"] = t.GetType().FullName, ["name"] = t.Name, ["description"] = t.Description };
            if (t is AIFunctionDeclaration d)
            {
                entry["jsonSchema"] = Worker.Parse(d.JsonSchema);
                entry["returnJsonSchema"] = Worker.Parse(d.ReturnJsonSchema);
            }
            if (t.AdditionalProperties is { Count: > 0 } extra)
            {
                entry["additionalProperties"] = JsonSerializer.SerializeToNode(extra, AIJsonUtilities.DefaultOptions);
            }
            declarations.Add(entry);
        }
        var results = new JsonArray();
        foreach (var message in messages)
        {
            foreach (var content in message.Contents.OfType<FunctionResultContent>())
            {
                results.Add(new JsonObject
                {
                    ["callId"] = content.CallId,
                    ["role"] = message.Role.Value,
                    ["result"] = Worker.ValueRep(content.Result),
                    ["exception"] = content.Exception is null ? null : Worker.Describe(content.Exception),
                });
            }
        }
        Requests.Add(new JsonObject { ["declarations"] = declarations, ["results"] = results });
        ChatMessage reply;
        if (tool is not null && Requests.Count == 1)
        {
            var args = new Dictionary<string, object?>();
            foreach (var (key, value) in arguments ?? [])
            {
                args[key] = value is null ? null : JsonSerializer.Deserialize<JsonElement>(value.ToJsonString());
            }
            reply = new ChatMessage(ChatRole.Assistant, [new FunctionCallContent(Worker.CallId, tool, args)]);
        }
        else
        {
            reply = new ChatMessage(ChatRole.Assistant, "done");
        }
        return Task.FromResult(new ChatResponse(reply));
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var response = await GetResponseAsync(messages, options, cancellationToken);
        foreach (var update in response.ToChatResponseUpdates())
        {
            yield return update;
        }
    }

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceKey is null && serviceType.IsInstanceOfType(this) ? this : null;

    public void Dispose()
    {
    }
}

/// <summary>What one agent run showed the model, and what the tool returned to the agent.</summary>
internal sealed record Turn(List<JsonObject> Requests, bool Invoked, object? Direct, Exception? DirectError)
{
    public JsonObject? FunctionResult()
    {
        if (Requests.Count < 2)
        {
            return null;
        }
        var results = Requests[1]["results"]!.AsArray().Select(r => r!.AsObject()).ToList();
        return results.FirstOrDefault(r => (string?)r["callId"] == Worker.CallId) ?? results.LastOrDefault();
    }
}

/// <summary>Warnings and errors the MCP C# SDK logged, kept as evidence for a message that never reached the agent.</summary>
internal sealed class CapturingLoggerFactory : ILoggerFactory
{
    public List<string> Lines { get; } = [];

    public ILogger CreateLogger(string categoryName) => new Logger(this, categoryName);

    public void AddProvider(ILoggerProvider provider)
    {
    }

    public void Dispose()
    {
    }

    private sealed class Logger(CapturingLoggerFactory owner, string category) : ILogger
    {
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => logLevel >= LogLevel.Warning;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
            Func<TState, Exception?, string> formatter)
        {
            if (!IsEnabled(logLevel))
            {
                return;
            }
            var line = $"{category} {logLevel}: {formatter(state, exception)}";
            if (exception is not null)
            {
                line += $" <- {exception.GetType().Name}: {exception.Message}";
            }
            lock (owner.Lines)
            {
                owner.Lines.Add(line.Length > 500 ? line[..500] : line);
            }
        }
    }
}

internal sealed class Run(JsonObject cfg, JsonObject output)
{
    readonly CapturingLoggerFactory logs = new();
    readonly List<JsonObject> serverLogs = [];
    readonly List<JsonObject> handlerSaw = [];
    readonly Dictionary<string, int> heard = new() { ["tools"] = 0, ["resources"] = 0, ["prompts"] = 0 };
    List<Root> roots = [new Root { Uri = Worker.RootUri, Name = "mcp-mirror" }];
    McpClient? client;
    IList<McpClientTool> tools = [];

    string Mode => (string?)cfg["mode"] ?? "tools";

    bool Hooks => (bool?)cfg["hooks"] ?? false;

    public async Task Close()
    {
        if (client is not null)
        {
            try
            {
                await client.DisposeAsync();
            }
            catch
            {
                // closing is best effort
            }
        }
    }

    public async Task Execute()
    {
        output["tools"] = new JsonArray();
        client = await McpClient.CreateAsync(Transport(), Options(), logs);
        tools = await client.ListToolsAsync();
        output["negotiatedMcpSpecVersion"] = client.NegotiatedProtocolVersion;
        var definitions = await TurnAsync(null, null);
        var declarations = definitions.Requests[0]["declarations"]!.DeepClone();
        switch (Mode)
        {
            case "tools":
                output["tools"] = declarations;
                output["retained"] = Retained();
                var results = new JsonArray();
                foreach (var call in cfg["calls"]?.AsArray() ?? [])
                {
                    results.Add(await Result(call!.AsObject()));
                }
                output["results"] = results;
                break;
            case "probe":
                output["probe"] = await Probe();
                break;
            case "notify":
                output["notify"] = await Notify();
                break;
            case "utility":
                output["utility"] = await Utility((string)cfg["utility"]!, cfg["utilitySpec"]?.AsObject() ?? []);
                break;
            case "client":
                output["client"] = await ClientFeatures(cfg["clientCalls"]?.AsArray() ?? []);
                break;
            default:
                throw new ArgumentException($"unknown mode {Mode}");
        }
        var sdk = new JsonArray();
        lock (logs.Lines)
        {
            foreach (var line in logs.Lines)
            {
                sdk.Add(line);
            }
        }
        output["sdkLog"] = sdk;
    }

    // ---------------------------------------------------------------------
    // Connection: the transport and McpClientOptions an application builds
    // ---------------------------------------------------------------------

    IClientTransport Transport()
    {
        var transport = (string?)cfg["transport"] ?? "stdio";
        var headers = new Dictionary<string, string>();
        foreach (var (key, value) in cfg["headers"]?.AsObject() ?? [])
        {
            headers[key] = value?.ToString() ?? "";
        }
        if (transport == "stdio")
        {
            return new StdioClientTransport(new StdioClientTransportOptions
            {
                Name = "mcp-mirror-server",
                Command = (string)cfg["command"]!,
                Arguments = (cfg["args"]?.AsArray() ?? []).Select(a => a!.ToString()).ToList(),
            }, logs);
        }
        if (transport is not ("http" or "sse"))
        {
            throw new ArgumentException($"unknown transport {transport}");
        }
        var options = new HttpClientTransportOptions
        {
            Name = "mcp-mirror-server",
            Endpoint = new Uri((string)cfg["url"]!),
            TransportMode = transport == "http" ? HttpTransportMode.StreamableHttp : HttpTransportMode.Sse,
            AdditionalHeaders = headers,
            ConnectionTimeout = TimeSpan.FromSeconds(30),
        };
        if (cfg["auth"] is JsonObject auth)
        {
            options.OAuth = OAuth(auth);
        }
        return new HttpClientTransport(options, logs);
    }

    /// <summary>
    /// ClientOAuthOptions for an auth/ conformance scenario. The redirect delegate is the
    /// browser leg: it visits the authorization URL and reads the code off the redirect.
    /// </summary>
    static ClientOAuthOptions OAuth(JsonObject auth)
    {
        var options = new ClientOAuthOptions
        {
            RedirectUri = new Uri((string?)auth["redirectUri"] ?? "http://localhost:3000/callback"),
            ClientId = (string?)auth["clientId"],
            ClientSecret = (string?)auth["clientSecret"],
            DynamicClientRegistration = new DynamicClientRegistrationOptions { ClientName = (string?)auth["clientName"] },
            AuthorizationCallbackHandler = async (context, cancellationToken) =>
            {
                using var http = new HttpClient(new HttpClientHandler { AllowAutoRedirect = false });
                using var response = await http.GetAsync(context.AuthorizationUri, cancellationToken);
                var location = response.Headers.Location;
                if (location is null)
                {
                    return null;
                }
                var query = HttpUtility.ParseQueryString(location.Query);
                return new AuthorizationResult { Code = query["code"], State = query["state"], Iss = query["iss"] };
            },
        };
        if ((string?)auth["clientMetadataUrl"] is { } metadata && options.ClientId is null)
        {
            options.ClientMetadataDocumentUri = new Uri(metadata);
        }
        return options;
    }

    McpClientOptions Options()
    {
        var handlers = new McpClientHandlers();
        var notifications = new List<KeyValuePair<string, Func<JsonRpcNotification, CancellationToken, ValueTask>>>
        {
            new(NotificationMethods.LoggingMessageNotification, (notification, _) =>
            {
                var message = notification.Params is null
                    ? null
                    : JsonSerializer.Deserialize<LoggingMessageNotificationParams>(notification.Params, Worker.Mcp);
                lock (serverLogs)
                {
                    serverLogs.Add(new JsonObject
                    {
                        ["level"] = message is null ? null : Worker.Tree(message.Level)?.ToString(),
                        ["data"] = message is null ? null : Worker.Parse(message.Data),
                    });
                }
                return ValueTask.CompletedTask;
            }),
        };
        if (Mode == "notify")
        {
            foreach (var (method, kind) in new[]
            {
                (NotificationMethods.ToolListChangedNotification, "tools"),
                (NotificationMethods.ResourceListChangedNotification, "resources"),
                (NotificationMethods.PromptListChangedNotification, "prompts"),
            })
            {
                notifications.Add(new(method, (_, _) =>
                {
                    lock (heard)
                    {
                        heard[kind]++;
                    }
                    return ValueTask.CompletedTask;
                }));
            }
        }
        handlers.NotificationHandlers = notifications;
        var options = new McpClientOptions { Handlers = handlers };
        if (Mode == "client" && Hooks)
        {
            handlers.SamplingHandler = (request, _, _) => ValueTask.FromResult(Sample(request));
            handlers.ElicitationHandler = (request, _) =>
            {
                var mode = request?.Mode ?? "form";
                Note("McpClientHandlers.ElicitationHandler", new JsonObject { ["mode"] = mode });
                var result = new ElicitResult { Action = "accept" };
                if (mode != "url")
                {
                    result.Content = new Dictionary<string, JsonElement>
                    {
                        ["name"] = JsonSerializer.SerializeToElement(Worker.Elicited),
                    };
                }
                return ValueTask.FromResult(result);
            };
            handlers.RootsHandler = (_, _) =>
            {
                Note("McpClientHandlers.RootsHandler", []);
                return ValueTask.FromResult(new ListRootsResult { Roots = roots.ToList() });
            };
            // The SDK fills sampling, form elicitation and roots in from the handlers;
            // sampling.tools, URL elicitation and roots.listChanged are declared here.
            options.Capabilities = new ClientCapabilities
            {
                Sampling = new SamplingCapability { Tools = new SamplingToolsCapability() },
                Elicitation = new ElicitationCapability
                {
                    Form = new FormElicitationCapability(),
                    Url = new UrlElicitationCapability(),
                },
                Roots = new RootsCapability { ListChanged = true },
            };
        }
        return options;
    }

    void Note(string hook, JsonObject facts)
    {
        facts["hook"] = hook;
        lock (handlerSaw)
        {
            handlerSaw.Add(facts);
        }
    }

    JsonArray DrainSaw()
    {
        lock (handlerSaw)
        {
            var seen = new JsonArray(handlerSaw.Select(s => (JsonNode)s).ToArray());
            handlerSaw.Clear();
            return seen;
        }
    }

    /// <summary>
    /// The stub model's answer (client_probe.sampling_reply): offered tools and no tool
    /// result yet, call the first tool; otherwise answer in text.
    /// </summary>
    CreateMessageResult Sample(CreateMessageRequestParams? request)
    {
        var offered = request?.Tools ?? [];
        bool afterTool = (request?.Messages ?? []).Any(m => m.Content.Any(b => b is ToolResultContentBlock));
        CreateMessageResult reply;
        if (offered.Count > 0 && !afterTool)
        {
            reply = new CreateMessageResult
            {
                Role = Role.Assistant,
                Model = "mcp-mirror-stub",
                StopReason = "toolUse",
                Content =
                [
                    new ToolUseContentBlock
                    {
                        Id = Worker.ToolUseId,
                        Name = offered[0].Name,
                        Input = JsonSerializer.SerializeToElement(new Dictionary<string, string> { ["word"] = "mirror" }),
                    },
                ],
            };
        }
        else
        {
            reply = new CreateMessageResult
            {
                Role = Role.Assistant,
                Model = "mcp-mirror-stub",
                StopReason = "endTurn",
                Content = [new TextContentBlock { Text = afterTool ? Worker.SampledAfterTool : Worker.Sampled }],
            };
        }
        Note("McpClientHandlers.SamplingHandler", new JsonObject { ["tools"] = offered.Count > 0, ["replied"] = reply.StopReason });
        return reply;
    }

    // ---------------------------------------------------------------------
    // The agent path
    // ---------------------------------------------------------------------

    /// <summary>One ChatClientAgent run against the scripted IChatClient.</summary>
    async Task<Turn> TurnAsync(string? tool, JsonObject? arguments, IList<AITool>? agentTools = null,
        CancellationToken cancellationToken = default)
    {
        var model = new ScriptedChatClient(tool, arguments);
        bool invoked = false;
        object? direct = null;
        Exception? directError = null;
        AIAgent agent = model.AsAIAgent(
            instructions: "mcp-mirror probe",
            name: "mcp_mirror",
            tools: agentTools ?? tools.Cast<AITool>().ToList());
        agent = agent.AsBuilder()
            .Use(async (_, context, next, ct) =>
            {
                invoked = true;
                try
                {
                    direct = await next(context, ct);
                    return direct;
                }
                catch (Exception error)
                {
                    directError = error;
                    throw;
                }
            })
            .Build();
        await agent.RunAsync("mcp-mirror", cancellationToken: cancellationToken);
        return new Turn(model.Requests, invoked, direct, directError);
    }

    async Task<string?> AgentText(string tool, JsonObject? arguments = null, IList<AITool>? agentTools = null,
        CancellationToken cancellationToken = default)
    {
        var turn = await TurnAsync(tool, arguments ?? [], agentTools, cancellationToken);
        if (turn.DirectError is not null)
        {
            return Worker.Describe(turn.DirectError);
        }
        return Worker.TextOf(turn.Direct);
    }

    /// <summary>What each McpClientTool keeps beside its AIFunction: the MCP Tool it was built from.</summary>
    JsonObject Retained()
    {
        var kept = new JsonObject();
        foreach (var tool in tools)
        {
            var protocol = Worker.Tree(tool.ProtocolTool)!.AsObject();
            protocol.Remove("inputSchema");
            protocol.Remove("outputSchema");
            kept[tool.Name] = protocol;
        }
        return kept;
    }

    async Task<JsonObject> Result(JsonObject call)
    {
        var tool = (string)call["tool"]!;
        var entry = new JsonObject { ["tool"] = tool };
        try
        {
            var turn = await TurnAsync(tool, call["arguments"]?.AsObject() ?? []);
            var response = turn.FunctionResult();
            if (response is null)
            {
                entry["ok"] = false;
                entry["error"] = "the next request carried no FunctionResultContent";
                return entry;
            }
            entry["ok"] = true;
            entry["agent"] = response.DeepClone();
            entry["invoked"] = turn.Invoked;
            entry["direct"] = Worker.ValueRep(turn.Direct);
            if (turn.DirectError is not null)
            {
                entry["directThrew"] = true;
                entry["exception"] = turn.DirectError.GetType().FullName;
                entry["error"] = Worker.Describe(turn.DirectError);
            }
        }
        catch (Exception error)
        {
            entry["ok"] = false;
            entry["threw"] = true;
            entry["exception"] = error.GetType().FullName;
            entry["error"] = Worker.Describe(error);
        }
        return entry;
    }

    // ---------------------------------------------------------------------
    // Resources and prompts: methods on the McpClient the application created
    // ---------------------------------------------------------------------

    async Task<JsonObject> Probe()
    {
        var mcp = client!;
        var probe = new JsonObject();
        var listed = new List<McpClientResource>();
        probe["resources.list"] = await Worker.Attempt(async () =>
        {
            listed.AddRange(await mcp.ListResourcesAsync());
            var entries = new JsonArray();
            foreach (var r in listed)
            {
                entries.Add(new JsonObject
                {
                    ["uri"] = r.Uri,
                    ["name"] = r.Name,
                    ["title"] = r.Title,
                    ["description"] = r.Description,
                    ["mimeType"] = r.MimeType,
                });
            }
            return new JsonObject { ["route"] = "adapter", ["api"] = "McpClient.ListResourcesAsync", ["entries"] = entries };
        });
        var textUri = listed.FirstOrDefault(r => (r.MimeType ?? "").StartsWith("text/"))?.Uri;
        var blobUri = listed.FirstOrDefault(r => r.MimeType is { Length: > 0 } m && !m.StartsWith("text/"))?.Uri;
        probe["resources.read.text"] = textUri is null ? Worker.Failed("no text resource listed") : await Worker.Attempt(() => Read(textUri));
        probe["resources.read.blob"] = blobUri is null ? Worker.Failed("no binary resource listed") : await Worker.Attempt(() => Read(blobUri));
        probe["resources.templates"] = await Worker.Attempt(async () =>
        {
            var entries = new JsonArray();
            foreach (var t in await mcp.ListResourceTemplatesAsync())
            {
                entries.Add(new JsonObject
                {
                    ["uriTemplate"] = t.UriTemplate,
                    ["name"] = t.Name,
                    ["description"] = t.Description,
                    ["mimeType"] = t.MimeType,
                });
            }
            return new JsonObject { ["route"] = "adapter", ["api"] = "McpClient.ListResourceTemplatesAsync", ["entries"] = entries };
        });
        var prompts = new List<McpClientPrompt>();
        probe["prompts.list"] = await Worker.Attempt(async () =>
        {
            prompts.AddRange(await mcp.ListPromptsAsync());
            var entries = new JsonArray();
            foreach (var p in prompts)
            {
                var arguments = new JsonArray();
                foreach (var a in p.ProtocolPrompt.Arguments ?? [])
                {
                    arguments.Add(new JsonObject { ["name"] = a.Name, ["description"] = a.Description, ["required"] = a.Required });
                }
                entries.Add(new JsonObject
                {
                    ["name"] = p.Name,
                    ["title"] = p.Title,
                    ["description"] = p.Description,
                    ["arguments"] = arguments,
                });
            }
            return new JsonObject { ["route"] = "adapter", ["api"] = "McpClient.ListPromptsAsync", ["entries"] = entries };
        });
        probe["prompts.get"] = prompts.Count == 0 ? Worker.Failed("no prompt listed") : await Worker.Attempt(async () =>
        {
            var first = prompts[0];
            var arguments = new Dictionary<string, object?>();
            foreach (var a in first.ProtocolPrompt.Arguments ?? [])
            {
                if (a.Required == true)
                {
                    arguments[a.Name] = "x";
                }
            }
            var result = await first.GetAsync(arguments);
            var messages = new JsonArray();
            foreach (var m in result.Messages)
            {
                var content = Worker.Tree(m.Content)?.AsObject();
                messages.Add(new JsonObject
                {
                    ["role"] = Worker.Tree(m.Role)?.ToString(),
                    ["type"] = content?["type"]?.DeepClone(),
                    ["text"] = content?["text"]?.DeepClone(),
                });
            }
            return new JsonObject
            {
                ["route"] = "adapter",
                ["api"] = "McpClientPrompt.GetAsync",
                ["description"] = result.Description,
                ["messages"] = messages,
            };
        });
        return probe;
    }

    async Task<JsonObject> Read(string uri)
    {
        var result = await client!.ReadResourceAsync(uri);
        var first = result.Contents[0];
        var answer = new JsonObject
        {
            ["route"] = "adapter",
            ["api"] = "McpClient.ReadResourceAsync",
            ["uri"] = first.Uri,
            ["mimeType"] = first.MimeType,
        };
        if (first is BlobResourceContents blob)
        {
            var bytes = blob.DecodedData.ToArray();
            answer["blob_sha256"] = Convert.ToHexStringLower(SHA256.HashData(bytes));
            answer["blob_bytes"] = bytes.Length;
        }
        else if (first is TextResourceContents text)
        {
            answer["text"] = text.Text;
        }
        return answer;
    }

    // ---------------------------------------------------------------------
    // Catalogue changes against fixtures/notify_server.py
    // ---------------------------------------------------------------------

    async Task<string?> CallRaw(string tool)
    {
        var result = await client!.CallToolAsync(tool, new Dictionary<string, object?>());
        return Worker.TextOf(result);
    }

    async Task<List<string>> Listing(string kind)
    {
        var names = kind switch
        {
            "tools" => (await client!.ListToolsAsync()).Select(t => t.Name).ToList(),
            "resources" => (await client!.ListResourcesAsync()).Select(r => r.Uri).ToList(),
            _ => (await client!.ListPromptsAsync()).Select(p => p.Name).ToList(),
        };
        names.Sort(StringComparer.Ordinal);
        return names;
    }

    static JsonArray Strings(IEnumerable<string?> values) => new(values.Select(v => (JsonNode?)v).ToArray());

    async Task<JsonObject> Notify()
    {
        var notify = new JsonObject { ["negotiated"] = output["negotiatedMcpSpecVersion"]?.DeepClone() };
        var changes = new[]
        {
            ("tools.list_changed", "add_tool", "tools", "SENTINEL_ADDED_TOOL", NotificationMethods.ToolListChangedNotification),
            ("resources.list_changed", "add_resource", "resources", "fixtures/added-", NotificationMethods.ResourceListChangedNotification),
            ("prompts.list_changed", "add_prompt", "prompts", "SENTINEL_ADDED_PROMPT", NotificationMethods.PromptListChangedNotification),
        };
        foreach (var (key, mutator, kind, sentinel, method) in changes)
        {
            notify[key] = await Worker.Attempt(async () =>
            {
                var before = await Listing(kind);
                int heardBefore;
                lock (heard)
                {
                    heardBefore = heard[kind];
                }
                var said = kind == "tools" ? await AgentText(mutator) : await CallRaw(mutator);
                for (int waited = 0; waited < 30; waited++)
                {
                    lock (heard)
                    {
                        if (heard[kind] > heardBefore)
                        {
                            break;
                        }
                    }
                    await Task.Delay(100);
                }
                await Task.Delay(200);
                var after = await Listing(kind);
                bool notified;
                lock (heard)
                {
                    notified = heard[kind] > heardBefore;
                }
                return new JsonObject
                {
                    ["hook"] = $"McpClientOptions.Handlers.NotificationHandlers[{method}]",
                    ["route"] = "adapter",
                    ["notified"] = notified,
                    ["before"] = Strings(before),
                    ["after"] = Strings(after),
                    ["sentinel"] = sentinel,
                    ["server_said"] = said,
                };
            });
        }
        notify["resources.subscribe"] = await Worker.Attempt(async () =>
        {
            int updates = 0;
            var said = new JsonArray();
            var answer = new JsonObject
            {
                ["route"] = "adapter",
                ["api"] = "McpClient.SubscribeToResourceAsync(uri, handler); disposing the returned handle unsubscribes",
                ["subscribed"] = false,
                ["subscribe_error"] = null,
                ["notified"] = false,
                ["unsubscribed"] = false,
                ["unsubscribe_error"] = null,
                ["notified_after_unsubscribe"] = null,
                ["server_said"] = said,
            };
            IAsyncDisposable? subscription = null;
            try
            {
                subscription = await client!.SubscribeToResourceAsync(Worker.WatchedUri, (_, _) =>
                {
                    Interlocked.Increment(ref updates);
                    return ValueTask.CompletedTask;
                });
                answer["subscribed"] = true;
            }
            catch (Exception error)
            {
                answer["subscribe_error"] = Worker.Describe(error);
            }
            said.Add(await CallRaw("update_resource"));
            for (int waited = 0; waited < 30 && Volatile.Read(ref updates) == 0; waited++)
            {
                await Task.Delay(100);
            }
            answer["notified"] = Volatile.Read(ref updates) > 0;
            if (subscription is null)
            {
                return answer;
            }
            try
            {
                await subscription.DisposeAsync();
                answer["unsubscribed"] = true;
            }
            catch (Exception error)
            {
                answer["unsubscribe_error"] = Worker.Describe(error);
                return answer;
            }
            int seen = Volatile.Read(ref updates);
            said.Add(await CallRaw("update_resource"));
            await Task.Delay(1500);
            answer["notified_after_unsubscribe"] = Volatile.Read(ref updates) > seen;
            return answer;
        });
        if (string.CompareOrdinal((string?)output["negotiatedMcpSpecVersion"] ?? "", "2026-07-28") >= 0)
        {
            notify["subscriptions.listen"] = Worker.Unsupported(
                "McpClient in the MCP C# SDK 2.2.0 has no subscriptions/listen method; RequestMethods.SubscriptionsListen "
                + "is only served, and McpClientTool and AsAIAgent take no listen option");
        }
        return notify;
    }

    // ---------------------------------------------------------------------
    // Server utilities, one question per worker run
    // ---------------------------------------------------------------------

    async Task<JsonObject> Utility(string feature, JsonObject spec)
    {
        switch (feature)
        {
            case "tools.list":
                return await Worker.Attempt(async () =>
                {
                    var turn = await TurnAsync(null, null);
                    var names = turn.Requests[0]["declarations"]!.AsArray().Select(d => (string?)d!["name"]);
                    return new JsonObject
                    {
                        ["route"] = "adapter",
                        ["api"] = "McpClient.ListToolsAsync -> AsAIAgent(tools: ...); AIFunctionDeclarations in ChatOptions.Tools",
                        ["names"] = Strings(names),
                    };
                });
            case "progress":
                return await Worker.Attempt(async () =>
                {
                    var events = new List<JsonObject>();
                    var progress = new SyncProgress(value =>
                    {
                        lock (events)
                        {
                            events.Add(new JsonObject
                            {
                                ["progress"] = value.Progress,
                                ["total"] = value.Total,
                                ["message"] = value.Message,
                            });
                        }
                    });
                    var withProgress = tools.Select(t => (AITool)(t.Name == "slow_with_progress" ? t.WithProgress(progress) : t)).ToList();
                    var text = await AgentText("slow_with_progress", null, withProgress);
                    await Task.Delay(300);
                    lock (events)
                    {
                        return new JsonObject
                        {
                            ["route"] = "adapter",
                            ["api"] = "McpClientTool.WithProgress(IProgress<ProgressNotificationValue>) handed to AsAIAgent",
                            ["events"] = new JsonArray(events.Select(e => (JsonNode)e).ToArray()),
                            ["text"] = text,
                        };
                    }
                });
            case "logging":
                return await Worker.Attempt(async () =>
                {
                    lock (serverLogs)
                    {
                        serverLogs.Clear();
                    }
                    var text = await AgentText("emit_logs");
                    await Task.Delay(300);
                    lock (serverLogs)
                    {
                        return new JsonObject
                        {
                            ["route"] = "adapter",
                            ["api"] = "McpClientOptions.Handlers.NotificationHandlers[notifications/message]",
                            ["messages"] = new JsonArray(serverLogs.Select(m => (JsonNode)m.DeepClone()).ToArray()),
                            ["text"] = text,
                        };
                    }
                });
            case "set_level":
                return await Worker.Attempt(async () =>
                {
                    var level = (string?)spec["set_level"] ?? "warning";
                    await client!.SetLoggingLevelAsync(JsonSerializer.Deserialize<LoggingLevel>($"\"{level}\"", Worker.Mcp));
                    var text = await AgentText("emit_logs");
                    return new JsonObject
                    {
                        ["route"] = "adapter",
                        ["api"] = "McpClient.SetLoggingLevelAsync",
                        ["level"] = level,
                        ["text"] = text,
                    };
                });
            case "completion":
                return await Worker.Attempt(async () =>
                {
                    var p = spec["completion_prompt"]!.AsArray().Select(v => (string)v!).ToList();
                    var r = spec["completion_resource"]!.AsArray().Select(v => (string)v!).ToList();
                    var prompt = await client!.CompleteAsync(new PromptReference { Name = p[0] }, p[1], p[2]);
                    var resource = await client!.CompleteAsync(new ResourceTemplateReference { Uri = r[0] }, r[1], r[2]);
                    return new JsonObject
                    {
                        ["route"] = "adapter",
                        ["api"] = "McpClient.CompleteAsync",
                        ["prompt"] = Strings(prompt.Completion.Values),
                        ["resource"] = Strings(resource.Completion.Values),
                    };
                });
            case "cancellation":
                return await Worker.Attempt(async () =>
                {
                    double after = (double?)spec["cancel_after_s"] ?? 1.0;
                    var wait = spec["cancel_wait_s"]?.DeepClone() ?? 4;
                    using var cancel = new CancellationTokenSource();
                    var started = System.Diagnostics.Stopwatch.StartNew();
                    var call = AgentText("cancellable_wait", new JsonObject { ["seconds"] = wait }, null, cancel.Token);
                    cancel.CancelAfter(TimeSpan.FromSeconds(after));
                    string outcome;
                    try
                    {
                        outcome = "returned: " + await call;
                    }
                    catch (OperationCanceledException error)
                    {
                        outcome = $"cancelled: {error.GetType().Name} from AIAgent.RunAsync";
                    }
                    double abandoned = started.Elapsed.TotalSeconds;
                    await Task.Delay(300);
                    return new JsonObject
                    {
                        ["route"] = "adapter",
                        ["mechanism"] = "the CancellationToken passed to AIAgent.RunAsync, which FunctionInvokingChatClient "
                            + "hands to McpClientTool.InvokeAsync and the SDK turns into notifications/cancelled",
                        ["abandoned_after_s"] = abandoned,
                        ["outcome"] = outcome.Length > 200 ? outcome[..200] : outcome,
                    };
                });
            case "tasks":
                return await Worker.Attempt(async () => new JsonObject
                {
                    ["route"] = "adapter",
                    ["text"] = await AgentText("task_capable") ?? "null",
                });
            case "caching":
                return await Worker.Attempt(async () =>
                {
                    var marks = new JsonArray();
                    double pause = (double?)spec["cache_pause_s"] ?? 2.0;
                    foreach (var mark in new[] { "first", "within_ttl", "after_ttl" })
                    {
                        if (mark == "after_ttl")
                        {
                            await Task.Delay(TimeSpan.FromSeconds(pause));
                        }
                        var startedAt = Worker.Stamp();
                        int count = (await client!.ListToolsAsync()).Count;
                        marks.Add(new JsonObject
                        {
                            ["call"] = mark,
                            ["started"] = startedAt,
                            ["finished"] = Worker.Stamp(),
                            ["count"] = count,
                        });
                    }
                    return new JsonObject
                    {
                        ["route"] = "adapter",
                        ["api"] = "McpClient.ListToolsAsync (re-fetches on every call; the SDK documents no internal cache)",
                        ["marks"] = marks,
                    };
                });
            default:
                return Worker.Unsupported($"unknown utility {feature}");
        }
    }

    // ---------------------------------------------------------------------
    // Client features against fixtures/client_features_server.py
    // ---------------------------------------------------------------------

    async Task<JsonObject> ClientFeatures(JsonArray calls)
    {
        var answers = new JsonObject();
        DrainSaw();
        foreach (var name in calls)
        {
            var tool = name!.ToString();
            var entry = new JsonObject();
            if (tool == "roots_changes_seen" && Hooks)
            {
                try
                {
                    // The RootsHandler answers from this list; the application tells the
                    // server it changed with the roots list_changed notification.
                    roots = [new Root { Uri = Worker.ChangedRootUri, Name = "mcp-mirror-changed" }];
                    await client!.SendNotificationAsync(NotificationMethods.RootsListChangedNotification);
                    await Task.Delay(300);
                }
                catch (Exception error)
                {
                    entry["root_change_error"] = Worker.Describe(error);
                }
            }
            var result = await Result(new JsonObject { ["tool"] = tool, ["arguments"] = new JsonObject() });
            foreach (var (key, value) in result.ToList())
            {
                result.Remove(key);
                entry[key] = value;
            }
            entry["handler_saw"] = DrainSaw();
            answers[tool] = entry;
        }
        return answers;
    }
}

/// <summary>IProgress that reports on the caller's thread, so no event is lost to a thread-pool post.</summary>
internal sealed class SyncProgress(Action<ProgressNotificationValue> report) : IProgress<ProgressNotificationValue>
{
    public void Report(ProgressNotificationValue value) => report(value);
}
