// mcp-mirror's Google ADK (Go) worker.
//
// `worker <config.json> <out.json>`, or `worker --versions`. The config names one MCP
// server; the worker builds ADK's real tool/mcptoolset Toolset for it and drives it the
// way an application does: an llmagent gets the toolset in Config.Toolsets and
// runner.NewInMemory runs it against a scripted model.LLM. ADK lists the MCP tools
// itself (Toolset.Tools -> mcpTool) and adds each FunctionDeclaration to the
// LLMRequest. The tool definitions are the functionDeclarations in the LLMRequest the
// model receives; a tool result is the functionResponse part in the next LLMRequest.
// No Gemini model runs and no provider request is serialized. Output goes to a file so
// nothing the stdio server prints can corrupt it.
//
// mcptoolset takes the caller's *mcp.Client (Config.Client) and transport
// (Config.Transport) and keeps the session it opens private. The worker builds the
// client the way mcptoolset's default does (Implementation "adk-mcp-client"), adds the
// handler options a probe needs, and watches the outgoing requests through a sending
// middleware only to read the negotiated protocol version.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/modelcontextprotocol/go-sdk/auth"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/modelcontextprotocol/go-sdk/oauthex"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/mcptoolset"
	"google.golang.org/genai"
)

const (
	appName       = "mcp-mirror"
	userID        = "mcp-mirror"
	callID        = "mcp-mirror-call"
	watchedURI    = "file:///fixtures/watched.txt"
	rootURI       = "file:///tmp/mcp-mirror-root"
	changedRoot   = "file:///tmp/mcp-mirror-root-changed"
	sampled       = "SENTINEL_SAMPLED"
	sampledAfter  = "SENTINEL_SAMPLED_AFTER_TOOL"
	toolUseID     = "SENTINEL_TOOL_USE_ID"
	elicited      = "SENTINEL_ELICITED"
	redirectURL   = "http://localhost:3000/callback"
	adkModule     = "google.golang.org/adk/v2"
	sdkModule     = "github.com/modelcontextprotocol/go-sdk"
	genaiModule   = "google.golang.org/genai"
	modern        = "2026-07-28"
	logTextLimit  = 500
	describeLimit = 1500
)

type obj = map[string]any

func main() {
	if len(os.Args) == 2 && os.Args[1] == "--versions" {
		data, _ := json.Marshal(versions())
		fmt.Println(string(data))
		return
	}
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: worker <config.json> <out.json> | worker --versions")
		os.Exit(64)
	}
	out := obj{"tools": []any{}}
	status := 0
	func() {
		defer func() {
			if r := recover(); r != nil {
				out["fatal"] = clip(fmt.Sprintf("panic: %v", r), describeLimit)
				status = 2
			}
		}()
		raw, err := os.ReadFile(os.Args[1])
		if err != nil {
			panic(err)
		}
		var cfg obj
		if err := json.Unmarshal(raw, &cfg); err != nil {
			panic(err)
		}
		w, err := newWorker(cfg)
		if err == nil {
			defer w.close()
			err = w.run(out)
		}
		if err != nil {
			out["fatal"] = clip(err.Error(), describeLimit)
			status = 2
		}
	}()
	data, err := json.Marshal(out)
	if err != nil {
		data, _ = json.Marshal(obj{"tools": []any{}, "fatal": "encoding the result: " + err.Error()})
		status = 2
	}
	if err := os.WriteFile(os.Args[2], data, 0o600); err != nil {
		fmt.Fprintln(os.Stderr, err)
		status = 2
	}
	os.Exit(status)
}

// versions reads the module versions linked into this binary.
func versions() obj {
	found := obj{"go": runtime.Version()}
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, dep := range info.Deps {
			switch dep.Path {
			case adkModule:
				found["adk"] = strings.TrimPrefix(dep.Version, "v")
			case sdkModule:
				found["go-sdk"] = strings.TrimPrefix(dep.Version, "v")
			case genaiModule:
				found["genai"] = strings.TrimPrefix(dep.Version, "v")
			}
		}
	}
	return found
}

func clip(text string, limit int) string {
	if len(text) > limit {
		return text[:limit]
	}
	return text
}

func stamp() string {
	return time.Now().UTC().Format("2006-01-02T15:04:05.000000+00:00")
}

func str(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprint(v)
}

func num(v any, fallback float64) float64 {
	if f, ok := v.(float64); ok {
		return f
	}
	return fallback
}

// attempt is {"ok": true, ...} or the failure, the way src/mcp_mirror/probe.py records it.
func attempt(body func() (obj, error)) obj {
	answer, err := body()
	if err != nil {
		var no unsupportedError
		if errors.As(err, &no) {
			return obj{"ok": false, "unsupported": no.reason}
		}
		return obj{"ok": false, "error": clip(err.Error(), describeLimit)}
	}
	answer["ok"] = true
	return answer
}

type unsupportedError struct{ reason string }

func (e unsupportedError) Error() string { return e.reason }

func unsupported(reason string) obj {
	return obj{"ok": false, "unsupported": reason}
}

// ---------------------------------------------------------------------------
// What the caller's client heard
// ---------------------------------------------------------------------------

// observer records what reached the handlers the worker registered, and the session
// the toolset opened, read off its own outgoing requests.
type observer struct {
	mu        sync.Mutex
	session   *mcp.ClientSession
	handshake obj
	events    []obj
	saw       []obj
}

func (o *observer) note(kind string, facts obj) {
	o.mu.Lock()
	defer o.mu.Unlock()
	entry := obj{"kind": kind, "t": stamp()}
	for k, v := range facts {
		entry[k] = v
	}
	o.events = append(o.events, entry)
}

func (o *observer) count(kind string) int {
	o.mu.Lock()
	defer o.mu.Unlock()
	n := 0
	for _, e := range o.events {
		if e["kind"] == kind {
			n++
		}
	}
	return n
}

// wait reports whether a new event of kind arrived after the first since, within limit.
func (o *observer) wait(kind string, since int, limit time.Duration) bool {
	deadline := time.Now().Add(limit)
	for time.Now().Before(deadline) {
		if o.count(kind) > since {
			return true
		}
		time.Sleep(50 * time.Millisecond)
	}
	return o.count(kind) > since
}

func (o *observer) of(kind string) []obj {
	o.mu.Lock()
	defer o.mu.Unlock()
	var found []obj
	for _, e := range o.events {
		if e["kind"] == kind {
			found = append(found, e)
		}
	}
	return found
}

func (o *observer) handled(hook string, facts obj) {
	o.mu.Lock()
	defer o.mu.Unlock()
	entry := obj{"hook": hook}
	for k, v := range facts {
		entry[k] = v
	}
	o.saw = append(o.saw, entry)
}

func (o *observer) drain() []obj {
	o.mu.Lock()
	defer o.mu.Unlock()
	saw := o.saw
	o.saw = nil
	if saw == nil {
		saw = []obj{}
	}
	return saw
}

// middleware only looks: it keeps the session and the handshake parameters it sees.
func (o *observer) middleware(next mcp.MethodHandler) mcp.MethodHandler {
	return func(ctx context.Context, method string, req mcp.Request) (mcp.Result, error) {
		if cs, ok := req.GetSession().(*mcp.ClientSession); ok {
			o.mu.Lock()
			o.session = cs
			o.mu.Unlock()
		}
		if method == "initialize" || method == "server/discover" {
			if data, err := json.Marshal(req.GetParams()); err == nil {
				var params obj
				if json.Unmarshal(data, &params) == nil {
					o.mu.Lock()
					o.handshake = obj{"method": method, "params": params}
					o.mu.Unlock()
				}
			}
		}
		return next(ctx, method, req)
	}
}

func (o *observer) negotiated() string {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.session == nil || o.session.InitializeResult() == nil {
		return ""
	}
	return o.session.InitializeResult().ProtocolVersion
}

// ---------------------------------------------------------------------------
// The toolset
// ---------------------------------------------------------------------------

type worker struct {
	cfg    obj
	mode   string
	obs    *observer
	client *mcp.Client
	set    tool.Toolset
	turns  int
}

func newWorker(cfg obj) (*worker, error) {
	w := &worker{cfg: cfg, mode: str(cfg["mode"]), obs: &observer{}}
	if w.mode == "" {
		w.mode = "tools"
	}
	transport, err := w.transport()
	if err != nil {
		return nil, err
	}
	w.client = mcp.NewClient(&mcp.Implementation{Name: "adk-mcp-client", Version: str(versions()["adk"])}, w.options())
	w.client.AddSendingMiddleware(w.obs.middleware)
	if w.mode == "client" {
		w.client.AddRoots(&mcp.Root{URI: rootURI, Name: "mcp-mirror"})
	}
	set, err := mcptoolset.New(mcptoolset.Config{Client: w.client, Transport: transport})
	if err != nil {
		return nil, fmt.Errorf("mcptoolset.New: %w", err)
	}
	w.set = set
	return w, nil
}

func (w *worker) close() {
	w.obs.mu.Lock()
	cs := w.obs.session
	w.obs.mu.Unlock()
	if cs != nil {
		_ = cs.Close()
	}
}

// options are the mcp.ClientOptions a probe registers on the client it hands
// mcptoolset; the tools mode registers none, like mcptoolset's default client.
func (w *worker) options() *mcp.ClientOptions {
	o := w.obs
	switch w.mode {
	case "client":
		return &mcp.ClientOptions{
			Capabilities: &mcp.ClientCapabilities{
				RootsV2:     &mcp.RootCapabilities{ListChanged: true},
				Elicitation: &mcp.ElicitationCapabilities{Form: &mcp.FormElicitationCapabilities{}, URL: &mcp.URLElicitationCapabilities{}},
			},
			CreateMessageWithToolsHandler: func(_ context.Context, req *mcp.CreateMessageWithToolsRequest) (*mcp.CreateMessageWithToolsResult, error) {
				reply := samplingReply(req.Params)
				o.handled("mcp.ClientOptions.CreateMessageWithToolsHandler", obj{"tools": len(req.Params.Tools) > 0, "replied": reply.StopReason})
				return reply, nil
			},
			ElicitationHandler: func(_ context.Context, req *mcp.ElicitRequest) (*mcp.ElicitResult, error) {
				mode := req.Params.Mode
				if mode == "" {
					mode = "form"
					if req.Params.URL != "" {
						mode = "url"
					}
				}
				o.handled("mcp.ClientOptions.ElicitationHandler", obj{"mode": mode})
				if mode == "url" {
					return &mcp.ElicitResult{Action: "accept"}, nil
				}
				return &mcp.ElicitResult{Action: "accept", Content: obj{"name": elicited}}, nil
			},
		}
	case "notify":
		return &mcp.ClientOptions{
			ToolListChangedHandler: func(context.Context, *mcp.ToolListChangedRequest) {
				o.note("ToolsListChanged", nil)
			},
			ResourceListChangedHandler: func(context.Context, *mcp.ResourceListChangedRequest) {
				o.note("ResourcesListChanged", nil)
			},
			PromptListChangedHandler: func(context.Context, *mcp.PromptListChangedRequest) {
				o.note("PromptsListChanged", nil)
			},
			ResourceUpdatedHandler: func(_ context.Context, req *mcp.ResourceUpdatedNotificationRequest) {
				o.note("ResourceUpdated", obj{"uri": req.Params.URI})
			},
		}
	case "utility":
		return &mcp.ClientOptions{
			LoggingMessageHandler: func(_ context.Context, req *mcp.LoggingMessageRequest) {
				o.note("log", obj{"level": string(req.Params.Level), "data": req.Params.Data, "logger": req.Params.Logger})
			},
			ProgressNotificationHandler: func(_ context.Context, req *mcp.ProgressNotificationClientRequest) {
				o.note("progress", obj{"progress": req.Params.Progress, "total": req.Params.Total, "message": req.Params.Message})
			},
		}
	}
	return nil
}

// samplingReply is client_probe.sampling_reply: offered tools and no tool result yet,
// call the first tool; otherwise answer in text, saying whether a result came back.
func samplingReply(params *mcp.CreateMessageWithToolsParams) *mcp.CreateMessageWithToolsResult {
	afterTool := false
	for _, message := range params.Messages {
		for _, block := range message.Content {
			if _, ok := block.(*mcp.ToolResultContent); ok {
				afterTool = true
			}
		}
	}
	if len(params.Tools) > 0 && !afterTool {
		return &mcp.CreateMessageWithToolsResult{
			Role:       "assistant",
			Content:    []mcp.Content{&mcp.ToolUseContent{ID: toolUseID, Name: params.Tools[0].Name, Input: obj{"word": "mirror"}}},
			Model:      "mcp-mirror-stub",
			StopReason: "toolUse",
		}
	}
	text := sampled
	if afterTool {
		text = sampledAfter
	}
	return &mcp.CreateMessageWithToolsResult{
		Role:       "assistant",
		Content:    []mcp.Content{&mcp.TextContent{Text: text}},
		Model:      "mcp-mirror-stub",
		StopReason: "endTurn",
	}
}

// headerTransport adds the fixed headers a ServerHandle carries to every request.
type headerTransport struct {
	headers map[string]string
	base    http.RoundTripper
}

func (h headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	for k, v := range h.headers {
		req.Header.Set(k, v)
	}
	return h.base.RoundTrip(req)
}

// transport is the mcp.Transport mcptoolset.Config.Transport takes for each kind.
func (w *worker) transport() (mcp.Transport, error) {
	headers := map[string]string{}
	if raw, ok := w.cfg["headers"].(obj); ok {
		for k, v := range raw {
			headers[k] = str(v)
		}
	}
	httpClient := &http.Client{Transport: headerTransport{headers: headers, base: http.DefaultTransport}}
	switch kind := str(w.cfg["transport"]); kind {
	case "", "stdio":
		var args []string
		if raw, ok := w.cfg["args"].([]any); ok {
			for _, a := range raw {
				args = append(args, str(a))
			}
		}
		cmd := exec.Command(str(w.cfg["command"]), args...)
		cmd.Stderr = os.Stderr
		return &mcp.CommandTransport{Command: cmd}, nil
	case "http":
		t := &mcp.StreamableClientTransport{Endpoint: str(w.cfg["url"]), HTTPClient: httpClient, MaxRetries: -1}
		if spec, ok := w.cfg["auth"].(obj); ok {
			handler, err := oauthHandler(spec)
			if err != nil {
				return nil, err
			}
			t.OAuthHandler = handler
		}
		return t, nil
	case "sse":
		return &mcp.SSEClientTransport{Endpoint: str(w.cfg["url"]), HTTPClient: httpClient}, nil
	default:
		return nil, fmt.Errorf("unknown transport %q", kind)
	}
}

// oauthHandler is the go-sdk's own authorization-code flow, on the
// StreamableClientTransport mcptoolset.Config.Transport takes. The fetcher does the
// browser's leg: it visits the authorization URL and reads the redirect.
func oauthHandler(spec obj) (*auth.AuthorizationCodeHandler, error) {
	config := &auth.AuthorizationCodeHandlerConfig{
		RedirectURL:              redirectURL,
		AuthorizationCodeFetcher: fetchCode,
	}
	if id := str(spec["clientId"]); id != "" {
		creds := &oauthex.ClientCredentials{ClientID: id}
		if secret := str(spec["clientSecret"]); secret != "" {
			creds.ClientSecretAuth = &oauthex.ClientSecretAuth{ClientSecret: secret}
		}
		config.PreregisteredClient = creds
	} else {
		if metadata := str(spec["clientMetadataUrl"]); metadata != "" {
			config.ClientIDMetadataDocumentConfig = &auth.ClientIDMetadataDocumentConfig{URL: metadata}
		}
		config.DynamicClientRegistrationConfig = &auth.DynamicClientRegistrationConfig{
			Metadata: &oauthex.ClientRegistrationMetadata{
				ClientName:    str(spec["clientName"]),
				RedirectURIs:  []string{redirectURL},
				GrantTypes:    []string{"authorization_code", "refresh_token"},
				ResponseTypes: []string{"code"},
			},
		}
	}
	return auth.NewAuthorizationCodeHandler(config)
}

func fetchCode(ctx context.Context, args *auth.AuthorizationArgs) (*auth.AuthorizationResult, error) {
	client := &http.Client{
		Timeout:       30 * time.Second,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, args.URL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	location, err := url.Parse(resp.Header.Get("Location"))
	if err != nil {
		return nil, err
	}
	query := location.Query()
	if query.Get("code") == "" {
		return nil, fmt.Errorf("authorization redirect carried no code (status %d, location %q)", resp.StatusCode, location)
	}
	return &auth.AuthorizationResult{Code: query.Get("code"), State: query.Get("state"), Iss: query.Get("iss")}, nil
}

// listContext is the agent.ReadonlyContext Toolset.Tools takes, for listing outside
// an agent run. mcptoolset reads only its context.Context.
type listContext struct{ context.Context }

func (listContext) UserContent() *genai.Content          { return nil }
func (listContext) InvocationID() string                 { return "" }
func (listContext) AgentName() string                    { return "mcp_mirror" }
func (listContext) ReadonlyState() session.ReadonlyState { return nil }
func (listContext) UserID() string                       { return userID }
func (listContext) AppName() string                      { return appName }
func (listContext) SessionID() string                    { return "" }
func (listContext) Branch() string                       { return "" }

func (w *worker) toolNames() ([]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	tools, err := w.set.Tools(listContext{ctx})
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(tools))
	for _, t := range tools {
		names = append(names, t.Name())
	}
	sort.Strings(names)
	return names, nil
}

// ---------------------------------------------------------------------------
// One agent run
// ---------------------------------------------------------------------------

type callSpec struct {
	tool      string
	arguments obj
}

// scriptedLLM answers from a script: the first request optionally asks for one tool
// call, every later one ends the run. It records each LLMRequest it is handed.
type scriptedLLM struct {
	call     *callSpec
	mu       sync.Mutex
	requests []obj
}

func (m *scriptedLLM) Name() string { return "mcp-mirror-scripted" }

func (m *scriptedLLM) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		declarations := []any{}
		if req.Config != nil {
			for _, t := range req.Config.Tools {
				if t == nil {
					continue
				}
				for _, d := range t.FunctionDeclarations {
					declarations = append(declarations, asJSON(d))
				}
			}
		}
		responses := []any{}
		for _, content := range req.Contents {
			if content == nil {
				continue
			}
			for _, part := range content.Parts {
				if part != nil && part.FunctionResponse != nil {
					responses = append(responses, asJSON(part.FunctionResponse))
				}
			}
		}
		m.mu.Lock()
		m.requests = append(m.requests, obj{"declarations": declarations, "responses": responses})
		first := len(m.requests) == 1
		m.mu.Unlock()
		part := &genai.Part{Text: "done"}
		if m.call != nil && first {
			args := m.call.arguments
			if args == nil {
				args = obj{}
			}
			part = &genai.Part{FunctionCall: &genai.FunctionCall{ID: callID, Name: m.call.tool, Args: args}}
		}
		yield(&model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{part}}, TurnComplete: true}, nil)
	}
}

// asJSON is a value as plain JSON, through its own serializer.
func asJSON(v any) any {
	data, err := json.Marshal(v)
	if err != nil {
		return obj{"unserializable": err.Error()}
	}
	var out any
	if err := json.Unmarshal(data, &out); err != nil {
		return obj{"unserializable": err.Error()}
	}
	return out
}

type turnResult struct {
	requests []obj
	direct   obj
	runErr   error
}

// functionResponse is the functionResponse the second LLMRequest carried for the call.
func (t *turnResult) functionResponse() obj {
	if len(t.requests) < 2 {
		return nil
	}
	responses, _ := t.requests[1]["responses"].([]any)
	var last obj
	for _, r := range responses {
		if entry, ok := r.(obj); ok {
			if entry["id"] == callID {
				return entry
			}
			last = entry
		}
	}
	return last
}

func (t *turnResult) declarations() []any {
	if len(t.requests) == 0 {
		return []any{}
	}
	declarations, _ := t.requests[0]["declarations"].([]any)
	return declarations
}

// turn is one llmagent run through runner.NewInMemory against the scripted model.
// AfterToolCallbacks sees what mcpTool.Run returned, result map or error.
func (w *worker) turn(ctx context.Context, call *callSpec) *turnResult {
	llm := &scriptedLLM{call: call}
	result := &turnResult{}
	var directMu sync.Mutex
	a, err := llmagent.New(llmagent.Config{
		Name:        "mcp_mirror",
		Model:       llm,
		Instruction: "mcp-mirror probe",
		Toolsets:    []tool.Toolset{w.set},
		AfterToolCallbacks: []llmagent.AfterToolCallback{
			func(_ agent.Context, _ tool.Tool, _, res map[string]any, err error) (map[string]any, error) {
				directMu.Lock()
				defer directMu.Unlock()
				direct := obj{"result": asJSON(res)}
				if err != nil {
					direct["error"] = err.Error()
					direct["error_type"] = fmt.Sprintf("%T", err)
				}
				result.direct = direct
				return nil, nil
			},
		},
	})
	if err != nil {
		result.runErr = fmt.Errorf("llmagent.New: %w", err)
		return result
	}
	r, err := runner.NewInMemory(appName, a)
	if err != nil {
		result.runErr = fmt.Errorf("runner.NewInMemory: %w", err)
		return result
	}
	w.turns++
	sessionID := fmt.Sprintf("mcp-mirror-%d", w.turns)
	message := genai.NewContentFromText("mcp-mirror", genai.RoleUser)
	for _, err := range r.Run(ctx, userID, sessionID, message, agent.RunConfig{}) {
		if err != nil && result.runErr == nil {
			result.runErr = err
		}
	}
	llm.mu.Lock()
	result.requests = llm.requests
	llm.mu.Unlock()
	directMu.Lock()
	defer directMu.Unlock()
	return result
}

func (w *worker) callTurn(tool string, arguments obj) *turnResult {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	return w.turn(ctx, &callSpec{tool: tool, arguments: arguments})
}

// entry is one tool call through the agent, in the shape the renderer reads.
func (w *worker) entry(tool string, arguments obj) obj {
	run := w.callTurn(tool, arguments)
	out := obj{"tool": tool}
	response := run.functionResponse()
	switch {
	case response != nil:
		out["ok"] = true
		out["agent"] = response
		out["direct"] = run.direct
	case run.runErr != nil:
		out["ok"] = false
		out["threw"] = true
		out["exception"] = fmt.Sprintf("%T", run.runErr)
		out["error"] = clip(run.runErr.Error(), describeLimit)
	default:
		out["ok"] = false
		out["error"] = "the next LLMRequest carried no functionResponse"
	}
	return out
}

// agentText is the text the tool answered with, as mcpTool.Run handed it to ADK.
func (w *worker) agentText(tool string, arguments obj) string {
	run := w.callTurn(tool, arguments)
	if run.direct == nil {
		if run.runErr != nil {
			return run.runErr.Error()
		}
		return ""
	}
	if e, ok := run.direct["error"].(string); ok {
		return e
	}
	if res, ok := run.direct["result"].(obj); ok {
		if text, ok := res["output"].(string); ok {
			return text
		}
		data, _ := json.Marshal(res["output"])
		return string(data)
	}
	data, _ := json.Marshal(run.direct)
	return string(data)
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

func (w *worker) run(out obj) error {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	definitions := w.turn(ctx, nil)
	cancel()
	if definitions.runErr != nil {
		return fmt.Errorf("agent run listing the tools: %w", definitions.runErr)
	}
	out["negotiatedMcpSpecVersion"] = w.obs.negotiated()
	w.obs.mu.Lock()
	if w.obs.handshake != nil {
		out["handshake"] = w.obs.handshake
	}
	w.obs.mu.Unlock()
	switch w.mode {
	case "tools":
		out["tools"] = definitions.declarations()
		results := []any{}
		if calls, ok := w.cfg["calls"].([]any); ok {
			for _, raw := range calls {
				call, _ := raw.(obj)
				arguments, _ := call["arguments"].(obj)
				results = append(results, w.entry(str(call["tool"]), arguments))
			}
		}
		out["results"] = results
	case "probe":
		out["probe"] = w.probe()
	case "notify":
		out["notify"] = w.notify(str(out["negotiatedMcpSpecVersion"]))
	case "utility":
		spec, _ := w.cfg["utilitySpec"].(obj)
		out["utility"] = w.utility(str(w.cfg["utility"]), spec)
	case "client":
		out["client"] = w.clientFeatures()
	default:
		return fmt.Errorf("unknown mode %q", w.mode)
	}
	return nil
}

const noSession = "mcptoolset.Toolset exposes only Tools(); the *mcp.ClientSession it opens stays inside its " +
	"connectionRefresher, so the caller has no session to call "

// probe: mcptoolset wraps only tools and hands the caller no session.
func (w *worker) probe() obj {
	return obj{
		"resources.list":      unsupported(noSession + "ListResources on"),
		"resources.read.text": unsupported(noSession + "ReadResource on"),
		"resources.read.blob": unsupported(noSession + "ReadResource on"),
		"resources.templates": unsupported(noSession + "ListResourceTemplates on"),
		"prompts.list":        unsupported(noSession + "ListPrompts on"),
		"prompts.get":         unsupported(noSession + "GetPrompt on"),
	}
}

// clientFeatures calls each client-features fixture tool through the agent; the
// handlers sit on the *mcp.Client handed to mcptoolset.Config.Client.
func (w *worker) clientFeatures() obj {
	answers := obj{}
	calls, _ := w.cfg["clientCalls"].([]any)
	for _, raw := range calls {
		tool := str(raw)
		var rootErr string
		if tool == "roots_changes_seen" {
			// Client.RemoveRoots and AddRoots notify every session the client holds.
			func() {
				defer func() {
					if r := recover(); r != nil {
						rootErr = fmt.Sprint(r)
					}
				}()
				w.client.RemoveRoots(rootURI)
				w.client.AddRoots(&mcp.Root{URI: changedRoot, Name: "mcp-mirror-changed"})
				time.Sleep(300 * time.Millisecond)
			}()
		}
		entry := w.entry(tool, obj{})
		entry["handler_saw"] = w.obs.drain()
		if rootErr != "" {
			entry["root_change_error"] = rootErr
		}
		answers[tool] = entry
	}
	return answers
}

// ---------------------------------------------------------------------------
// Catalogue changes against fixtures/notify_server.py
// ---------------------------------------------------------------------------

func (w *worker) notify(negotiated string) obj {
	o := w.obs
	out := obj{"negotiated": negotiated}
	out["tools.list_changed"] = attempt(func() (obj, error) {
		before, err := w.toolNames()
		if err != nil {
			return nil, err
		}
		since := o.count("ToolsListChanged")
		said := w.agentText("add_tool", obj{})
		heard := o.wait("ToolsListChanged", since, 3*time.Second)
		after, err := w.toolNames()
		if err != nil {
			return nil, err
		}
		return obj{
			"hook":        "mcp.ClientOptions.ToolListChangedHandler on the client passed as mcptoolset.Config.Client",
			"notified":    heard,
			"before":      before,
			"after":       after,
			"sentinel":    "SENTINEL_ADDED_TOOL",
			"server_said": said,
		}, nil
	})
	for _, change := range []struct{ key, tool, kind, noun string }{
		{"resources.list_changed", "add_resource", "ResourcesListChanged", "resources"},
		{"prompts.list_changed", "add_prompt", "PromptsListChanged", "prompts"},
	} {
		since := o.count(change.kind)
		said := w.agentText(change.tool, obj{})
		heard := o.wait(change.kind, since, 3*time.Second)
		answer := unsupported("mcptoolset lists only tools, so the agent has no " + change.noun + " list to refresh")
		answer["handler_heard"] = heard
		answer["hook"] = "mcp.ClientOptions." + map[string]string{
			"ResourcesListChanged": "ResourceListChangedHandler",
			"PromptsListChanged":   "PromptListChangedHandler",
		}[change.kind] + " on the client passed as mcptoolset.Config.Client"
		answer["server_said"] = said
		out[change.key] = answer
	}
	subscribe := unsupported(noSession + "Subscribe on")
	subscribe["server_said"] = []string{w.agentText("update_resource", obj{})}
	out["resources.subscribe"] = subscribe
	if negotiated >= modern {
		out["subscriptions.listen"] = attempt(func() (obj, error) {
			// The go-sdk Client.Connect opens subscriptions/listen itself when list-changed
			// handlers are set; reaching this point means the stream opened.
			o.mu.Lock()
			mark := len(o.events)
			o.mu.Unlock()
			for _, tool := range []string{"add_tool", "add_resource", "add_prompt", "update_resource"} {
				w.agentText(tool, obj{})
			}
			deadline := time.Now().Add(3 * time.Second)
			kinds := map[string]bool{}
			for time.Now().Before(deadline) {
				o.mu.Lock()
				for _, e := range o.events[mark:] {
					kinds[str(e["kind"])] = true
				}
				o.mu.Unlock()
				if len(kinds) >= 4 {
					break
				}
				time.Sleep(50 * time.Millisecond)
			}
			events := []string{}
			o.mu.Lock()
			for _, e := range o.events[mark:] {
				events = append(events, str(e["kind"]))
			}
			o.mu.Unlock()
			return obj{
				"route": "passthrough",
				"api": "go-sdk Client.Connect opens subscriptions/listen for the list-changed handlers set on the " +
					"client passed as mcptoolset.Config.Client; no resource subscription rides it",
				"opened":     true,
				"open_error": nil,
				"honored":    nil,
				"events":     events,
			}, nil
		})
	}
	return out
}

// ---------------------------------------------------------------------------
// Server utilities, one question per worker run
// ---------------------------------------------------------------------------

func (w *worker) utility(feature string, spec obj) obj {
	o := w.obs
	switch feature {
	case "tools.list":
		return attempt(func() (obj, error) {
			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			defer cancel()
			run := w.turn(ctx, nil)
			if run.runErr != nil {
				return nil, run.runErr
			}
			names := []string{}
			for _, d := range run.declarations() {
				if entry, ok := d.(obj); ok {
					names = append(names, str(entry["name"]))
				}
			}
			return obj{"route": "adapter", "api": "llmagent.Config.Toolsets functionDeclarations in the LLMRequest", "names": names}, nil
		})
	case "progress":
		return attempt(func() (obj, error) {
			since := len(o.of("progress"))
			text := w.agentText("slow_with_progress", obj{})
			time.Sleep(300 * time.Millisecond)
			events := []obj{}
			for _, e := range o.of("progress")[since:] {
				events = append(events, e)
			}
			return obj{
				"route": "passthrough",
				"api": "mcp.ClientOptions.ProgressNotificationHandler on the client passed as mcptoolset.Config.Client; " +
					"mcpTool.Run calls CallTool with no progress token",
				"events": events,
				"text":   text,
			}, nil
		})
	case "logging":
		return attempt(func() (obj, error) {
			text := w.agentText("emit_logs", obj{})
			time.Sleep(300 * time.Millisecond)
			messages := []obj{}
			for _, e := range o.of("log") {
				messages = append(messages, obj{"level": e["level"], "data": e["data"]})
			}
			return obj{
				"route":    "passthrough",
				"api":      "mcp.ClientOptions.LoggingMessageHandler on the client passed as mcptoolset.Config.Client",
				"messages": messages,
				"text":     text,
			}, nil
		})
	case "set_level":
		return unsupported(noSession + "SetLoggingLevel on")
	case "completion":
		return unsupported(noSession + "Complete on")
	case "cancellation":
		return attempt(func() (obj, error) {
			after := time.Duration(num(spec["cancel_after_s"], 1.0) * float64(time.Second))
			wait := spec["cancel_wait_s"]
			if wait == nil {
				wait = 4.0
			}
			ctx, cancel := context.WithCancel(context.Background())
			done := make(chan *turnResult, 1)
			started := time.Now()
			go func() { done <- w.turn(ctx, &callSpec{tool: "cancellable_wait", arguments: obj{"seconds": wait}}) }()
			var outcome string
			select {
			case run := <-done:
				outcome = "returned: " + fmt.Sprint(run.direct)
			case <-time.After(after):
				cancel()
				outcome = "abandoned: cancelled the context passed to Runner.Run"
				select {
				case run := <-done:
					if run.direct != nil {
						outcome += fmt.Sprintf("; the tool call ended with %v", run.direct["error"])
					}
				case <-time.After(5 * time.Second):
					outcome += "; the run had not returned 5s later"
				}
			}
			cancel()
			abandoned := time.Since(started).Seconds()
			time.Sleep(300 * time.Millisecond)
			return obj{
				"route": "adapter",
				"mechanism": "cancelling the context.Context passed to Runner.Run; mcpTool.Run hands it to " +
					"ClientSession.CallTool",
				"abandoned_after_s": abandoned,
				"outcome":           clip(outcome, 200),
			}, nil
		})
	case "tasks":
		return attempt(func() (obj, error) {
			return obj{"route": "adapter", "text": w.agentText("task_capable", obj{})}, nil
		})
	case "caching":
		return attempt(func() (obj, error) {
			pause := time.Duration(num(spec["cache_pause_s"], 2.0) * float64(time.Second))
			marks := []obj{}
			for _, mark := range []string{"first", "within_ttl", "after_ttl"} {
				if mark == "after_ttl" {
					time.Sleep(pause)
				}
				startedAt := stamp()
				names, err := w.toolNames()
				if err != nil {
					return nil, err
				}
				marks = append(marks, obj{"call": mark, "started": startedAt, "finished": stamp(), "count": len(names)})
			}
			return obj{"route": "adapter", "api": "mcptoolset Toolset.Tools (calls ListTools on every call; no cache option)", "marks": marks}, nil
		})
	}
	return unsupported("unknown utility " + feature)
}
