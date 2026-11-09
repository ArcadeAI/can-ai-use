// POST /feedback: the site's Feedback form. A message is stored in D1 and never
// published; there is no email, token or third party behind it.

export interface FeedbackDb {
  prepare(query: string): {
    bind(...values: unknown[]): {
      first<T = Record<string, unknown>>(): Promise<T | null>;
      run(): Promise<unknown>;
    };
  };
}

export const MAX_FEEDBACK_BYTES = 8_192;
export const MAX_MESSAGE = 4_000;
// Per sender, per hour. Generous for a person, useless for a script.
export const HOURLY_LIMIT = 10;

const KINDS = new Set(["capability", "behaviour", "framework", "agent"]);
const SUBJECT = /^[A-Za-z0-9._-]{1,100}$/;

const json = (status: number, body: unknown): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: {"Content-Type": "application/json", "Cache-Control": "no-store"},
  });

const text = (value: unknown, max: number): string | null => {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed ? trimmed.slice(0, max) : null;
};

async function clientId(request: Request): Promise<string> {
  // Salted by the day, so the stored value cannot be matched to an IP across days.
  const ip = request.headers.get("CF-Connecting-IP") ?? "unknown";
  const day = new Date().toISOString().slice(0, 10);
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(`canaiuse-feedback:${day}:${ip}`));
  return [...new Uint8Array(digest)].slice(0, 16).map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function handleFeedback(request: Request, db: FeedbackDb | undefined): Promise<Response> {
  if (request.method !== "POST") return json(405, {error: "POST only"});
  if (!db) return json(503, {error: "Feedback is not available right now."});
  // Only the site's own pages may post; another origin gets nothing.
  const origin = request.headers.get("Origin");
  if (origin && origin !== new URL(request.url).origin) return json(403, {error: "Cross-origin feedback is refused."});
  if (!(request.headers.get("Content-Type") ?? "").includes("application/json")) {
    return json(415, {error: "Send JSON."});
  }
  const raw = await request.text();
  if (new TextEncoder().encode(raw).length > MAX_FEEDBACK_BYTES) return json(413, {error: "Too long."});
  let body: Record<string, unknown>;
  try {
    body = JSON.parse(raw);
  } catch {
    return json(400, {error: "Not JSON."});
  }
  if (!body || typeof body !== "object") return json(400, {error: "Not JSON."});
  // A field people cannot see: a bot that fills it is told it worked and is not stored.
  if (text(body.website, 200)) return json(200, {ok: true});
  const kind = text(body.kind, 20);
  const subject = text(body.subject, 100);
  const message = text(body.message, MAX_MESSAGE);
  if (!kind || !KINDS.has(kind) || !subject || !SUBJECT.test(subject)) return json(400, {error: "Unknown subject."});
  if (!message) return json(400, {error: "Write a message."});
  const client = await clientId(request);
  const recent = await db
    .prepare("SELECT COUNT(*) AS n FROM feedback WHERE client = ? AND created_at > strftime('%Y-%m-%dT%H:%M:%SZ', 'now', '-1 hour')")
    .bind(client)
    .first<{n: number}>();
  if ((recent?.n ?? 0) >= HOURLY_LIMIT) return json(429, {error: "Too many messages; try again in an hour."});
  await db
    .prepare("INSERT INTO feedback (kind, subject, about, revision, page, message, contact, client) VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
    .bind(kind, subject, text(body.about, 300), text(body.revision, 20), text(body.page, 500), message, text(body.contact, 200), client)
    .run();
  return json(200, {ok: true});
}
