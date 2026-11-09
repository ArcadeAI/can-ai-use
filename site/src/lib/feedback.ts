/** What the Feedback form posts to /feedback (worker/feedback.ts). Stored privately. */
export interface FeedbackPayload {
  kind: "capability" | "behaviour" | "framework" | "agent";
  subject: string;
  /** The cell or note the reader was looking at, e.g. "#10 · OpenAI Agents SDK 0.23.1". */
  about?: string;
  revision?: string;
  page: string;
  message: string;
  contact?: string;
  /** Hidden from people; a bot that fills it is dropped by the Worker. */
  website?: string;
}

export type FeedbackResult = {ok: true} | {ok: false; error: string};

export async function sendFeedback(
  payload: FeedbackPayload,
  post: typeof fetch = fetch,
): Promise<FeedbackResult> {
  try {
    const response = await post("/feedback", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const body = (await response.json().catch(() => ({}))) as {error?: string};
    return response.ok ? {ok: true} : {ok: false, error: body.error ?? `Not sent (${response.status}).`};
  } catch {
    return {ok: false, error: "Not sent: the site could not be reached."};
  }
}
