import {Button} from "@heroui/react";
import {useId, useState} from "react";

import {type FeedbackPayload, sendFeedback} from "../lib/feedback";

/**
 * The site's Feedback form: a message, an optional way to reply, and what the reader was
 * looking at. It posts to /feedback, which stores it privately; nothing is published.
 */
export function FeedbackForm({
  kind,
  subject,
  about,
  revision,
  onClearAbout,
}: {
  kind: FeedbackPayload["kind"];
  subject: string;
  about?: string;
  revision?: string;
  onClearAbout?: () => void;
}) {
  const id = useId().replace(/:/g, "");
  const [message, setMessage] = useState("");
  const [contact, setContact] = useState("");
  const [website, setWebsite] = useState("");
  const [state, setState] = useState<"idle" | "sending" | "sent" | {error: string}>("idle");

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!message.trim() || state === "sending") return;
    setState("sending");
    const result = await sendFeedback({
      kind,
      subject,
      about,
      revision,
      page: window.location.href,
      message,
      contact: contact || undefined,
      website: website || undefined,
    });
    if (result.ok) {
      setMessage("");
      setState("sent");
    } else {
      setState({error: result.error});
    }
  };

  if (state === "sent") {
    return (
      <div className="feedback-form feedback-sent" role="status">
        <p>Thanks, sent. It goes to the maintainer privately; nothing is published.</p>
        <Button size="sm" variant="ghost" onPress={() => setState("idle")}>Send another</Button>
      </div>
    );
  }

  return (
    <form className="feedback-form" onSubmit={submit} aria-label="Feedback">
      <p>
        Something here look wrong, or missing? Tell us what you saw. It goes to the
        maintainer privately; leave a way to reach you if you want an answer.
      </p>
      {about ? (
        <p className="feedback-about">
          About <strong>{about}</strong>
          {onClearAbout ? (
            <button type="button" className="feedback-about-clear" onClick={onClearAbout} aria-label="Not about this cell">
              ×
            </button>
          ) : null}
        </p>
      ) : null}
      <label htmlFor={`${id}-message`} className="sr-only">Message</label>
      <textarea
        id={`${id}-message`}
        name="message"
        required
        maxLength={4000}
        rows={4}
        placeholder="What looks wrong, and how you know (a link, a version, the output you saw)"
        value={message}
        onChange={(event) => setMessage(event.target.value)}
      />
      <div className="feedback-row">
        <label htmlFor={`${id}-contact`} className="sr-only">How to reach you (optional)</label>
        <input
          id={`${id}-contact`}
          name="contact"
          maxLength={200}
          placeholder="How to reach you (optional)"
          value={contact}
          onChange={(event) => setContact(event.target.value)}
        />
        <input
          className="feedback-trap"
          name="website"
          tabIndex={-1}
          autoComplete="off"
          aria-hidden="true"
          value={website}
          onChange={(event) => setWebsite(event.target.value)}
        />
        <Button type="submit" size="sm" isDisabled={!message.trim() || state === "sending"}>
          {state === "sending" ? "Sending…" : "Send"}
        </Button>
      </div>
      {typeof state === "object" ? <p className="feedback-error" role="alert">{state.error}</p> : null}
    </form>
  );
}
