-- Reader feedback from the site's Feedback form. Private: nothing here is published.
-- client is a salted hash of the sender's IP, kept only to rate-limit; the IP is not stored.
CREATE TABLE feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
  kind TEXT NOT NULL,
  subject TEXT NOT NULL,
  about TEXT,
  revision TEXT,
  page TEXT,
  message TEXT NOT NULL,
  contact TEXT,
  client TEXT NOT NULL
);
CREATE INDEX feedback_client_time ON feedback (client, created_at);
