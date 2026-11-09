import {ToggleButton, ToggleButtonGroup} from "@heroui/react";
import {useEffect, useState} from "react";

import {siteFileHref} from "../lib/data";
import {ArrowUpRightIcon} from "./Icons";
import type {
  Harness,
  HarnessCell,
  HarnessDatabase,
  HarnessRow,
  SpecIndex,
} from "../types";

/**
 * What each word on this page means. The words themselves come from the data
 * (build_data.py reads them off each cell's code and wire evidence); this only
 * says what a reader should take them to mean.
 */
const VOCABULARY: Record<string, string> = {
  requested:
    "The harness sent it on the wire, or answered what the fixture asked, in every judged run.",
  negotiated:
    "The revision the handshake settled on. The revisions the harness asked for are listed beneath it.",
  "never called": "Nothing for it reached the fixture in any judged run.",
  "blocked by prompt":
    "The harness held the call behind a permission prompt instead of sending it.",
};

const EXTRA_LABELS: Record<string, string> = {
  "ran without asking":
    "The destructive call reached the server with no permission prompt, in default approval mode.",
  "none offered": "The harness offered its model none of the fixtures' tools.",
  "first page only": "It listed the first page of tools and never followed the cursor.",
  "no re-list": "It did not list tools again after the server said the list changed.",
  "dropped slow server": "It gave up on a server that took 8 seconds to start, and never listed its tools.",
  "not answered":
    "The server asked it for something (a form or a model completion) and got an error or no answer.",
  "missed new tool": "The tool the server added was not in the next session's list.",
  "requested in part": "Seen on some servers or in some runs, not all.",
  "not measured": "No judged run answers the cell yet. The note says why.",
  "stdio / HTTP":
    "The harness answered differently per transport, so the cell shows one band each. A value like A / B is A against the older fixture line and B against the newer.",
};

export function useHarnessData(specIndex: SpecIndex | null) {
  const [data, setData] = useState<HarnessDatabase | null>(
    () => window.__MCP_MIRROR_HARNESSES__ ?? null,
  );
  const [error, setError] = useState<string | null>(null);
  const file = specIndex?.harnesses?.file ?? "harnesses.json";

  useEffect(() => {
    if (data) {
      return;
    }
    const controller = new AbortController();
    fetch(`./data/${file}`, {signal: controller.signal})
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Harness data returned HTTP ${response.status}`);
        }
        return response.json() as Promise<HarnessDatabase>;
      })
      .then(setData)
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") {
          return;
        }
        setError(reason instanceof Error ? reason.message : "Unknown error");
      });
    return () => controller.abort();
  }, [data, file]);

  return {data, error};
}

// A published harness's captures ship with the site (scripts/copy_captures.mjs), so the
// link is local. A preview harness's captures are linked in the repository instead.
export function captureHref(repo: string, cell: HarnessCell, published: boolean): string {
  if (published) return siteFileHref(cell.capture);
  return `${repo}/blob/main/${cell.capture}${cell.line ? `#L${cell.line}` : ""}`;
}

/**
 * A cell's evidence note under its one-line summary, collapsed until a reader opens
 * it. Nothing is dropped: the full evidence and the capture it was read from are here.
 */
export function HarnessEvidence({
  evidence,
  href,
  captureName,
  line,
}: {
  evidence: string;
  href: string;
  captureName: string;
  line: number | null;
}) {
  return (
    <details className="harness-evidence">
      <summary>Run evidence</summary>
      <p>
        {evidence}{" "}
        <a
          className="harness-note-capture"
          href={href}
          target="_blank"
          rel="noopener noreferrer"
        >
          {captureName}
          {line ? `:${line}` : ""}
          <ArrowUpRightIcon width={11} height={11} />
        </a>
      </p>
    </details>
  );
}

/** What each word means, for a tooltip or the legend's own list. */
export function termMeaning(word: string): string {
  return VOCABULARY[word] ?? EXTRA_LABELS[word] ?? "";
}

/** The words an agent cell can carry, the data's vocabulary first. */
export function legendTerms(data: HarnessDatabase): string[] {
  const vocabulary = data.vocabulary.length ? data.vocabulary : Object.keys(VOCABULARY);
  return [...vocabulary, ...Object.keys(EXTRA_LABELS).filter((word) => !vocabulary.includes(word))];
}

/**
 * The agent legend, compact like the framework legend: one swatch per code with what
 * it means, then the words a cell can carry, each with its meaning on hover and in
 * the list under "What the words mean". Shared by the agents pages and the
 * capability pages.
 */
export function HarnessLegend({
  data,
  titleId,
  title = "Legend",
}: {
  data: HarnessDatabase;
  titleId: string;
  title?: string;
}) {
  const terms = legendTerms(data);
  return (
    <section className="harness-legend" aria-labelledby={titleId}>
      <h2 id={titleId}>{title}</h2>
      <ul className="harness-legend-codes">
        {Object.entries(data.support_codes).map(([code, meaning]) => (
          <li key={code} data-code={code}>
            <i className={`support-${code}`} /> {meaning}
          </li>
        ))}
      </ul>
      <ul className="harness-legend-terms" aria-label="Words a cell can carry">
        {terms.map((word) => (
          <li key={word}>
            <abbr className="harness-legend-term" title={termMeaning(word)}>
              {word}
            </abbr>
          </li>
        ))}
      </ul>
      <details className="harness-legend-meanings">
        <summary>What the words mean</summary>
        <dl>
          {terms.map((word) => (
            <div key={word}>
              <dt>{word}</dt>
              <dd>{termMeaning(word)}</dd>
            </div>
          ))}
        </dl>
      </details>
    </section>
  );
}

/** Filtered | All, styled like the framework toggle on capability pages. */
export function HarnessToggle({
  showAll,
  onChange,
  rule,
}: {
  showAll: boolean;
  onChange: (showAll: boolean) => void;
  rule: string;
}) {
  return (
    <div className="harness-toggle" data-region="harness-filter">
      <ToggleButtonGroup
        aria-label="Agents shown"
        disallowEmptySelection
        selectionMode="single"
        selectedKeys={[showAll ? "all" : "filtered"]}
        size="sm"
        onSelectionChange={(keys) => {
          const selected = Array.from(keys)[0];
          if (selected === "filtered" || selected === "all") {
            onChange(selected === "all");
          }
        }}
      >
        <ToggleButton id="filtered">Filtered</ToggleButton>
        <ToggleButtonGroup.Separator />
        <ToggleButton id="all">All</ToggleButton>
      </ToggleButtonGroup>
      <p className="filter-rule">{rule}</p>
    </div>
  );
}
