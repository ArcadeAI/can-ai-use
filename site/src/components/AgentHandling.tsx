import {Link} from "@heroui/react";
import {useId, useState} from "react";

import {previewRequested} from "../lib/data";
import {
  agentsHref,
  harnessRowFor,
  harnessRule,
  shownHarnesses,
  visibleHarnesses,
} from "../lib/harnesses";
import {HARNESS_LINKS} from "../lib/harnessSources";
import type {SpecIndex} from "../types";
import {HarnessLegend, HarnessToggle, useHarnessData} from "./HarnessParts";
import {HarnessRowTable} from "./HarnessRowTable";

/**
 * The agent-side half of a capability: the paired harness row, as its own table
 * below the framework grid. It asks the harness row's question and uses the harness
 * vocabulary, since a harness cell and a framework cell measure different things.
 */
export function AgentHandling({
  capabilityId,
  specIndex,
}: {
  capabilityId: string;
  specIndex: SpecIndex | null;
}) {
  const {data} = useHarnessData(specIndex);
  const [showAll, setShowAll] = useState(false);
  const headingId = useId();
  const legendId = useId();
  const scope = `agents-${headingId.replace(/[^a-zA-Z0-9_-]/g, "")}`;
  if (!data) return null;
  const row = harnessRowFor(capabilityId, data, HARNESS_LINKS);
  if (!row) return null;
  const visible = visibleHarnesses(data, previewRequested(window.location.search));
  if (!visible.length) return null;
  const harnesses = shownHarnesses(visible, HARNESS_LINKS, showAll);

  return (
    <section
      className="agent-handling"
      aria-labelledby={headingId}
      data-harness-row={row.id}
    >
      <h2 id={headingId}>How agents handle this</h2>
      <p className="agent-handling-question">
        <strong>
          <Link className="agent-handling-row-link" href={agentsHref(row.id)}>
            {row.title}
          </Link>
          .
        </strong>{" "}
        {row.question || row.observe}
      </p>
      <HarnessToggle
        showAll={showAll}
        onChange={setShowAll}
        rule={harnessRule(data, HARNESS_LINKS)}
      />
      <HarnessRowTable data={data} row={row} harnesses={harnesses} noteScope={scope} />
      <HarnessLegend data={data} titleId={legendId} title="Agent legend" />
    </section>
  );
}
