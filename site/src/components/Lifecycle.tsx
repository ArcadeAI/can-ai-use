import {Link} from "@heroui/react";

import type {Lifecycle} from "../types";

export const LIFECYCLE_LABELS: Record<string, string> = {
  archived: "Archived",
  discontinued: "Discontinued",
  transitioning: "Being replaced",
};

const label = (lifecycle: Lifecycle) => LIFECYCLE_LABELS[lifecycle.status] ?? lifecycle.status;

/** The row's project status under its title: what changed, since when, and the source. */
export function LifecycleNote({lifecycle}: {lifecycle?: Lifecycle}) {
  if (!lifecycle) return null;
  return (
    <p className="lifecycle-note" data-status={lifecycle.status}>
      <strong>{label(lifecycle)}</strong>
      {lifecycle.since ? <> since <time dateTime={lifecycle.since}>{lifecycle.since}</time></> : null}.{" "}
      {lifecycle.note}
      {lifecycle.source ? (
        <>
          {" "}
          <Link href={lifecycle.source} target="_blank" rel="noopener noreferrer">Source</Link>
        </>
      ) : null}
    </p>
  );
}
