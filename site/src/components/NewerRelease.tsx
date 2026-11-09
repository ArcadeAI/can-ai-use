import {shortDate} from "../lib/data";
import type {VersionEntry} from "../types";

interface Versioned {
  current_version: string;
  stable_version?: string;
  version_list: VersionEntry[];
}

/** The newest stable release when it came out after the measured one; null when the row is current. */
export function newerRelease(row: Versioned): VersionEntry | null {
  if (!row.stable_version || row.stable_version === row.current_version) return null;
  const find = (version: string) => row.version_list.find((entry) => entry.version === version);
  const measured = find(row.current_version);
  const stable = find(row.stable_version);
  if (!measured?.release_date || !stable?.release_date) return null;
  return stable.release_date > measured.release_date ? stable : null;
}

/** "· 2.1.303 out" after a score-list version, so a reader knows the row trails a release. */
export function NewerReleaseMark({row}: {row: Versioned}) {
  const newer = newerRelease(row);
  if (!newer) return null;
  return (
    <small
      className="agents-score-date newer-release"
      data-newer-version={newer.version}
      title={`${newer.version} was released ${shortDate(newer.release_date!)}; this row measures ${row.current_version}`}
    >
      {" "}· {newer.version} out
    </small>
  );
}

/** The "Latest release" fact on a framework or agent page; nothing when the row is current. */
export function NewerReleaseFact({row}: {row: Versioned}) {
  const newer = newerRelease(row);
  if (!newer) return null;
  return (
    <>
      <dt>Latest release</dt>
      <dd className="newer-release" data-newer-version={newer.version}>
        <strong>{newer.version}</strong>, released {shortDate(newer.release_date!)}. Not measured yet.
      </dd>
    </>
  );
}
