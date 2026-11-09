import {Link} from "@heroui/react";

import {InlineCode} from "./CapabilityText";
import {capabilityHref} from "../lib/data";
import {newestAdditions} from "../lib/home";
import type {SpecIndex} from "../types";

/**
 * caniuse's "Latest features": the capabilities the newest MCP revision adds. Each link
 * opens that revision's snapshot, since the capability is not in the older ones.
 */
export function NewCapabilities({specIndex, titleId}: {specIndex: SpecIndex | null; titleId?: string}) {
  const added = newestAdditions(specIndex);
  return (
    <section className="home-section home-section-new" aria-labelledby={titleId}>
      <h2 id={titleId}>{added ? `New in ${added.label}` : "New capabilities"}</h2>
      {added ? (
        <ul className="home-list">
          {added.capabilities.map((capability) => (
            <li data-capability-id={capability.id} key={capability.id}>
              <Link href={`?spec=${encodeURIComponent(added.spec)}${capabilityHref(capability.id)}`}>
                <span className="capability-title">
                  <InlineCode text={capability.title} />
                </span>
              </Link>
            </li>
          ))}
        </ul>
      ) : (
        <p className="home-section-note">No revision has added capabilities yet.</p>
      )}
    </section>
  );
}
