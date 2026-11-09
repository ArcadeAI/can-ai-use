import {
  Button,
  Link,
  Tabs,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
} from "@heroui/react";
import {useEffect, useId, useMemo, useState} from "react";

import {
  capabilityHref,
  currentSupport,
  measuredRange,
  parseSupport,
  reproduceCommand,
  shortDate,
  supportCounts,
} from "../lib/data";
import {defaultsRule, filteredDatabase, recordedCommands} from "../lib/frameworks";
import {FRAMEWORK_DEFAULTS} from "../lib/frameworkSources";
import type {Capability, MirrorDatabase, SupportCode} from "../types";
import {CapabilityGrid, type GridMode} from "./CapabilityGrid";
import {CapabilityTitle, DocsLink, InlineCode} from "./CapabilityText";
import {CopyCodeBlock, writeClipboardText} from "./CopyCodeBlock";
import {FeedbackForm} from "./FeedbackForm";
import {StarIcon} from "./Icons";

type DetailKey = "notes" | "test" | "issues" | "resources" | "feedback";

/** "A, B and C": the first `limit` names, popular frameworks first. */
function namedFrameworks(database: MirrorDatabase, agentIds: string[], limit = 3): string {
  const rank = (agentId: string) => {
    const index = FRAMEWORK_DEFAULTS.default_frameworks.indexOf(agentId);
    return index === -1 ? Number.MAX_SAFE_INTEGER : index;
  };
  const names = [...agentIds]
    .sort((left, right) => rank(left) - rank(right))
    .slice(0, limit)
    .map((agentId) => database.agents[agentId].name);
  return names.length > 1 ? `${names.slice(0, -1).join(", ")} and ${names.at(-1)}` : names[0];
}

/**
 * The capability's finding in one line: which frameworks lose it, named, popular ones
 * first. Counted over every framework, whatever the table below is filtered to.
 */
function baselineHeadline(database: MirrorDatabase, capability: Capability): string {
  const byCode: Record<SupportCode, string[]> = {y: [], a: [], n: [], x: [], u: []};
  for (const agentId of Object.keys(database.agents)) {
    byCode[currentSupport(database, capability, agentId).code].push(agentId);
  }
  const answered = byCode.y.length + byCode.a.length + byCode.n.length;
  if (!answered) {
    // Nothing observed, but a row where every adapter is blocked is answered, not open.
    return byCode.x.length && !byCode.u.length ? "No adapter can reach this revision" : "Not yet measured";
  }
  if (byCode.y.length === answered) return `Kept by all ${answered} frameworks`;
  if (byCode.n.length === answered) return `Dropped by all ${answered} frameworks`;
  const lost = byCode.n.length ? byCode.n : byCode.a;
  const verb = byCode.n.length ? "Dropped" : "Changed";
  const changed = byCode.n.length && byCode.a.length ? `; changed by ${byCode.a.length}` : "";
  const including = lost.length > 3 ? ", including " : ": ";
  return `${verb} by ${lost.length} of ${answered}${including}${namedFrameworks(database, lost)}${changed}`;
}

function focusNote(prefix: string, note: number) {
  window.requestAnimationFrame(() => {
    window.requestAnimationFrame(() => {
      const target = document.getElementById(`${prefix}-note-${note}`);
      target?.focus({preventScroll: true});
      target?.scrollIntoView({behavior: "smooth", block: "nearest"});
    });
  });
}

function ExampleComparison({capability}: {capability: Capability}) {
  if (!capability.example) return null;
  return (
    <div className="feature-example-comparison">
      <section>
        <h4>{capability.example.before_label ?? "Server publishes"}</h4>
        <CopyCodeBlock value={capability.example.before.join("\n")} />
      </section>
      <section>
        <h4>{capability.example.after_label ?? "Adapter produces"}</h4>
        <CopyCodeBlock value={capability.example.after.join("\n")} />
      </section>
    </div>
  );
}

function TestPanel({
  database,
  capability,
}: {
  database: MirrorDatabase;
  capability: Capability;
}) {
  const entries = Object.entries(database.agents);
  const [agentId, setAgentId] = useState(entries[0]?.[0] ?? "");
  const agent = database.agents[agentId];
  const testedVersion =
    agent?.version_list.find((entry) => entry.era === 0)?.version ??
    agent?.current_version ??
    "";
  const [version, setVersion] = useState(testedVersion);

  useEffect(() => setVersion(testedVersion), [agentId, testedVersion]);

  // Offering a scan command here would imply the command reproduces a measurement. For a
  // feature no scan covers, it would just run and quietly produce nothing for this row.
  if (capability.measurement_state === "not_measured") {
    return (
      <div className="feature-test-panel">
        <p>
          No scan produces this row yet, so there is nothing to reproduce. The
          feature is listed because MCP {capability.measured.mcp_spec} defines
          it, not because anything has been observed about these frameworks.
        </p>
        {capability.probe ? (
          <>
            <p>Measuring it would take a probe that does this:</p>
            <blockquote className="feature-probe">{capability.probe}</blockquote>
          </>
        ) : null}
      </div>
    );
  }

  return (
    <div className="feature-test-panel">
      <p>
        Reproduce this adapter comparison locally. No model call or model API
        key is required.
      </p>
      <div className="feature-test-controls">
        <label>
          Framework
          <select value={agentId} onChange={(event) => setAgentId(event.target.value)}>
            {entries.map(([id, candidate]) => (
              <option value={id} key={id}>{candidate.name}</option>
            ))}
          </select>
        </label>
        <label>
          Version
          <select value={version} onChange={(event) => setVersion(event.target.value)}>
            {agent?.version_list.map((entry) => (
              <option value={entry.version} key={entry.version}>
                {entry.version}{entry.era === 0 ? " (tested)" : ""}
              </option>
            ))}
          </select>
        </label>
      </div>
      <CopyCodeBlock
        className="command-box"
        value={
          agent
            ? // The tested release reruns the scan that produced its cell; another
              // release gets the generic command for that version.
              (version === testedVersion
                ? recordedCommands(agentId, database.agents, [capability]).join(" && ")
                : "") || reproduceCommand(agentId, agent, version, capability)
            : ""
        }
      />
    </div>
  );
}

export function FeatureModule({
  database,
  capability,
  starred,
  onToggleStar,
  gridMode,
  onGridModeChange,
  detail = false,
}: {
  database: MirrorDatabase;
  capability: Capability;
  starred: boolean;
  onToggleStar: () => void;
  gridMode: GridMode;
  onGridModeChange: (mode: GridMode) => void;
  detail?: boolean;
}) {
  const reactId = useId().replace(/:/g, "");
  const scope = `feature-${capability.id}-${reactId}`;
  const [selectedDetail, setSelectedDetail] = useState<DetailKey>("notes");
  // One toggle for both: Filtered trims each column to the releases around the tested
  // one and drops frameworks under the download threshold; All shows everything.
  const [showAll, setShowAll] = useState(false);
  const shown = useMemo(
    () => (showAll ? database : filteredDatabase(database, FRAMEWORK_DEFAULTS)),
    [database, showAll],
  );
  const [linkCopied, setLinkCopied] = useState(false);
  const counts = supportCounts(database, capability);
  const frameworkCount = Object.keys(database.agents).length;
  // Releases with a measured cell: neither unreachable at this revision nor unmeasured.
  const measuredCount = frameworkCount - counts.x - counts.u;
  // The row's frameworks were captured on different days; say the span, not one day.
  const tested = measuredRange(database, capability);
  const shownCapability = shown.data[capability.id] ?? capability;
  const notes = Object.entries(shownCapability.notes_by_num ?? {}).sort(
    ([left], [right]) => Number(left) - Number(right),
  );
  const noteReferences = useMemo(() => {
    const references = new Map<number, string[]>();
    Object.entries(shown.agents).forEach(([agentId, agent]) => {
      agent.version_list.forEach(({version}) => {
        parseSupport(capability.stats[agentId]?.[version]).noteRefs.forEach(
          (note) => {
            const labels = references.get(note) ?? [];
            labels.push(`${agent.name} ${version}`);
            references.set(note, labels);
          },
        );
      });
    });
    return references;
  }, [capability, shown.agents]);
  // Cells the data marks as answering differently at different result boundaries.
  const boundaryDependent = (shownCapability.boundary_dependent ?? [])
    .map((agentId) => [agentId, shown.agents[agentId]] as const)
    .filter(([, agent]) => Boolean(agent));
  const issues = capability.known_issues ?? [];
  const resources = capability.links ?? [];
  // The note a reader chose to report; the Feedback form names it.
  const [feedbackAbout, setFeedbackAbout] = useState<string | undefined>();
  const reportNote = (label: string) => {
    setFeedbackAbout(label);
    setSelectedDetail("feedback");
  };

  const activateNote = (note: number) => {
    setSelectedDetail("notes");
    focusNote(scope, note);
  };

  // A link to #/cap/<id>?note=N, from a framework page or a grid cell, opens on that note,
  // showing every framework when the filtered view leaves it out.
  useEffect(() => {
    if (!detail) {
      return;
    }
    const openLinkedNote = () => {
      const query = window.location.hash.split("?")[1] ?? "";
      const note = new URLSearchParams(query).get("note");
      if (!note || !(note in (capability.notes_by_num ?? {}))) {
        return;
      }
      const filtered = filteredDatabase(database, FRAMEWORK_DEFAULTS).data[capability.id];
      if (!(note in (filtered?.notes_by_num ?? {}))) {
        setShowAll(true);
      }
      activateNote(Number(note));
    };
    openLinkedNote();
    window.addEventListener("hashchange", openLinkedNote);
    return () => window.removeEventListener("hashchange", openLinkedNote);
  }, [capability, database, detail]);

  const permalink = () => {
    const url = `${window.location.href.split("#")[0]}${capabilityHref(capability.id)}`;
    void writeClipboardText(url).then(() => {
      setLinkCopied(true);
      window.setTimeout(() => setLinkCopied(false), 1600);
    });
  };

  return (
    <article
      className={`feature-module${detail ? " is-detail" : " is-search-result"}`}
      data-capability-id={capability.id}
      data-measurement-state={capability.measurement_state ?? "measured"}
      aria-labelledby={`${scope}-title`}
    >
      <aside className="feature-action-rail" aria-label="Capability actions">
        <Tooltip delay={250}>
          <Tooltip.Trigger>
            <Button
              aria-label="Copy a link to this capability"
              isIconOnly
              size="sm"
              variant="ghost"
              onPress={permalink}
            >
              <span className="permalink-symbol" aria-hidden="true">#</span>
            </Button>
          </Tooltip.Trigger>
          <Tooltip.Content>{linkCopied ? "Link copied" : "Copy link"}</Tooltip.Content>
        </Tooltip>
        <Button
          aria-label={starred ? "Remove star" : "Star this capability"}
          isIconOnly
          size="sm"
          variant="ghost"
          onPress={onToggleStar}
        >
          <StarIcon filled={starred} />
        </Button>
      </aside>

      <div className="feature-module-main">
        <header className="feature-module-header">
          <div className="feature-heading-copy">
            <div className="feature-title-line">
              <h2 id={`${scope}-title`}>
                {detail ? (
                  <CapabilityTitle capability={capability} />
                ) : (
                  <Link href={capabilityHref(capability.id)}>
                    <CapabilityTitle capability={capability} />
                  </Link>
                )}
              </h2>
              <DocsLink capability={capability} />
            </div>
            <section
              className="measurement-baseline"
              aria-label="Measurement baseline"
              data-region="measurement-baseline"
            >
              <strong>{baselineHeadline(database, capability)}</strong>
              {capability.measured.run_date ? (
                <span>
                  MCP {capability.measured.mcp_spec} · tested{" "}
                  {tested && tested.earliest !== tested.latest ? (
                    <>
                      <time dateTime={tested.earliest}>{tested.earliest}</time> to{" "}
                      <time dateTime={tested.latest}>{tested.latest}</time>
                    </>
                  ) : (
                    <time dateTime={tested?.latest ?? capability.measured.run_date}>
                      {shortDate(tested?.latest ?? capability.measured.run_date)}
                    </time>
                  )}{" "}
                  · {measuredCount} releases ·{" "}
                  <code>{capability.measured.fixture ?? "custom fixture"}</code>
                </span>
              ) : (
                <span>
                  MCP {capability.measured.mcp_spec} · defined by the
                  specification · no scan covers it yet ·{" "}
                  {counts.x > 0
                    ? `${counts.u} releases unmeasured, ${counts.x} cannot reach this revision`
                    : `${frameworkCount} releases unmeasured`}
                </span>
              )}
            </section>
          </div>
          <p className="feature-observed-summary" aria-label="Observed support summary">
            <span>Observed</span>
            <strong>{counts.y} present · {counts.a} changed/retained · {counts.n} dropped</strong>
            <small>
              {measuredCount === frameworkCount
                ? `${frameworkCount} tested releases`
                : [
                    `${measuredCount} of ${frameworkCount} releases tested`,
                    counts.x > 0 ? `${counts.x} cannot reach this revision` : "",
                    counts.u > 0 ? `${counts.u} unmeasured` : "",
                  ]
                    .filter(Boolean)
                    .join(" · ")}
            </small>
          </p>
        </header>

        <div className="feature-description" data-region="description">
          <InlineCode text={capability.description} />
        </div>

        <div className="feature-display-controls" data-region="display-controls">
          <span className="alignment-label">Current aligned</span>
          <ToggleButtonGroup
            aria-label="Version cell detail"
            disallowEmptySelection
            selectionMode="single"
            selectedKeys={[gridMode]}
            size="sm"
            onSelectionChange={(keys) => {
              const mode = Array.from(keys)[0];
              if (mode === "support" || mode === "date") onGridModeChange(mode);
            }}
          >
            <ToggleButton id="support">Result</ToggleButton>
            <ToggleButtonGroup.Separator />
            <ToggleButton id="date">Release date</ToggleButton>
          </ToggleButtonGroup>
          <ToggleButtonGroup
            aria-label="Frameworks and versions shown"
            disallowEmptySelection
            selectionMode="single"
            selectedKeys={[showAll ? "all" : "filtered"]}
            size="sm"
            onSelectionChange={(keys) => {
              const value = Array.from(keys)[0];
              if (value === "all" || value === "filtered") {
                setShowAll(value === "all");
              }
            }}
          >
            <ToggleButton id="filtered">Filtered</ToggleButton>
            <ToggleButtonGroup.Separator />
            <ToggleButton id="all">All</ToggleButton>
          </ToggleButtonGroup>
          <p className="filter-rule" data-region="filter-rule">
            {showAll ? `All ${Object.keys(database.agents).length} frameworks` : defaultsRule(database, FRAMEWORK_DEFAULTS)}
            {showAll ? null : (
              <span>
                {" "}· {Object.keys(shown.agents).length} of{" "}
                {Object.keys(database.agents).length} shown
              </span>
            )}
          </p>
        </div>

        <div className="feature-matrix" data-region="framework-matrix">
          <CapabilityGrid
            database={shown}
            capability={shownCapability}
            hideToolbar
            mode={gridMode}
            onModeChange={onGridModeChange}
            showAllVersions={showAll}
            showCaption={false}
            noteTargetPrefix={scope}
            onNoteActivate={activateNote}
          />
        </div>

        <Tabs
          className="feature-tabs"
          selectedKey={selectedDetail}
          variant="secondary"
          onSelectionChange={(key) => setSelectedDetail(String(key) as DetailKey)}
        >
          <Tabs.ListContainer>
            <Tabs.List aria-label={`${capability.title.replaceAll("`", "")} details`}>
              <Tabs.Tab id="notes" aria-label="Notes">
                <Tabs.Indicator /><span>Notes</span>
              </Tabs.Tab>
              <Tabs.Tab id="test" aria-label="Test a feature">
                <Tabs.Indicator />
                <span className="tab-label-full">Test a feature</span>
                <span className="tab-label-short" aria-hidden="true">Test</span>
              </Tabs.Tab>
              <Tabs.Tab id="issues" aria-label={`Known issues, ${issues.length}`}>
                <Tabs.Indicator />
                <span className="tab-label-full">Known issues ({issues.length})</span>
                <span className="tab-label-short" aria-hidden="true">Issues ({issues.length})</span>
              </Tabs.Tab>
              <Tabs.Tab id="resources" aria-label={`Resources, ${resources.length}`}>
                <Tabs.Indicator />
                <span className="tab-label-full">Resources ({resources.length})</span>
                <span className="tab-label-short" aria-hidden="true">Resources ({resources.length})</span>
              </Tabs.Tab>
              <Tabs.Tab id="feedback" aria-label="Feedback">
                <Tabs.Indicator /><span>Feedback</span>
              </Tabs.Tab>
            </Tabs.List>
          </Tabs.ListContainer>

          <Tabs.Panel id="notes">
            <div className="feature-panel-content">
              {capability.notes ? <p><InlineCode text={capability.notes} /></p> : null}
              {boundaryDependent.length ? (
                <p className="boundary-dependent-callout">
                  <strong>Depends on the result boundary.</strong>{" "}
                  {boundaryDependent
                    .map(([, agent]) => agent.name)
                    .join(", ")}{" "}
                  {boundaryDependent.length === 1 ? "answers" : "answer"} this
                  differently depending on whether the tool is invoked directly or
                  through the framework&apos;s own agent path. Both boundaries are
                  listed on the Frameworks page; the note below says what each one
                  yields.
                </p>
              ) : null}
              {notes.length ? (
                <ul className="numbered-notes">
                  {notes.map(([number, note]) => {
                    const labels = noteReferences.get(Number(number)) ?? [];
                    const visibleLabel = `#${number} · ${labels.join(", ")}`;
                    return (
                    <li
                      id={`${scope}-note-${number}`}
                      key={number}
                      tabIndex={-1}
                      aria-label={`${visibleLabel}: ${note}`}
                    >
                      <strong className="note-reference-label">{visibleLabel}</strong>
                      <span><InlineCode text={note} /></span>
                      <button
                        type="button"
                        className="note-report"
                        onClick={() => reportNote(visibleLabel)}
                        aria-label={`Report a problem with note ${number}`}
                      >
                        Report
                      </button>
                    </li>
                    );
                  })}
                </ul>
              ) : (
                <p>No additional notes are recorded.</p>
              )}
            </div>
          </Tabs.Panel>
          <Tabs.Panel id="test">
            <div className="feature-panel-content">
              <TestPanel database={database} capability={capability} />
            </div>
          </Tabs.Panel>
          <Tabs.Panel id="issues">
            <div className="feature-panel-content">
              {issues.length ? (
                <ul>{issues.map((issue) => <li key={issue}><InlineCode text={issue} /></li>)}</ul>
              ) : <p>No known issues are recorded.</p>}
            </div>
          </Tabs.Panel>
          <Tabs.Panel id="resources">
            <div className="feature-panel-content">
              {resources.length ? (
                <ul>
                  {resources.map((resource) => (
                    <li key={resource.url}>
                      <Link href={resource.url} target="_blank" rel="noopener noreferrer">
                        {resource.title}<Link.Icon />
                      </Link>
                    </li>
                  ))}
                </ul>
              ) : <p>No additional resources are recorded.</p>}
              <ExampleComparison capability={capability} />
            </div>
          </Tabs.Panel>
          <Tabs.Panel id="feedback">
            <div className="feature-panel-content feature-feedback">
              <FeedbackForm
                kind="capability"
                subject={capability.id}
                revision={database.mcp_spec}
                about={feedbackAbout}
                onClearAbout={() => setFeedbackAbout(undefined)}
              />
              <DocsLink capability={capability} />
            </div>
          </Tabs.Panel>
        </Tabs>

      </div>
    </article>
  );
}
