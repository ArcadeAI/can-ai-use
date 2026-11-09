import {AgentHandling} from "../components/AgentHandling";
import {FeatureModule} from "../components/FeatureModule";
import type {GridMode} from "../components/CapabilityGrid";
import type {Capability, MirrorDatabase, SpecIndex} from "../types";

export function CapabilityPage({
  database,
  capability,
  starred,
  onToggleStar,
  gridMode,
  onGridModeChange,
  specIndex,
}: {
  database: MirrorDatabase;
  capability: Capability;
  starred: boolean;
  onToggleStar: () => void;
  gridMode: GridMode;
  onGridModeChange: (mode: GridMode) => void;
  specIndex: SpecIndex | null;
}) {
  return (
    <main className="site-shell page-main capability-page" id="main-content">
      <FeatureModule
        database={database}
        capability={capability}
        starred={starred}
        onToggleStar={onToggleStar}
        gridMode={gridMode}
        onGridModeChange={onGridModeChange}
        detail
      />
      <AgentHandling capabilityId={capability.id} specIndex={specIndex} />
    </main>
  );
}
