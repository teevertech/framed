import { useState } from "react";
import MemberLegend from "@/components/MemberLegend";
import PlaybackControls from "@/components/PlaybackControls";

interface PanelConfig {
  wall_length_ft: number;
  opening_type: string;
  opening_width_in: number;
  opening_center_pct: number;
  seed: number;
}

interface Props {
  onGenerate: (config: PanelConfig) => void;
  onRunSequence: () => void;
  loading: boolean;

  collisionPenalty: number;
  onCollisionPenaltyChange: (val: number) => void;

  playing: boolean;
  stepIndex: number;
  totalSteps: number;
  speed: number;
  onPlay: () => void;
  onPause: () => void;
  onStepForward: () => void;
  onStepBack: () => void;
  onSeek: (step: number) => void;
  onSpeedChange: (speed: number) => void;

  darkMode: boolean;
  onToggleDark: (val: boolean) => void;
}

export default function Sidebar({
  onGenerate,
  onRunSequence,
  loading,
  collisionPenalty,
  onCollisionPenaltyChange,
  playing,
  stepIndex,
  totalSteps,
  speed,
  onPlay,
  onPause,
  onStepForward,
  onStepBack,
  onSeek,
  onSpeedChange,
  darkMode,
  onToggleDark,
}: Props) {
  const [panelCfg, setPanelCfg] = useState<PanelConfig>({
    wall_length_ft: 12,
    opening_type: "window",
    opening_width_in: 36,
    opening_center_pct: 50,
    seed: 0,
  });

  function randomizeSeed() {
    setPanelCfg((p) => ({ ...p, seed: Math.floor(Math.random() * 10_000) }));
  }

  return (
    // sticky + max-h + overflow-y-auto keeps the sidebar from pushing
    // below the viewport on shorter screens. Full layout pass later.
    <aside className="w-56 shrink-0 sticky top-4 max-h-[calc(100vh-2rem)]
                      overflow-y-auto space-y-3 text-sm pb-2">
      {/* Panel config */}
      <div className="card p-4 space-y-3">
        <h2 className="font-medium text-c-text-1">Panel config</h2>

        <Field label="Wall length" value={`${panelCfg.wall_length_ft} ft`}>
          <input
            type="range"
            min={8} max={20} step={1}
            value={panelCfg.wall_length_ft}
            onChange={(e) => setPanelCfg((p) => ({ ...p, wall_length_ft: Number(e.target.value) }))}
            className="w-full"
          />
        </Field>

        <Field label="Opening type">
          <select
            value={panelCfg.opening_type}
            onChange={(e) => setPanelCfg((p) => ({ ...p, opening_type: e.target.value }))}
            className="w-full"
          >
            <option value="window">Window</option>
            <option value="door">Door</option>
          </select>
        </Field>

        <Field label="Opening width" value={`${panelCfg.opening_width_in} in`}>
          <input
            type="range"
            min={24} max={60} step={2}
            value={panelCfg.opening_width_in}
            onChange={(e) => setPanelCfg((p) => ({ ...p, opening_width_in: Number(e.target.value) }))}
            className="w-full"
          />
        </Field>

        <Field label="Opening position" value={`${panelCfg.opening_center_pct}%`}>
          <input
            type="range"
            min={20} max={80} step={1}
            value={panelCfg.opening_center_pct}
            onChange={(e) => setPanelCfg((p) => ({ ...p, opening_center_pct: Number(e.target.value) }))}
            className="w-full"
          />
        </Field>

        {/* Seed row with randomize button */}
        <div>
          <span className="text-c-text-2 text-xs block mb-1">Seed</span>
          <div className="flex gap-1.5">
            <input
              type="number"
              min={0}
              value={panelCfg.seed}
              onChange={(e) => setPanelCfg((p) => ({ ...p, seed: Number(e.target.value) }))}
              className="flex-1 min-w-0"
            />
            <button
              onClick={randomizeSeed}
              title="Random seed"
              aria-label="Randomize seed"
              className="px-2 py-1 rounded-md border border-c-border text-base leading-none
                         hover:bg-c-subtle transition-colors"
            >
              🎲
            </button>
          </div>
        </div>

        <Field label="Collision penalty (k)" value={collisionPenalty.toFixed(1)}>
          <input
            type="range"
            min={0} max={6} step={0.5}
            value={collisionPenalty}
            onChange={(e) => onCollisionPenaltyChange(Number(e.target.value))}
            className="w-full"
          />
        </Field>

        <button onClick={() => onGenerate(panelCfg)} disabled={loading} className="btn-primary">
          {loading ? "Generating…" : "Generate panel"}
        </button>

        <button onClick={onRunSequence} disabled={loading} className="btn-accent">
          {loading ? "Running…" : "Run comparison"}
        </button>
      </div>

      {/* Playback — only shown once there are results */}
      {totalSteps > 0 && (
        <div className="card p-4">
          <h2 className="font-medium text-c-text-1 mb-3">Playback</h2>
          <PlaybackControls
            playing={playing}
            stepIndex={stepIndex}
            totalSteps={totalSteps}
            speed={speed}
            onPlay={onPlay}
            onPause={onPause}
            onStepForward={onStepForward}
            onStepBack={onStepBack}
            onSeek={onSeek}
            onSpeedChange={onSpeedChange}
          />
        </div>
      )}

      {/* Legend */}
      <MemberLegend />

    </aside>
  );
}

function Field({
  label,
  value,
  children,
}: {
  label: string;
  value?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="flex justify-between mb-1">
        <span className="text-c-text-2 text-xs">{label}</span>
        {value && <span className="text-c-text-3 text-xs">{value}</span>}
      </div>
      {children}
    </div>
  );
}
