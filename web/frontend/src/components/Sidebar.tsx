import { useState } from "react";
import MemberLegend from "@/components/MemberLegend";
import PlaybackControls from "@/components/PlaybackControls";

interface OpeningConfig {
  type: "window" | "door";
  width_in: number;
}

interface PanelConfig {
  wall_length_ft: number;
  openings: OpeningConfig[];
  seed: number;
}

interface Props {
  onGenerate: (config: PanelConfig) => void;
  onRandomPanel: () => void;
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

const DEFAULT_WINDOW_WIDTHS = [24, 30, 32, 36, 42, 48];
const DEFAULT_DOOR_WIDTHS = [32, 36];
const MAX_OPENINGS = 4;

function defaultWidthFor(type: "window" | "door"): number {
  return type === "window" ? 36 : 36;
}

export default function Sidebar({
  onGenerate,
  onRandomPanel,
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
    openings: [{ type: "window", width_in: 36 }],
    seed: 0,
  });

  function randomizeSeed() {
    setPanelCfg((p) => ({ ...p, seed: Math.floor(Math.random() * 10_000) }));
  }

  function updateOpening(idx: number, patch: Partial<OpeningConfig>) {
    setPanelCfg((p) => {
      const next = [...p.openings];
      const current = next[idx];
      // If switching type, snap width to a valid default for that type
      if (patch.type && patch.type !== current.type) {
        patch.width_in = defaultWidthFor(patch.type);
      }
      next[idx] = { ...current, ...patch };
      return { ...p, openings: next };
    });
  }

  function addOpening() {
    if (panelCfg.openings.length >= MAX_OPENINGS) return;
    setPanelCfg((p) => ({
      ...p,
      openings: [...p.openings, { type: "window", width_in: 36 }],
    }));
  }

  function removeOpening(idx: number) {
    if (panelCfg.openings.length <= 1) return;
    setPanelCfg((p) => ({
      ...p,
      openings: p.openings.filter((_, i) => i !== idx),
    }));
  }

  return (
    <aside className="w-56 shrink-0 sticky top-4 max-h-[calc(100vh-2rem)]
                      overflow-y-auto space-y-3 text-sm pb-2">
      {/* Panel config */}
      <div className="card p-4 space-y-3">
        <h2 className="font-medium text-c-text-1">Panel config</h2>

        <Field label="Wall length" value={`${panelCfg.wall_length_ft} ft`}>
          <input
            type="range"
            min={8} max={16} step={1}
            value={panelCfg.wall_length_ft}
            onChange={(e) => setPanelCfg((p) => ({ ...p, wall_length_ft: Number(e.target.value) }))}
            className="w-full"
          />
        </Field>

        {/* Openings list */}
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-c-text-2 text-xs">
              Openings ({panelCfg.openings.length})
            </span>
            {panelCfg.openings.length < MAX_OPENINGS && (
              <button
                onClick={addOpening}
                className="text-xs text-c-accent hover:underline"
              >
                + Add
              </button>
            )}
          </div>

          <div className="space-y-2">
            {panelCfg.openings.map((op, idx) => {
              const widths = op.type === "window"
                ? DEFAULT_WINDOW_WIDTHS
                : DEFAULT_DOOR_WIDTHS;
              return (
                <div
                  key={idx}
                  className="rounded-md border border-c-border p-2 space-y-1.5
                             bg-c-base transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-c-text-3 text-[10px] uppercase tracking-wider">
                      Opening {idx + 1}
                    </span>
                    {panelCfg.openings.length > 1 && (
                      <button
                        onClick={() => removeOpening(idx)}
                        className="text-c-text-3 hover:text-red-500 text-xs
                                   leading-none transition-colors"
                        title="Remove opening"
                        aria-label={`Remove opening ${idx + 1}`}
                      >
                        ✕
                      </button>
                    )}
                  </div>

                  <select
                    value={op.type}
                    onChange={(e) =>
                      updateOpening(idx, {
                        type: e.target.value as "window" | "door",
                      })
                    }
                    className="w-full text-xs"
                  >
                    <option value="window">Window</option>
                    <option value="door">Door</option>
                  </select>

                  <div className="flex items-center gap-1.5">
                    <select
                      value={op.width_in}
                      onChange={(e) =>
                        updateOpening(idx, { width_in: Number(e.target.value) })
                      }
                      className="flex-1 text-xs"
                    >
                      {widths.map((w) => (
                        <option key={w} value={w}>
                          {w}"
                        </option>
                      ))}
                    </select>
                    <span className="text-c-text-3 text-[10px] whitespace-nowrap">wide</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

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

        <div className="flex gap-1.5">
          <button
            onClick={() => onGenerate(panelCfg)}
            disabled={loading}
            className="btn-primary flex-1"
          >
            {loading ? "…" : "Generate"}
          </button>
          <button
            onClick={onRandomPanel}
            disabled={loading}
            title="Random wall length, openings, types, and widths"
            className="btn-primary px-2"
          >
            🎲
          </button>
        </div>

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
