import { useCallback, useState } from "react";
import { generatePanel, runSequence } from "@/api/client";
import { useAnimation } from "@/hooks/useAnimation";
import PanelRenderer from "@/components/PanelRenderer";
import PlaybackControls from "@/components/PlaybackControls";
import ScoreCards from "@/components/ScoreCards";
import Sidebar from "@/components/Sidebar";
import type { PanelData, SequenceResponse } from "@/types/panel";

export default function App() {
  const [panel, setPanel] = useState<PanelData | null>(null);
  const [results, setResults] = useState<SequenceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [speed, setSpeed] = useState(1.5);

  // Two independent animation tracks: greedy nearest (left) and policy (right).
  const nearestAnim = useAnimation(results?.greedy_nearest.steps ?? null, speed);
  const policyAnim = useAnimation(
    results?.policy?.steps ?? results?.greedy_cost_aware.steps ?? null,
    speed,
  );

  const handleGenerate = useCallback(
    async (config: {
      wall_length_ft: number;
      opening_type: string;
      opening_width_in: number;
      opening_center_pct: number;
      seed: number;
    }) => {
      setLoading(true);
      setError(null);
      setResults(null);
      try {
        // Convert center percentage to inches.
        const wallIn = config.wall_length_ft * 12;
        const centerIn = (config.opening_center_pct / 100) * wallIn;

        const p = await generatePanel({
          wall_length_ft: config.wall_length_ft,
          opening_type: config.opening_type,
          opening_width_in: config.opening_width_in,
          opening_center_x_in: centerIn,
          seed: config.seed,
        });
        setPanel(p);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to generate panel");
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const handleRunSequence = useCallback(
    async (config: { model_name: string; collision_penalty: number }) => {
      if (!panel) return;
      setLoading(true);
      setError(null);
      try {
        const r = await runSequence({
          panel,
          collision_penalty_multiplier: config.collision_penalty,
          model_name: config.model_name || undefined,
        });
        setResults(r);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to run sequence");
      } finally {
        setLoading(false);
      }
    },
    [panel],
  );

  // Synchronize play/pause across both panels.
  const handlePlay = useCallback(() => {
    nearestAnim.play();
    policyAnim.play();
  }, [nearestAnim, policyAnim]);

  const handlePause = useCallback(() => {
    nearestAnim.pause();
    policyAnim.pause();
  }, [nearestAnim, policyAnim]);

  const handleStepForward = useCallback(() => {
    nearestAnim.stepForward();
    policyAnim.stepForward();
  }, [nearestAnim, policyAnim]);

  const handleStepBack = useCallback(() => {
    nearestAnim.stepBack();
    policyAnim.stepBack();
  }, [nearestAnim, policyAnim]);

  const handleSeek = useCallback(
    (step: number) => {
      nearestAnim.seekTo(step);
      policyAnim.seekTo(step);
    },
    [nearestAnim, policyAnim],
  );

  const rightLabel = results?.policy ? "Trained policy" : "Cost-aware greedy";
  const rightSublabel = results?.policy ? "" : "(no model selected)";

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="border-b border-gray-200 bg-white px-6 py-3">
        <div className="flex items-center gap-3">
          <h1 className="font-medium text-gray-900">framed</h1>
          <span className="text-xs text-gray-400">panel assembly sequencer</span>
        </div>
      </header>

      {/* Main content */}
      <div className="flex gap-4 p-4">
        <Sidebar onGenerate={handleGenerate} onRunSequence={handleRunSequence} loading={loading} />

        <main className="flex-1 min-w-0 space-y-3">
          {/* Error banner */}
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-4 py-2">
              {error}
            </div>
          )}

          {/* Score cards */}
          <ScoreCards results={results} />

          {/* Side-by-side animation */}
          <div className="bg-white border border-gray-200 rounded-xl p-4">
            {panel ? (
              <>
                <div className="grid grid-cols-2 gap-3">
                  <PanelRenderer
                    panel={panel}
                    frame={nearestAnim.frame}
                    label="Greedy nearest"
                  />
                  <PanelRenderer
                    panel={panel}
                    frame={policyAnim.frame}
                    label={rightLabel}
                    sublabel={rightSublabel}
                  />
                </div>

                {results && (
                  <div className="mt-3">
                    <PlaybackControls
                      playing={nearestAnim.playing}
                      stepIndex={nearestAnim.stepIndex}
                      totalSteps={nearestAnim.totalSteps}
                      speed={speed}
                      onPlay={handlePlay}
                      onPause={handlePause}
                      onStepForward={handleStepForward}
                      onStepBack={handleStepBack}
                      onSeek={handleSeek}
                      onSpeedChange={setSpeed}
                    />
                  </div>
                )}
              </>
            ) : (
              <div className="flex items-center justify-center h-64 text-gray-400 text-sm">
                Generate a panel to get started
              </div>
            )}
          </div>

          {/* Member info */}
          {panel && (
            <div className="bg-white border border-gray-200 rounded-xl px-4 py-3">
              <div className="text-xs font-medium text-gray-900 mb-1">
                Panel: {panel.members.length} members
                <span className="text-gray-400 font-normal ml-2">
                  {(panel.wall_length / 12).toFixed(0)} ft × {(panel.wall_height / 12).toFixed(0)} ft
                </span>
              </div>
              <div className="flex flex-wrap gap-1">
                {panel.members.map((m) => {
                  const placed = nearestAnim.frame?.placedIds.has(m.id);
                  return (
                    <span
                      key={m.id}
                      className={`text-[11px] px-2 py-0.5 rounded ${
                        placed
                          ? "bg-amber-100 text-amber-800"
                          : "bg-gray-100 text-gray-500"
                      }`}
                      title={`${m.kind} — prereqs: ${m.prerequisites.join(", ") || "none"}`}
                    >
                      {m.id}
                    </span>
                  );
                })}
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
