import { useCallback, useEffect, useState } from "react";
import { generatePanel, listModels, runSequence } from "@/api/client";
import { useAnimation } from "@/hooks/useAnimation";
import CommonInfoBar from "@/components/CommonInfoBar";
import ComparisonCard from "@/components/ComparisonCard";
import ModelPicker from "@/components/ModelPicker";
import Sidebar from "@/components/Sidebar";
import { applyTheme, darkTheme, lightTheme } from "@/theme";
import type { ModelsResponse, PanelData, SequenceResponse } from "@/types/panel";

export default function App() {
  const [panel, setPanel] = useState<PanelData | null>(null);
  const [results, setResults] = useState<SequenceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [speed, setSpeed] = useState(1.5);

  const [models, setModels] = useState<ModelsResponse>({});
  const [selectedModel, setSelectedModel] = useState("");
  const [collisionPenalty, setCollisionPenalty] = useState(4.0);

  const [darkMode, setDarkMode] = useState<boolean>(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("framed-dark") === "true";
    }
    return false;
  });

  // Apply theme by writing CSS variables directly onto <html>'s inline
  // style — bypasses any stylesheet-related processing entirely.
  useEffect(() => {
    applyTheme(darkMode ? darkTheme : lightTheme);
    localStorage.setItem("framed-dark", String(darkMode));
  }, [darkMode]);

  useEffect(() => {
    listModels()
      .then((m) => {
        setModels(m);
        for (const [run, ckpts] of Object.entries(m)) {
          if (ckpts.includes("final_model")) {
            setSelectedModel(`${run}/final_model`);
            return;
          }
        }
        const firstRun = Object.keys(m)[0];
        if (firstRun) {
          const firstCkpt = m[firstRun]?.[0];
          if (firstCkpt) setSelectedModel(`${firstRun}/${firstCkpt}`);
        }
      })
      .catch(() => {});
  }, []);

  const nearestAnim = useAnimation(results?.greedy_nearest.steps ?? null, speed);
  const policyAnim = useAnimation(
    results?.policy?.steps ?? results?.greedy_cost_aware.steps ?? null,
    speed,
  );

  useEffect(() => {
    if (results) {
      nearestAnim.play();
      policyAnim.play();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [results]);

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

  const handleRunSequence = useCallback(async () => {
    if (!panel) return;
    setLoading(true);
    setError(null);
    try {
      const r = await runSequence({
        panel,
        collision_penalty_multiplier: collisionPenalty,
        model_name: selectedModel || undefined,
      });
      setResults(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to run sequence");
    } finally {
      setLoading(false);
    }
  }, [panel, collisionPenalty, selectedModel]);

  const handlePlay        = useCallback(() => { nearestAnim.play();        policyAnim.play();        }, [nearestAnim, policyAnim]);
  const handlePause       = useCallback(() => { nearestAnim.pause();       policyAnim.pause();       }, [nearestAnim, policyAnim]);
  const handleStepForward = useCallback(() => { nearestAnim.stepForward(); policyAnim.stepForward(); }, [nearestAnim, policyAnim]);
  const handleStepBack    = useCallback(() => { nearestAnim.stepBack();    policyAnim.stepBack();    }, [nearestAnim, policyAnim]);
  const handleSeek        = useCallback((step: number) => { nearestAnim.seekTo(step); policyAnim.seekTo(step); }, [nearestAnim, policyAnim]);

  const rightResult = results?.policy ?? results?.greedy_cost_aware ?? null;
  const rightLabel  = results?.policy ? "Trained policy" : "Cost-aware greedy";

  return (
    <div className="min-h-screen bg-c-base">
      <header className="border-b border-c-border bg-c-surface px-6 py-3 transition-colors">
        <div className="flex items-center gap-3">
          <h1 className="font-medium text-c-text-1">framed</h1>
          <span className="text-xs text-c-text-3">panel assembly sequencer</span>
        </div>
      </header>

      <div className="flex gap-4 p-4">
        <Sidebar
          onGenerate={handleGenerate}
          onRunSequence={handleRunSequence}
          loading={loading}
          collisionPenalty={collisionPenalty}
          onCollisionPenaltyChange={setCollisionPenalty}
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
          darkMode={darkMode}
          onToggleDark={setDarkMode}
        />

        <main className="flex-1 min-w-0 space-y-3">
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-4 py-2">
              {error}
            </div>
          )}

          {panel && (
            <CommonInfoBar panel={panel} collisionPenalty={collisionPenalty} />
          )}

          {panel ? (
            <div className="grid grid-cols-2 gap-3">
              <ComparisonCard
                panel={panel}
                frame={nearestAnim.frame}
                label="Greedy nearest"
                result={results?.greedy_nearest ?? null}
              />
              <ComparisonCard
                panel={panel}
                frame={policyAnim.frame}
                label={rightLabel}
                result={rightResult}
                comparisonReward={results?.greedy_nearest.total_reward}
              >
                <ModelPicker
                  models={models}
                  selectedModel={selectedModel}
                  onSelect={setSelectedModel}
                />
              </ComparisonCard>
            </div>
          ) : (
            <div className="card flex items-center justify-center h-64 text-c-text-3 text-sm">
              Generate a panel to get started
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
