import { useEffect, useState } from "react";
import type { ModelInfo } from "@/types/panel";
import { listModels } from "@/api/client";

interface PanelConfig {
  wall_length_ft: number;
  opening_type: string;
  opening_width_in: number;
  opening_center_pct: number;
  seed: number;
}

interface SequenceConfig {
  model_name: string;
  collision_penalty: number;
}

interface Props {
  onGenerate: (config: PanelConfig) => void;
  onRunSequence: (config: SequenceConfig) => void;
  loading: boolean;
}

export default function Sidebar({ onGenerate, onRunSequence, loading }: Props) {
  const [panelCfg, setPanelCfg] = useState<PanelConfig>({
    wall_length_ft: 12,
    opening_type: "window",
    opening_width_in: 36,
    opening_center_pct: 50,
    seed: 0,
  });

  const [seqCfg, setSeqCfg] = useState<SequenceConfig>({
    model_name: "",
    collision_penalty: 4.0,
  });

  const [models, setModels] = useState<ModelInfo[]>([]);

  useEffect(() => {
    listModels()
      .then((m) => {
        setModels(m);
        // Auto-select first "final_model" if available.
        const final = m.find((x) => x.name.includes("final_model"));
        if (final) setSeqCfg((prev) => ({ ...prev, model_name: final.name }));
        else if (m.length > 0) setSeqCfg((prev) => ({ ...prev, model_name: m[0]!.name }));
      })
      .catch(() => {});
  }, []);

  return (
    <aside className="w-56 shrink-0 space-y-3 text-sm">
      {/* Panel config */}
      <div className="bg-white border border-gray-200 rounded-xl p-4 space-y-3">
        <h2 className="font-medium text-gray-900">Panel config</h2>

        <Field label="Wall length" value={`${panelCfg.wall_length_ft} ft`}>
          <input
            type="range"
            min={8}
            max={20}
            step={1}
            value={panelCfg.wall_length_ft}
            onChange={(e) =>
              setPanelCfg((p) => ({ ...p, wall_length_ft: Number(e.target.value) }))
            }
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
            min={24}
            max={60}
            step={2}
            value={panelCfg.opening_width_in}
            onChange={(e) =>
              setPanelCfg((p) => ({ ...p, opening_width_in: Number(e.target.value) }))
            }
            className="w-full"
          />
        </Field>

        <Field label="Opening position" value={`${panelCfg.opening_center_pct}%`}>
          <input
            type="range"
            min={20}
            max={80}
            step={1}
            value={panelCfg.opening_center_pct}
            onChange={(e) =>
              setPanelCfg((p) => ({ ...p, opening_center_pct: Number(e.target.value) }))
            }
            className="w-full"
          />
        </Field>

        <Field label="Seed">
          <input
            type="number"
            min={0}
            value={panelCfg.seed}
            onChange={(e) => setPanelCfg((p) => ({ ...p, seed: Number(e.target.value) }))}
            className="w-full"
          />
        </Field>

        <button
          onClick={() => onGenerate(panelCfg)}
          disabled={loading}
          className="w-full py-2 font-medium rounded-lg bg-gray-900 text-white hover:bg-gray-800 disabled:opacity-50 transition-colors"
        >
          {loading ? "Generating…" : "Generate panel"}
        </button>
      </div>

      {/* Sequencing controls */}
      <div className="bg-white border border-gray-200 rounded-xl p-4 space-y-3">
        <h2 className="font-medium text-gray-900">Sequencing</h2>

        <Field label="Trained model">
          <select
            value={seqCfg.model_name}
            onChange={(e) => setSeqCfg((s) => ({ ...s, model_name: e.target.value }))}
            className="w-full"
          >
            {models.length === 0 && <option value="">No models found</option>}
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Collision penalty (k)" value={seqCfg.collision_penalty.toFixed(1)}>
          <input
            type="range"
            min={0}
            max={6}
            step={0.5}
            value={seqCfg.collision_penalty}
            onChange={(e) =>
              setSeqCfg((s) => ({ ...s, collision_penalty: Number(e.target.value) }))
            }
            className="w-full"
          />
        </Field>

        <button
          onClick={() => onRunSequence(seqCfg)}
          disabled={loading}
          className="w-full py-2 font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {loading ? "Running…" : "Run comparison"}
        </button>
      </div>
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
        <span className="text-gray-500 text-xs">{label}</span>
        {value && <span className="text-gray-400 text-xs">{value}</span>}
      </div>
      {children}
    </div>
  );
}
