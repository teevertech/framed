import type { ModelsResponse } from "@/types/panel";

interface Props {
  models: ModelsResponse;
  selectedModel: string;
  onSelect: (name: string) => void;
}

export default function ModelPicker({ models, selectedModel, onSelect }: Props) {
  const runs = Object.keys(models);

  if (runs.length === 0) {
    return <p className="text-xs text-c-text-3 italic">No models found</p>;
  }

  const slashIdx     = selectedModel.lastIndexOf("/");
  const selectedRun  = slashIdx >= 0 ? selectedModel.slice(0, slashIdx) : (runs[0] ?? "");
  const selectedCkpt = slashIdx >= 0 ? selectedModel.slice(slashIdx + 1) : selectedModel;
  const runMeta      = models[selectedRun];
  const checkpoints  = runMeta?.checkpoints ?? [];

  function handleRunChange(run: string) {
    const ckpts = models[run]?.checkpoints ?? [];
    const ckpt  = ckpts.find((c) => c.name === "final_model")?.name
                ?? ckpts[0]?.name
                ?? "";
    onSelect(`${run}/${ckpt}`);
  }

  return (
    <div className="space-y-2">
      <div>
        <p className="text-[10px] text-c-text-3 mb-0.5">Model run</p>
        <select
          value={selectedRun}
          onChange={(e) => handleRunChange(e.target.value)}
          className="w-full"
        >
          {runs.map((r) => <option key={r} value={r}>{r}</option>)}
        </select>
      </div>
      <div>
        <p className="text-[10px] text-c-text-3 mb-0.5">Checkpoint</p>
        <select
          value={selectedCkpt}
          onChange={(e) => onSelect(`${selectedRun}/${e.target.value}`)}
          className="w-full"
        >
          {checkpoints.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name}{c.timestep > 0 ? ` (${(c.timestep / 1000).toFixed(0)}k)` : ""}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
