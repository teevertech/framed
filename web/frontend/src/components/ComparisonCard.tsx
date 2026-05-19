import PanelRenderer from "@/components/PanelRenderer";
import type { PanelData, SequenceResult } from "@/types/panel";

type FrameProp = Parameters<typeof PanelRenderer>[0]["frame"];

interface Props {
  panel: PanelData;
  frame: FrameProp;
  label: string;
  result: SequenceResult | null;
  comparisonReward?: number;
  children?: React.ReactNode;
}

export default function ComparisonCard({
  panel,
  frame,
  label,
  result,
  comparisonReward,
  children,
}: Props) {
  const travel = result
    ? result.steps.reduce((sum, s) => sum + s.travel_time, 0)
    : null;

  const improvement =
    result && comparisonReward !== undefined && comparisonReward !== 0
      ? ((result.total_reward - comparisonReward) / Math.abs(comparisonReward)) * 100
      : null;

  return (
    <div className="card p-4 flex flex-col gap-3">
      <p className="text-xs font-medium text-c-text-2 text-center">{label}</p>

      <PanelRenderer panel={panel} frame={frame} label="" />

      {result && (
        <div className="grid grid-cols-2 gap-x-3 gap-y-1">
          <StatRow label="Reward"     value={result.total_reward.toFixed(1)} />
          <StatRow label="Collisions" value={String(result.collision_count)} />
          <StatRow label="Travel"     value={`${travel!.toFixed(1)}s`} />
          {improvement !== null && (
            <StatRow
              label="vs nearest"
              value={`${improvement >= 0 ? "+" : ""}${improvement.toFixed(1)}%`}
              valueClass={improvement >= 0 ? "text-green-600" : "text-red-500"}
            />
          )}
        </div>
      )}

      {children && <div>{children}</div>}
    </div>
  );
}

function StatRow({
  label,
  value,
  valueClass = "text-c-text-1",
}: {
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <>
      <span className="text-xs text-c-text-3">{label}</span>
      <span className={`text-sm font-medium ${valueClass}`}>{value}</span>
    </>
  );
}
