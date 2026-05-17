import type { SequenceResponse } from "@/types/panel";

interface Props {
  results: SequenceResponse | null;
}

export default function ScoreCards({ results }: Props) {
  if (!results) {
    return (
      <div className="grid grid-cols-3 gap-2">
        {["Greedy nearest", "Cost-aware", "Trained policy"].map((label) => (
          <div key={label} className="bg-gray-100 rounded-lg p-3">
            <div className="text-[11px] text-gray-400">{label}</div>
            <div className="text-lg font-medium text-gray-300">—</div>
          </div>
        ))}
      </div>
    );
  }

  const nearest = results.greedy_nearest.total_reward;
  const costAware = results.greedy_cost_aware.total_reward;
  const policy = results.policy?.total_reward ?? null;

  const improvement =
    policy !== null && Math.abs(nearest) > 0.01
      ? ((policy - nearest) / Math.abs(nearest)) * 100
      : null;

  return (
    <div className="grid grid-cols-3 gap-2">
      <Card
        label="Greedy nearest"
        value={nearest.toFixed(1)}
        sub={`${results.greedy_nearest.collision_count} collisions`}
      />
      <Card
        label="Cost-aware"
        value={costAware.toFixed(1)}
        sub={`${results.greedy_cost_aware.collision_count} collisions`}
      />
      <Card
        label="Trained policy"
        value={policy !== null ? policy.toFixed(1) : "—"}
        sub={
          improvement !== null
            ? `${improvement > 0 ? "+" : ""}${improvement.toFixed(1)}% vs nearest`
            : policy === null
              ? "no model selected"
              : undefined
        }
        highlight={improvement !== null && improvement > 0}
        collisions={results.policy?.collision_count}
      />
    </div>
  );
}

function Card({
  label,
  value,
  sub,
  highlight,
  collisions,
}: {
  label: string;
  value: string;
  sub?: string;
  highlight?: boolean;
  collisions?: number;
}) {
  return (
    <div className="bg-gray-100 rounded-lg p-3">
      <div className="text-[11px] text-gray-400">{label}</div>
      <div className={`text-lg font-medium ${highlight ? "text-blue-600" : "text-gray-600"}`}>
        {value}
      </div>
      {sub && (
        <div className={`text-[11px] ${highlight ? "text-green-600" : "text-gray-400"}`}>
          {sub}
        </div>
      )}
      {collisions !== undefined && (
        <div className="text-[11px] text-gray-400">{collisions} collisions</div>
      )}
    </div>
  );
}
