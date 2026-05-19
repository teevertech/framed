import type { PanelData } from "@/types/panel";

interface Props {
  panel: PanelData;
  collisionPenalty: number;
}

export default function CommonInfoBar({ panel, collisionPenalty }: Props) {
  const widthFt  = (panel.wall_length / 12).toFixed(0);
  const heightFt = (panel.wall_height / 12).toFixed(0);
  const openingType = panel.members.some((m) => m.kind === "sill_plate") ? "window" : "door";

  return (
    <div className="card px-4 py-3 flex items-center text-sm text-c-text-2">
      <span>{widthFt} ft × {heightFt} ft</span>
      <Dot />
      <span>{panel.members.length} members</span>
      <Dot />
      <span>{openingType}</span>
      <Dot />
      <span>k = {collisionPenalty.toFixed(1)}</span>
    </div>
  );
}

function Dot() {
  return <span className="text-c-border mx-2">·</span>;
}
