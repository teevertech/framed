import type { AnimationFrame } from "@/hooks/useAnimation";
import type { MemberData, PanelData } from "@/types/panel";
import { KIND_COLORS } from "@/types/panel";

interface Props {
  panel: PanelData;
  frame: AnimationFrame | null;
  label?: string;
  sublabel?: string;
}

const GHOST_FILL = "#EDEDED";
const GHOST_STROKE = "#C0C0C0";
const TARGET_FILL = "#FFD54F";
const TARGET_STROKE = "#F57F17";
const PATH_CLEAR = "#43A047";
const PATH_COLLIDE = "#E53935";
const ROBOT_COLOR = "#1565C0";
const WALL_BG = "#F7F4EE";
const WALL_STROKE = "#BDBDBD";

/** Build the three vertex coordinates of the robot triangle, in wall units. */
function robotTriangle(cx: number, cy: number, size: number, dx: number, dy: number) {
  const len = Math.hypot(dx, dy);
  const angle = len > 0.001 ? Math.atan2(dy, dx) : 0;
  const tipX = cx + size * Math.cos(angle);
  const tipY = cy + size * Math.sin(angle);
  const lx = cx + size * 0.6 * Math.cos(angle + 2.3);
  const ly = cy + size * 0.6 * Math.sin(angle + 2.3);
  const rx = cx + size * 0.6 * Math.cos(angle - 2.3);
  const ry = cy + size * 0.6 * Math.sin(angle - 2.3);
  return `${tipX},${tipY} ${lx},${ly} ${rx},${ry}`;
}

function MemberRect({
  m,
  placed,
  isTarget,
  wh,
  strokeWidth,
}: {
  m: MemberData;
  placed: boolean;
  isTarget: boolean;
  wh: number;
  strokeWidth: number;
}) {
  const [xMin, yMin, xMax, yMax] = m.bounds;
  // Flip y: SVG y=0 is top, panel y=0 is bottom.
  const x = xMin;
  const y = wh - yMax;
  const w = xMax - xMin;
  const h = yMax - yMin;

  let fill: string;
  let stroke: string;
  let sw: number;
  let opacity = 1;

  if (placed) {
    fill = KIND_COLORS[m.kind] ?? "#DEB887";
    stroke = "#5D4037";
    sw = strokeWidth;
  } else if (isTarget) {
    fill = TARGET_FILL;
    stroke = TARGET_STROKE;
    sw = strokeWidth * 1.5;
    opacity = 0.85;
  } else {
    fill = GHOST_FILL;
    stroke = GHOST_STROKE;
    sw = strokeWidth * 0.6;
  }

  return (
    <rect
      x={x}
      y={y}
      width={w}
      height={h}
      fill={fill}
      stroke={stroke}
      strokeWidth={sw}
      opacity={opacity}
    />
  );
}

export default function PanelRenderer({ panel, frame, label, sublabel }: Props) {
  const { wall_length: wl, wall_height: wh } = panel;
  const margin = wh * 0.05;
  const vbX = -margin;
  const vbY = -margin;
  const vbW = wl + 2 * margin;
  const vbH = wh + 2 * margin;

  // Stroke widths and the robot size are expressed in wall units (inches)
  // so they scale with the SVG viewport.
  const memberStroke = wh * 0.004;
  const pathStroke = wh * 0.006;
  const robotSize = wh * 0.04;

  // Compute robot direction from the partial path or last completed path.
  let robotDx = 1;
  let robotDy = 0;
  if (frame?.partialPath) {
    robotDx = frame.partialPath.to[0] - frame.partialPath.from[0];
    robotDy = -(frame.partialPath.to[1] - frame.partialPath.from[1]);
  } else if (frame && frame.completedPaths.length > 0) {
    const last = frame.completedPaths[frame.completedPaths.length - 1]!;
    robotDx = last.to[0] - last.from[0];
    robotDy = -(last.to[1] - last.from[1]);
  }

  return (
    <div>
      {(label || sublabel) && (
        <div className="text-center mb-1">
          {label && <span className="text-xs text-gray-600 font-medium">{label}</span>}
          {sublabel && <span className="text-xs text-gray-400 ml-1">{sublabel}</span>}
        </div>
      )}
      <svg
        viewBox={`${vbX} ${vbY} ${vbW} ${vbH}`}
        className="w-full border border-gray-200 rounded-lg"
        style={{ background: WALL_BG }}
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Wall outline */}
        <rect
          x={0}
          y={0}
          width={wl}
          height={wh}
          fill="none"
          stroke={WALL_STROKE}
          strokeWidth={memberStroke}
        />

        {/* Members */}
        {panel.members.map((m) => (
          <MemberRect
            key={m.id}
            m={m}
            placed={frame?.placedIds.has(m.id) ?? false}
            isTarget={frame?.targetId === m.id}
            wh={wh}
            strokeWidth={memberStroke}
          />
        ))}

        {/* Completed paths */}
        {frame?.completedPaths.map((p, i) => {
          const color = p.collided ? PATH_COLLIDE : PATH_CLEAR;
          return (
            <line
              key={`path-${i}`}
              x1={p.from[0]}
              y1={wh - p.from[1]}
              x2={p.to[0]}
              y2={wh - p.to[1]}
              stroke={color}
              strokeWidth={pathStroke}
              opacity={0.7}
              strokeLinecap="round"
            />
          );
        })}

        {/* Partial (growing) path */}
        {frame?.partialPath && (
          <line
            x1={frame.partialPath.from[0]}
            y1={wh - frame.partialPath.from[1]}
            x2={frame.partialPath.to[0]}
            y2={wh - frame.partialPath.to[1]}
            stroke={frame.collidedThis ? PATH_COLLIDE : PATH_CLEAR}
            strokeWidth={pathStroke}
            opacity={0.6}
            strokeLinecap="round"
          />
        )}

        {/* Robot triangle */}
        {frame && (
          <polygon
            points={robotTriangle(
              frame.robotXY[0],
              wh - frame.robotXY[1],
              robotSize,
              robotDx,
              robotDy,
            )}
            fill={ROBOT_COLOR}
            stroke="white"
            strokeWidth={memberStroke * 0.5}
          />
        )}
      </svg>
    </div>
  );
}
