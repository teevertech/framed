import { KIND_COLORS } from "@/types/panel";

const LEGEND_ITEMS = [
  { kind: "bottom_plate",   label: "Plate"        },
  { kind: "common_stud",    label: "Common stud"  },
  { kind: "king_stud",      label: "King stud"    },
  { kind: "jack_stud",      label: "Jack stud"    },
  { kind: "header",         label: "Header"       },
  { kind: "sill_plate",     label: "Sill plate"   },
  { kind: "top_cripple",    label: "Cripple"      },
];

export default function MemberLegend() {
  return (
    <div className="card p-4 space-y-3">
      <h2 className="font-medium text-c-text-1 text-sm">Legend</h2>

      {/* Lumber types */}
      <div>
        <p className="text-[10px] text-c-text-3 uppercase tracking-wide mb-1.5">
          Lumber
        </p>
        <div className="space-y-1">
          {LEGEND_ITEMS.map(({ kind, label }) => (
            <div key={kind} className="flex items-center gap-2">
              <span
                className="w-3 h-3 rounded-sm shrink-0 border border-c-border"
                style={{ backgroundColor: KIND_COLORS[kind] }}
              />
              <span className="text-xs text-c-text-2">{label}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="border-t border-c-divider" />

      {/* Path indicators */}
      <div>
        <p className="text-[10px] text-c-text-3 uppercase tracking-wide mb-1.5">
          Robot path
        </p>
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <PathSwatch color="#43A047" />
            <span className="text-xs text-c-text-2">Clear path</span>
          </div>
          <div className="flex items-center gap-2">
            <PathSwatch color="#E53935" dashed />
            <span className="text-xs text-c-text-2">Collision detour</span>
          </div>
          <div className="flex items-center gap-2">
            <RobotSwatch />
            <span className="text-xs text-c-text-2">Robot position</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function PathSwatch({ color, dashed = false }: { color: string; dashed?: boolean }) {
  return (
    <svg width="20" height="12" viewBox="0 0 20 12" className="shrink-0">
      <line
        x1="0" y1="6" x2="20" y2="6"
        stroke={color}
        strokeWidth="2"
        strokeDasharray={dashed ? "3 2" : undefined}
        strokeLinecap="round"
      />
    </svg>
  );
}

function RobotSwatch() {
  return (
    <svg width="14" height="12" viewBox="0 0 14 12" className="shrink-0">
      <polygon points="7,1 13,11 1,11" fill="#1565C0" />
    </svg>
  );
}
