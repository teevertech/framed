export interface MemberData {
  id: string;
  kind: string;
  position: [number, number];
  size: [number, number];
  prerequisites: string[];
  bounds: [number, number, number, number];
  center: [number, number];
}

export interface PanelData {
  wall_length: number;
  wall_height: number;
  members: MemberData[];
}

export interface SequenceStep {
  member_id: string;
  member_index: number;
  from_xy: [number, number];
  to_xy: [number, number];
  travel_time: number;
  collided: boolean;
  reward: number;
  cumulative_reward: number;
}

export interface SequenceResult {
  total_reward: number;
  collision_count: number;
  steps: SequenceStep[];
}

export interface SequenceResponse {
  greedy_nearest: SequenceResult;
  greedy_cost_aware: SequenceResult;
  policy: SequenceResult | null;
}

export interface ModelInfo {
  name: string;
  path: string;
}

/** Lumber colors keyed by MemberKind value string. */
export const KIND_COLORS: Record<string, string> = {
  bottom_plate: "#C4A265",
  top_plate: "#C4A265",
  common_stud: "#DEB887",
  king_stud: "#CD853F",
  jack_stud: "#D2691E",
  header: "#7B5E2A",
  sill_plate: "#B8895A",
  top_cripple: "#E8C99A",
  bottom_cripple: "#E8C99A",
};
