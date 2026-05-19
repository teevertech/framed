import type { ModelsResponse, PanelData, SequenceResponse } from "@/types/panel";

const BASE = "/api";

async function post<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error ?? `HTTP ${res.status}`);
  }
  return res.json();
}

async function get<T>(url: string): Promise<T> {
  const res = await fetch(`${BASE}${url}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/**
 * Generate a panel from explicit parameters (multi-opening).
 */
export async function generatePanel(params: {
  wall_length_ft: number;
  openings: Array<{ type: "window" | "door"; width_in: number }>;
  seed: number;
}): Promise<PanelData> {
  return post("/panels/generate", params);
}

/**
 * Generate a fully random panel for generalization demos.
 * The backend randomizes wall length, opening count, types, and widths.
 */
export async function randomPanel(params?: {
  seed?: number;
  max_openings?: number;
}): Promise<PanelData> {
  return post("/panels/random", params ?? {});
}

/**
 * Run greedy baselines + trained policy on a panel.
 */
export async function runSequence(params: {
  panel: PanelData;
  robot_speed?: number;
  collision_penalty_multiplier?: number;
  model_name?: string;
}): Promise<SequenceResponse> {
  return post("/sequence/run", params);
}

/**
 * List available trained model runs with rich metadata.
 */
export async function listModels(): Promise<ModelsResponse> {
  const data = await get<{ models: ModelsResponse }>("/models");
  return data.models;
}
