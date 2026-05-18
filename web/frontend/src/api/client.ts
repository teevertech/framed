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

export async function generatePanel(params: {
  wall_length_ft: number;
  opening_type: string;
  opening_width_in: number;
  opening_center_x_in?: number;
  seed: number;
}): Promise<PanelData> {
  return post("/panels/generate", params);
}

export async function runSequence(params: {
  panel: PanelData;
  robot_speed?: number;
  collision_penalty_multiplier?: number;
  model_name?: string;
}): Promise<SequenceResponse> {
  return post("/sequence/run", params);
}

export async function listModels(): Promise<ModelsResponse> {
  const data = await get<{ models: ModelsResponse }>("/models");
  return data.models;
}
