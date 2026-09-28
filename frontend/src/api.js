/**
 * Thin API client — proxied to FastAPI at http://localhost:8000
 */
const BASE = '';

export async function fetchModels() {
  const res = await fetch(`${BASE}/api/models`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json(); // { models: [...] }
}

export async function runInfer({ gloss, checkpoint, numBeams = 4, threshold = 0.5 }) {
  const res = await fetch(`${BASE}/api/infer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      gloss,
      checkpoint,
      num_beams: numBeams,
      threshold,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchMetrics(experiment) {
  const res = await fetch(`${BASE}/api/metrics/${experiment}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
