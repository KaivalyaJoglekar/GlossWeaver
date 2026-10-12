async function request(path, options = {}) {
  let response;
  try { response = await fetch(path, options); }
  catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error('Cannot reach the API. Start the backend, then refresh model status.');
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = Array.isArray(data.detail)
      ? data.detail.map(item => `${item.loc.at(-1)}: ${item.msg}`).join('; ')
      : data.detail;
    throw new Error(message || `Request failed (${response.status})`);
  }
  return data;
}
export const fetchModels = () => request('/api/models');
export const fetchMetrics = () => request('/api/metrics');
export const runInfer = ({ gloss, checkpoint, numBeams = 4, threshold = 0.5, alternatives = 1, signal }) => request('/api/infer', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, signal,
  body: JSON.stringify({ gloss, checkpoint, num_beams: numBeams, threshold, alternatives }),
});
export const fetchDatasets = () => request('/api/datasets');
export const fetchHealth = () => request('/api/health');
export const fetchEvaluations = () => request('/api/evaluations');
export const fetchExamples = (id, category = '') => request(`/api/evaluations/${encodeURIComponent(id)}/examples?category=${encodeURIComponent(category)}`);
export const runCompare = ({ signal, ...payload }) => request('/api/compare', { method: 'POST', headers: { 'Content-Type': 'application/json' }, signal, body: JSON.stringify(payload) });
