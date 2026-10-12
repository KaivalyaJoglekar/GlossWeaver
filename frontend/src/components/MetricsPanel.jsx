import { useEffect, useState } from 'react';
import { fetchMetrics } from '../api';
import './MetricsPanel.css';
const COLUMNS = [['bleu', 'BLEU', 2], ['chrf', 'chrF', 2], ['rouge_l', 'ROUGE-L', 3], ['meteor', 'METEOR', 3], ['bertscore_f1', 'BERTScore', 3], ['content_word_recall', 'Content recall', 3], ['exact_match', 'Exact match', 3]];
const defaultEvaluation = next => JSON.stringify(next.evaluations.find(item => item.dataset === 'balanced_development') || next.evaluations.find(item => item.dataset === 'reconstruction_grouped') || next.evaluations[0] || {});
const format = (value, digits) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
export default function MetricsPanel() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [evaluation, setEvaluation] = useState('');
  const [model, setModel] = useState('');
  const [filter, setFilter] = useState('');
  const [sort, setSort] = useState('name');
  async function load() {
    setError('');
    try {
      const next = await fetchMetrics(); setData(next);
      setEvaluation(current => current || defaultEvaluation(next));
    } catch (e) { setError(e.message); }
  }
  useEffect(() => { let active = true; fetchMetrics().then(next => {
    if (active) { setData(next); setEvaluation(defaultEvaluation(next)); }
  }).catch(e => { if (active) setError(e.message); }); return () => { active = false; }; }, []);
  const selection = evaluation ? JSON.parse(evaluation) : {};
  const rows = (data?.rows.filter(row => row.dataset === selection.dataset && row.split === selection.split && row.display_name.toLowerCase().includes(filter.toLowerCase())) || []).sort((a, b) => sort === 'name' ? a.display_name.localeCompare(b.display_name) : (b[sort] ?? -Infinity) - (a[sort] ?? -Infinity));
  const active = rows.find(row => (row.evaluation_id || row.model) === model) || rows.find(row => row.model === 'e4_glossweaver') || rows[0];
  return <div>
    <div className="page-header"><h1 className="page-title">Evaluation results.</h1><p className="page-subtitle">Compare saved evaluations within the same dataset and split. Historical scores do not imply that model weights are currently installed.</p></div>
    <div className="metrics-toolbar"><div><label className="form-label" htmlFor="evaluation-select">Dataset and evaluation split</label><select id="evaluation-select" className="form-select" value={evaluation} onChange={e => { setEvaluation(e.target.value); setModel(''); }} disabled={!data}>
      {!data && <option value="">Loading evaluations…</option>}{data?.evaluations.map(item => <option key={JSON.stringify(item)} value={JSON.stringify(item)}>{item.dataset} · {item.split}</option>)}</select></div><div><label className="form-label" htmlFor="metric-filter">Filter models</label><input id="metric-filter" className="form-input" value={filter} onChange={e => setFilter(e.target.value)} placeholder="Model name" /></div><div><label className="form-label" htmlFor="metric-sort">Sort results</label><select id="metric-sort" className="form-select" value={sort} onChange={e => setSort(e.target.value)}><option value="name">Model name</option>{COLUMNS.map(([key,label]) => <option key={key} value={key}>{label} · descending</option>)}</select></div><button className="text-button" onClick={load}>Refresh results</button></div>
    {error && <p className="error-banner" role="alert">{error}</p>}
    {!data && !error && <p role="status">Loading saved results…</p>}
    {data && <><p className="benchmark-note">{data.provenance} {selection.dataset === 'challenge' && 'The nine-example challenge is a development/regression set.'} {selection.split === 'development_diagnostic' && 'Development diagnostics, not final research results. The balanced validation set also guided checkpoint selection.'} {selection.dataset === 'reconstruction_grouped' && 'This sample has been inspected during development; it is not an untouched final benchmark.'}</p>
      {!rows.length ? <p className="history-empty">No evaluation rows exist for this selection.</p> : <div className="table-responsive"><table className="benchmark-table"><caption>Saved reconstruction metrics · {selection.dataset} / {selection.split}</caption><thead><tr><th scope="col">Model</th><th scope="col">Examples</th>{COLUMNS.map(([key, label]) => <th scope="col" key={key}>{label}</th>)}</tr></thead><tbody>
        {rows.map((row, index) => <tr key={row.evaluation_id || `${row.model}-${index}`}><th scope="row"><span>{row.display_name}</span><small>{row.evaluation_id && `Run ${row.evaluation_id} · seed ${row.seed} · `}{row.model === 'e0_copy' ? 'Copy baseline available' : row.checkpoint_available ? 'Weights available' : 'Saved result · weights unavailable'}</small></th><td>{row.examples}</td>{COLUMNS.map(([key, , digits]) => <td key={key}>{format(row[key], digits)}</td>)}</tr>)}
      </tbody></table></div>}
      <p className="form-hint">— means the metric was not stored. Content recall checks lexical overlap; it does not verify semantic correctness.</p>
      {active && <section className="diagnostics-section"><div className="pane-heading"><div><h2>Output grammar diagnostics</h2><p className="form-hint">Heuristic features compared with the primary reference. These do not measure the classifier’s accuracy.</p></div><select className="form-select" aria-label="Model for grammar diagnostics" value={active.evaluation_id || active.model} onChange={e => setModel(e.target.value)}>{rows.map((row, index) => <option key={row.evaluation_id || `${row.model}-${index}`} value={row.evaluation_id || row.model}>{row.display_name}{row.evaluation_id ? ` · ${row.evaluation_id}` : ' · historical'}</option>)}</select></div>
        <div className="diagnostic-list">{Object.entries(active.grammar_diagnostics || {}).map(([category, item]) => <div className="diagnostic-row" key={category}><span>{category.replaceAll('_', ' ').toLowerCase()}</span><div className="diagnostic-track"><span style={{ width: `${item.f1 * 100}%` }} /></div><span>F1 {format(item.f1, 3)}</span><span>n = {item.support}</span></div>)}</div>
        {!Object.keys(active.grammar_diagnostics || {}).length && <p className="form-hint">No grammar diagnostics were saved for this evaluation.</p>}
      </section>}
    </>}
  </div>;
}
