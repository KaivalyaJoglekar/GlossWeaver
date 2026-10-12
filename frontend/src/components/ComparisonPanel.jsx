import { useEffect, useRef, useState } from 'react';
import { runCompare } from '../api';
import SemanticValidation from './SemanticValidation';
export default function ComparisonPanel({ models }) {
  const [text, setText] = useState(''); const [selected, setSelected] = useState([]); const [beams, setBeams] = useState(4);
  const [results, setResults] = useState(null); const [loading, setLoading] = useState(false); const [error, setError] = useState('');
  const controller = useRef(null);
  useEffect(() => () => controller.current?.abort(), []);
  async function compare(event) {
    event.preventDefault(); if (loading) return;
    const active = new AbortController(); controller.current = active;
    setLoading(true); setError(''); setResults(null);
    try { setResults(await runCompare({ gloss: text, models: selected, num_beams: beams, signal: active.signal })); }
    catch (e) { if (e.name !== 'AbortError') setError(e.message); }
    finally { controller.current = null; setLoading(false); }
  }
  function toggle(key) { setSelected(current => current.includes(key) ? current.filter(k => k !== key) : [...current, key]); setResults(null); }
  return <><div className="page-header"><h1 className="page-title">Model Comparison</h1><p className="page-subtitle">One input. Shared decoding settings. Independent results from the checkpoints you select.</p></div><form onSubmit={compare} className="comparison-form"><label htmlFor="compare-text" className="form-label">Shared input</label><textarea id="compare-text" className="form-textarea" placeholder="Enter a fragmented sentence…" maxLength={2000} value={text} disabled={loading} onChange={e => { setText(e.target.value); setResults(null); }} required /><fieldset disabled={loading}><legend>Select up to four models</legend><div className="model-options">{models.map(m => <label key={m.checkpoint} title={m.unavailability_reason || m.description}><input type="checkbox" checked={selected.includes(m.checkpoint)} disabled={!m.available || (!selected.includes(m.checkpoint) && selected.length >= 4)} onChange={() => toggle(m.checkpoint)} /><span>{m.name}<small>{m.available ? m.architecture : 'Weights unavailable'}</small></span></label>)}</div></fieldset><div className="comparison-actions"><label>Beam width <input type="number" min="1" max="10" value={beams} onChange={e => { setBeams(Number(e.target.value)); setResults(null); }} disabled={loading} /></label><button className="btn-primary" disabled={loading || !text.trim() || !selected.length || !Number.isInteger(beams) || beams < 1 || beams > 10}>{loading ? 'Comparing checkpoints…' : 'Compare models →'}</button></div></form>
  {models.filter(m => m.available && m.debug.model_type !== 'copy').length < 2 && <p className="form-hint">Fewer than two neural checkpoints are installed. You can compare with the copy baseline or train another matched configuration.</p>}{error && <p className="error-banner" role="alert">{error}</p>}<div aria-live="polite" aria-busy={loading} className="comparison-results">{loading && <p role="status">Running selected models sequentially to limit memory use…</p>}{results?.results.map(item => <article className="comparison-result" key={item.model}><div className="pane-heading"><h2>{models.find(m => m.checkpoint === item.model)?.name || item.model}</h2>{item.result && <span className="quiet-label">{item.result.elapsed_ms.toLocaleString()} ms</span>}</div>{item.status === 'error' ? <p className="error-banner">{item.error}</p> : <><p className="result-text">{item.result.reconstruction}</p><code>{item.result.model_identifier}</code><p className="form-hint">{item.result.grammar_aware ? `Grammar features: ${item.result.grammar_active.join(', ') || 'none above threshold'}` : 'Auxiliary grammar predictions unavailable for this checkpoint.'}</p><SemanticValidation semantic={item.result.semantic} /></>}</article>)}</div></>;
}
