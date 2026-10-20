import { useEffect, useRef, useState } from 'react';
import { fetchModels, runInfer } from '../api';
import GrammarBars from './GrammarBars';
import SemanticValidation from './SemanticValidation';
import TransformationText from './TransformationText';
import './InferencePanel.css';

const SAMPLES = [
  ['Past tense', 'ME GO STORE YESTERDAY'],
  ['Negation', 'BOY NOT WANT FOOD'],
  ['Word order', 'WOMAN BUY CAR RED'],
  ['Prepositions', 'SHE PUT BOOK ON TABLE'],
  ['Everyday text', 'i going school tomorrow'],
  ['Question', 'WHERE YOU GO TOMORROW'],
  ['Ambiguous', 'SHE SEE MAN TELESCOPE'],
];

export default function InferencePanel() {
  const [models, setModels] = useState([]);
  const [checkpoint, setCheckpoint] = useState('');
  const [modelError, setModelError] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [gloss, setGloss] = useState('');
  const [numBeams, setNumBeams] = useState(4);
  const [threshold, setThreshold] = useState(0.5);
  const [alternatives, setAlternatives] = useState(1);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [copyMessage, setCopyMessage] = useState('');
  const controller = useRef(null);
  const mounted = useRef(true);

  async function loadModels() {
    setRefreshing(true); setModelError('');
    try {
      const data = await fetchModels();
      if (!mounted.current) return;
      const working = data.models.filter(m => m.available);
      setModels(working);
      setCheckpoint(current => working.some(m => m.checkpoint === current)
        ? current : (working.find(m => m.default)?.checkpoint || working[0]?.checkpoint || ''));
    } catch (e) { if (mounted.current) setModelError(e.message); }
    finally { if (mounted.current) setRefreshing(false); }
  }
  useEffect(() => {
    mounted.current = true;
    fetchModels().then(data => {
      if (!mounted.current) return;
      const working = data.models.filter(m => m.available);
      setModels(working);
      setCheckpoint(working.find(m => m.default)?.checkpoint || working[0]?.checkpoint || '');
    }).catch(e => { if (mounted.current) setModelError(e.message); });
    return () => { mounted.current = false; controller.current?.abort(); };
  }, []);

  const selected = models.find(m => m.checkpoint === checkpoint);
  const neuralReady = models.some(m => m.available && m.debug.model_type !== 'copy');
  const validSettings = Number.isInteger(numBeams) && numBeams >= 1 && numBeams <= 10
    && Number.isFinite(threshold) && threshold >= 0 && threshold <= 1 && Number.isInteger(alternatives) && alternatives >= 1 && alternatives <= Math.min(5, numBeams);
  const canRun = gloss.trim().length > 0 && gloss.length <= 2000 && selected?.available && validSettings && !loading;

  function editInput(value) { setGloss(value); setResult(null); setError(''); setCopyMessage(''); }
  async function handleRun(event) {
    event?.preventDefault();
    if (!canRun || controller.current) return;
    const activeController = new AbortController();
    controller.current = activeController;
    setLoading(true); setError(''); setResult(null); setCopyMessage('');
    try {
      const data = await runInfer({ gloss: gloss.trim(), checkpoint, numBeams, threshold, alternatives, signal: activeController.signal });
      if (mounted.current) {
        const run = { ...data, instant: event?.type === 'keydown' || event?.nativeEvent?.submitter?.matches(':focus-visible'), id: crypto.randomUUID(), tested_at: new Date().toISOString(), num_beams: numBeams, threshold };
        setResult(run); setHistory(previous => [run, ...previous].slice(0, 20));
      }
    } catch (e) { if (mounted.current && e.name !== 'AbortError') setError(e.message); }
    finally { controller.current = null; if (mounted.current) setLoading(false); }
  }
  async function copyResult() {
    try { await navigator.clipboard.writeText(result.reconstruction); setCopyMessage('Copied to clipboard'); }
    catch { setCopyMessage('Clipboard unavailable. Select and copy the sentence manually.'); }
  }
  function exportHistory() {
    const blob = new Blob([JSON.stringify({ project: 'GlossWeaver', runs: history }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'glossweaver-session.json'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return <div className="workbench">
    <div className="page-header">
      <h1 className="page-title">Reconstruction Studio</h1>
      <p className="page-subtitle">Enter a gloss or incomplete sentence. Compare the result with what you meant.</p>
    </div>
    <div className="model-status" role="status">
      <span className={`status-dot ${neuralReady ? 'ready' : ''}`} />
      <div><strong>{models.length ? (neuralReady ? 'Model active · Neural inference ready' : 'Copy baseline ready') : 'Checking model availability'}</strong>
        {!neuralReady && <p>Trained weights are required to restore grammar.</p>}</div>
      <button className="text-button" onClick={loadModels} disabled={refreshing || loading}>{refreshing ? 'Checking…' : 'Refresh status'}</button>
    </div>
    {modelError && <p className="error-banner" role="alert">{modelError}</p>}
    <div className="infer-grid">
      <form className="editor-pane" onSubmit={handleRun}>
        <div className="pane-heading"><h2>Your input</h2><span className="quiet-label">Short text or gloss</span></div>
        <label className="sr-only" htmlFor="gloss-input">Text to reconstruct</label>
        <textarea id="gloss-input" className="form-textarea" placeholder="Enter a fragmented sentence…" rows={4}
          value={gloss} onChange={e => editInput(e.target.value)} disabled={loading} maxLength={2000} spellCheck={false}
          aria-describedby="input-help" onKeyDown={e => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') handleRun(e); }} />
        <div className="input-meta" id="input-help"><span>Short, single sentences work best.</span><span>{gloss.length}/2,000</span></div>
        <div className="sample-section"><p className="form-label">Try an example</p><div className="examples-list">
          {SAMPLES.map(([label, text]) => <button key={label} type="button" className="example-btn" disabled={loading} onClick={() => editInput(text)} title={text}>{label}</button>)}
        </div></div>
        <label className="form-label" htmlFor="model-select">Reconstruction model</label>
        <select id="model-select" className="form-select" value={checkpoint} disabled={loading || !models.length}
          onChange={e => { setCheckpoint(e.target.value); setResult(null); setError(''); }}>
          {!models.length && <option value="">Waiting for API…</option>}
          {models.map(m => <option key={m.checkpoint} value={m.checkpoint}>{m.name}</option>)}
        </select>
        {selected?.debug.model_type === 'pretrained' && <p className="form-hint">{selected.debug.training_data_description}</p>}
        {selected?.debug.model_type === 'copy' && <p className="form-hint">This baseline normalizes your text. It cannot fill missing grammar.</p>}
        <details className="advanced-options"><summary>Advanced settings</summary><div className="form-row">
          <div><label className="form-label" htmlFor="beam-input">Beam width</label><input id="beam-input" className="form-input" type="number" min="1" max="10" value={numBeams}
            disabled={loading} onChange={e => setNumBeams(e.target.value === '' ? '' : Number(e.target.value))} /></div>
          <div><label className="form-label" htmlFor="threshold-input">Feature threshold</label><input id="threshold-input" className="form-input" type="number" min="0" max="1" step="0.05" value={threshold}
            disabled={loading} onChange={e => setThreshold(e.target.value === '' ? '' : Number(e.target.value))} /></div>
        <div><label className="form-label" htmlFor="alternative-input">Candidate count</label><input id="alternative-input" className="form-input" type="number" min="1" max={Math.min(5, numBeams)} value={alternatives} disabled={loading} onChange={e => setAlternatives(Number(e.target.value))} /></div></div><p className="form-hint">The threshold controls displayed grammar features, not output correctness.</p></details>
        {!validSettings && <p className="error-banner">Use beams 1–10, threshold 0–1, and candidate count 1–5 within the beam width.</p>}
        <div className="submit-row"><button className="btn-primary" type="submit" disabled={!canRun}>{loading ? 'Reconstructing…' : 'Reconstruct sentence'}<span aria-hidden="true">→</span></button>
          <button type="button" className="text-button" disabled={loading || !gloss} onClick={() => editInput('')}>Clear</button></div>
        <p className="shortcut-hint">⌘ / Ctrl + Enter to reconstruct</p>
      </form>
      <section className="output-pane" aria-labelledby="output-title" aria-busy={loading}>
        <div className="pane-heading"><h2 id="output-title">Reconstruction</h2>{result && <button className="text-button" onClick={copyResult}>Copy sentence</button>}</div>
        <div aria-live="polite">
          {loading && <div className="output-empty"><div className="loading-lines" aria-hidden="true"><span /><span /><span /></div><p>Reconstructing your input…</p><p className="form-hint">Controls are locked until this run finishes.</p></div>}
          {!loading && error && <p className="error-banner" role="alert">{error}</p>}
          {!loading && !result && !error && <div className="output-empty"><p>Your sentence appears here.</p><p className="form-hint">Enter text on the left, then reconstruct.</p></div>}
          {!loading && result && <div className="result-view" data-instant={result.instant ? "true" : "false"}>
            <TransformationText result={result} />
            <div className="result-meta"><span>{result.debug.model_name}</span><span>{result.elapsed_ms.toLocaleString()} ms</span></div>
            <code className="checkpoint-label">{result.model_identifier}</code>
            <SemanticValidation semantic={result.semantic} />
            {!!result.alternatives?.length && <section className="alternatives-section"><h3>Alternative interpretations available</h3><p className="form-hint">Unique beam candidates, ranked by potential polarity / quantity / time warnings, then beam order. These may be paraphrases rather than distinct interpretations.</p>{result.alternatives.map(candidate => <details key={candidate.text}><summary>{candidate.text}</summary><SemanticValidation semantic={candidate.semantic} /></details>)}</section>}
            {!!result.changes?.length && <details className="advanced-options"><summary>Original → reconstructed · text alignment</summary><div className="alignment-view">{result.changes.map((change, index) => <span key={index} className={`edit-${change.operation}`} title={change.category}>{change.generated || change.source}</span>)}</div><p className="form-hint">Surface differences only. Uncertain edits are uncategorized text changes.</p></details>}
            {result.notices.map(notice => <p className="result-notice" key={notice}>{notice}</p>)}
            <details className="advanced-options"><summary>Submitted input and settings</summary><p className="submitted-input">{result.gloss}</p><p className="form-hint">{result.checkpoint} · beams {result.num_beams} · threshold {result.threshold}</p></details>
          </div>}
          {copyMessage && <p className="form-hint">{copyMessage}</p>}
        </div>
        {result?.grammar_aware && <div className="grammar-section"><h3>Predicted grammar features</h3><p className="form-hint">Auxiliary classifier estimates, not a translation-confidence score.</p><GrammarBars grammar={result.grammar} threshold={result.threshold} grammarAware /></div>}
      </section>
    </div>
    <details className="session-section"><summary>Session history <span className="quiet-label">{history.length} runs</span></summary><section aria-labelledby="session-title">
      <div className="pane-heading"><div><h2 id="session-title">This session</h2><p className="form-hint">Up to 20 runs, kept in memory until you reload. Export to keep your observations.</p></div>
        <div className="session-actions"><button className="text-button" disabled={!history.length} onClick={exportHistory}>Export JSON</button><button className="text-button" disabled={!history.length} onClick={() => setHistory([])}>Clear history</button></div></div>
      {!history.length ? <p className="history-empty">Your first reconstruction starts the comparison log.</p> : <div className="history-list">{history.map(run => <div className="history-row" key={run.id}>
        <div><span className="quiet-label">{run.debug.model_name}</span><p className="history-input">{run.gloss}</p></div><p>{run.reconstruction}</p>
        <button className="text-button" disabled={loading} onClick={() => { setGloss(run.gloss); setResult(run); setError(''); setCopyMessage(''); }}>Review run</button>
      </div>)}</div>}
    </section></details>
  </div>;
}
