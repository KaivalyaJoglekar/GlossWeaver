import React, { useState, useEffect } from 'react';
import { fetchModels, runInfer } from '../api';
import GrammarBars from './GrammarBars';
import './InferencePanel.css';

const EXAMPLES = [
  'ME GO STORE YESTERDAY',
  'X-I WANT BOOK',
  'DOG RUN FAST PARK',
  'WOMAN BUY CAR RED',
  'TEACHER EXPLAIN STUDENT MATH',
  'CHILD PLAY OUTSIDE SUNNY',
  'BOY THROW BALL STRONG',
  'MOTHER COOK FOOD TONIGHT',
];

export default function InferencePanel() {
  const [models, setModels] = useState([]);
  const [modelsErr, setModelsErr] = useState(null);
  const [checkpoint, setCheckpoint] = useState('');
  const [gloss, setGloss] = useState('');
  const [numBeams, setNumBeams] = useState(4);
  const [threshold, setThreshold] = useState(0.5);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    fetchModels()
      .then(d => {
        setModels(d.models);
        const first = d.models.find(m => m.default && m.available) || d.models.find(m => m.available);
        if (first) setCheckpoint(first.checkpoint);
      })
      .catch(e => setModelsErr(e.message));
  }, []);

  const selectedModel = models.find(m => m.checkpoint === checkpoint);
  const canRun = gloss.trim() && checkpoint && selectedModel?.available && !loading;

  const handleRun = async () => {
    if (!canRun) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await runInfer({ gloss: gloss.trim(), checkpoint, numBeams, threshold });
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      handleRun();
    }
  };

  const handleCopy = async () => {
    if (!result) return;
    await navigator.clipboard.writeText(result.reconstruction).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  function getModelBadge(m) {
    if (!m) return null;
    if (!m.available) return { cls: 'badge-error', label: 'Checkpoint unavailable' };
    if (m.name.includes('GlossWeaver') || m.name.includes('Grammar-Aware')) {
      return { cls: 'badge-accent', label: 'Grammar-Aware Neural Model' };
    }
    if (m.name.includes('Copy')) {
      return { cls: 'badge-neutral', label: 'Baseline Heuristic' };
    }
    return { cls: 'badge-neutral', label: 'Standard T5 Sequence-to-Sequence' };
  }

  const modelBadge = getModelBadge(selectedModel);

  return (
    <div className="infer-container animate-fade-in">
      <div className="page-header">
        <h1 className="page-title">ASL Gloss → Fluent English</h1>
        <p className="page-subtitle">
          Reconstruct natural, grammatically complete English sentences from ASL gloss text.
        </p>
      </div>

      <div className="infer-grid">
        {/* Input Card */}
        <div className="card">
          <div className="card-header">
            <div className="card-title-group">
              <span className="card-icon">✏️</span>
              <h2 className="card-title">Gloss Input</h2>
            </div>
            {modelBadge && (
              <span className={`badge ${modelBadge.cls}`}>
                {modelBadge.label}
              </span>
            )}
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="model-select">
              Model Selection
            </label>
            {modelsErr ? (
              <div className="error-banner">Failed to load models: {modelsErr}</div>
            ) : (
              <select
                id="model-select"
                className="form-select"
                value={checkpoint}
                onChange={e => {
                  setCheckpoint(e.target.value);
                  setResult(null);
                  setError(null);
                }}
                disabled={!models.length}
              >
                {!models.length && <option value="">Loading models...</option>}
                {models.map(m => (
                  <option key={m.checkpoint} value={m.checkpoint} disabled={!m.available}>
                    {m.name} {!m.available ? '(unavailable)' : ''}
                  </option>
                ))}
              </select>
            )}
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="gloss-input">
              Gloss Notation
              <span className="shortcut-hint">Ctrl + Enter to run</span>
            </label>
            <textarea
              id="gloss-input"
              className="form-textarea"
              rows={4}
              placeholder="e.g. ME GO STORE YESTERDAY"
              spellCheck={false}
              value={gloss}
              onChange={e => setGloss(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <p className="form-hint">Enter capitalized ASL gloss tokens without punctuation.</p>
          </div>

          <div className="form-group">
            <span className="form-label">Quick Sample Inputs</span>
            <div className="examples-list">
              {EXAMPLES.map(ex => (
                <button
                  key={ex}
                  type="button"
                  className="example-btn"
                  onClick={() => {
                    setGloss(ex);
                    setResult(null);
                    setError(null);
                  }}
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>

          <details className="advanced-options">
            <summary className="advanced-summary">Advanced Inference Parameters</summary>
            <div className="advanced-content">
              <div className="form-row">
                <div className="form-group">
                  <label className="form-label" htmlFor="beam-input">Beam Width</label>
                  <input
                    id="beam-input"
                    type="number"
                    className="form-input"
                    value={numBeams}
                    min={1}
                    max={10}
                    onChange={e => setNumBeams(parseInt(e.target.value, 10) || 4)}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label" htmlFor="threshold-input">Grammar Threshold</label>
                  <input
                    id="threshold-input"
                    type="number"
                    className="form-input"
                    value={threshold}
                    min={0}
                    max={1}
                    step={0.05}
                    onChange={e => setThreshold(parseFloat(e.target.value) || 0.5)}
                  />
                </div>
              </div>
            </div>
          </details>

          <button
            type="button"
            className="btn-primary"
            disabled={!canRun}
            onClick={handleRun}
          >
            {loading ? (
              <>
                <span className="spinner-sm" />
                <span>Reconstructing...</span>
              </>
            ) : (
              <span>Reconstruct English</span>
            )}
          </button>
        </div>

        {/* Output Card */}
        <div className="output-column">
          <div className="card">
            <div className="card-header">
              <div className="card-title-group">
                <span className="card-icon">💬</span>
                <h2 className="card-title">Reconstruction Result</h2>
              </div>
              {result && (
                <button type="button" className="btn-copy" onClick={handleCopy}>
                  {copied ? '✓ Copied' : 'Copy'}
                </button>
              )}
            </div>

            {loading && (
              <div className="state-empty">
                <div className="spinner" />
                <p className="state-title">Reconstructing text...</p>
                <p className="state-desc">Running sequence generation on selected checkpoint</p>
              </div>
            )}

            {!loading && !result && !error && (
              <div className="state-empty">
                <span className="empty-symbol">🔤</span>
                <p className="state-title">Ready for reconstruction</p>
                <p className="state-desc">Enter gloss text and click Reconstruct to generate fluent English.</p>
              </div>
            )}

            {!loading && error && (
              <div className="error-box animate-fade-in">
                <div className="error-title">Inference Error</div>
                <div className="error-message">{error}</div>
              </div>
            )}

            {!loading && result && (
              <div className="result-view animate-fade-in">
                <div className="result-gloss-chip">
                  <span className="chip-label">INPUT</span>
                  <span className="chip-value">{result.gloss}</span>
                </div>

                <div className="result-arrow">↓</div>

                <div className="result-text-card">
                  <span className="result-label">RECONSTRUCTED ENGLISH</span>
                  <p className="result-text">{result.reconstruction}</p>
                </div>

                <details className="advanced-options">
                  <summary className="advanced-summary">Active checkpoint details</summary>
                  <div className="advanced-content">
                    <p><strong>Model:</strong> {result.debug.model_name}</p>
                    <p><strong>Experiment:</strong> {result.debug.experiment_name}</p>
                    <p><strong>Checkpoint:</strong> {result.debug.checkpoint_path}</p>
                    <p><strong>Modified:</strong> {result.debug.checkpoint_modified_time || 'Not applicable'}</p>
                    <p><strong>Training data:</strong> {result.debug.training_data_description}</p>
                  </div>
                </details>
              </div>
            )}
          </div>

          {result && (
            <div className="card animate-fade-in">
              <div className="card-header">
                <div className="card-title-group">
                  <span className="card-icon">📊</span>
                  <h2 className="card-title">Grammar Auxiliary Predictions</h2>
                </div>
                <span className="subtle-status">
                  {result.grammar_aware ? 'Active Model' : 'Not Supported for this Checkpoint'}
                </span>
              </div>
              <GrammarBars
                grammar={result.grammar}
                threshold={threshold}
                grammarAware={result.grammar_aware}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
