import React, { useState, useEffect } from 'react';
import { fetchMetrics } from '../api';
import './MetricsPanel.css';

const EXPERIMENTS = [
  { id: 'e0', label: 'E0', name: 'Copy Baseline' },
  { id: 'e1', label: 'E1', name: 'T5 Baseline' },
  { id: 'e2', label: 'E2', name: 'T5 + Augmentation' },
  { id: 'e3', label: 'E3', name: 'Grammar-Aware T5' },
  { id: 'e4', label: 'E4', name: 'GlossWeaver (Full)' },
];

const GRAMMAR_LABELS = [
  'ARTICLE',
  'PREPOSITION',
  'AUXILIARY',
  'PRONOUN',
  'TENSE_ASPECT',
  'AGREEMENT_INFLECTION',
];

const METRICS_COLUMNS = [
  { key: 'sacrebleu', label: 'SacreBLEU', format: v => v.toFixed(2) },
  { key: 'chrf', label: 'chrF', format: v => v.toFixed(2) },
  { key: 'rouge_l', label: 'ROUGE-L', format: v => v.toFixed(4) },
  { key: 'meteor', label: 'METEOR', format: v => v.toFixed(4) },
  { key: 'bertscore_f1', label: 'BERTScore F1', format: v => v.toFixed(4) },
  { key: 'exact_match', label: 'Exact Match', format: v => `${(v * 100).toFixed(1)}%` },
];

export default function MetricsPanel() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedExp, setSelectedExp] = useState('e4');

  useEffect(() => {
    Promise.all(
      EXPERIMENTS.map(exp =>
        fetchMetrics(exp.id)
          .then(m => ({ ...exp, metrics: m }))
          .catch(() => ({ ...exp, metrics: null }))
      )
    )
      .then(results => {
        setData(results.filter(r => r.metrics));
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="metrics-state-container animate-fade-in">
        <div className="spinner" />
        <p className="state-text">Loading benchmark results...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="metrics-state-container animate-fade-in">
        <p className="error-banner">Failed to load metrics: {error}</p>
      </div>
    );
  }

  // Calculate highest score for each metric
  const bestScores = {};
  METRICS_COLUMNS.forEach(({ key }) => {
    const scores = data.map(d => d.metrics?.[key] ?? -Infinity);
    bestScores[key] = Math.max(...scores);
  });

  const activeExperimentData = data.find(d => d.id === selectedExp);

  return (
    <div className="metrics-container animate-fade-in">
      <div className="page-header">
        <h1 className="page-title">Experimental Benchmarks</h1>
        <p className="page-subtitle">
          Evaluation results across 2,500 held-out test examples from the ASLG-PC12 dataset.
        </p>
      </div>

      {/* Summary Table */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-header">
          <div className="card-title-group">
            <span className="card-icon">📈</span>
            <h2 className="card-title">Comparative Performance Summary</h2>
          </div>
          <span className="dataset-tag">ASLG-PC12 Test Split (N = 2,500)</span>
        </div>

        <div className="table-responsive">
          <table className="benchmark-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>System</th>
                {METRICS_COLUMNS.map(col => (
                  <th key={col.key}>{col.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map(row => (
                <tr key={row.id} className={row.id === 'e4' ? 'featured-row' : ''}>
                  <td>
                    <span className="id-tag">{row.label}</span>
                  </td>
                  <td className="system-name">{row.name}</td>
                  {METRICS_COLUMNS.map(({ key, format }) => {
                    const value = row.metrics?.[key];
                    const isBest = value !== undefined && Math.abs(value - bestScores[key]) < 1e-5;
                    return (
                      <td key={key} className={isBest ? 'best-cell' : ''}>
                        {value !== undefined ? format(value) : '—'}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Grammar Diagnostics */}
      <div className="card">
        <div className="card-header">
          <div className="card-title-group">
            <span className="card-icon">🔬</span>
            <h2 className="card-title">Grammar Category Diagnostics</h2>
          </div>
          <div className="tab-pill-group">
            {data.map(d => (
              <button
                key={d.id}
                type="button"
                className={`tab-pill ${selectedExp === d.id ? 'active' : ''}`}
                onClick={() => setSelectedExp(d.id)}
              >
                {d.label}
              </button>
            ))}
          </div>
        </div>

        {activeExperimentData && (
          <div className="diagnostics-grid animate-fade-in" key={selectedExp}>
            {GRAMMAR_LABELS.map(label => {
              const diag = activeExperimentData.metrics?.grammar_diagnostics?.[label];
              if (!diag) return null;
              return (
                <div key={label} className="diag-card">
                  <div className="diag-header">
                    <span className="diag-title">{label.replace('_', ' ')}</span>
                    <span className="diag-f1">F1: {(diag.f1 * 100).toFixed(1)}%</span>
                  </div>

                  <div className="diag-bars">
                    {['precision', 'recall'].map(metric => (
                      <div key={metric} className="diag-bar-item">
                        <div className="diag-bar-info">
                          <span className="metric-name">{metric}</span>
                          <span className="metric-val">{(diag[metric] * 100).toFixed(1)}%</span>
                        </div>
                        <div className="diag-track">
                          <div
                            className="diag-fill"
                            style={{ width: `${(diag[metric] * 100).toFixed(1)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="diag-support-info">
                    Support: {diag.support.toLocaleString()} test instances
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
