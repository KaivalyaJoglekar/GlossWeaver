import React from 'react';
import './AboutPanel.css';

export default function AboutPanel() {
  return (
    <div className="about-container animate-fade-in">
      <div className="page-header">
        <h1 className="page-title">About GlossWeaver</h1>
        <p className="page-subtitle">
          Grammar-aware sequence reconstruction of fluent English from telegraphic ASL glosses.
        </p>
      </div>

      <div className="about-grid">
        {/* Research Problem */}
        <div className="card">
          <div className="card-header">
            <div className="card-title-group">
              <span className="card-icon">🎯</span>
              <h2 className="card-title">Research Overview</h2>
            </div>
          </div>
          <p className="about-text">
            GlossWeaver evaluates whether <strong>grammar-aware auxiliary multi-task supervision</strong> and{' '}
            <strong>controlled telegraphic augmentation</strong> improve Gloss-to-Text reconstruction over a standard
            T5-small sequence-to-sequence baseline.
          </p>
          <div className="example-box">
            <span className="example-label">EXAMPLE TASK</span>
            <div className="example-content">
              <span className="example-gloss">ME GO STORE YESTERDAY</span>
              <span className="example-arrow">→</span>
              <span className="example-target">I went to the store yesterday.</span>
            </div>
          </div>
          <p className="about-disclaimer">
            Scope: text-only reconstruction. Video sign language recognition and speech synthesis are out of scope.
          </p>
        </div>

        {/* Experiment Ladder */}
        <div className="card">
          <div className="card-header">
            <div className="card-title-group">
              <span className="card-icon">🔬</span>
              <h2 className="card-title">Experiment Ladder</h2>
            </div>
          </div>
          <table className="ladder-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>System</th>
                <th>Grammar Loss</th>
                <th>Augmentation</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['E0', 'Copy / Normalization', false, false],
                ['E1', 'T5-small Baseline', false, false],
                ['E2', 'T5-small + Augmentation', false, true],
                ['E3', 'Grammar-Aware T5', true, false],
                ['E4', 'GlossWeaver (Full)', true, true],
              ].map(([id, name, gl, aug]) => (
                <tr key={id} className={id === 'E4' ? 'featured-ladder' : ''}>
                  <td>
                    <span className="ladder-id">{id}</span>
                  </td>
                  <td className="ladder-name">{name}</td>
                  <td className={gl ? 'status-true' : 'status-false'}>{gl ? 'Yes' : 'No'}</td>
                  <td className={aug ? 'status-true' : 'status-false'}>{aug ? 'Yes' : 'No'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Auxiliary Labels */}
        <div className="card">
          <div className="card-header">
            <div className="card-title-group">
              <span className="card-icon">🏷️</span>
              <h2 className="card-title">Auxiliary Grammar Supervision</h2>
            </div>
          </div>
          <p className="about-text">
            The grammar-aware architecture shares an encoder with a 6-category classification head during training,
            predicting linguistic elements omitted in glosses:
          </p>
          <div className="tags-container">
            {[
              ['ARTICLE', 'Articles (a, an, the)'],
              ['PREPOSITION', 'Prepositions (in, on, to, for)'],
              ['AUXILIARY', 'Auxiliary verbs (is, was, were, have)'],
              ['PRONOUN', 'Pronouns (I, me, he, they, you)'],
              ['TENSE_ASPECT', 'Verb morphology & tense inflection'],
              ['AGREEMENT_INFLECTION', 'Subject-verb number agreement'],
            ].map(([label, desc]) => (
              <div key={label} className="tag-row">
                <span className="tag-name">{label.replace('_', ' ')}</span>
                <span className="tag-desc">{desc}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Dataset */}
        <div className="card">
          <div className="card-header">
            <div className="card-title-group">
              <span className="card-icon">📊</span>
              <h2 className="card-title">Dataset: ASLG-PC12</h2>
            </div>
          </div>
          <p className="about-text">
            Experiments utilize the verified Hugging Face revision <code>cb7cd272</code> of ASLG-PC12 containing 81,017
            cleaned, non-overlapping pairs.
          </p>
          <div className="split-stats">
            <div className="split-item">
              <span className="split-val">64,813</span>
              <span className="split-lbl">Train (80%)</span>
            </div>
            <div className="split-item">
              <span className="split-val">8,101</span>
              <span className="split-lbl">Validation (10%)</span>
            </div>
            <div className="split-item">
              <span className="split-val">8,103</span>
              <span className="split-lbl">Test (10%)</span>
            </div>
          </div>
          <p className="about-disclaimer">
            Note: ASLG-PC12 glosses are rule-generated and serve as a standardized large-scale benchmark for controlled
            morphological recovery experiments.
          </p>
        </div>
      </div>
    </div>
  );
}
