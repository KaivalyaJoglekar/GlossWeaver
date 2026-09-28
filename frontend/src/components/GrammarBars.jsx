import React from 'react';
import './GrammarBars.css';

const LABEL_ORDER = [
  'ARTICLE',
  'PREPOSITION',
  'AUXILIARY',
  'PRONOUN',
  'TENSE_ASPECT',
  'AGREEMENT_INFLECTION',
  'WORD_ORDER',
];

const LABEL_DESCRIPTIONS = {
  ARTICLE: 'Recovers missing articles (a, an, the)',
  PREPOSITION: 'Recovers relational prepositions (in, on, to)',
  AUXILIARY: 'Recovers auxiliary verbs (is, was, have)',
  PRONOUN: 'Recovers subject/object pronouns (I, he, they)',
  TENSE_ASPECT: 'Inflects verb tense and aspect (went, running)',
  AGREEMENT_INFLECTION: 'Adjusts subject-verb agreement (runs, eats)',
  WORD_ORDER: 'Recovers English constituent and modifier order',
};

export default function GrammarBars({ grammar, threshold = 0.5, grammarAware }) {
  if (!grammarAware) {
    return (
      <div className="grammar-unavailable">
        <p className="unavailable-text">
          Auxiliary grammar supervision is exclusive to <strong>E3 (Grammar-Aware T5)</strong> and <strong>E4 (GlossWeaver)</strong>.
        </p>
      </div>
    );
  }

  if (!grammar || Object.keys(grammar).length === 0) {
    return <p className="grammar-empty">No grammar diagnostics produced for this sample.</p>;
  }

  const items = LABEL_ORDER.filter(label => label in grammar).map(label => ({
    label,
    score: grammar[label],
    isActive: grammar[label] >= threshold,
  }));

  return (
    <div className="grammar-list">
      {items.map(({ label, score, isActive }) => {
        const pct = Math.round(score * 100);
        return (
          <div key={label} className={`grammar-card ${isActive ? 'active' : ''}`}>
            <div className="grammar-header">
              <span className="grammar-name">{label.replace('_', ' ')}</span>
              <span className="grammar-value">{pct}%</span>
            </div>
            <div className="grammar-desc">{LABEL_DESCRIPTIONS[label]}</div>
            <div className="grammar-track">
              <div
                className="grammar-fill"
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
