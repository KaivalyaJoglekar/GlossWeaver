export default function SemanticValidation({ semantic }) {
  if (!semantic) return null;
  return <section className="semantic-section" aria-label="Meaning preservation"><div className="pane-heading"><h3>Meaning preservation</h3><span className={`badge ${semantic.issues.length ? 'warning' : 'success'}`}>{semantic.issues.length ? 'Verification recommended' : 'No issue detected'}</span></div>
    <div className="check-list">{semantic.checks.map(check => <span key={check.category} className={`check ${check.status}`}><span aria-hidden="true">{check.status === 'potential_issue' ? '△' : check.status === 'unavailable' ? '–' : '✓'}</span> {check.category.replaceAll('_', ' ')}{check.status === 'unavailable' && ' · unavailable'}</span>)}</div>
    {semantic.issues.map((issue, index) => <p className={`semantic-notice ${issue.severity}`} key={index}>{issue.explanation}</p>)}<p className="form-hint">{semantic.advisory}</p></section>;
}
