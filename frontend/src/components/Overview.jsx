import { ArrowDownIcon, ArrowTopRightIcon } from '@radix-ui/react-icons';

export default function Overview({ models, datasets, health, metrics, navigate, error, refresh }) {
  const ready = models.filter(m => m.available && m.debug.model_type !== 'copy');
  const defaultModel = models.find(m => m.default);
  const latest = metrics?.rows.filter(r => r.evaluation_id).at(-1);

  return <div className="overview">
    <div className="overview-intro">
      <div className="hero">
        <p className="hero-eyebrow">The English reconstruction laboratory</p>
        <h1>From fragments <br />to fluency.</h1>
        <p className="hero-description">Reconstruct natural English from incomplete language. Compare models, inspect alternatives, and check that the meaning holds.</p>
        <div className="hero-actions">
          <button className="btn-primary" onClick={() => navigate('infer')}>Open Reconstruction Studio <ArrowTopRightIcon aria-hidden="true" /></button>
          <button className="text-button" onClick={() => navigate('models')}>Explore models →</button>
        </div>
      </div>
      <div className="token-composition" aria-label="Illustrative reconstruction">
        <span className="token-label">Source fragment</span>
        <div className="token-source">{'ME GO STORE YESTERDAY'.split(' ').map(word => <span key={word}>{word}</span>)}</div>
        <div className="token-bridge" aria-hidden="true"><ArrowDownIcon /></div>
        <div className="token-output"><span className="token-label">Reconstructed English</span><p>I <span>went</span> to the store yesterday.</p></div>
        <small>Illustrative example. Actual model outputs may differ.</small>
      </div>
    </div>
    <div className="overview-status">
      <div><span>Default checkpoint</span><strong>{defaultModel?.name || 'Checking availability'}</strong></div>
      <div><span>Neural checkpoints</span><strong>{models.length ? `${ready.length} available` : 'Checking…'}</strong></div>
      <div><span>Dataset manifests</span><strong>{datasets ? `${datasets.datasets.filter(d => d.available).length} prepared` : 'Checking…'}</strong></div>
      <div><span>API readiness</span><strong className={health ? 'success-text' : ''}>{health?.status === 'ok' ? 'Online' : 'Connecting'}</strong></div>
    </div>
    {error && <p role="alert" className="error-banner">{error} <button className="text-button" onClick={refresh}>Retry connection</button></p>}
    <div className="overview-lower">
      <div><h2>Keep the meaning in view.</h2><p>Inspect negation, quantities, time references, and content changes alongside every reconstruction.</p><button className="text-button" onClick={() => navigate('compare')}>Enter the Comparison Lab →</button></div>
      <div><h2>Explore the evidence.</h2><p>{latest ? `Latest saved run: ${latest.display_name} on ${latest.dataset}, ${latest.examples} examples.` : 'Saved evaluations appear here when available. Unexecuted experiments carry no scores.'}</p><button className="text-button" onClick={() => navigate('metrics')}>Inspect saved research →</button></div>
    </div>
  </div>;
}
