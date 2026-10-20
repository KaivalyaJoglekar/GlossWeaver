import { useEffect, useState } from 'react';
import InferencePanel from './components/InferencePanel';
import { fetchModels, fetchHealth } from './api';
import NavigationIcon from './components/NavigationIcon';
import { ArrowLeftIcon, ArrowRightIcon } from '@radix-ui/react-icons';
import './App.css';

const PAGES = [['infer', 'Reconstruction Studio']];

export default function App() {
  const [collapsed, setCollapsed] = useState(false);
  const [models, setModels] = useState([]);
  const [health, setHealth] = useState(null);
  const [, setError] = useState('');

  async function refresh() {
    const requests = await Promise.allSettled([fetchModels(), fetchHealth()]);
    if (requests[0].status === 'fulfilled') setModels(requests[0].value.models);
    if (requests[1].status === 'fulfilled') setHealth(requests[1].value);
    setError(requests.filter(r => r.status === 'rejected').map(r => r.reason.message).filter((v, i, a) => a.indexOf(v) === i).join(' '));
  }

  useEffect(() => {
    let active = true;
    Promise.allSettled([fetchModels(), fetchHealth()]).then(requests => {
      if (!active) return;
      if (requests[0].status === 'fulfilled') setModels(requests[0].value.models);
      if (requests[1].status === 'fulfilled') setHealth(requests[1].value);
      setError(requests.filter(r => r.status === 'rejected').map(r => r.reason.message).filter((v, i, a) => a.indexOf(v) === i).join(' '));
    });
    return () => { active = false; };
  }, []);

  return (
    <div className={`app app-shell ${collapsed ? 'sidebar-collapsed' : ''}`}>
      <a className="skip-link" href="#main">Skip to content</a>
      <aside className="sidebar">
        <div className="sidebar-brand">
          <span className="brand-symbol" aria-hidden="true">g<span>↗</span></span>
          <span className="brand-name">GlossWeaver<small>RECONSTRUCTION LAB</small></span>
        </div>
        <button
          className="collapse-button"
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          onClick={() => setCollapsed(!collapsed)}
        >
          {collapsed ? <ArrowRightIcon aria-hidden="true" /> : <ArrowLeftIcon aria-hidden="true" />}
        </button>
        <nav aria-label="Main navigation">
          {PAGES.map(([id, label]) => (
            <button key={id} className="sidebar-link active" aria-current="page" title={label}>
              <span aria-hidden="true"><NavigationIcon page={id} /></span>
              <span className="nav-label">{label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-status">
          <span className={`status-dot ${health ? 'ready' : ''}`} />
          <span>
            {health ? 'API online' : 'API disconnected'}
            <small>{models.find(m => m.default)?.checkpoint || 'Awaiting registry'}</small>
          </span>
        </div>
        <span className="sidebar-version">GlossWeaver 2.0</span>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <span>Laboratory <span className="topbar-divider">/</span> Reconstruction Studio</span>
          <span className="topbar-note">Local inference · Neural edition</span>
        </header>
        <main id="main" className="main-container">
          <InferencePanel />
        </main>
        <footer className="footer">
          <span>GlossWeaver · Intelligent English Reconstruction Laboratory</span>
          <span>Verify meaning. Preserve context.</span>
        </footer>
      </div>
    </div>
  );
}

