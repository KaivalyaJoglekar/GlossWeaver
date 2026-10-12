import { lazy, Suspense, useEffect, useState } from 'react';
import InferencePanel from './components/InferencePanel';
import Overview from './components/Overview';
import ModelLibrary from './components/ModelLibrary';
import ComparisonPanel from './components/ComparisonPanel';
import AboutPanel from './components/AboutPanel';
import DataPanel from './components/DataPanel';
import { fetchModels, fetchDatasets, fetchHealth, fetchMetrics } from './api';
import NavigationIcon from './components/NavigationIcon';
import { ArrowLeftIcon, ArrowRightIcon } from '@radix-ui/react-icons';
import './App.css';
const ResearchPanel = lazy(() => import('./components/ResearchPanel'));
const PAGES = [['overview', 'Overview'], ['infer', 'Reconstruction Studio'], ['compare', 'Model Comparison'], ['metrics', 'Research & Evaluation'], ['models', 'Model Library'], ['data', 'Datasets'], ['about', 'Settings / About']];
export default function App() {
  const [page, setPage] = useState('overview'); const [collapsed, setCollapsed] = useState(false);
  const [models, setModels] = useState([]); const [datasets, setDatasets] = useState(null); const [health, setHealth] = useState(null); const [metrics, setMetrics] = useState(null); const [error, setError] = useState('');
  async function refresh() {
    const requests = await Promise.allSettled([fetchModels(), fetchDatasets(), fetchHealth(), fetchMetrics()]);
    const setters = [d => setModels(d.models), setDatasets, setHealth, setMetrics];
    requests.forEach((r, i) => { if (r.status === 'fulfilled') setters[i](r.value); });
    setError(requests.filter(r => r.status === 'rejected').map(r => r.reason.message).filter((v, i, a) => a.indexOf(v) === i).join(' '));
  }
  useEffect(() => { let active = true; Promise.allSettled([fetchModels(), fetchDatasets(), fetchHealth(), fetchMetrics()]).then(requests => {
    if (!active) return;
    if (requests[0].status === 'fulfilled') setModels(requests[0].value.models);
    if (requests[1].status === 'fulfilled') setDatasets(requests[1].value);
    if (requests[2].status === 'fulfilled') setHealth(requests[2].value);
    if (requests[3].status === 'fulfilled') setMetrics(requests[3].value);
    setError(requests.filter(r => r.status === 'rejected').map(r => r.reason.message).filter((v, i, a) => a.indexOf(v) === i).join(' '));
  }); return () => { active = false; }; }, []);
  const current = PAGES.find(p => p[0] === page);
  return <div className={`app app-shell ${collapsed ? 'sidebar-collapsed' : ''}`}><a className="skip-link" href="#main">Skip to content</a><aside className="sidebar"><div className="sidebar-brand"><span className="brand-symbol" aria-hidden="true">g<span>↗</span></span><span className="brand-name">GlossWeaver<small>RECONSTRUCTION LAB</small></span></div><button className="collapse-button" aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'} onClick={() => setCollapsed(!collapsed)}>{collapsed ? <ArrowRightIcon aria-hidden="true" /> : <ArrowLeftIcon aria-hidden="true" />}</button><nav aria-label="Main navigation">{PAGES.map(([id, label]) => <button key={id} className={`sidebar-link ${page === id ? 'active' : ''}`} aria-current={page === id ? 'page' : undefined} title={label} onClick={() => setPage(id)}><span aria-hidden="true"><NavigationIcon page={id} /></span><span className="nav-label">{label}</span></button>)}</nav><div className="sidebar-status"><span className={`status-dot ${health ? 'ready' : ''}`} /><span>{health ? 'API online' : 'API disconnected'}<small>{models.find(m => m.default)?.checkpoint || 'Awaiting registry'}</small></span></div><span className="sidebar-version">GlossWeaver 2.0</span></aside><div className="workspace"><header className="topbar"><span>Laboratory <span className="topbar-divider">/</span> {current[1]}</span><span className="topbar-note">Local inference · Research edition</span></header><main id="main" className="main-container"><Suspense fallback={<p role="status">Loading research tools…</p>}>{page === 'overview' && <Overview models={models} datasets={datasets} health={health} metrics={metrics} navigate={setPage} error={error} refresh={refresh} />}{<section hidden={page !== 'infer'}><InferencePanel /></section>}{page === 'compare' && <ComparisonPanel models={models} />}{page === 'metrics' && <ResearchPanel />}{page === 'models' && <ModelLibrary models={models} error={error} refresh={refresh} />}{page === 'data' && <DataPanel />}{page === 'about' && <><div className="page-header"><h1 className="page-title">Settings / About</h1><p className="page-subtitle">A local research workbench. Generation settings live beside each reconstruction; service settings are configured through environment variables.</p></div><button className="text-button" onClick={refresh}>Refresh service status</button><AboutPanel /></>}</Suspense></main><footer className="footer"><span>GlossWeaver · Intelligent English Reconstruction Laboratory</span><span>Verify meaning. Preserve context.</span></footer></div></div>;
}
