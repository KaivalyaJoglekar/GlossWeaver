export default function Header({ tabs, activeTab, onTabChange }) {
  function keyboard(event, index) {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next === undefined) return;
    event.preventDefault(); onTabChange(tabs[next].id);
    document.getElementById(`tab-${tabs[next].id}`).focus();
  }
  return <header className="header"><div className="header-inner">
    <div className="brand"><span className="brand-name">GlossWeaver</span></div>
    <nav className="nav-tabs" role="tablist" aria-label="Workbench sections">{tabs.map((tab, index) => <button key={tab.id} id={`tab-${tab.id}`} role="tab" aria-selected={activeTab === tab.id} aria-controls={`panel-${tab.id}`} tabIndex={activeTab === tab.id ? 0 : -1} className={`nav-tab-btn ${activeTab === tab.id ? 'active' : ''}`} onClick={() => onTabChange(tab.id)} onKeyDown={e => keyboard(e, index)}>{tab.label}</button>)}</nav>
  </div></header>;
}
