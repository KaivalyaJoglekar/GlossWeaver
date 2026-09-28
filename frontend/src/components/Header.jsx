import React from 'react';
import './Header.css';

export default function Header({ tabs, activeTab, onTabChange }) {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="brand">
          <div className="brand-logo">GW</div>
          <span className="brand-name">GlossWeaver</span>
          <span className="brand-badge">Research</span>
        </div>

        <nav className="nav-tabs" role="tablist" aria-label="Navigation Tabs">
          {tabs.map(tab => (
            <button
              key={tab.id}
              role="tab"
              aria-selected={activeTab === tab.id}
              className={`nav-tab-btn ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => onTabChange(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>
    </header>
  );
}
