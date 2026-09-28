import React, { useState } from 'react';
import Header from './components/Header';
import InferencePanel from './components/InferencePanel';
import MetricsPanel from './components/MetricsPanel';
import AboutPanel from './components/AboutPanel';
import Background from './components/Background';
import './App.css';

const TABS = [
  { id: 'infer', label: 'Inference' },
  { id: 'metrics', label: 'Metrics' },
  { id: 'about', label: 'About' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('infer');

  return (
    <div className="app">
      <Background />
      <Header tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="main-container">
        {activeTab === 'infer' && <InferencePanel />}
        {activeTab === 'metrics' && <MetricsPanel />}
        {activeTab === 'about' && <AboutPanel />}
      </main>
      <footer className="footer">
        <span>GlossWeaver</span>
        <span className="footer-divider">·</span>
        <span>Research Preview</span>
        <span className="footer-divider">·</span>
        <span>ASLG-PC12 Benchmark (T5-small Backbone)</span>
      </footer>
    </div>
  );
}
