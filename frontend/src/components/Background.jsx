import React from 'react';
import './Background.css';

export default function Background() {
  return (
    <div className="ambient-background" aria-hidden="true">
      <div className="ambient-glow" />
      <div className="subtle-grid" />
    </div>
  );
}
