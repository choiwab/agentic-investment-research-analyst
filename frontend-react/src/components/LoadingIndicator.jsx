import React from 'react';
import '../styles/LoadingIndicator.css';

function LoadingIndicator() {
  return (
    <div className="loading-indicator">
      <div className="message assistant-message">
        <div className="message-avatar">🤖</div>
        <div className="message-content">
          <div className="loading-dots">
            <span className="loading-text">Analyzing your query</span>
            <span className="dot">.</span>
            <span className="dot">.</span>
            <span className="dot">.</span>
          </div>
          <div className="loading-steps">
            <div className="step">🔍 Processing intent</div>
            <div className="step">📊 Gathering data</div>
            <div className="step">🧠 Generating analysis</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default LoadingIndicator;
