import React from 'react';
import '../styles/Sidebar.css';

function Sidebar({ isOpen, onToggle, onNewChat }) {
  return (
    <>
      <button className="sidebar-toggle" onClick={onToggle}>
        {isOpen ? '◀' : '▶'}
      </button>

      <aside className={`sidebar ${isOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <h2>📊 Research AI</h2>
        </div>

        <button className="new-chat-button" onClick={onNewChat}>
          ➕ New Chat
        </button>

        <div className="sidebar-section">
          <h3>🎯 What I Can Do</h3>
          <ul className="capabilities-list">
            <li>
              <strong>Company Analysis</strong>
              <span>Deep dive into stocks, earnings, and financials</span>
            </li>
            <li>
              <strong>Market Insights</strong>
              <span>Economic trends and market conditions</span>
            </li>
            <li>
              <strong>Financial Education</strong>
              <span>Learn financial concepts and terminology</span>
            </li>
          </ul>
        </div>

        <div className="sidebar-section">
          <h3>💡 Example Queries</h3>
          <div className="example-list">
            <div className="example-category">
              <strong>Companies:</strong>
              <p>"Analyze Apple's recent performance"</p>
              <p>"Compare Tesla and Ford"</p>
            </div>
            <div className="example-category">
              <strong>Markets:</strong>
              <p>"Current inflation outlook"</p>
              <p>"S&P 500 trends"</p>
            </div>
            <div className="example-category">
              <strong>Education:</strong>
              <p>"What is dividend yield?"</p>
              <p>"Explain market cap"</p>
            </div>
          </div>
        </div>

        <div className="sidebar-footer">
          <p>Powered by OpenAI & LangGraph</p>
        </div>
      </aside>
    </>
  );
}

export default Sidebar;
