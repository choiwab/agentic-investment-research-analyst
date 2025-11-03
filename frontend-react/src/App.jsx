import React, { useState, useEffect } from 'react';
import ChatInterface from './components/ChatInterface';
import Sidebar from './components/Sidebar';
import './styles/App.css';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    // Load welcome message
    setMessages([
      {
        role: 'assistant',
        content: `# Welcome to Equity Research AI Assistant 📊

I can help you with:

**📈 Company Analysis**
- "Analyze Tesla stock"
- "What's Apple's financial performance?"
- "Compare MSFT earnings"

**🌍 Market Insights**
- "What are current inflation trends?"
- "S&P 500 outlook"
- "How are interest rates affecting markets?"

**🎓 Financial Education**
- "What is P/E ratio?"
- "Explain dividend yield"
- "How does compound interest work?"

Ask me anything about stocks, markets, or financial concepts!`,
        timestamp: new Date().toISOString(),
      }
    ]);
  }, []);

  const handleNewChat = () => {
    setMessages([
      {
        role: 'assistant',
        content: '# New Chat Started\n\nHow can I assist you with your equity research today?',
        timestamp: new Date().toISOString(),
      }
    ]);
  };

  return (
    <div className="app">
      <Sidebar
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        onNewChat={handleNewChat}
      />
      <div className={`main-content ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <ChatInterface
          messages={messages}
          setMessages={setMessages}
        />
      </div>
    </div>
  );
}

export default App;
