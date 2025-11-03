import React, { useState, useRef } from 'react';
import '../styles/ChatInput.css';

function ChatInput({ onSendMessage, disabled }) {
  const [inputValue, setInputValue] = useState('');
  const textareaRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim() && !disabled) {
      onSendMessage(inputValue);
      setInputValue('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleInput = (e) => {
    setInputValue(e.target.value);
    // Auto-resize textarea
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
  };

  const exampleQueries = [
    "Analyze Tesla stock",
    "What are current inflation trends?",
    "What is P/E ratio?",
  ];

  return (
    <div className="chat-input-container">
      <div className="example-queries">
        {exampleQueries.map((query, index) => (
          <button
            key={index}
            className="example-query-button"
            onClick={() => !disabled && onSendMessage(query)}
            disabled={disabled}
          >
            {query}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="chat-input-form">
        <textarea
          ref={textareaRef}
          value={inputValue}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder="Ask about stocks, markets, or financial concepts..."
          disabled={disabled}
          rows={1}
          className="chat-textarea"
        />
        <button
          type="submit"
          disabled={disabled || !inputValue.trim()}
          className="send-button"
        >
          {disabled ? '⏳' : '📤'}
        </button>
      </form>
    </div>
  );
}

export default ChatInput;
