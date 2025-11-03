import React from 'react';
import Message from './Message';
import LoadingIndicator from './LoadingIndicator';
import '../styles/MessageList.css';

function MessageList({ messages, isLoading, messagesEndRef }) {
  return (
    <div className="message-list">
      {messages.map((message, index) => (
        <Message
          key={index}
          message={message}
        />
      ))}

      {isLoading && <LoadingIndicator />}

      <div ref={messagesEndRef} />
    </div>
  );
}

export default MessageList;
