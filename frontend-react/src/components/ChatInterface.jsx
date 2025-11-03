import React, { useState, useRef, useEffect } from 'react';
import MessageList from './MessageList';
import ChatInput from './ChatInput';
import { runEquityResearch } from '../services/api';
import '../styles/ChatInterface.css';

function ChatInterface({ messages, setMessages }) {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (userMessage) => {
    if (!userMessage.trim() || isLoading) return;

    // Add user message
    const newUserMessage = {
      role: 'user',
      content: userMessage,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, newUserMessage]);
    setIsLoading(true);
    setError(null);

    try {
      // Call API
      const result = await runEquityResearch(userMessage);

      // Format assistant response based on intent
      const assistantMessage = {
        role: 'assistant',
        content: formatResponse(result),
        timestamp: new Date().toISOString(),
        raw_result: result,
        intent: result.intent,
      };

      setMessages((prev) => [...prev, assistantMessage]);

      // Show processing time if available
      if (result.processing_time) {
        console.log(`Processing time: ${result.processing_time.toFixed(2)}s`);
      }

    } catch (err) {
      setError(err.message);
      const errorMessage = {
        role: 'assistant',
        content: `## Error\n\n${err.message}`,
        timestamp: new Date().toISOString(),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatResponse = (result) => {
    const intent = result.intent;

    console.log('[DEBUG] Formatting response:', { intent, result });

    // Handle irrelevant queries
    if (intent === 'irrelevant') {
      return `## Not a Financial Query

I'm specialized in equity research and financial analysis. I can help with:

- **Company Analysis**: Stock analysis, earnings, financials
- **Market Insights**: Economic trends, market conditions
- **Financial Education**: Definitions, concepts, explanations

Please ask me a finance-related question!`;
    }

    // Handle finance-education
    if (intent === 'finance-education') {
      return `## Financial Education

${result.answer || result.output_from_websearch || 'No answer available.'}

---

*Want to learn more? Ask me another financial concept!*`;
    }

    // Handle finance-market
    if (intent === 'finance-market') {
      let response = `## Market Analysis\n\n`;

      // Use text_summary if available (primary field from backend)
      if (result.text_summary) {
        response += `${result.text_summary}\n\n`;
      }

      // Add web search sources if available
      if (result.output_from_websearch) {
        try {
          // Try to parse the web search results
          const wsString = result.output_from_websearch.replace(/'/g, '"');
          const wsData = JSON.parse(wsString);

          if (wsData.results && wsData.results.length > 0) {
            response += `### Sources\n\n`;
            wsData.results.forEach((item, idx) => {
              if (item.url) {
                response += `${idx + 1}. [${new URL(item.url).hostname}](${item.url})\n`;
              }
            });
            response += '\n';
          }
        } catch (e) {
          // If parsing fails, skip showing web search results
          console.warn('Could not parse web search results:', e);
        }
      }

      // Add table if available
      if (result.table) {
        response += `${result.table}\n\n`;
      }

      // Add qualitative summary if available
      if (result.qualitative_summary) {
        response += `### Key Insights\n\n${result.qualitative_summary}\n\n`;
      }

      // Add quantitative summary if available
      if (result.quantitative_summary) {
        response += `### Market Data\n\n${result.quantitative_summary}\n\n`;
      }

      // Add insight outlook if available
      if (result.insight_outlook) {
        response += `### Market Outlook\n\n${result.insight_outlook}\n\n`;
      }

      // Add investment outlook if available
      if (result.investment_outlook) {
        response += `### Investment Perspective\n\n${result.investment_outlook}\n\n`;
      }

      // Add executive summary if available
      if (result.executive_summary) {
        response += `### Summary\n\n${result.executive_summary}\n\n`;
      }

      // Add PDF link if available - use API base URL
      if (result.pdf_path) {
        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const pdfUrl = `${apiUrl}/outputs/${result.pdf_path.split('/').pop()}`;
        response += `---\n\n📄 [Download Full Report](${pdfUrl})\n`;
      }

      return response;
    }

    // Handle finance-company (comprehensive report)
    if (intent === 'finance-company') {
      let response = `## ${result.ticker || 'Company'} Analysis\n\n`;

      // Executive Summary
      if (result.executive_summary) {
        response += `### Executive Summary\n\n${result.executive_summary}\n\n`;
      }

      // Financial Analysis
      if (result.financial_analysis) {
        response += `### Financial Analysis\n\n${result.financial_analysis}\n\n`;
      }

      // News & Sentiment
      if (result.news_sentiment_analysis) {
        response += `### News & Sentiment\n\n${result.news_sentiment_analysis}\n\n`;
      }

      // Investment Outlook
      if (result.investment_outlook) {
        response += `### Investment Outlook\n\n${result.investment_outlook}\n\n`;
      }

      // Recommendation
      if (result.recommendation) {
        response += `### Recommendation\n\n**${result.recommendation}**`;
        if (result.price_target) {
          response += ` | Target: ${result.price_target}`;
        }
        response += '\n\n';
      }

      // PDF link if available - use API base URL
      if (result.pdf_path) {
        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const pdfUrl = `${apiUrl}/outputs/${result.pdf_path.split('/').pop()}`;
        response += `---\n\n📄 [Download Full Report](${pdfUrl})\n`;
      }

      return response;
    }

    // Fallback - try to format nicely instead of raw JSON
    console.warn('[WARNING] Unknown intent or missing data, using fallback formatting');

    // Try to extract any readable content
    let fallbackResponse = `## Analysis Result\n\n`;

    const textFields = [
      'text_summary',
      'answer',
      'executive_summary',
      'qualitative_summary',
      'financial_analysis',
      'investment_outlook',
    ];

    for (const field of textFields) {
      if (result[field] && typeof result[field] === 'string') {
        fallbackResponse += `${result[field]}\n\n`;
      }
    }

    // Add table if available
    if (result.table && typeof result.table === 'string') {
      fallbackResponse += `${result.table}\n\n`;
    }

    // Add PDF link if available - use API base URL
    if (result.pdf_path) {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const pdfUrl = `${apiUrl}/outputs/${result.pdf_path.split('/').pop()}`;
      fallbackResponse += `---\n\n📄 [Download Full Report](${pdfUrl})\n`;
    }

    // If still no content found, show formatted JSON as last resort
    if (fallbackResponse === `## Analysis Result\n\n`) {
      fallbackResponse += '```json\n' + JSON.stringify(result, null, 2) + '\n```';
    }

    return fallbackResponse;
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h1>📊 Equity Research AI Assistant</h1>
        <p className="subtitle">Powered by OpenAI GPT-4o-mini & LangGraph</p>
      </div>

      <MessageList
        messages={messages}
        isLoading={isLoading}
        messagesEndRef={messagesEndRef}
      />

      {error && (
        <div className="error-banner">
          ⚠️ {error}
        </div>
      )}

      <ChatInput
        onSendMessage={handleSendMessage}
        disabled={isLoading}
      />
    </div>
  );
}

export default ChatInterface;
