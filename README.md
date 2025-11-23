# Agentic Investment Research Analyst

An AI-powered multi-agent system that automates end-to-end equity research, delivering real-time financial analysis, sentiment insights, and actionable investment reports.

## Overview

This system uses LangGraph and LangChain to orchestrate specialized AI agents that collect financial news, extract key metrics, analyze sentiment, and compile comprehensive investment reports. It features both a React web interface and a Streamlit chatbot for interactive equity research.

## Key Features

- **Multi-Agent Architecture**: Specialized agents for news scraping, metrics extraction, sentiment analysis, and report compilation
- **Real-Time Data Integration**: Pulls live financial data from multiple APIs (Finnhub, Alpha Vantage, Yahoo Finance)
- **Sentiment Analysis**: Advanced NLP using FinBERT and custom sentiment models
- **Automated Report Generation**: PDF reports with financial metrics, visualizations, and trade ideas
- **Interactive Interfaces**: Modern React frontend and Streamlit chatbot
- **Intelligent Routing**: Intent classification to handle company analysis, market insights, and educational queries
- **Fallback Mechanisms**: Multi-source data fetching with intelligent fallback to ensure data completeness

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interfaces                         │
│  ┌────────────────────┐      ┌──────────────────────┐       │
│  │  React Frontend    │      │ Streamlit Chatbot    │       │
│  │  (Port 3000)       │      │ (Port 8501)          │       │
│  └─────────┬──────────┘      └──────────┬───────────┘       │
└────────────┼────────────────────────────┼───────────────────┘
             │                            │
             └──────────┬─────────────────┘
                        │
             ┌──────────▼──────────┐
             │  FastAPI Backend    │
             │  (Port 8000)        │
             └──────────┬──────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌───────────┐   ┌──────────┐   ┌───────────────┐
│ LangGraph │   │  Ollama  │   │  Data APIs    │
│ Agents    │   │ (LLM)    │   │ - Finnhub     │
│ Pipeline  │   │          │   │ - AlphaVantage│
└───────────┘   └──────────┘   │ - Yahoo       │
                                │ - Tavily      │
                                └───────────────┘
```

## Agent Pipeline

1. **Preprocessor**: Classifies user intent and extracts ticker symbols
2. **News Scraper**: Fetches latest company news from Finnhub and Tavily
3. **Sentiment Extractor**: Analyzes news sentiment using FinBERT
4. **Metrics Extractor**: Gathers financial metrics with multi-API fallback
5. **Research Compiler**: Generates comprehensive PDF reports with visualizations

## Tech Stack

### Backend
- **LangGraph & LangChain**: Agent orchestration and workflow management
- **FastAPI**: REST API server
- **Python 3.12+**: Core programming language
- **Ollama**: Local LLM inference (llama3.1)
- **OpenAI GPT-4**: Advanced language processing
- **FinBERT**: Financial sentiment analysis
- **BeautifulSoup**: Web scraping

### Frontend
- **React 18**: Modern UI framework
- **Vite**: Fast build tool
- **Axios**: HTTP client
- **React Markdown**: Markdown rendering
- **Streamlit**: Alternative chatbot interface

### Data Sources
- **Finnhub**: News and company data
- **Alpha Vantage**: Financial metrics
- **Yahoo Finance**: Backup metrics and charts
- **Tavily**: Web search and news aggregation

### DevOps
- **Docker & Docker Compose**: Containerization
- **MongoDB**: Data storage (optional)
- **Apache Airflow**: ETL pipeline orchestration (optional)

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- Ollama installed locally
- API keys for:
  - OpenAI
  - Finnhub
  - Alpha Vantage
  - Tavily

### 1. Clone and Setup Environment

```bash
git clone https://github.com/choiwab/agentic-investment-research-analyst.git
cd agentic-investment-research-analyst

# Create virtual environment
python -m venv agenticenv
source agenticenv/bin/activate  # On Windows: agenticenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
OPENAI_API_KEY=your_openai_key
FINNHUB_API_KEY=your_finnhub_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
TAVILY_API_KEY=your_tavily_key
FASTAPI_BASE_URL=http://localhost:8000
```

### 3. Start Ollama

```bash
ollama serve
# In another terminal
ollama pull llama3.1
```

### 4. Start Backend API

```bash
export PYTHONPATH="${PWD}/backend/app:${PYTHONPATH}"
python -m uvicorn backend.app.api_controller:app --reload
```

The API will be available at [http://localhost:8000](http://localhost:8000)

### 5. Choose Your Frontend

#### Option A: React Frontend (Recommended)

```bash
cd frontend-react
npm install
cp .env.example .env
# Edit .env and set VITE_API_URL=http://localhost:8000
npm run dev
```

Access at [http://localhost:3000](http://localhost:3000)

#### Option B: Streamlit Chatbot

```bash
streamlit run frontend/equity_research_chatbot.py
```

Access at [http://localhost:8501](http://localhost:8501)

### Using the Launch Script

For convenience, use the React launcher:

```bash
chmod +x launch_react.sh
./launch_react.sh
```

## Usage Examples

### Company Analysis
```
"Analyze Tesla stock"
"Give me an investment report for Apple"
"Should I invest in Microsoft?"
```

### Market Insights
```
"What are current inflation trends?"
"How is the tech sector performing?"
"Analyze S&P 500 trends"
```

### Educational Queries
```
"What is P/E ratio?"
"Explain earnings per share"
"How does market cap work?"
```

## API Endpoints

### Core Endpoints
- `POST /api/research` - Main research endpoint
- `GET /api/tickers` - Get available tickers
- `GET /api/company/{ticker}` - Get company info
- `GET /health` - Health check

### Example Request

```bash
curl -X POST http://localhost:8000/api/research \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze Apple stock"}'
```

## Project Structure

```
agentic-investment-research-analyst/
├── backend/
│   └── app/
│       ├── agents/              # LangGraph agent implementations
│       │   ├── preprocessor.py
│       │   ├── news_scraper.py
│       │   ├── sentiment_extractor.py
│       │   ├── metric_extractor.py
│       │   ├── research_compiler.py
│       │   └── equity_research_graph.py
│       ├── api_controller.py    # FastAPI routes
│       └── main.py
├── frontend-react/              # React web application
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   └── App.jsx
│   └── package.json
├── frontend/                    # Streamlit chatbot
│   └── equity_research_chatbot.py
├── etl/                         # Airflow ETL pipelines
├── research_outputs/            # Generated PDF reports
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Docker orchestration
└── .env.example                 # Environment template
```

## Configuration

### Model Configuration

Edit model settings in [backend/app/agents/equity_research_graph.py](backend/app/agents/equity_research_graph.py):

```python
# Use OpenAI GPT-4
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Or use Ollama (local)
llm = ChatOllama(model="llama3.1", temperature=0)
```

### Sentiment Analysis

Configure sentiment thresholds in [backend/app/agents/config/sentiment_config.py](backend/app/agents/config/sentiment_config.py)

## Deployment

### Docker Deployment

```bash
# React frontend + Backend
docker-compose -f docker-compose.react.yml up -d

# Full stack with all services
docker-compose up -d
```

### Cloud Deployment

#### Backend (Railway/Render)
1. Push to GitHub
2. Connect repository to Railway/Render
3. Set environment variables
4. Deploy with `uvicorn backend.app.api_controller:app --host 0.0.0.0 --port $PORT`

#### Frontend (Vercel/Netlify)
```bash
cd frontend-react
npm run build
# Deploy dist/ folder
```

## Development

### Running Tests

```bash
# Backend tests
pytest backend/tests/

# Frontend tests
cd frontend-react
npm test
```

### Code Formatting

```bash
# Python
black backend/
flake8 backend/

# JavaScript
cd frontend-react
npm run lint
```

## Troubleshooting

### Common Issues

#### "Module not found" Error
```bash
export PYTHONPATH="${PWD}/backend/app:${PYTHONPATH}"
```

#### "Cannot connect to Ollama"
```bash
ollama serve
ollama pull llama3.1
```

#### CORS Errors
Ensure backend CORS settings in [backend/app/api_controller.py](backend/app/api_controller.py) include your frontend URL.

#### API Rate Limits
The system has built-in fallback mechanisms. If one API fails, it automatically tries alternatives.

### Debug Mode

Run FastAPI with debug logging:
```bash
LOG_LEVEL=debug python -m uvicorn backend.app.api_controller:app --reload
```

## Performance

- **First query**: ~25-30 seconds (model loading)
- **Subsequent queries**: ~10-15 seconds
- **Educational queries**: ~5 seconds
- **Full company analysis**: ~20-30 seconds

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Recent Updates

- ✅ Added Yahoo Finance API fallback with deep merge logic
- ✅ Improved ticker extraction with refined regex patterns
- ✅ Enhanced sentiment analysis with insider_sentiment priority
- ✅ Implemented multi-source data aggregation
- ✅ Added comprehensive error handling and retry mechanisms

## Roadmap

- [ ] Add support for portfolio analysis
- [ ] Implement real-time stock price tracking
- [ ] Add comparison analysis between companies
- [ ] Integrate technical analysis indicators
- [ ] Support for international markets
- [ ] Mobile app development

## License

MIT License - see [LICENSE](LICENSE) for details

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain)
- Sentiment analysis powered by [FinBERT](https://huggingface.co/ProsusAI/finbert)
- Financial data from Finnhub, Alpha Vantage, and Yahoo Finance

## Support

For issues or questions:
- Open an issue on [GitHub Issues](https://github.com/choiwab/agentic-investment-research-analyst/issues)
- Check existing documentation in [START_HERE.md](START_HERE.md)
- Review [DEVELOPERS.md](DEVELOPERS.md) for technical details

---

**Made with AI for AI-powered equity research**
